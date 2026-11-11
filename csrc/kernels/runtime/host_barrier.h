// DeepEP-owned NDR Host barrier. Compiled into libdeepep_kernels, never SHMEM.
// Uses the locked SDK's existing transport APIs without modifying the SDK.
// Host callers must serialize this collective with communication kernels.
ACLSHMEM_DEVICE void deepep_barrier_store(__gm__ int32_t* address, int32_t value)
{
    AscendC::GlobalTensor<int32_t> signal;
    signal.SetGlobalBuffer(address);
    signal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    signal.SetValue(0, value);
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                    AscendC::DcciDst::CACHELINE_OUT>(signal);
}

ACLSHMEM_DEVICE void deepep_barrier_wait(__gm__ int32_t* address, int32_t generation,
                                       uint64_t wait_cycles, int rank, int predecessor, int slot)
{
    AscendC::GlobalTensor<int32_t> signal;
    signal.SetGlobalBuffer(address);
    signal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    const uint64_t begin = AscendC::GetSystemCycle();
    while (true) {
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                        AscendC::DcciDst::CACHELINE_OUT>(signal);
        const int32_t value = signal.GetValue(0);
        if (value == generation || value == generation + 1) return;
        if (static_cast<uint64_t>(AscendC::GetSystemCycle()) - begin >= wait_cycles) {
            AscendC::printf("[DeepEP barrier TIMEOUT] rank=%d predecessor=%d slot=%d expected=%d observed=%d\n",
                            rank, predecessor, slot, generation, value);
            AscendC::Trap();
            return;
        }
    }
}

ACLSHMEM_DEVICE void deepep_host_barrier(aclshmem_team_t tid, uint64_t wait_cycles = 120000000000ULL,
                                      bool trace = false)
{
    if (tid == -1) return;
    auto state = aclshmemi_get_state();
    auto team = state->team_pools[tid];
    const int me = state->team_pools[ACLSHMEM_TEAM_WORLD]->mype;
    if ASCEND_IS_AIV {
        if (trace) AscendC::printf("[DeepEP barrier] rank=%d vec=%u sub=%u entry team_size=%d\n",
                                  me, AscendC::GetBlockIdx(), AscendC::GetSubBlockIdx(), team->size);
    }
    bool remote = false;
    for (int i = 0; i < team->size; ++i) {
        const int peer = team->pe_mapping[i];
        if (!(state->topo_list[peer] & ACLSHMEM_TRANSPORT_MTE)) remote = true;
    }
    if (!remote) {
        if ASCEND_IS_AIV {
            if (trace) AscendC::printf("[DeepEP barrier] rank=%d vec=%u all_mte enter\n", me, AscendC::GetBlockIdx());
        }
        aclshmemi_sync<false>(tid);
        if ASCEND_IS_AIV {
            if (trace) AscendC::printf("[DeepEP barrier] rank=%d vec=%u all_mte return\n", me, AscendC::GetBlockIdx());
        }
        return;
    }

    // All participating cores rendezvous, but only AIV0 owns the generation,
    // QPs and SDK UB scratch. AIC and the second AIV must not issue RDMA.
    aclshmemi_sync_core<false>();
    if ASCEND_IS_AIV {
        if (trace) AscendC::printf("[DeepEP barrier] rank=%d vec=%u entry_core_sync return\n", me, AscendC::GetBlockIdx());
        if (AscendC::GetBlockIdx() == 0) {
            auto pool = aclshmemi_get_team_sync_pool(tid);
            auto counter = aclshmemi_get_team_sync_counter(tid);
            auto scratch = reinterpret_cast<__ubuf__ uint8_t*>(state->rdma_config.aclshmem_ub);
            const uint32_t event = state->rdma_config.sync_id;
            int index = -1;
            for (int i = 0; i < team->size; ++i) {
                const int peer = team->pe_mapping[i];
                if (peer == me) index = i;
                if (!(state->topo_list[peer] & ACLSHMEM_TRANSPORT_MTE)) {
                    if (!(state->topo_list[peer] & ACLSHMEM_TRANSPORT_ROCE)) {
                        AscendC::Trap();
                        return;
                    }
                    // Complete prior writes on every per-peer QP before arrival.
                    const uint32_t qps = aclshmemi_qp_info_fetch()->qp_num;
                    if (trace) AscendC::printf("[DeepEP barrier] rank=%d drain peer=%d qps=%u enter\n", me, peer, qps);
                    for (uint32_t qp = 0; qp < qps; ++qp)
                        aclshmemx_roce_qp_quiet(peer, qp, scratch, event);
                    if (trace) AscendC::printf("[DeepEP barrier] rank=%d drain peer=%d return\n", me, peer);
                }
            }
            if (index < 0) { AscendC::Trap(); return; }
            dcci_cacheline(reinterpret_cast<__gm__ uint8_t*>(counter));
            const int32_t previous = aclshmemi_load(reinterpret_cast<__gm__ int32_t*>(counter));
            // The SDK polling convention also accepts generation+1.
            if (previous >= INT32_MAX - 1) { AscendC::Trap(); return; }
            const int32_t generation = previous + 1;
            deepep_barrier_store(reinterpret_cast<__gm__ int32_t*>(counter), generation);

            // Binary dissemination: one cacheline per round, always local polling.
            // Mapping, rather than start/stride arithmetic, supports explicit teams.
            int slot = 0;
            for (int shift = 1; shift < team->size; shift <<= 1, ++slot) {
                if (slot >= SYNC_ARRAY_SIZE / ACLSHMEM_SYNCBIT_SIZE) { AscendC::Trap(); return; }
                const int peer = team->pe_mapping[(index + shift) % team->size];
                const int predecessor = team->pe_mapping[(index - shift + team->size) % team->size];
                if (trace) AscendC::printf("[DeepEP barrier] rank=%d slot=%d generation=%d publish peer=%d transport=%u\n",
                                          me, slot, generation, peer, state->topo_list[peer]);
                if (state->topo_list[peer] & ACLSHMEM_TRANSPORT_MTE) {
                    deepep_barrier_store(reinterpret_cast<__gm__ int32_t*>(aclshmem_ptr(pool + slot, peer)), generation);
                } else {
                    // Pass LOCAL symmetric addresses to the SDK for translation.
                    // No RDMA get, remote Scalar dereference or remote atomic.
                    aclshmemx_roce_qp_put_nbi(
                        reinterpret_cast<__gm__ uint8_t*>(pool + slot),
                        reinterpret_cast<__gm__ uint8_t*>(counter), scratch, sizeof(int32_t), peer, 0, event);
                    aclshmemx_roce_qp_quiet(peer, 0, scratch, event);
                }
                if (trace) AscendC::printf("[DeepEP barrier] rank=%d slot=%d publish complete; wait predecessor=%d\n", me, slot, predecessor);
                deepep_barrier_wait(reinterpret_cast<__gm__ int32_t*>(pool + slot), generation, wait_cycles, me, predecessor, slot);
                if (trace) AscendC::printf("[DeepEP barrier] rank=%d slot=%d wait complete\n", me, slot);
            }
        }
    }
    if ASCEND_IS_AIV {
        if (trace) AscendC::printf("[DeepEP barrier] rank=%d vec=%u exit_core_sync enter\n", me, AscendC::GetBlockIdx());
    }
    aclshmemi_sync_core<false>();
    if ASCEND_IS_AIV {
        if (trace) AscendC::printf("[DeepEP barrier] rank=%d vec=%u exit_core_sync return\n", me, AscendC::GetBlockIdx());
    }
}
