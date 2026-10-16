// Copyright (c) 2026, Lu Lu
#pragma once

// A descriptor is {source VA, target VA, immutable header UB address, reserved}.
// Prepare addresses once in route order; the SIMT loop contains no mode/peer
// branches and builds all eight WQE words without repeating SDK lookups.
__simt_vf__ __launch_bounds__(32) inline void BuildCombinePutWqesVf(__ubuf__ uint64_t* output,
                                                                    __ubuf__ uint64_t* descriptors, uint32_t count,
                                                                    uint32_t bytes)
{
    for (uint32_t i = threadIdx.x; i < count; i += 32U) {
        auto* descriptor = descriptors + uint64_t(i) * 4U;
        auto* header = reinterpret_cast<__ubuf__ uint64_t*>(descriptor[2]);
        auto* wqe = output + uint64_t(i) * 8U;
        wqe[0] = header[0];
        wqe[1] = header[1];
        wqe[2] = header[2];
        wqe[3] = header[3];
        wqe[4] = header[4];
        wqe[5] = descriptor[1];
        wqe[6] = bytes;
        wqe[7] = descriptor[0];
    }
}

__aicore__ inline void BuildCombinePutWqes(__ubuf__ uint8_t* output, __ubuf__ uint64_t* descriptors, uint32_t count,
                                           uint32_t bytes)
{
    if (!count) return;
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    asc_vf_call<BuildCombinePutWqesVf>(dim3(32), reinterpret_cast<__ubuf__ uint64_t*>(output), descriptors, count,
                                       bytes);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
}

// Per cross lane: 2 KiB WQEs, 1 KiB descriptors, 4 KiB header cache,
// then the existing pressure SO scratch. All fit the existing 8 KiB lane.
constexpr uint32_t kCrossDescriptorsOffset = 2048U;
constexpr uint32_t kCrossHeadersOffset = 3072U;
constexpr uint32_t kCrossHeaderCapacity = 64U;
static_assert(kCrossHeadersOffset + kCrossHeaderCapacity * 64U <= kWqeBatchScratchBytes - 512U);

__aicore__ inline __ubuf__ uint64_t* CrossPutHeader(__ubuf__ uint8_t* scratch, uint32_t lane, uint32_t index)
{
    return reinterpret_cast<__ubuf__ uint64_t*>(scratch + lane * kWqeBatchScratchBytes + kCrossHeadersOffset +
                                                index * 64U);
}

__aicore__ inline void CacheCombinePutHeader(__ubuf__ uint64_t* header, GM_ADDR workspace, uint32_t peer, uint32_t qp)
{
    auto* info = aclshmemi_udma_qp_info_fetch();
    const uint32_t slot = aclshmemi_udma_compute_slot(peer, static_cast<uint32_t>(-1));
    auto* remote = reinterpret_cast<__gm__ uint8_t*>(aclshmem_ptr(workspace, peer));
    aclshmemi_udma_fill_sqe_base_ctx<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE, __ubuf__ aclshmemi_sqe_ctx_t*,
                                     kPayloadRoNoCqeConfig>(reinterpret_cast<__ubuf__ aclshmemi_sqe_ctx_t*>(header),
                                                            remote, aclshmemi_udma_get_mem_info(info, slot, qp), 0U);
    header[7] = reinterpret_cast<uint64_t>(remote);
}

// Local URMA keeps the existing 128-row submit quantum and its single tail
// CQE. Reuse the existing 16 KiB scratch; gather metadata stays in copy UB.
constexpr uint32_t kLocalDescriptorsOffset = 8192U;
constexpr uint32_t kLocalHeaderOffset = 12288U;
static_assert(kLocalDescriptorsOffset + 128U * 4U * sizeof(uint64_t) == kLocalHeaderOffset);
static_assert(kLocalHeaderOffset + 128U <= kWqeScratchBytes);
struct LocalPutBatch {
    GM_ADDR workspace = nullptr;
    __ubuf__ uint8_t* scratch = nullptr;
    __gm__ aclshmemi_udma_wq_ctx_t* context = nullptr;
    uint32_t peer = 0U, slot = 0U, count = 0U, bytes = 0U;
    bool strong_tail = false;
    __aicore__ inline void Init(GM_ADDR base, uint32_t target, uint32_t row_bytes, __ubuf__ uint8_t* ub,
                                bool require_strong_tail = false)
    {
        workspace = base;
        peer = target;
        bytes = row_bytes;
        scratch = ub;
        strong_tail = require_strong_tail;
    }

    __aicore__ inline bool PrepareContext()
    {
        if (context) return true;
        auto* info = aclshmemi_udma_qp_info_fetch();
        slot = aclshmemi_udma_compute_slot(peer, static_cast<uint32_t>(-1));
        context = aclshmemi_udma_get_qp_ctx(info, slot, 0U);
        if (!context || context->wqe_size != kWqeBytes || context->depth != shm::UDMA_SQ_BASKBLK_CNT) {
            aclshmemi_kernel_abort("PUT combine local SQ ABI mismatch\n");
            return false;
        }
        CacheCombinePutHeader(reinterpret_cast<__ubuf__ uint64_t*>(scratch + kLocalHeaderOffset), workspace, peer, 0U);
        return true;
    }

    __aicore__ inline void FlushRows(__ubuf__ uint64_t* rows, uint32_t row_count)
    {
        if (!row_count) return;
        auto* tail_header = reinterpret_cast<__ubuf__ uint64_t*>(scratch + kLocalHeaderOffset + 64U);
        // EP8 return batches close every RO prefix with SO + CQE. This also
        // covers intermediate credit checkpoints, not only the window tail.
        // Other callers retain their upstream tail configuration.
        auto* destination = reinterpret_cast<__gm__ uint8_t*>(rows[(row_count - 1U) * 4U + 1U]);
        auto* memory = aclshmemi_udma_get_mem_info(aclshmemi_udma_qp_info_fetch(), slot, 0U);
        if (strong_tail) {
            aclshmemi_udma_fill_sqe_base_ctx<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE,
                                             __ubuf__ aclshmemi_sqe_ctx_t*, kCompletionSoConfig>(
                reinterpret_cast<__ubuf__ aclshmemi_sqe_ctx_t*>(tail_header), destination, memory, row_count - 1U);
        } else {
            aclshmemi_udma_fill_sqe_base_ctx<uint8_t, aclshmemi_udma_opcode_t::UDMA_OP_WRITE,
                                             __ubuf__ aclshmemi_sqe_ctx_t*, kPayloadRoConfig>(
                reinterpret_cast<__ubuf__ aclshmemi_sqe_ctx_t*>(tail_header), destination, memory, row_count - 1U);
        }
        rows[(row_count - 1U) * 4U + 2U] = reinterpret_cast<uint64_t>(tail_header);
        BuildCombinePutWqes(scratch, rows, row_count, bytes);
        aclshmemi_udma_flush_aggregate_wqes(scratch, context, slot, 0U, EVENT_ID0, row_count, kWqeBytes);
    }

    __aicore__ inline void Flush()
    {
        FlushRows(reinterpret_cast<__ubuf__ uint64_t*>(scratch + kLocalDescriptorsOffset), count);
        count = 0U;
    }

    __aicore__ inline void Add(GM_ADDR destination, GM_ADDR source)
    {
        auto* header = reinterpret_cast<__ubuf__ uint64_t*>(scratch + kLocalHeaderOffset);
        if (!PrepareContext()) return;
        auto* row = reinterpret_cast<__ubuf__ uint64_t*>(scratch + kLocalDescriptorsOffset) + count * 4U;
        row[0] = reinterpret_cast<uint64_t>(source);
        row[1] = header[7] + uint64_t(destination - workspace);
        row[2] = reinterpret_cast<uint64_t>(header);
        if (++count == 128U) Flush();
    }
};
