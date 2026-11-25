// Copyright (c) 2026, Lu Lu
#pragma once

#include <cstddef>
#include <cstdint>

#include <deep_ep/common/compiled.hpp>
#if defined(__CCE__) || defined(__CCE_IS_AICORE__)
#include <deep_ep/comm/transport.hpp>
#endif

namespace deep_ep {
namespace comm {
struct CommArgs {
    int rank = 0;  // attr rank_id, global rank
    int localRank = -1;
    int rankSize = 0;        // global rank size
    int localRankSize = -1;  // 此参数是指fullmesh互联的卡数
    uint32_t extraFlag = 0;  // legacy collective flag map; unused by LL
    __gm__ uint8_t* peerMems[DISPATCH_DEDUP_MAX_RANK_SIZE] = {};  // 传入初始化获得的buff，所有allreduce都是同一个参数

    // Legacy collective fields. LL never writes or reads them, but the
    // 256-rank CommArgs size and offsets are a frozen Host/device ABI.
    __gm__ uint8_t* creditMems[DISPATCH_DEDUP_MAX_RANK_SIZE] = {};
    int64_t sendCountMatrix[DISPATCH_DEDUP_MAX_RANK_SIZE * DISPATCH_DEDUP_MAX_RANK_SIZE] = {};  // for all2allv
    int64_t dfx[DFX_COUNT] = {};
    __gm__ uint8_t* dumpAddr = nullptr;
    int32_t magics[DISPATCH_DEDUP_MAX_RANK_SIZE] = {0};

    uint64_t fftsVal = 0;
    __gm__ uint8_t* udmaInfoPtr = nullptr;
    __gm__ uint8_t* udmaRegistryPtr = nullptr;
    __gm__ uint8_t* sdmaWorkspacePtr = nullptr;
    __gm__ uint8_t* udmaFullmeshPtr = nullptr;
    uint64_t udmaRegistrationGeneration = 0U;
    uint64_t peerMemBytes = 0U;
    int32_t commDomain = 0;
    uint32_t reserved0 = 0U;  // LL v4: physical ranks per host (production sets 16).
    // LL ABI v4: RDMA uses addresses[0:2] for symmetric workspace/staging,
    // config[0:2] for backend/QP count. MTE and other paths keep this tail zero.
    __gm__ uint8_t* reservedAddresses[4] = {};
    uint32_t reservedConfig[2] = {};
};

static_assert(sizeof(CommArgs) == 529944U, "256-rank communication ABI; rebuild Host and kernel together");
static_assert(offsetof(CommArgs, reservedAddresses) == 529904U, "reserved ABI tail offset");

}  // namespace comm
}  // namespace deep_ep

namespace deep_ep {
using comm::CommArgs;

#if defined(__CCE__) || defined(__CCE_IS_AICORE__)
namespace comm {
// Combine/clock build the same device view dispatch receives explicitly.
__aicore__ inline TransportView MakeTransportView(const __gm__ CommArgs* args)
{
    TransportView view{};
    view.peer_words = reinterpret_cast<__gm__ const uint64_t*>(&args->peerMems[0]);
    view.rdma_remote_base = args->reservedAddresses[0];
    view.rdma_stage_base = args->reservedAddresses[1];
    view.rank = uint32_t(args->rank);
    view.num_ranks = uint32_t(args->rankSize);
    view.ranks_per_host = args->reserved0 ? uint32_t(args->reserved0) : kNumRanksPerHost;
    view.stage_block_num = uint32_t(args->reservedConfig[1]);
    view.backend = uint32_t(args->reservedConfig[0]);
    view.peer_mem_bytes = uint64_t(args->peerMemBytes);
    return view;
}
}  // namespace comm
#endif
}  // namespace deep_ep
