// Copyright (c) 2026, Lu Lu

// UBX route-plan Combine device entry shared by the Ahead-of-Time device
// library and the DeepJIT translation unit.
//
// One ordinary <<<>>> global entry and one SuperKernel sub-entry call the same
// body. The body receives its logical execution context instead of reading
// AscendC::GetBlockIdx()/GetBlockNum() directly, because inside a fused
// SuperKernel task those accessors describe the fused grid.
#pragma once

#include <cstddef>
#include <cstdint>

#include "kernel_operator.h"
#include "shmem.h"
#include "../../kernels/common/core_context.h"
#include "../launch.hpp"
#include "combine_server_dedup_kernel.h"
#include "dispatch_server_dedup_magic.h"
#include "ep_memory_server_dedup_combine_tiling.h"
#include "ubx_jit_spec.h"

namespace UbxCombineEntry {
// Mirrors the ordinary global entry's parameter list, one field per argument
// and in the same order. GM_ADDR is pointer sized, so every field is already
// naturally aligned to the four-byte ABI minimum.
struct ArgsBlock {
    GM_ADDR comm;
    GM_ADDR expert;
    GM_ADDR destination;
    GM_ADDR counts;
    GM_ADDR weights;
    GM_ADDR types;
    GM_ADDR relay;
    GM_ADDR mask;
    GM_ADDR output;
    GM_ADDR tiling;
};
static_assert(sizeof(ArgsBlock) == 10U * sizeof(GM_ADDR), "SuperKernel argument view must match the global ABI");

template <uint32_t Group>
__aicore__ inline void RunGroup(const ArgsBlock& block, deepep::CoreContext core, uint32_t bs, uint32_t h,
                                uint32_t topK, uint32_t experts, uint8_t dtype, bool dedup, bool skMode)
{
    const auto* args = reinterpret_cast<const __gm__ UbxDispatch::CommArgs*>(block.comm);
    util_set_ffts_config(args->fftsVal);
    uint32_t magic;
    {
        AscendC::TPipe pipe;
        magic = UbxDispatch::RotateDispatchServerDedupMagic(args, core.index, &pipe);
        // A SuperKernel sub-task is an ordinary function to the compiler, so
        // the implicit destructor no longer guarantees the pipeline barrier.
        if (skMode) pipe.DestroyWithoutPipeAll();
    }
    if (skMode) AscendC::PipeBarrier<PIPE_ALL>();

    const auto* td = reinterpret_cast<const volatile __gm__ UbxCombine::EpServerDedupCombineTilingData*>(block.tiling);
    // The Host filled the tiling from the same immutable description; rechecking
    // it here keeps an unsupported or stale configuration from running.
    if (td->bs != bs || td->h != h || td->topK != topK || td->moeExpertNum != experts || td->dtype != dtype ||
        td->tilingKey != Group) {
        AscendC::Trap();
        return;
    }
    if (!UbxDispatch::LlTilingKeyValid(args->rankSize, td->tilingKey) ||
        UbxDispatch::EpServerDedupEnabled(td->numMaxTokensPerRank, td->bs, args->rankSize) != dedup) {
        AscendC::Trap();
        return;
    }

    UbxCombine::CombineMemoryServerDedupKernel<Group> kernel;
    kernel.Init(block.comm, block.expert, block.destination, block.counts, block.weights, block.types, block.relay,
                block.mask, block.output, bs, h, topK, experts, dtype, magic, core, dedup);
    kernel.Process();
    if (skMode) kernel.Destroy();
}

__aicore__ inline void Run(const ArgsBlock& block, deepep::CoreContext core, bool skMode)
{
    const auto* args = reinterpret_cast<const __gm__ UbxDispatch::CommArgs*>(block.comm);
    const auto* td = reinterpret_cast<const volatile __gm__ UbxCombine::EpServerDedupCombineTilingData*>(block.tiling);

#ifdef DEEPEP_UBX_JIT_SPECIALIZED
    // Specialized kernels only run the one configuration in their cache key.
    constexpr uint32_t bs = DEEPEP_UBX_JIT_SPEC_BS;
    constexpr uint32_t h = DEEPEP_UBX_JIT_SPEC_H;
    constexpr uint32_t topK = DEEPEP_UBX_JIT_SPEC_TOPK;
    constexpr uint32_t experts = DEEPEP_UBX_JIT_SPEC_EXPERTS;
    constexpr uint32_t world = DEEPEP_UBX_JIT_SPEC_WORLD;
    constexpr uint32_t rank = DEEPEP_UBX_JIT_SPEC_RANK;
    constexpr uint32_t cores = DEEPEP_UBX_JIT_SPEC_CORES;
    constexpr uint32_t capacity = DEEPEP_UBX_JIT_SPEC_CAPACITY;
    constexpr uint8_t dtype = DEEPEP_UBX_JIT_SPEC_BF16 ? UbxDispatch::DISPATCH_DEDUP_DATA_TYPE_BFP16
                                                       : UbxDispatch::DISPATCH_DEDUP_DATA_TYPE_FP16;
    constexpr bool dedup = bool(DEEPEP_UBX_JIT_SPEC_DEDUP);
    const bool specDedup = UbxDispatch::EpServerDedupEnabled(td->numMaxTokensPerRank, td->bs, args->rankSize);
    if (uint32_t(args->rankSize) != world || uint32_t(args->rank) != rank || core.count != cores || td->bs != bs ||
        td->h != h || td->topK != topK || td->moeExpertNum != experts || td->tilingKey != DEEPEP_UBX_JIT_SPEC_GROUP ||
        td->dtype != dtype || uint32_t(td->numMaxTokensPerRank) != capacity ||
        (specDedup ? 1U : 0U) != (dedup ? 1U : 0U)) {
        AscendC::Trap();
        return;
    }
    RunGroup<DEEPEP_UBX_JIT_SPEC_GROUP>(block, core, bs, h, topK, experts, dtype, dedup, skMode);
#else
    const uint32_t bs = td->bs;
    const uint32_t h = td->h;
    const uint32_t topK = td->topK;
    const uint32_t experts = td->moeExpertNum;
    const uint8_t dtype = td->dtype;
    const bool dedup = UbxDispatch::EpServerDedupEnabled(td->numMaxTokensPerRank, bs, args->rankSize);
    switch (td->tilingKey) {
        case 2U:
            RunGroup<2U>(block, core, bs, h, topK, experts, dtype, dedup, skMode);
            break;
        case 4U:
            RunGroup<4U>(block, core, bs, h, topK, experts, dtype, dedup, skMode);
            break;
        case 16U:
            RunGroup<16U>(block, core, bs, h, topK, experts, dtype, dedup, skMode);
            break;
        default:
            AscendC::Trap();
            break;
    }
#endif
}
}  // namespace UbxCombineEntry

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ubx_combine_route_plan(
    GM_ADDR comm, GM_ADDR expert, GM_ADDR destination, GM_ADDR counts, GM_ADDR weights, GM_ADDR types, GM_ADDR relay,
    GM_ADDR mask, GM_ADDR output, GM_ADDR tiling)
{
    const UbxCombineEntry::ArgsBlock block{comm, expert, destination, counts, weights,
                                           types, relay, mask, output, tiling};
    UbxCombineEntry::Run(block, deepep::OrdinaryCoreContext(), false);
}

// Equivalent SuperKernel sub-entries. The template parameter only gives the
// framework four distinct symbols; every instance runs the same body. The JIT
// ELF carries the same ordinary entry, SK sub-entries and bind record; the SK
// capability is not coupled to the compilation route.
template <uint32_t splitNum>
__sk__ __vector__ void ubx_combine_route_plan_sk(const UbxCombineEntry::ArgsBlock* block, sk::SkSystemArgs* sysArgs)
{
    sysArgs->SkInit();
    deepep::CoreContext core{sysArgs->SkGetBlockIdx(), sysArgs->SkGetNumBlocks()};
    // Unpack the framework-provided argument block once, like the ordinary
    // entry's register arguments, instead of re-reading each field from GM
    // through the sub-function body.
    const UbxCombineEntry::ArgsBlock local = *block;
    UbxCombineEntry::Run(local, core, true);
}

// Default capability mask: no early start and no disabled DCCI. The operator
// keeps its publish/consume protocol and does not rely on the compiler's
// end-of-kernel cache flush.
SK_BIND(ubx_combine_route_plan, 0, ubx_combine_route_plan_sk<0>, ubx_combine_route_plan_sk<1>,
        ubx_combine_route_plan_sk<2>, ubx_combine_route_plan_sk<3>);
