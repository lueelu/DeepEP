// Copyright (c) 2026, Lu Lu
// Modified by SimpleBrightMan 2026

#pragma once

#include "kernels/mega_moe_urma/mega_moe_urma_tiling.hpp"

namespace MegaMoeUrma {

// Pure host/device scheduling arithmetic. Compute groups contain at most M256
// rows of one expert, independently of the peer/CV communication boundaries.
constexpr uint32_t kComputeCores = kAic;
// Minimum reducer count, used only to size UB for the largest per-core quota.
constexpr uint32_t kReduceWorkers = kLocalWorkers - kCommunicationWorkers;
constexpr uint32_t kComputeTileM = 256U;
constexpr uint32_t kComputeTileN = 256U;
// Routed GMM1 logical columns are paired gate128/up128 in one physical N256
// result. Shared experts keep their separate planar N256 gate/up projections.
constexpr uint32_t kGmm1LogicalTileN = 128U;
// Keep matrix scheduling at M256; only the AIV1 epilogue works in M128
// slices. The W4 AIV0 prologue defines its input/output banks independently.
constexpr uint32_t kActivationTileM = 128U;
// The H3584 specialization has 112 E8M0 bytes per input row. GM stays
// compact while DMA uses 32-byte-aligned rows in phase-private activation UB.
constexpr uint32_t kInputScaleBytes = 3584U / 32U;
constexpr uint32_t kInputScaleUbStride = ((kInputScaleBytes + 31U) / 32U) * 32U;
constexpr uint32_t kInputScaleUbBytes = kComputeTileM * kInputScaleUbStride;
// The platform exposes 248KiB of usable UB on a nominal 256KiB device.
// Reserve the highest usable 8KiB for persistent peer counters, not scratch.
constexpr uint32_t kRequiredUbBytes = 248U * 1024U;
constexpr uint32_t kAiv1PersistentUbBase = 240U * 1024U;
static_assert(kAiv1PersistentUbBase + 2U * kMaxPeerChunks * sizeof(uint32_t) == kRequiredUbBytes);
struct ActivationUbContract {
    static constexpr uint32_t elements = kActivationTileM * kComputeTileN;
    static constexpr uint32_t gateOffset = 0U;
    static constexpr uint32_t upOffset = elements * sizeof(uint16_t);
    static constexpr uint32_t inputEnd = upOffset + elements * sizeof(uint16_t);
    // Gate/up are dead before quantization reuses their storage.
    static constexpr uint32_t activationOffset = 0U;
    static constexpr uint32_t quantOffset = elements * sizeof(uint16_t);
    static constexpr uint32_t scaleElements = elements / 32U;
    static constexpr uint32_t scaleOffset = quantOffset + elements;
    static constexpr uint32_t maxExpOffset = scaleOffset + scaleElements;
    static constexpr uint32_t reciprocalOffset = maxExpOffset + scaleElements * sizeof(uint16_t);
    static constexpr uint32_t scratchEnd = reciprocalOffset + scaleElements * sizeof(uint16_t);
};
static_assert(ActivationUbContract::inputEnd == 128U * 1024U);
static_assert(ActivationUbContract::scratchEnd == 101U * 1024U);
static_assert(ActivationUbContract::inputEnd <= kRequiredUbBytes &&
              ActivationUbContract::scratchEnd <= kRequiredUbBytes);
static_assert(kInputScaleUbBytes == 32768U && kInputScaleUbBytes <= ActivationUbContract::inputEnd);

struct ActivationRowSlice {
    uint32_t rowBegin;
    uint32_t rows;
};

struct ComputeBatchRange {
    uint32_t begin;
    uint32_t end;
};

#if defined(__NPU_DEVICE__)
#define MEGAMOE_URMA_COMPUTE_INLINE __aicore__ inline
#else
#define MEGAMOE_URMA_COMPUTE_INLINE inline
#endif

MEGAMOE_URMA_COMPUTE_INLINE constexpr uint32_t CommunicationWorkers(uint32_t world)
{
    return world - 1U;
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr uint32_t ActivationWorkers(uint32_t world)
{
    return kAic - CommunicationWorkers(world);
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr uint32_t ReduceWorkers(uint32_t world)
{
    return kLocalWorkers - CommunicationWorkers(world);
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr uint32_t LocalWorkerIndex(uint32_t block, uint32_t subblock)
{
    return 2U * block + subblock;
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr uint64_t LocalOwnedRows(uint64_t rows, uint32_t worker)
{
    return worker < rows ? (rows - 1U - worker) / kLocalWorkers + 1U : 0U;
}

// Contiguous partitions preserve the old strided schedule's per-worker count,
// including empty workers and remainder rows. Pack can therefore DMA
// their route metadata in batches without changing timer payload accounting.
MEGAMOE_URMA_COMPUTE_INLINE constexpr uint64_t LocalRowBegin(uint64_t rows, uint32_t worker)
{
    const uint64_t remainder = rows % kLocalWorkers;
    return static_cast<uint64_t>(worker) * (rows / kLocalWorkers) + (worker < remainder ? worker : remainder);
}

// Peer owners never reduce. All AIV0 and all noncommunication AIV1 workers
// enter Reduce independently after releasing their compute resources.
// Return ReduceWorkers(world) as the sentinel for a communication worker.
MEGAMOE_URMA_COMPUTE_INLINE constexpr uint32_t ReduceWorkerIndex(uint32_t block, uint32_t subblock,
                                                                 uint32_t world = kWorld)
{
    const uint32_t communicationWorkers = CommunicationWorkers(world);
    return subblock == 0U
               ? block
               : (block >= communicationWorkers ? kAic + block - communicationWorkers : ReduceWorkers(world));
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr bool UseLegacyReduce(const Config& config)
{
    return config.hidden == 3584U && config.topk == 8U && config.bs <= 8192U && config.maxPeerChunks <= 512U;
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr uint64_t ReduceOwnedRows(uint64_t rows, uint32_t worker, uint32_t world = kWorld)
{
    const uint32_t workers = ReduceWorkers(world);
    return worker < workers && worker < rows ? (rows - 1U - worker) / workers + 1U : 0U;
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr uint64_t ReduceRowBegin(uint64_t rows, uint32_t worker, uint32_t world = kWorld)
{
    const uint32_t workers = ReduceWorkers(world);
    const uint64_t remainder = rows % workers;
    return static_cast<uint64_t>(worker) * (rows / workers) + (worker < remainder ? worker : remainder);
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr bool IsSelfExpandedRow(uint64_t row, const Config& config)
{
    return row >= config.selfSendRow && row - config.selfSendRow < config.selfRows;
}

// Call only for a row in the self interval. Stable expert ordering is identical
// on both sides, so the inverse transform also maps self GMM2 rows to Reduce.
MEGAMOE_URMA_COMPUTE_INLINE constexpr uint64_t SelfReceiveRow(uint64_t expandedRow, const Config& config)
{
    return config.selfRecvRow + (expandedRow - config.selfSendRow);
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr uint64_t SelfReturnRow(uint64_t receivedRow, const Config& config)
{
    return config.selfSendRow + (receivedRow - config.selfRecvRow);
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr uint64_t ComputeBatchTaskCount(uint32_t batchCount)
{
    return static_cast<uint64_t>(batchCount) * 2U;
}

// Legacy schedule, retained as the host-testable zero-lookahead reference.
// The device consumes the explicit host-planned task array instead. A batch
// contains all M256 groups of one expert, including empty experts.
MEGAMOE_URMA_COMPUTE_INLINE constexpr ComputeBatchTask ComputeBatchTaskAt(uint64_t step, uint32_t batchCount)
{
    const uint64_t taskCount = ComputeBatchTaskCount(batchCount);
    if (step >= taskCount) return {ComputeBatchStage::None, 0U};
    if (step == 0U) return {ComputeBatchStage::Gmm1, 0U};
    if (step == taskCount - 1U) return {ComputeBatchStage::Gmm2, batchCount - 1U};
    if ((step & 1U) != 0U) return {ComputeBatchStage::Gmm1, static_cast<uint32_t>((step + 1U) / 2U)};
    return {ComputeBatchStage::Gmm2, static_cast<uint32_t>(step / 2U - 1U)};
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr ComputeBatchRange ComputeBatchFragments(const ComputeBatch& batch)
{
    return {batch.fragmentBegin, batch.fragmentBegin + batch.fragmentCount};
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr ActivationRowSlice ActivationSlice(uint32_t tileRowBegin, uint32_t tileRows,
                                                                         uint32_t offset)
{
    const uint32_t remaining = offset < tileRows ? tileRows - offset : 0U;
    return {tileRowBegin + offset, remaining < kActivationTileM ? remaining : kActivationTileM};
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr uint32_t ComputeTileCount(uint32_t rows, uint32_t columns)
{
    return ((rows + kComputeTileM - 1U) / kComputeTileM) * ((columns + kComputeTileN - 1U) / kComputeTileN);
}

MEGAMOE_URMA_COMPUTE_INLINE constexpr uint32_t Gmm1ComputeTileCount(uint32_t rows, const Config& c)
{
    return ((rows + kComputeTileM - 1U) / kComputeTileM) *
           ((c.intermediate + kGmm1LogicalTileN - 1U) / kGmm1LogicalTileN);
}

// All stage task IDs are global prefix sums. Sparse experts cannot repeatedly
// give the same physical core their first tile. AIC and its AIV0 call this with
// identical inputs; the AIV1 activation team uses ActivationWorkers(world).
MEGAMOE_URMA_COMPUTE_INLINE constexpr uint32_t FirstOwnedComputeTile(uint64_t tileBegin, uint32_t worker,
                                                                     uint32_t workerCount)
{
    const uint32_t startOwner = tileBegin % workerCount;
    return worker >= startOwner ? worker - startOwner : worker + workerCount - startOwner;
}

// Physical core -> forward tile-owner residue. GMM1 balances the opposite
// end of the core array; GMM2 retains its ascending owner assignment.
MEGAMOE_URMA_COMPUTE_INLINE constexpr uint32_t RoutedOwnerResidue(bool gmm1, uint32_t core)
{
    return gmm1 ? kAic - 1U - core : core;
}

struct OwnedComputeGroup {
    uint32_t group;
    uint32_t firstTile;
    uint32_t tileCount;
};

// ownedTile is a batch-relative tile offset, starting at FirstOwnedComputeTile.
// All nonempty M256 groups have the same N tile count, even with a short M tail.
// Advance by tileCount * workerCount to visit only this worker's next group;
// consuming all owned tiles at once also supports groups wider than workerCount.
// Call only with nonzero tilesPerGroup/workerCount and an in-range ownedTile.
MEGAMOE_URMA_COMPUTE_INLINE constexpr OwnedComputeGroup OwnedComputeGroupAt(uint64_t ownedTile, uint32_t tilesPerGroup,
                                                                            uint32_t workerCount)
{
    const uint32_t group = static_cast<uint32_t>(ownedTile / tilesPerGroup);
    const uint32_t firstTile = ownedTile % tilesPerGroup;
    return {group, firstTile, (tilesPerGroup - 1U - firstTile) / workerCount + 1U};
}

// Each nonempty fragment is at most M256, so all groups of an expert have the
// same N tile count, including its short final M group. Return groupCount when
// this worker has no tile; no hardware context may be opened in that case.
MEGAMOE_URMA_COMPUTE_INLINE constexpr uint32_t LastOwnedComputeGroup(uint64_t tileBegin, uint32_t groupCount,
                                                                     uint32_t tilesPerGroup, uint32_t worker,
                                                                     uint32_t workerCount)
{
    if (groupCount == 0U || tilesPerGroup == 0U) return groupCount;
    const uint64_t tileCount = static_cast<uint64_t>(groupCount) * tilesPerGroup;
    const uint64_t first = FirstOwnedComputeTile(tileBegin, worker, workerCount);
    if (first >= tileCount) return groupCount;
    const uint64_t last = first + ((tileCount - 1U - first) / workerCount) * workerCount;
    return static_cast<uint32_t>(last / tilesPerGroup);
}

#undef MEGAMOE_URMA_COMPUTE_INLINE
}  // namespace MegaMoeUrma
