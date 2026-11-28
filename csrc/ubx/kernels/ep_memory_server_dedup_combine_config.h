// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef COMBINE_DEDUP_EP_MEMORY_SERVER_DEDUP_COMBINE_CONFIG_H
#define COMBINE_DEDUP_EP_MEMORY_SERVER_DEDUP_COMBINE_CONFIG_H

#include <cstdint>
#include <ubx/kernels/ll_core_config.h>
#include <ubx/kernels/ll_rdma_layout.h>

#include <ubx/kernels/combine_core_layout.h>

namespace UbxCombine {
// Keep the existing vector repeat bounds on devices with larger UB.
constexpr uint32_t kEpServerDedupCombineMaxUbBytes = 240U * 1024U;
// Reserve event 7 for metadata/control; keep one more payload event spare.
constexpr uint32_t kEpServerDedupCombineM4MaxInputLanes = 6U;
constexpr uint32_t kEpServerDedupCombineM5MaxInputLanes = 7U;  // MTE2_V(7) belongs to probe
constexpr uint32_t kEpServerDedupCombineM5MaxOutputLanes = 2U;
constexpr uint32_t kEpServerDedupCombineTableRows = 512U;
// Even before metadata is reserved, one admitted input/output pair has fewer
// than 256 blocks, so packing never narrows an oversized vector repeat count.
static_assert(kEpServerDedupCombineMaxUbBytes / (480U + 512U) <= 255U, "M3 vector repeat budget");
// Prefix count, a DIRECT comparison mask and the shared routing table.
constexpr uint32_t kEpServerDedupCombineResidentUbBytes = 32U + 256U;
constexpr uint32_t kEpServerDedupCombineFixedUbBytes =
    kEpServerDedupCombineResidentUbBytes + kEpServerDedupCombineTableRows * 12U;
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define COMBINE_DEDUP_EP_COMBINE_CONSTEXPR __aicore__ inline constexpr
#else
#define COMBINE_DEDUP_EP_COMBINE_CONSTEXPR constexpr
#endif
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombineAlign(uint32_t n, uint32_t a)
{
    return (n + a - 1U) / a * a;
}
// The active count is queried on the host and agreed across ranks.
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombineRecvCoreCount(uint32_t cores)
{
    return cores / 2U;
}
// One SDK workspace per live NIC slot and one independent Quiet scratch.
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombineRdmaBytes(uint32_t h)
{
    const uint32_t packet = ((2U * h + 479U) / 480U) * 512U;
    return (UbxRdma::kCoreBytes / packet + 1U) * 128U;
}
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombineRowUbBytes(uint32_t tokens)
{
    return 3U * EpCombineAlign(tokens == 0U ? 4U : tokens * 4U, 256U) + EpCombineAlign((tokens + 7U) / 8U + 32U, 256U);
}
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombinePayloadLaneCount(uint32_t tokens, uint32_t h, uint32_t ubBytes)
{
    if (h == 0U || h > ubBytes / 2U || tokens > ubBytes / 12U) {
        return 0U;
    }
    const uint32_t fixed = kEpServerDedupCombineFixedUbBytes + EpCombineRdmaBytes(h) + EpCombineRowUbBytes(tokens);
    if (fixed >= ubBytes) return 0U;
    const uint32_t available = ubBytes - fixed;
    // M3 vector packing reads every 480B payload block, including its tail.
    // M1 uses the same input lanes and drains them before M3. Output lanes
    // remain distinct until their MTE3 consumers have completed.
    const uint32_t blocks = (h * 2U + 479U) / 480U;
    const uint32_t lanes = available / (blocks * (480U + 512U));
    return lanes > 8U ? 8U : lanes;
}
constexpr uint32_t kEpServerDedupCombineM4TableRows = 32U;
// During server selection M4's table caches and low-precision packing buffer
// are idle. Alias only that phase range, never inputs or in-flight outputs.
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombineM4ScanBytes(uint32_t rows)
{
    return rows * 12U + 2U * EpCombineAlign(rows * 4U, 256U) + 3U * EpCombineAlign((rows + 7U) / 8U, 32U);
}
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombineM4ScanRows(uint32_t h, uint32_t k)
{
    const uint32_t available =
        384U + EpCombineAlign(kEpServerDedupCombineM4TableRows * k * 8U, 32U) + ((h * 2U + 479U) / 480U) * 480U;
    uint32_t rows = 32U;
    while (rows < 512U && EpCombineM4ScanBytes(rows * 2U) <= available) rows *= 2U;
    return rows;
}
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t
EpCombineM4TaskBytes(uint32_t tokens, uint32_t recvCores = UbxDispatch::kLlDefaultPlanCoreNum / 2U)
{
    return EpCombineAlign(((tokens + recvCores - 1U) / recvCores) * 4U, 32U);
}
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombineReadyTaskBytes(uint32_t tokens, uint32_t recvCores)
{
    return EpCombineAlign(((tokens + recvCores - 1U) / recvCores) * 4U, 256U);
}
// Both caches already exist in M4. Table2 needs a padded 32B row;
// table3 needs an aligned K*8B row. Do not consume payload lane space.
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombineReadyCacheCapacity(uint32_t k)
{
    if (k == 0U) return 0U;
    const uint32_t table2 = (kEpServerDedupCombineM4TableRows * 12U) / 32U;
    const uint32_t table3 =
        EpCombineAlign(kEpServerDedupCombineM4TableRows * k * 8U, 32U) / EpCombineAlign(k * 8U, 32U);
    return table2 < table3 ? table2 : table3;
}
// M5 input events use IDs 0..6; MTE2_V(7) is reserved for probe.
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t
EpCombineM5TokenCount(uint32_t bs, uint32_t recvCores = UbxDispatch::kLlDefaultPlanCoreNum / 2U)
{
    return bs / recvCores + (bs % recvCores != 0U);
}
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombineM5FlagBytes(uint32_t h, uint32_t k)
{
    // CANN 9.1 dav_3510 ReduceSum(mask=1, stride=1) loads a full 256B
    // vector at every 32B step. Reserve its final vector beyond the flags.
    return k * ((h * 2U + 479U) / 480U) * 32U + 256U;
}
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombineM5WorkBytes(uint32_t h, uint32_t k)
{
    // The intermediate B floats are reduced with full 256B vector loads.
    return EpCombineAlign(k * ((h * 2U + 479U) / 480U) * 4U, 256U);
}
// Offsets are relative to the scratch after resident control buffers. Only
// phase metadata changes meaning; output slots stay live across M4 -> M5.
struct EpCombineM45WorkspaceLayout {
    uint32_t computeBytes = 0U, phaseOffset = 0U, phaseBytes = 0U;
    uint32_t residentBytes = 0U, scratchBytes = 0U;
    uint32_t inputOffset = 0U, outputOffset = 0U, finalInputOffset = 0U;
    uint32_t inputBytes = 0U, outputBytes = 0U;
    uint32_t inputLanes = 0U, outputLanes = 0U, finalOutputLanes = 0U;
    uint32_t totalBytes = 0U;
};
// Available only BEFORE M4 initializes or issues payload lanes, after M1/M2
// retirement. Planning finishes before the same bytes become inputs/outputs.
// No change to layout, lane counts or the M4 -> M5 live-output handoff.
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR uint32_t EpCombineRouteSnapshotBytes(const EpCombineM45WorkspaceLayout& layout)
{
    return layout.inputLanes && layout.outputLanes && layout.inputOffset < layout.scratchBytes
               ? layout.scratchBytes - layout.inputOffset
               : 0U;
}
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR EpCombineM45WorkspaceLayout EpCombineM45Layout(
    uint32_t tokens, uint32_t bs, uint32_t h, uint32_t k, uint32_t recvCores, uint32_t ubBytes)
{
    EpCombineM45WorkspaceLayout result{};
    if (recvCores == 0U || recvCores > UbxDispatch::kLlMaxRecvCoreNum ||
        bs == 0U || h == 0U || h > ubBytes / 12U || k == 0U || k > 32U)
        return result;
    const uint64_t owned = EpCombineM5TokenCount(bs, recvCores);
    const uint64_t partial = EpCombineM4TaskBytes(tokens, recvCores) +
                             128U +  // final vector packing load guard
                             EpCombineReadyTaskBytes(tokens, recvCores) + 384U +
                             EpCombineM5FlagBytes(h, k) + EpCombineM5WorkBytes(h, k) + 32U +
                             EpCombineAlign(kEpServerDedupCombineM4TableRows * k * 8U, 32U) +
                             ((h * 2U + 479U) / 480U) * 480U;
    const uint64_t final = 2U * ((owned * 4U + 31U) / 32U * 32U) + (owned * k * 4U + 31U) / 32U * 32U +
                           EpCombineM5FlagBytes(h, k) + EpCombineM5WorkBytes(h, k) + 32U;
    const uint64_t phase = partial > final ? partial : final;
    const uint32_t compute = EpCombineAlign(h * 4U, 256U);
    result.residentBytes = kEpServerDedupCombineResidentUbBytes + EpCombineRdmaBytes(h);
    constexpr uint32_t computeBuffers = 3U;
    const uint64_t fixed = result.residentBytes + computeBuffers * compute + phase;
    const uint32_t input = EpCombineAlign(((h * 2U + 479U) / 480U) * 480U, 128U);
    const uint32_t output = ((h * 2U + 479U) / 480U) * 512U;
    if (fixed + input + output > ubBytes) return result;
    result.scratchBytes = ubBytes - result.residentBytes;
    const uint32_t reads = uint32_t((ubBytes - fixed - output) / input);
    result.inputLanes = reads > kEpServerDedupCombineM4MaxInputLanes ? kEpServerDedupCombineM4MaxInputLanes : reads;
    const uint32_t writes = uint32_t((ubBytes - fixed - result.inputLanes * input) / output);
    result.outputLanes = writes > 8U ? 8U : writes;
    result.finalOutputLanes = result.outputLanes > kEpServerDedupCombineM5MaxOutputLanes
                                  ? kEpServerDedupCombineM5MaxOutputLanes
                                  : result.outputLanes;
    result.computeBytes = compute;
    result.phaseOffset = computeBuffers * compute;
    result.phaseBytes = uint32_t(phase);
    result.inputOffset = result.phaseOffset + result.phaseBytes;
    result.finalInputOffset = result.phaseOffset + uint32_t(final);
    result.outputOffset = result.inputOffset + result.inputLanes * input;
    result.inputBytes = input;
    result.outputBytes = output;
    result.totalBytes = result.residentBytes + result.outputOffset + result.outputLanes * output;
    return result;
}
struct PutRowPlan {
    uint32_t rows = 0, lanes = 0;
    EpCombineM45WorkspaceLayout receive{};
};

// First try all rows on this rank. Only a rank whose live UB does not fit
// goes through the tiled loop. Sender and receiver reuse the same arena.
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR PutRowPlan PlanPutRows(uint32_t rows, uint32_t bs, uint32_t h,
                                                          uint32_t k, uint32_t recvCores, uint32_t ubBytes)
{
    for (;;) {
        const uint32_t lanes = EpCombinePayloadLaneCount(rows, h, ubBytes);
        if (lanes) {
            const auto receive = EpCombineM45Layout(rows, bs, h, k, recvCores, ubBytes);
            if (receive.inputLanes) return {rows, lanes, receive};
        }
        if (rows <= 1U) return {};
        rows = rows > 2048U ? 2048U : rows / 2U;
    }
}
// M5 reuses all space after its own metadata except the retained physical
// output slots. Each input remembers which M4 sends must retire before overwrite.
struct EpCombineM5WorkspaceLayout {
    uint32_t inputLanes = 0U;
    uint32_t inputOffsets[kEpServerDedupCombineM5MaxInputLanes] = {};
    uint32_t inputRetireMasks[kEpServerDedupCombineM5MaxInputLanes] = {};
    uint32_t outputSlots[kEpServerDedupCombineM5MaxOutputLanes] = {};
};
COMBINE_DEDUP_EP_COMBINE_CONSTEXPR EpCombineM5WorkspaceLayout
EpCombineM5Layout(const EpCombineM45WorkspaceLayout& shared, uint32_t partialTasks)
{
    EpCombineM5WorkspaceLayout result{};
    if (shared.outputLanes == 0U || shared.inputBytes == 0U) return result;
    // Keep the two latest M4 sends; with fewer sends include unused slots.
    const uint32_t retained = partialTasks < shared.finalOutputLanes ? partialTasks : shared.finalOutputLanes;
    const uint32_t first = (partialTasks - retained) % shared.outputLanes;
    for (uint32_t lane = 0; lane < shared.finalOutputLanes; ++lane)
        result.outputSlots[lane] = (first + lane) % shared.outputLanes;
    const uint32_t limit = shared.scratchBytes;
    uint32_t cursor = shared.finalInputOffset;
    while (result.inputLanes < kEpServerDedupCombineM5MaxInputLanes && cursor + shared.inputBytes <= limit) {
        bool overlapsOutput = false;
        for (uint32_t lane = 0; lane < shared.finalOutputLanes; ++lane) {
            const uint32_t begin = shared.outputOffset + result.outputSlots[lane] * shared.outputBytes;
            if (cursor < begin + shared.outputBytes && cursor + shared.inputBytes > begin) {
                cursor = begin + shared.outputBytes;
                overlapsOutput = true;
                break;
            }
        }
        if (overlapsOutput) continue;
        const uint32_t inputLane = result.inputLanes++;
        result.inputOffsets[inputLane] = cursor;
        for (uint32_t slot = 0; slot < shared.outputLanes; ++slot) {
            const uint32_t begin = shared.outputOffset + slot * shared.outputBytes;
            if (cursor < begin + shared.outputBytes && cursor + shared.inputBytes > begin)
                result.inputRetireMasks[inputLane] |= 1U << slot;
        }
        cursor += shared.inputBytes;
    }
    return result;
}
// Three FP32 H buffers alone bound H, and therefore K*N. The admitted
// per-token flag clear fits CANN's 12-bit DataCopyPad blockCount (4095).
static_assert(32U * ((2U * (kEpServerDedupCombineMaxUbBytes / 12U) + 479U) / 480U) <= 4095U,
              "M5 flag clear blockCount budget");
#undef COMBINE_DEDUP_EP_COMBINE_CONSTEXPR
}  // namespace UbxCombine
#endif
