// Copyright (c) 2026, Lu Lu
#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.hpp>

#include <cstdint>

namespace deep_ep {
constexpr uint32_t kLlCountWireRecords = 15U;
constexpr uint32_t kLlCountWireBlockBytes = 512U;
constexpr uint32_t kLlCountWireFlagOffset = 480U;
constexpr __aicore__ __forceinline__ uint32_t CountWireBytes(uint32_t records)
{
    return (records + kLlCountWireRecords - 1U) / kLlCountWireRecords * kLlCountWireBlockBytes;
}
__aicore__ __forceinline__ uint32_t CountWireRecordOffset(uint32_t record)
{
    return record / kLlCountWireRecords * kLlCountWireBlockBytes + record % kLlCountWireRecords * 32U;
}
// Same E*512B reservation, physical source-major frames; logical prefixes stay
// expert-major. Legacy/local frames intersect a count-core and target rank;
// Direct remote frames cover all experts of that destination rank.
struct LlCountFrame {
    uint64_t recordOffset, commitOffset, frameOffset;
    uint32_t firstExpert, records;
    uint32_t sourceOffset, wireBytes, flagWords;
};
__aicore__ __forceinline__ LlCountFrame CountFrame(uint32_t experts, uint32_t world, uint32_t countCores,
    uint32_t destination, uint32_t source, uint32_t localExpert,
    bool directRanks = false)
{
    const uint32_t local = experts / world, expert = destination * local + localExpert;
    if (directRanks && !LlSameHost(destination, source)) {
        // One source/core owns all experts of this remote rank. The compact
        // body and its trailing 32B commit travel in the same SO WQE.
        const uint64_t frame = uint64_t(source) * local * 512U;
        return {frame + uint64_t(localExpert) * 32U,
                frame + uint64_t(local) * 32U, frame, 0U, local, 0U, (local + 1U) * 32U, 1U};
    }
    const uint32_t base = experts / countCores, extra = experts % countCores;
    const uint32_t wideEnd = (base + 1U) * extra;
    const uint32_t owner = expert < wideEnd ? expert / (base + 1U) : extra + (expert - wideEnd) / base;
    const uint32_t begin = owner * base + (owner < extra ? owner : extra);
    const uint32_t end = begin + base + (owner < extra ? 1U : 0U);
    const uint32_t first = begin > destination * local ? begin - destination * local : 0U;
    const uint32_t last = end < (destination + 1U) * local ? end - destination * local : local;
    // All same-host destinations share the immutable image of this worker's
    // complete (host-clamped) range. Send whole blocks intersecting each frame.
    const uint32_t hostFirst = source / kNumRanksPerHost * kNumRanksPerHost * local;
    const uint32_t inputFirst = directRanks && begin < hostFirst ? hostFirst : begin;
    const uint32_t firstRecord = destination * local + first - inputFirst;
    const uint32_t record = expert - inputFirst;
    const uint32_t firstBlock = firstRecord / kLlCountWireRecords;
    const uint32_t lastBlock = (firstRecord + last - first - 1U) / kLlCountWireRecords;
    const uint64_t frame = (uint64_t(source) * local + first) * 512U;
    const uint32_t sourceOffset = firstBlock * kLlCountWireBlockBytes;
    return {frame + CountWireRecordOffset(record) - sourceOffset,
            frame + (record / kLlCountWireRecords - firstBlock) * kLlCountWireBlockBytes + kLlCountWireFlagOffset,
            frame, first, last - first, sourceOffset,
            (lastBlock - firstBlock + 1U) * kLlCountWireBlockBytes, 8U};
}
}

namespace deep_ep {

struct EpServerDedupDataLayout {
    uint64_t quantCount = 0;
    uint64_t scaleBytes = 0;
    uint64_t headerPayloadOffset = 0, readIndexPayloadOffset = 0;
    uint64_t headerWireOffset = 0, relayReadIndexWireOffset = 0;
    uint64_t payloadBytes = 0;
    uint64_t blockCount = 0;
    uint64_t packedPayloadBytes = 0;
    uint64_t tokenStride = 0;
    uint64_t returnStride = 0;
    uint64_t sourceOffset = 0, sourceBytes = 0;
    uint64_t dispatchOffset = 0, dispatchBytes = 0;
    uint64_t relayOffset = 0, relayBytes = 0, requiredBytes = 0;
    // SmoothInt8 carries two wire formats in one physical slot. The fields above
    // describe the INT8 wire (source-quantized intra-host payload and terminal
    // output). The BF16 entry wire used across machines is stored separately.
    // Both formats share the largest slot stride and the same slot tail so a
    // single tail flag probe can tell them apart.
    uint64_t int8BlockCount = 0, int8Stride = 0, int8PackedPayloadBytes = 0;
    uint64_t wire16BlockCount = 0, wire16Stride = 0, wire16PackedPayloadBytes = 0;
    uint64_t wire16PayloadBytes = 0, wire16HeaderPayloadOffset = 0, wire16ReadIndexPayloadOffset = 0;
    uint64_t int8TailOffset = 0;
    uint32_t probeStartBlock = 0, probeBlocks = 0;
};

constexpr __aicore__ __forceinline__ uint64_t DispatchDedupAlign(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

// Host validates positive uint32 shapes and the count UB bound before calling;
// E<=1024, BS*K<2^14 and H<2^32 bound the entire domain below 2^59 bytes.
// Public buffer products are also checked against size_t by the Host builder.
constexpr __aicore__ __forceinline__ EpServerDedupDataLayout DispatchServerDedupDataLayout(uint32_t bs, uint32_t h,
                                                                                    uint32_t topK, uint32_t experts,
                                                                                    EpServerDedupInputMode mode)
{
    const bool prequantized = mode == EpServerDedupInputMode::PrequantizedFp8Packs;
    const bool smoothInt8 = mode == EpServerDedupInputMode::SmoothInt8;
    // One-byte payloads keep the header right after the packed data; only plain
    // BF16 uses its own two-byte width.
    const bool int8Wire = prequantized || smoothInt8;
    EpServerDedupDataLayout layout{};
    layout.quantCount = DispatchDedupAlign(h, 256U);
    layout.scaleBytes = prequantized ? ((uint64_t(h) + 127U) / 128U) * 4U : (smoothInt8 ? 4U : 0U);
    layout.headerPayloadOffset =
        DispatchDedupAlign(int8Wire ? layout.quantCount + layout.scaleBytes : uint64_t(h) * 2U, 32U);
    layout.readIndexPayloadOffset = layout.headerPayloadOffset + 32U;
    const uint64_t readIndexBytes = uint64_t(topK) * 8U;
    if (layout.readIndexPayloadOffset % 480U + readIndexBytes > 480U)
        layout.readIndexPayloadOffset = DispatchDedupAlign(layout.readIndexPayloadOffset, 480U);
    layout.headerWireOffset = layout.headerPayloadOffset / 480U * 512U + layout.headerPayloadOffset % 480U;
    layout.relayReadIndexWireOffset =
        layout.readIndexPayloadOffset / 480U * 512U + layout.readIndexPayloadOffset % 480U;
    layout.payloadBytes = DispatchDedupAlign(layout.readIndexPayloadOffset + readIndexBytes, 32U);
    layout.blockCount = (layout.payloadBytes + 479U) / 480U;
    layout.packedPayloadBytes = layout.blockCount * 480U;
    layout.tokenStride = layout.blockCount * 512U;
    layout.returnStride = (uint64_t(h) * 2U + 479U) / 480U * 512U;
    if (smoothInt8) {
        // Entry wire: raw BF16 token plus the same header/read-index tail.
        const uint64_t readIndexBytes16 = uint64_t(topK) * 8U;
        const uint64_t header16 = DispatchDedupAlign(uint64_t(h) * 2U, 32U);
        uint64_t index16 = header16 + 32U;
        if (index16 % 480U + readIndexBytes16 > 480U) index16 = DispatchDedupAlign(index16, 480U);
        const uint64_t payload16 = DispatchDedupAlign(index16 + readIndexBytes16, 32U);
        const uint64_t blocks16 = (payload16 + 479U) / 480U;
        layout.int8BlockCount = layout.blockCount;
        layout.int8Stride = layout.tokenStride;
        layout.int8PackedPayloadBytes = layout.packedPayloadBytes;
        layout.wire16BlockCount = blocks16;
        layout.wire16PayloadBytes = payload16;
        layout.wire16HeaderPayloadOffset = header16;
        layout.wire16ReadIndexPayloadOffset = index16;
        layout.wire16PackedPayloadBytes = blocks16 * 480U;
        layout.wire16Stride = blocks16 * 512U;
        // Same-slot dual format: both wires end at one shared slot tail. The
        // tail probe covers the intersection of the two block ranges so the
        // first flag block of either format (which carries its marker) is
        // always included.
        if (blocks16 >= layout.int8BlockCount) {
            layout.int8TailOffset = (blocks16 - layout.int8BlockCount) * 512U;
            layout.probeStartBlock = uint32_t(blocks16 - layout.int8BlockCount);
            layout.probeBlocks = uint32_t(layout.int8BlockCount);
            layout.blockCount = blocks16;
            layout.packedPayloadBytes = layout.wire16PackedPayloadBytes;
            layout.tokenStride = layout.wire16Stride;
        } else {
            layout.int8TailOffset = 0U;
            layout.probeStartBlock = 0U;
            layout.probeBlocks = uint32_t(blocks16);
        }
    }
    layout.sourceOffset = kEpServerDedupControlBytes;
    layout.sourceBytes = uint64_t(bs) * topK * layout.returnStride;
    layout.dispatchOffset = DispatchDedupAlign(layout.sourceOffset + layout.sourceBytes, 512U);
    layout.dispatchBytes = uint64_t(experts) * bs * layout.tokenStride;
    layout.relayOffset = DispatchDedupAlign(layout.dispatchOffset + layout.dispatchBytes, 512U);
    // Historical Combine reservation, unused by Dispatch. The new pull Combine
    // must define its stage/result allocation before consuming the five tables.
    layout.relayBytes = uint64_t(8U) * bs * topK * layout.returnStride;
    layout.requiredBytes = layout.relayOffset + layout.relayBytes;
    return layout;
}

__aicore__ __forceinline__ uint64_t DispatchServerDedupDispatchOffset(const EpServerDedupDataLayout& layout,
                                                                         uint32_t source, uint32_t destination,
                                                                         uint32_t localExpert, uint32_t q,
                                                                         uint32_t ranks, uint32_t localExperts,
                                                                         uint32_t bs)
{
    return layout.dispatchOffset +
           (((uint64_t(source) + destination) % ranks * localExperts + localExpert) * bs + q) * layout.tokenStride;
}
struct LlPrequantPadding {
    uint32_t inputOffset, inputBytes;
    uint32_t tailBlockOffset;
    uint64_t tailBlockMask;
    uint32_t tailOffset, tailBytes;
};

// LL admits nonaligned H and rounds packed scale storage to whole 4B packs.
// Vector bases remain 32B aligned. The input-padding boundary may include the
// last <=31 valid input bytes, which the following MTE2 load overwrites. A
// masked uint32 Duplicate clears only the partial scale-tail block, then a
// contiguous Duplicate clears the remaining tail; scale bytes are untouched.
__aicore__ __forceinline__ LlPrequantPadding LlPlanPrequantPadding(
    uint32_t hidden, uint32_t quantCount, uint32_t scaleBytes, uint32_t packedBytes)
{
    const uint32_t end = quantCount + scaleBytes;
    const uint32_t tail = (end + 31U) / 32U * 32U;
    const uint32_t firstWord = (end % 32U) / sizeof(uint32_t);
    const uint32_t input = hidden / 32U * 32U;
    return {input, quantCount - input, end / 32U * 32U,
            firstWord ? (uint64_t(0xffU) << firstWord) & 0xffU : 0U,
            tail, packedBytes - tail};
}
}  // namespace deep_ep

