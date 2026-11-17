#pragma once
#include <cstdint>
#include "ll_compile_attrs.h"
#include "ll_group_policy.h"

namespace DispatchDedup {
constexpr uint32_t kLlCountWireRecords = 15U;
constexpr uint32_t kLlCountWireBlockBytes = 512U;
constexpr uint32_t kLlCountWireFlagOffset = 480U;
__aicore__ __forceinline__ uint32_t CountWireBytes(uint32_t records)
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
    const uint32_t hostFirst = source / kLlPhysicalRanksPerHost * kLlPhysicalRanksPerHost * local;
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
