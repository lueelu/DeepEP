#pragma once
#include <cstdint>

namespace DispatchDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_SENDER_INPUT_INLINE __aicore__ inline
#else
#define LL_SENDER_INPUT_INLINE inline
#endif

// Cache one token in the existing input queue (plain/prequantized) or quantBuf
// (quantizing). Packet tails are mutable, so a cache hit must retire the prior
// Vector pack read before Scalar changes that tail. A miss retains the original
// queue/quantization dependencies. No packet lane is used as cached storage.
class LlSenderInputReuse {
    uint32_t token_ = 0;
    bool valid_ = false;
public:
    template <class Operations>
    LL_SENDER_INPUT_INLINE void Prepare(uint32_t token, Operations& operations) {
        if (valid_ && token == token_) {
            operations.WaitPackedRead();
            return;
        }
        if (valid_) operations.ReleaseInput();
        operations.LoadInput(token);
        token_ = token;
        valid_ = true;
    }
    template <class Operations>
    LL_SENDER_INPUT_INLINE void Finish(Operations& operations) {
        if (valid_) operations.ReleaseInput();
        valid_ = false;
    }
};

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
LL_SENDER_INPUT_INLINE LlPrequantPadding LlPlanPrequantPadding(
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

// A fresh prequantized payload zeroes unused relay entries. Reusing that input
// must preserve those defined bytes as a larger first-hop becomes a smaller
// first-hop or DIRECT. Only previously written entries need clearing; the
// caller's existing Scalar->Vector dependency also orders these stores.
template <class Words>
LL_SENDER_INPUT_INLINE void LlClearStalePrequantTail(Words index, uint32_t previousEntries, uint32_t entries)
{
    for (uint32_t stale = entries + 1U; stale <= previousEntries; ++stale) {
        index.SetValue(stale * 2U, 0U);
        index.SetValue(stale * 2U + 1U, 0U);
    }
}

#undef LL_SENDER_INPUT_INLINE
}
