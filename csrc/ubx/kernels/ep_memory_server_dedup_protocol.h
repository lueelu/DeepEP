// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#ifndef DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_PROTOCOL_H
#define DISPATCH_DEDUP_EP_MEMORY_SERVER_DEDUP_PROTOCOL_H

#include <cstddef>
#include <cstdint>
#include "ll_group_policy.h"

namespace UbxDispatch {
// Routing group is selected by Host tiling key; this constant is capacity only.
constexpr uint32_t kLlRanksPerGroup = kLlMaxRanksPerGroup;
constexpr uint32_t kEpServerDedupAlignmentBytes = 32U;
constexpr uint32_t kEpServerDedupMaxTopK = 32U;

// The wire header and TokenType output use the same values.
enum class EpServerDedupTokenType : uint32_t { Direct = 1U, FirstHop = 2U, Terminal = 3U };

// FIRST_HOP packets carry the PUT Combine route in the words that used to be
// padding. A TERMINAL row publishes owner/inbox columns only when putTag
// matches, so packets produced by an older Dispatch are never misread.
constexpr uint32_t kEpServerDedupPutRouteTag = 0x50555431U;  // 'PUT1'

struct alignas(32) EpServerDedupPacketHeader {
    uint32_t srcRank;
    uint32_t srcTokenId;
    uint32_t topK;  // Original route index / returnK, not the shape K.
    uint32_t type;
    // FIRST_HOP-only. putOwnerRank is the actual FIRST_HOP target selected by
    // the sender; putInboxBase is (srcRank * B + srcTokenId) * shapeK, where B
    // is this launch's real token count, not the admitted capacity M.
    // Member m (1-based, 0 is FIRST_HOP itself) writes putInboxBase + m.
    uint32_t putOwnerRank;
    uint32_t putInboxBase;
    uint32_t putTag;
    uint32_t reserved;
};

// Block 0: index=N, weightBits=selfWeight. Member: index=expert|(q<<16).
// A RelayReadIndex row contains exactly K blocks; only blocks 0..N are valid.
struct EpServerDedupReadIndexBlock {
    uint32_t index;
    uint32_t weightBits;
};

struct EpServerDedupDirectTask {
    uint32_t tokenOffset;
    uint32_t originalK;
    uint32_t expertId;
    uint32_t q;
};
static_assert(sizeof(EpServerDedupTokenType) == 4U, "token type ABI");
static_assert(sizeof(EpServerDedupPacketHeader) == 32U, "packet header ABI");
static_assert(alignof(EpServerDedupPacketHeader) == 32U, "packet header alignment");
static_assert(offsetof(EpServerDedupPacketHeader, srcTokenId) == 4U, "source token offset");
static_assert(offsetof(EpServerDedupPacketHeader, topK) == 8U, "return k offset");
static_assert(offsetof(EpServerDedupPacketHeader, type) == 12U, "packet role offset");
static_assert(offsetof(EpServerDedupPacketHeader, putOwnerRank) == 16U, "PUT owner offset");
static_assert(offsetof(EpServerDedupPacketHeader, putInboxBase) == 20U, "PUT inbox base offset");
static_assert(offsetof(EpServerDedupPacketHeader, putTag) == 24U, "PUT tag offset");
static_assert(offsetof(EpServerDedupPacketHeader, reserved) == 28U, "reserved header word offset");
static_assert(sizeof(EpServerDedupReadIndexBlock) == 8U, "read index block ABI");
static_assert(offsetof(EpServerDedupReadIndexBlock, weightBits) == 4U, "weight bits offset");
static_assert(sizeof(EpServerDedupDirectTask) == 16U, "source direct task ABI");
}  // namespace UbxDispatch
#endif
