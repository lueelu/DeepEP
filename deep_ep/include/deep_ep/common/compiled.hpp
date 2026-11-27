// Copyright (c) 2026, Lu Lu
#pragma once

#include <cstddef>
#include <cstdint>

#if !defined(__CCE__)
#define __aicore__
#define __forceinline__ inline
#define __gm__
#define __ubuf__
#define __simd_vf__
#define __simd_callee__
#endif

namespace deep_ep {
// UB allocation granularity shared by every static dispatch region.
constexpr int kNumUbAlignmentBytes = 32;
constexpr uint32_t kNumRanksPerHost = 16U;
constexpr uint32_t kNumMaxRanksPerGroup = 16U;
constexpr __aicore__ __forceinline__ uint32_t LlGroupForWorld(uint32_t world) { return world <= 16U ? 4U : 16U; }
constexpr __aicore__ __forceinline__ uint8_t LlTilingKeyForWorld(uint32_t world) { return uint8_t(LlGroupForWorld(world)); }
constexpr __aicore__ __forceinline__ bool LlTilingKeyValid(uint32_t world, uint8_t key) { return key == LlTilingKeyForWorld(world); }
constexpr __aicore__ __forceinline__ bool LlSameHost(uint32_t rank, uint32_t peer, uint32_t ranksPerHost = kNumRanksPerHost) {
    return ranksPerHost != 0U && rank / ranksPerHost == peer / ranksPerHost;
}
}

namespace deep_ep {
// Count frames and the full Dispatch C API are one wire. The remote/local
// histogram split is gone, so this protocol is the only LL count identity.
constexpr uint32_t kLlCoreProtocolVersion = 36U; // multi-host direct counts, 512B DataAsFlag frames, raw C API
constexpr __aicore__ __forceinline__ bool LlCoreCountValid(uint32_t cores) {
    return cores >= 16U && cores % 2U == 0U;
}
}

namespace deep_ep {
namespace rdma {
constexpr uint32_t kBackend = 1U;
constexpr uint32_t kCoreBytes = 128U * 1024U;
// ABI v3 tail: addresses[0]=unbiased symmetric workspace, [1]=per-core staging;
// config[0]=backend, [1]=QP count. Remote peerMems are NULL in RDMA mode.
}
}  // namespace deep_ep


namespace deep_ep {
// The dedup resource layout is fixed for the lifetime of the binary. The
// execution path is chosen at runtime from the per-rank capacity instead.
constexpr uint32_t kEpServerDedupDedupThreshold = 32U;
// Only the per-rank capacity and communicator size select the execution path;
// the actual launch B no longer participates.
constexpr __aicore__ __forceinline__ bool EpServerDedupEnabled(uint32_t numMaxTokensPerRank, uint32_t world = 16U)
{
    return world > 16U || numMaxTokensPerRank > kEpServerDedupDedupThreshold;
}
}  // namespace deep_ep

namespace deep_ep {
constexpr uint32_t kEpServerDedupMaxTopK = 32U;

// The wire header and TokenType output use the same values.
enum class EpServerDedupTokenType : uint32_t { Direct = 1U, FirstHop = 2U, Terminal = 3U };

struct alignas(32) EpServerDedupPacketHeader {
    uint32_t srcRank;
    uint32_t srcTokenId;
    uint32_t topK;  // Original route index / returnK, not the shape K.
    uint32_t type;
    uint32_t padding[4];
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
static_assert(sizeof(EpServerDedupReadIndexBlock) == 8U, "read index block ABI");
static_assert(offsetof(EpServerDedupReadIndexBlock, weightBits) == 4U, "weight bits offset");
static_assert(sizeof(EpServerDedupDirectTask) == 16U, "source direct task ABI");
}  // namespace deep_ep

// Host/device wire constants shared with the 256-rank CommArgs ABI.

namespace deep_ep {

constexpr int DISPATCH_DEDUP_MAX_RANK_SIZE = 256;

constexpr int DFX_COUNT = 50;

// Flag tail reserved at the end of every symmetric window.

constexpr int DISPATCH_DEDUP_FLAG_BUFF_BYTES = 4 * 1024 * 1024;

}  // namespace deep_ep


namespace deep_ep {
// Wire FP8 does not imply internal quantization. Values also form the tiling ABI.
// SmoothInt8 is not a prequantized input: BF16 tokens enter and the kernel emits
// per-token dynamic INT8 output with one FP32 dequant scale per expanded row.
enum class EpServerDedupInputMode : uint8_t {
    Plain16 = 0,
    SmoothInt8 = 1,
    PrequantizedFp8Packs = 2,
};
}  // namespace deep_ep

namespace deep_ep {
constexpr uint64_t kEpServerDedupPeerBytes = 8ULL * 1024U * 1024U * 1024U;
constexpr uint64_t kEpServerDedupWindowBytes = kEpServerDedupPeerBytes / 2U;
constexpr uint64_t kEpServerDedupControlBytes = 2ULL * 1024U * 1024U;
// Runtime owns the domain's final flag region. Leave the same tail unused in
// window 0 so both windows have identical capacity and shape-independent stride.
constexpr uint64_t kEpServerDedupWindowUsableBytes =
    kEpServerDedupWindowBytes - DISPATCH_DEDUP_FLAG_BUFF_BYTES;
}  // namespace deep_ep
