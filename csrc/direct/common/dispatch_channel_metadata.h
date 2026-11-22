#pragma once

#include <cstdint>

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define ASCEND_DEEPEP_METADATA_TILE_FN __aicore__ inline
#else
#define ASCEND_DEEPEP_METADATA_TILE_FN inline constexpr
#endif

namespace ascend_deepep {
// Fixed peer/channel slots: only valid rows are transmitted. Keep this layout
// identical to the existing compact dispatch plan; no cross-channel compaction.
inline constexpr uint32_t kChannelMetadataMaxTopk = 128;
inline constexpr uint32_t kChannelMetadataMaxRanks = 256;

struct ChannelMetadataTile {
    uint32_t tokens;
    uint32_t index_offset;
    uint32_t weight_offset;
    uint32_t bytes;
};

inline constexpr uint32_t kChannelMetadataDescriptorWords = 4;

struct RemoteMetadataTile {
    ChannelMetadataTile input;
    uint32_t plan_tokens;
};

// Grow only the plan cache into otherwise unused private UB. Input tile size
// and sender scratch are unchanged. Every peer's plan starts on a 32-byte boundary.
ASCEND_DEEPEP_METADATA_TILE_FN RemoteMetadataTile ExtendMetadataPlanCache(ChannelMetadataTile tile, uint32_t peers,
                                                                          uint32_t channel_stride, uint32_t budget)
{
    if (!tile.tokens || !peers || budget < tile.bytes) return {};
    const uint32_t input_bytes = tile.bytes - tile.index_offset;
    const uint32_t available = (budget - input_bytes) / (peers * 4U) / 8U * 8U;
    const uint32_t wanted = channel_stride > tile.tokens ? channel_stride : tile.tokens;
    const uint32_t rows = wanted < available ? wanted : available;
    const uint32_t indices_bytes = tile.weight_offset - tile.index_offset;
    tile.index_offset = rows * peers * 4U;
    tile.weight_offset = tile.index_offset + indices_bytes;
    tile.bytes = tile.index_offset + input_bytes;
    return {tile, rows};
}

ASCEND_DEEPEP_METADATA_TILE_FN ChannelMetadataTile MakeChannelMetadataTile(uint32_t tokens, uint32_t world,
                                                                           uint32_t topk, uint32_t index_bytes,
                                                                           bool weighted)
{
    const uint32_t heads = (tokens * world * 4U + 31U) / 32U * 32U;
    const uint32_t indices = (tokens * topk * index_bytes + 31U) / 32U * 32U;
    const uint32_t weights = weighted ? (tokens * topk * 4U + 31U) / 32U * 32U : 0;
    return {tokens, heads, heads + indices, heads + indices + weights};
}

// Zero means automatic: prefer 128 tokens, reducing the tile only for UB capacity.
// Explicit choices fail on the host instead of silently changing an experiment.
ASCEND_DEEPEP_METADATA_TILE_FN ChannelMetadataTile SelectChannelMetadataTile(uint32_t requested, uint32_t world,
                                                                             uint32_t topk, uint32_t index_bytes,
                                                                             bool weighted, uint32_t ub_bytes)
{
    if (!world || world > kChannelMetadataMaxRanks || !topk || topk > kChannelMetadataMaxTopk ||
        (index_bytes != 4 && index_bytes != 8) ||
        (requested && requested != 32 && requested != 64 && requested != 128 && requested != 256))
        return {};
    uint32_t tokens = requested ? requested : 128;
    for (;;) {
        const auto tile = MakeChannelMetadataTile(tokens, world, topk, index_bytes, weighted);
        if (tile.bytes <= ub_bytes) return tile;
        if (requested || tokens == 32) return {};
        tokens /= 2;
    }
}
}  // namespace ascend_deepep

#undef ASCEND_DEEPEP_METADATA_TILE_FN
