#pragma once

#include <cstdint>

namespace ascend_deepep {
// Fixed peer/channel slots: only valid rows are transmitted. Keep this layout
// identical to the existing compact dispatch plan; no cross-channel compaction.
inline constexpr uint32_t kChannelMetadataTileTokens = 32;
inline constexpr uint32_t kChannelMetadataMaxTopk = 128;
inline constexpr uint32_t kChannelMetadataMaxRanks = 256;
inline constexpr uint32_t kChannelMetadataHeadBytes = kChannelMetadataTileTokens * kChannelMetadataMaxRanks * 4;
inline constexpr uint32_t kChannelMetadataIndexBytes = kChannelMetadataTileTokens * kChannelMetadataMaxTopk * 8;
inline constexpr uint32_t kChannelMetadataWeightBytes = kChannelMetadataTileTokens * kChannelMetadataMaxTopk * 4;
inline constexpr uint32_t kChannelMetadataUbBytes =
    kChannelMetadataHeadBytes + kChannelMetadataIndexBytes + kChannelMetadataWeightBytes;
} // namespace ascend_deepep
