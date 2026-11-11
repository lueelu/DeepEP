// Copyright (c) 2026, Lu Lu
// Modified by liuboxi, 2026

#ifndef DISPATCH_DEDUP_EP_PACKET_RECEIVE_LAYOUT_H
#define DISPATCH_DEDUP_EP_PACKET_RECEIVE_LAYOUT_H

#include <cstdint>

namespace DispatchDedup {
namespace PacketReceiveLayout {
// A transport layout hint, independent of application-level packet types.
// Only the first float of the first Flag block carries the hint.
// NDR protocol10 sends a whole packet in one SO-ordered WQE. Keep checking ALL
// flag blocks: this includes the last 32B tail, while preserving MTE/relay
// readiness and the first-block 1.5 marker. Never reduce this to the first flag.
constexpr uint32_t kMarkerOffset = 480U;
constexpr float kCompact = 1.0f;
constexpr float kKeepFlags = 1.5f;
enum class State { Pending, Compact, KeepFlags };

#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#endif
    constexpr State
    Decode(float sum, uint32_t blocks)
{
    return sum == float(blocks * 8U)
               ? State::Compact
               : (sum == float(blocks * 8U) + (kKeepFlags - kCompact) ? State::KeepFlags : State::Pending);
}
}  // namespace PacketReceiveLayout
}  // namespace DispatchDedup
#endif
