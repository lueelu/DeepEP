#pragma once

namespace ascend_deepep::ops {
// Per-launch observation only: never consulted when selecting the next launch.
inline bool& LastDispatchChannelMetadata()
{
    static thread_local bool enabled = false;
    return enabled;
}
inline unsigned& LastDispatchChannelMetadataTileTokens()
{
    static thread_local unsigned tokens = 0;
    return tokens;
}
}  // namespace ascend_deepep::ops
