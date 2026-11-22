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
inline unsigned& LastDispatchOverlapOptions()
{
    static thread_local unsigned options = 0;
    return options;
}
inline unsigned& LastDispatchMetadataOptions()
{
    static thread_local unsigned options = 0;
    return options;
}
}  // namespace ascend_deepep::ops
