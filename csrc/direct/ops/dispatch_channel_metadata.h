#pragma once

namespace ascend_deepep::ops {
// Per-launch observation only: never consulted when selecting the next launch.
inline bool& LastDispatchChannelMetadata()
{
    static thread_local bool enabled = false;
    return enabled;
}
}  // namespace ascend_deepep::ops
