#pragma once

#include <torch/extension.h>

namespace ascend_deepep::ops {
// Diagnostics retain only the last launch on this host thread. The allocator
// records its execution stream, including when an older row is overwritten.
inline c10::optional<torch::Tensor>& DispatchDeviceTimingSlot()
{
    static thread_local c10::optional<torch::Tensor> tensor;
    return tensor;
}
inline c10::optional<torch::Tensor> TakeDispatchDeviceTiming()
{
    auto result = DispatchDeviceTimingSlot();
    DispatchDeviceTimingSlot().reset();
    return result;
}
}  // namespace ascend_deepep::ops
