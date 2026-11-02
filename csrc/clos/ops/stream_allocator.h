#pragma once

#include <initializer_list>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>

namespace ascend_deepep::ops {

// Allocation and execution streams are independent. Initialization and all
// custom kernels stay on the execution stream, including temporary buffers.
class StreamAllocator {
public:
    explicit StreamAllocator(c10::optional<int64_t> allocation_stream_id)
        : execution_(c10_npu::getCurrentNPUStream()),
          allocation_(allocation_stream_id.has_value()
                          ? c10_npu::NPUStream::unpack3(*allocation_stream_id, execution_.device_index(),
                                                        c10::DeviceType::PrivateUse1)
                          : execution_)
    {
    }

    void Record(std::initializer_list<c10::optional<torch::Tensor>> tensors) const
    {
        for (const auto& tensor : tensors) {
            if (!tensor.has_value() || !tensor->defined()) continue;
            TORCH_CHECK(tensor->device() == execution_.device(), "V1 tensors must be on the current NPU device");
            // recordStream ignores foreign DataPtr deleters, such as SHMEM
            // from_blob views. Their lifetime is still owned by Buffer.
            c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), execution_);
        }
    }

    torch::Tensor Empty(at::IntArrayRef sizes, const torch::TensorOptions& options) const
    {
        TORCH_CHECK(options.device() == execution_.device(), "V1 allocation must be on the current NPU device");
        auto tensor = [&] {
            c10_npu::NPUStreamGuard guard(allocation_);
            return torch::empty(sizes, options);
        }();
        Record({tensor});
        return tensor;
    }

    torch::Tensor Zeros(at::IntArrayRef sizes, const torch::TensorOptions& options) const
    {
        auto tensor = Empty(sizes, options);
        tensor.zero_();
        return tensor;
    }

    torch::Tensor Full(at::IntArrayRef sizes, const at::Scalar& value, const torch::TensorOptions& options) const
    {
        auto tensor = Empty(sizes, options);
        tensor.fill_(value);
        return tensor;
    }

private:
    c10_npu::NPUStream execution_;
    c10_npu::NPUStream allocation_;
};

}  // namespace ascend_deepep::ops
