#pragma once

#include "kernel_operator.h"
#include "common/platform.h"

namespace ascend_deepep {

// Kernels partition this arena by byte offset and manage their own hard events.
// Do not allocate another buffer from this pipe after releasing its event state.
class UbWorkspace {
public:
    __aicore__ inline UbWorkspace()
    {
        pipe_.InitBuffer(buffer_, static_cast<uint32_t>(platform::kKernelUbBytes));
        bytes_ = buffer_.Get<uint8_t>();
        // Preserve the previous resource wrapper's initialization/release order.
        // The saved UB view remains in use; kernels explicitly drain their events.
        pipe_.Destroy();
    }

    template <class T>
    __aicore__ inline AscendC::LocalTensor<T> GetBufferByByte(uint32_t offset) const
    {
        return bytes_[offset].template ReinterpretCast<T>();
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_;
    AscendC::LocalTensor<uint8_t> bytes_;
};

}  // namespace ascend_deepep
