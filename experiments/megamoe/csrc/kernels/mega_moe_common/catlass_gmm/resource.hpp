// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#ifndef MEGA_MOE_CATLASS_RESOURCE_H
#define MEGA_MOE_CATLASS_RESOURCE_H
#include "catlass/arch/arch.hpp"

namespace Catlass::Gemm::Block {
// Non-owning views of the Wave stage's scratch memory. Creating a CATLASS TPipe
// per slice would reset events used by the surrounding persistent kernel.
template <AscendC::TPosition Position, uint32_t Bytes>
struct MegaMoeBuffer {
    template <class Element>
    __aicore__ inline AscendC::LocalTensor<Element> GetBufferByByte(uint32_t offset) const
    {
        AscendC::LocalTensor<uint8_t> bytes(Position, 0, Bytes);
        return bytes[offset].template ReinterpretCast<Element>();
    }
};
struct MegaMoeResource {
    using Arch = Catlass::Arch::Ascend950;
    MegaMoeBuffer<AscendC::TPosition::A1, Arch::L1_SIZE> l1Buf;
    MegaMoeBuffer<AscendC::TPosition::A2, Arch::L0A_SIZE> l0ABuf;
    MegaMoeBuffer<AscendC::TPosition::B2, Arch::L0B_SIZE> l0BBuf;
    MegaMoeBuffer<AscendC::TPosition::CO1, Arch::L0C_SIZE> l0CBuf;
    MegaMoeBuffer<AscendC::TPosition::C2, Arch::BIAS_SIZE> btBuf;
    MegaMoeBuffer<AscendC::TPosition::VECCALC, Arch::UB_SIZE> ubBuf;
};
}  // namespace Catlass::Gemm::Block
#endif
