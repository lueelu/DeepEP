// Copyright (c) 2026, Lu Lu

#pragma once
#include "kernel_operator.h"

#include "ll_transport.h"
#include "ep_memory_server_dedup_buffer_layout.h"

namespace UbxCombine {
// Each receive core polls and clears its own ready row in the active window.
template <uint32_t Group>
class CombineGroupReady {
public:
    __aicore__ inline void Init(const __gm__ UbxDispatch::CommArgs* args, uint64_t control, uint32_t core,
                                uint32_t send, uint32_t recv, AscendC::LocalTensor<float> input,
                                AscendC::LocalTensor<float> work, AscendC::LocalTensor<float> indices,
                                AscendC::LocalTensor<float> result, AscendC::LocalTensor<float> publication)
    {
        args_ = args;
        control_ = control;
        core_ = core;
        send_ = send;
        recv_ = recv;
        observed_ = 0U;
        input_ = input.ReinterpretCast<uint32_t>();
        work_ = work.ReinterpretCast<int32_t>();
        indices_ = indices.ReinterpretCast<uint32_t>();
        result_ = result.ReinterpretCast<uint8_t>();
        publication_ = publication.ReinterpretCast<uint32_t>();

        if (core < send) return;

        AscendC::CreateVecIndex(indices_.ReinterpretCast<int32_t>(), int32_t(0), 64U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(indices_.ReinterpretCast<int32_t>(), indices_.ReinterpretCast<int32_t>(), int32_t(32), 64U);
        Sync<AscendC::HardEvent::V_S>();
    }
    __aicore__ inline bool All() const
    {
        return observed_ == ((1U << Group) - 1U);
    }
    __aicore__ inline uint32_t Poll()
    {
        if (All()) return observed_;

        const uint64_t offset = COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET + (core_ - send_) * 512U;
        const uint32_t expected = 0x3f800000U;
        observed_ |= uint32_t(ReadMask(offset, Group, expected));

        return observed_;
    }
    __aicore__ inline void Wait()
    {
        while (!All()) {
            Poll();
        }
    }
    __aicore__ inline void Retire()
    {
        Wait();
        // Includes delayed, unused peers. Never clear before their write arrived.
        AscendC::Duplicate(indices_, uint32_t(0), 128U);
        Sync<AscendC::HardEvent::V_MTE3>();
        auto dest = Tensor(uint32_t(DEEPEP_UBX_LAUNCH_RANK(args_)),
                           COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET + (core_ - send_) * 512U);
        AscendC::DataCopyPad(dest, indices_, {1U, 512U, 0U, 0U, 0U});
    }

private:
    template <AscendC::HardEvent E>
    __aicore__ inline void Sync()
    {
        // 7 is separate from the live input/output lane tickets (0..6).
        AscendC::SetFlag<E>(7U);
        AscendC::WaitFlag<E>(7U);
    }
    __aicore__ inline AscendC::GlobalTensor<uint32_t> Tensor(uint32_t peer, uint64_t offset)
    {
        if (!UbxTransport::SameHost(args_, peer) || !args_->peerMems[peer]) {
            AscendC::Trap();
        }
        AscendC::GlobalTensor<uint32_t> t;
        t.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(args_->peerMems[peer] + control_ + offset));

        return t;
    }
    __aicore__ inline uint64_t ReadMask(uint64_t offset, uint32_t count, uint32_t expected)
    {
        Sync<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopyPad(input_, Tensor(uint32_t(DEEPEP_UBX_LAUNCH_RANK(args_)), offset), {1U, count * 32U, 0U, 0U, 0U},
                             {false, 0U, 0U, uint32_t(0)});
        Sync<AscendC::HardEvent::MTE2_V>();

        AscendC::Duplicate(work_, int32_t(~expected), 64U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(work_, input_.ReinterpretCast<int32_t>(), indices_, 0U, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compares(result_, work_, int32_t(expected), AscendC::CMPMODE::EQ, 64U);
        Sync<AscendC::HardEvent::V_S>();
        const auto mask = result_.ReinterpretCast<uint64_t>().GetValue(0U) & ((uint64_t(1) << count) - 1U);

        return mask;
    }

    const __gm__ UbxDispatch::CommArgs* args_ = nullptr;
    AscendC::LocalTensor<int32_t> work_;
    AscendC::LocalTensor<uint32_t> input_;
    AscendC::LocalTensor<uint32_t> indices_, publication_;
    AscendC::LocalTensor<uint8_t> result_;
    uint64_t control_ = 0;
    uint32_t core_ = 0, send_ = 0, recv_ = 0, epoch_ = 0, observed_ = 0;
};
}  // namespace UbxCombine
