// Copyright (c) 2026, Lu Lu

#pragma once
#include "ll_transport.h"

#include "combine_wqe_prepare.h"

namespace UbxCombine {
// One quiet per used peer before any GM slot is recycled. M5 may overlap
// the final outstanding NIC reads; Flush still retires them at kernel exit.
class TransferBatch {
    static constexpr uint32_t kWords = (UbxDispatch::DISPATCH_DEDUP_MAX_RANK_SIZE + 31U) / 32U;
    const __gm__ UbxTransport::Args* args_ = nullptr;
    uint32_t bytes_ = 0U, count_ = 0U, capacity_ = 0U;
    uint32_t peers_[kWords]{};
    GM_ADDR stage_ = nullptr;

    __aicore__ inline void WaitStage()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(7U);
    }
    AscendC::LocalTensor<uint8_t> privateUb_;
    bool partial_ = false;
    __aicore__ inline void Retire(AscendC::LocalTensor<uint8_t> scratch)
    {
        if (!count_) return;
        for (uint32_t word = 0U; word < kWords; ++word) {
            const uint32_t mask = peers_[word];
            if (!mask) continue;
            for (uint32_t bit = 0U; bit < 32U; ++bit)
                if (mask & (1U << bit)) {
                    const uint32_t peer = word * 32U + bit;

                    UbxTransport::Quiet(peer, scratch);
                }
            peers_[word] = 0U;
        }
        count_ = 0U;
    }
    __aicore__ inline GM_ADDR Next(uint32_t peer, uint64_t offset)
    {
        UbxTransport::Check(args_, peer, offset, bytes_);
        return stage_ + uint64_t(count_) * bytes_;
    }
    __aicore__ inline void Submit(uint32_t peer, uint64_t offset, GM_ADDR src, AscendC::LocalTensor<uint8_t> scratch)
    {
        using namespace AscendC;
        SetFlag<HardEvent::MTE3_S>(7U);

        PreparedWrite write;
        bool prepared = false;
        if (partial_) {
            prepared = write.Prepare(peer, UbxTransport::RemoteAddress(args_, peer, offset), src, bytes_, privateUb_);
        }

        WaitStage();

        if (prepared)
            write.Publish();
        else
            UbxTransport::Put(args_, peer, UbxTransport::RemoteAddress(args_, peer, offset), src, bytes_, scratch);

        peers_[peer / 32U] |= 1U << (peer % 32U);
        if (++count_ == capacity_) Retire(scratch);
    }

public:
    __aicore__ inline void Init(const __gm__ UbxTransport::Args* args, uint32_t bytes)
    {
        if (Pending() || !bytes || bytes % 512U || bytes > UbxRdma::kCoreBytes) {
            AscendC::Trap();
            return;
        }
        args_ = args;
        bytes_ = bytes;
        capacity_ = UbxRdma::kCoreBytes / bytes;
        if (uint32_t(args->rankSize) > UbxDispatch::kLlPhysicalRanksPerHost)
            stage_ = UbxTransport::Stage(args, capacity_ * bytes);
    }
    __aicore__ inline void SetPartialScratch(AscendC::LocalTensor<uint8_t> scratch)
    {
        privateUb_ = scratch;
        partial_ = true;
    }
    __aicore__ inline bool Pending() const
    {
        return count_ != 0U;
    }
    __aicore__ inline void Enqueue(uint32_t peer, uint64_t offset, AscendC::LocalTensor<uint8_t> packet)
    {
        const auto dst = Next(peer, offset);

        AscendC::DataCopy(UbxTransport::Tensor(dst, false), packet, bytes_);

        Submit(peer, offset, dst, packet);
    }

    __aicore__ inline void Flush(AscendC::LocalTensor<uint8_t> scratch)
    {
        Retire(scratch);
    }
};
}  // namespace UbxCombine
