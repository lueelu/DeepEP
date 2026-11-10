#pragma once
#include "kernels/mega_moe_urma/timer.hpp"
namespace MegaMoeUrma {
// Diagnostic-only observer of peer -> THIS rank. Every peer submits CVs in
// order on one QP; its final SO epoch certifies all prior X/scale PUTs to us.
// No CQ access, remote loads, credit updates, new signals or quiet.
class DispatchReceiveObserver {
public:
    __aicore__ inline void Init(Timer& timer)
    {
#ifdef DEEPEP_MEGAMOE_TIMER
        if (timer.DiagnosticEnabled()) timer_ = &timer;
#else
        (void)timer;
#endif
    }
    // Arm only after route summary is acquired and local Dispatch submission
    // is finished, when the owner enters its Combine phase. Empty incoming
    // edges are explicitly recorded, never polled as CV UINT_MAX.
    __aicore__ inline void Arm(const Arguments& a, uint32_t peer, uint32_t rows)
    {
#ifdef DEEPEP_MEGAMOE_TIMER
        if (!timer_) return;
        peer_ = peer;
        rows_ = rows;
        epoch_ = a.config.epoch;
        bytes_ = uint64_t(rows) * (a.config.hidden + a.config.hidden / 32U);
        if (rows == 0U) {
            timer_->RecordDispatchReceive(peer, 0U, 0U, 0U);
            return;
        }
        cv_ = (rows - 1U) / a.config.cvRows;
        flag_ = reinterpret_cast<__gm__ uint64_t*>(a.symmetric + a.config.dispatchReadyOffset +
                                                   (uint64_t(peer) * a.config.maxPeerChunks + cv_) * kFlagBytes);
        pending_ = true;
#else
        (void)a;
        (void)peer;
        (void)rows;
#endif
    }
    __aicore__ inline void Poll()
    {
#ifdef DEEPEP_MEGAMOE_TIMER
        if (!pending_ || AscendC::ReadGmByPassDCache(flag_) != epoch_) return;
        timer_->RecordDispatchReceive(peer_, cv_, rows_, bytes_);
        pending_ = false;
#endif
    }

private:
#ifdef DEEPEP_MEGAMOE_TIMER
    Timer* timer_ = nullptr;
    __gm__ uint64_t* flag_ = nullptr;
    uint64_t epoch_ = 0U, bytes_ = 0U;
    uint32_t peer_ = 0U, cv_ = 0U, rows_ = 0U;
    bool pending_ = false;
#endif
};
}  // namespace MegaMoeUrma
