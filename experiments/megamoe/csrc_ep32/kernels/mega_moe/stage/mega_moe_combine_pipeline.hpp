// Copyright (c) 2026, Lu Lu
#ifndef MEGA_MOE_COMBINE_PIPELINE_H
#define MEGA_MOE_COMBINE_PIPELINE_H

#include "mega_moe_gmm2_combine.hpp"

namespace MegaMoeImpl {

// Only consecutive Combine tasks share this ring. Every actual transition to
// activation, Dispatch, or the export tail drains it before stage UB/event reuse.
// A failed readiness probe does not touch UB and must not drain the ring.
class CrossExpertCombinePipeline {
public:
    __aicore__ inline bool Enabled(const Params& params) const
    {
        return UseHierarchicalRoce(params.tilingData);
    }

    // The caller has acquired this slice's GMM2-ready counter. Ownership and
    // route metadata are identical to RunTokenCombineSlice<true>.
    __aicore__ inline void IssueSlice(const MoeStageCommonConfig& common, const Params& params,
                                      const ExpertLoopState& state, uint32_t rowStart, const WorkRange& owned,
                                      AscendTimerDevice& timer)
    {
        if (!initialized_) {
            slots_ = params.tilingData->bf16Bs96 ? 4U : 2U;
            buffers_ = slots_ == 4U ? InitWaveCombineBuffers<COMBINE_NO_QUANT, false, 4U>(common, scratch_, true)
                                    : InitWaveCombineBuffers<COMBINE_NO_QUANT, false, 2U>(common, scratch_, true);
            initialized_ = true;
        }
        GlobalTensor<bfloat16_t> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.workspaceInfo.gmm2MmadResPtr));
        const uint64_t offset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;
        for (uint32_t i = 0U; i < owned.count; ++i) {
            const uint32_t slot = nextSlot_;
            Retire(slot, timer);
            const uint32_t row = rowStart + owned.start + i;
            const uint64_t globalRow = static_cast<uint64_t>(state.globalTokenStartIndex) + row;
            expert_[slot] = state.expertIdx;
            row_[slot] = row;
            wave_[slot] = timer.CurrentWave();
            const int issue = timer.BeginCombineRow(state.expertIdx, row, slot, AscendTimer::CombineRowPhase::Issue);
            PreloadWaveCombineMetaInfo(params, scratch_, globalRow, 1U, 0U);
            lifetime_[slot] = timer.BeginCombineRow(state.expertIdx, row, slot, AscendTimer::CombineRowPhase::Lifetime);
            // Retire already consumed the old completion event, so the sender
            // must use first-use semantics even when wrapping across experts.
            SendWaveCombineToken<COMBINE_NO_QUANT, false>(common, buffers_, scratch_, params, source, offset, globalRow,
                                                          scratch_.metaInfoTensor, slot);
            active_[slot] = true;
            timer.EndCombineRow(issue);
            nextSlot_ = (slot + 1U) % slots_;
        }
    }

    __aicore__ inline void Drain(AscendTimerDevice& timer)
    {
        for (uint32_t i = 0; i < slots_; ++i) Retire((nextSlot_ + i) % slots_, timer);
        nextSlot_ = 0U;
    }

private:
    WaveCombineBufferConfig buffers_{};
    WaveCombineScratch scratch_{};
    bool initialized_ = false;
    uint32_t slots_ = 2U;
    bool active_[4]{};
    uint32_t nextSlot_ = 0U;
    uint32_t expert_[4]{};
    uint32_t row_[4]{};
    int64_t wave_[4]{};
    int lifetime_[4]{};

    __aicore__ inline void Retire(uint32_t slot, AscendTimerDevice& timer)
    {
        if (!active_[slot]) return;
        const int wait = timer.BeginCombineRowAtWave(expert_[slot], row_[slot], slot,
                                                     AscendTimer::CombineRowPhase::ReuseWait, wave_[slot]);
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(slot));
        // Observe completion on Scalar as well, before allowing V/Scalar to
        // overwrite stage UB. This orders through MTE2, not a global PIPE_ALL.
        SyncFuncStatic<HardEvent::MTE2_S, EVENT_ID0>();
        timer.EndCombineRow(wait);
        timer.EndCombineRow(lifetime_[slot]);
        lifetime_[slot] = -1;
        active_[slot] = false;
    }
};

}  // namespace MegaMoeImpl
#endif
