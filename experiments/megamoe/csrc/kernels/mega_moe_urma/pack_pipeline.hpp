// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "kernels/mega_moe_urma/compute_contract.hpp"
#include "kernels/mega_moe_urma/route_runtime.hpp"
#include "kernel_operator.h"

namespace MegaMoeUrma {

constexpr uint32_t kPackPipelineSlots = 6U;
constexpr uint32_t kPackTokenCapacity = 8192U;
constexpr uint32_t kPackScaleCapacity = kPackTokenCapacity / 32U;
constexpr uint32_t kPackSlotBytes = kPackTokenCapacity + kPackScaleCapacity;
constexpr uint32_t kPackOrderRows = 1024U;
constexpr uint32_t kPackOrderOffset = kPackPipelineSlots * kPackSlotBytes;
constexpr uint32_t kPackUbBytes = kPackOrderOffset + kPackOrderRows * sizeof(int64_t);
// Each inverse DMA needs a 32B-aligned immutable UB source. Reuse the old
// 8KiB index window for 96 int32 indices and two 96-row inverse source banks.
// The bank cursor survives Submit/expert/round boundaries, like the data ring.
constexpr uint32_t kPackMetadataRows = 96U;
constexpr uint32_t kPackMetadataBanks = 2U;
constexpr uint32_t kPackInverseOffset = kPackOrderOffset + kPackMetadataRows * sizeof(int32_t);
constexpr uint32_t kPackInverseStride = 32U;
constexpr uint32_t kPackInverseBankBytes = kPackMetadataRows * kPackInverseStride;
constexpr uint32_t kPackMetadataEventBase = 6U;
// Activation and scale gather use the low 128KiB. Pack may run between their
// tasks on AIV1; keep its data above that region, and close all event credits
// before switching to a different DMA stage. AIV0 releases this region before opening its W4 context.
constexpr uint32_t kPackUbBase = ActivationUbContract::inputEnd;
static_assert(kPackSlotBytes % 32U == 0U && kPackOrderOffset % 32U == 0U);
static_assert(kPackUbBase + kPackUbBytes <= kRequiredUbBytes, "Pack and activation must fit the minimum UB");
static_assert(kPackUbBase + kPackUbBytes <= kAiv1PersistentUbBase, "Pack must not overlap peer counters");
static_assert(kPackInverseOffset % 32U == 0U && kPackInverseBankBytes % 32U == 0U);
static_assert(kPackInverseOffset + kPackMetadataBanks * kPackInverseBankBytes <= kPackUbBytes);

// Build the padded DMA sources in V, rather than issuing per-row Scalar UB stores.
// q fits uint32 under the supported BS/TopK capacity; inverse retains int64 ABI.
__simd_vf__ inline void PackInverseRowsVF(__ubuf__ uint32_t* dst, uint32_t rowBegin, uint32_t rows)
{
    constexpr uint32_t lanes = AscendC::GetVecLen() / sizeof(uint32_t);
    auto full = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<int32_t> signedIndex;
    AscendC::Reg::RegTensor<uint32_t> values, offsets, zero;
    AscendC::Reg::Duplicate(zero, 0U, full);
    for (uint32_t first = 0U; first < rows; first += lanes) {
        uint32_t remaining = rows - first;
        auto active = AscendC::Reg::UpdateMask<uint32_t>(remaining);
        AscendC::Reg::Arange(signedIndex, static_cast<int32_t>(first));
        auto& index = reinterpret_cast<AscendC::Reg::RegTensor<uint32_t>&>(signedIndex);
        AscendC::Reg::Adds(values, index, rowBegin, active);
        AscendC::Reg::ShiftLefts(offsets, index, static_cast<int16_t>(3), active);
        AscendC::Reg::Scatter(dst, values, offsets, active);
        AscendC::Reg::Adds(offsets, offsets, 1U, active);
        AscendC::Reg::Scatter(dst, zero, offsets, active);
    }
}

// A ring of DMA slots shared across consecutive Pack segments/rounds.
// Close explicitly before handing these event IDs to activation/dequant/Reduce.
class PackPipeline {
public:
    __aicore__ inline bool Active() const
    {
        return active_;
    }
    template <class Completion>
    __aicore__ inline void Submit(const Arguments& args, uint64_t begin, uint64_t count, Completion& completion)
    {
        if (count == 0U) return;
        using namespace AscendC;
        const uint32_t h = args.config.hidden;
        const uint32_t scales = h / 32U;
        GlobalTensor<uint8_t> x, xScale, expandedX, expandedScale, selfX, selfScale;
        GlobalTensor<int32_t> routes;
        GlobalTensor<int64_t> inverse;
        x.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.x));
        xScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.xScale));
        expandedX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.expandedX));
        expandedScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.expandedScale));
        selfX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.symmetric + args.config.receiveXOffset));
        selfScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args.symmetric + args.config.receiveScaleOffset));
        routes.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args.expertRoutes));
        inverse.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(args.reduceIndex));
        LocalTensor<int32_t> routeIndices(TPosition::VECCALC, kPackUbBase + kPackOrderOffset, kPackMetadataRows);
        const auto* info = RouteInfo(args);
        if (!active_) {
            for (uint32_t slot = 0U; slot < kPackPipelineSlots; ++slot)
                SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(slot));
            active_ = true;
            nextSlot_ = 0U;
        }
        const uint64_t retireAfter = count < kPackPipelineSlots - 1U ? count : kPackPipelineSlots - 1U;
        uint32_t low = 0U, high = args.config.world * args.config.localExperts;
        while (low < high) {
            const uint32_t mid = (low + high) / 2U;
            if (info->outgoing[mid + 1U] <= begin)
                low = mid + 1U;
            else
                high = mid;
        }
        uint32_t expert = low;
        for (uint64_t batch = 0U; batch < count;) {
            const uint64_t first = begin + batch;
            while (info->outgoing[expert + 1U] <= first) ++expert;
            const uint64_t expertRows = info->outgoing[expert + 1U] - first;
            const uint64_t remaining = count - batch < expertRows ? count - batch : expertRows;
            const uint32_t batchRows =
                static_cast<uint32_t>(remaining < kPackMetadataRows ? remaining : kPackMetadataRows);
            // MTE3_MTE2 credits protect token buffers, not inverse metadata.
            // Wait only when this particular bank is reused. The other bank's
            // previous batch may remain in flight while the next segment
            // submits its first five rows and retires the old Pack quota.
            // Batch limits are 96 rows or the current expert boundary; bank
            // reuse and Close wait for MTE3_S before their metadata buffers are reused.
            const uint32_t inverseBank = nextInverseBank_;
            RetireInverseMetadata(inverseBank);
            LocalTensor<int64_t> inverseRows(TPosition::VECCALC,
                                             kPackUbBase + kPackInverseOffset + inverseBank * kPackInverseBankBytes,
                                             kPackInverseBankBytes / sizeof(int64_t));
            // This dependency protects the preceding batch's scalar UB reads.
            // The MTE2_S wait occurs once per index batch, never once per token.
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            const uint64_t bucketRow =
                uint64_t(expert) * args.config.sourceRowCapacity + first - info->outgoing[expert];
            DataCopyPad(routeIndices, routes[bucketRow],
                        {1U, static_cast<uint32_t>(batchRows * sizeof(int32_t)), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
            asc_vf_call<PackInverseRowsVF>(reinterpret_cast<__ubuf__ uint32_t*>(inverseRows.GetPhyAddr()),
                                           static_cast<uint32_t>(first), batchRows);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            for (uint32_t index = 0U; index < batchRows; ++index) {
                const uint64_t localRow = batch + index;
                const uint64_t row = begin + localRow;
                const uint32_t slot = nextSlot_;
                const auto event = static_cast<TEventID>(slot);
                LocalTensor<uint8_t> token(TPosition::VECCALC, kPackUbBase + slot * kPackSlotBytes, kPackTokenCapacity);
                LocalTensor<uint8_t> scale(TPosition::VECCALC, kPackUbBase + slot * kPackSlotBytes + kPackTokenCapacity,
                                           kPackScaleCapacity);
                const uint32_t flat = static_cast<uint32_t>(routeIndices.GetValue(index));
                const uint64_t tokenRow = flat / args.config.topk;
                WaitFlag<HardEvent::MTE3_MTE2>(event);
                DataCopyPad(token, x[tokenRow * h], {1U, h, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
                DataCopyPad(scale, xScale[tokenRow * scales], {1U, scales, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
                SetFlag<HardEvent::MTE2_MTE3>(event);
                WaitFlag<HardEvent::MTE2_MTE3>(event);
                if (IsSelfExpandedRow(row, args.config)) {
                    const uint64_t receiveRow = SelfReceiveRow(row, args.config);
                    DataCopyPad(selfX[receiveRow * h], token, {1U, h, 0U, 0U, 0U});
                    DataCopyPad(selfScale[receiveRow * scales], scale, {1U, scales, 0U, 0U, 0U});
                } else {
                    DataCopyPad(expandedX[row * h], token, {1U, h, 0U, 0U, 0U});
                    DataCopyPad(expandedScale[row * scales], scale, {1U, scales, 0U, 0U, 0U});
                }
                // Byte-exact DMA avoids dirty Scalar cache-line writeback when
                // different Pack owners update adjacent inverse entries.
                DataCopyPad(inverse[flat], inverseRows[index * kPackInverseStride / sizeof(int64_t)],
                            {1U, static_cast<uint32_t>(sizeof(int64_t)), 0U, 0U, 0U});
                SetFlag<HardEvent::MTE3_MTE2>(event);
                nextSlot_ = (nextSlot_ + 1U) % kPackPipelineSlots;
                // Retire only the PREVIOUS segment's captured fence, after this
                // segment has submitted five rows (or its entire short tail).
                if (localRow + 1U == retireAfter) completion.RetirePending();
            }
            SetFlag<HardEvent::MTE3_S>(static_cast<TEventID>(kPackMetadataEventBase + inverseBank));
            inverseMetadataPending_[inverseBank] = true;
            nextInverseBank_ = (inverseBank + 1U) % kPackMetadataBanks;
            batch += batchRows;
        }
    }
    __aicore__ inline void Close()
    {
        if (!active_) return;
        using namespace AscendC;
        for (uint32_t bank = 0U; bank < kPackMetadataBanks; ++bank) RetireInverseMetadata(bank);
        for (uint32_t slot = 0U; slot < kPackPipelineSlots; ++slot)
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(slot));
        // Prove credit waits were consumed before another stage rearms them.
        // This also certifies all six producer MTE3 stores, without PIPE_ALL.
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        active_ = false;
    }

private:
    __aicore__ inline void RetireInverseMetadata(uint32_t bank)
    {
        if (!inverseMetadataPending_[bank]) return;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(static_cast<AscendC::TEventID>(kPackMetadataEventBase + bank));
        inverseMetadataPending_[bank] = false;
    }
    uint32_t nextSlot_ = 0U;
    bool active_ = false;
    uint32_t nextInverseBank_ = 0U;
    bool inverseMetadataPending_[kPackMetadataBanks] = {false, false};
};

struct NoPackCompletion {
    __aicore__ inline void RetirePending() {}
};

// Blocking compatibility entry point, used by standalone full Expand callers.
__aicore__ inline void ExpandRows(const Arguments& args, uint64_t begin, uint64_t count)
{
    PackPipeline pipeline;
    NoPackCompletion completion;
    pipeline.Submit(args, begin, count, completion);
    pipeline.Close();
}

// Retain the original full-expand entry point for standalone callers.
__aicore__ inline void Expand(const Arguments& args, uint32_t worker)
{
    const uint64_t rows = static_cast<uint64_t>(args.config.bs) * args.config.topk;
    ExpandRows(args, LocalRowBegin(rows, worker), LocalOwnedRows(rows, worker));
}

}  // namespace MegaMoeUrma
