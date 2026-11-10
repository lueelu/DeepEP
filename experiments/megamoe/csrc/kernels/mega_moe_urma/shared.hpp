#pragma once
#include "kernels/mega_moe_urma/compute.hpp"
#include "kernels/mega_moe_urma/shared_contract.hpp"
#include "kernels/mega_moe_urma/shared_mx.hpp"
#include "kernels/mega_moe_urma/tile_completion.hpp"
#include "kernels/mega_moe_urma/shared_add.hpp"
#include "kernels/mega_moe_urma/tail_completion.hpp"
#include "kernels/mega_moe_urma/dispatch_completion.hpp"
namespace MegaMoeUrma {
using SharedGmm1 = SharedMxGmm<true>;
using SharedGmm2 = SharedMxGmm<false>;
__aicore__ inline Arguments SharedView(const Arguments& a)
{
    Arguments v = a;
    v.config.intermediate = a.sharedIntermediate;
    v.weight1 = a.sharedWeight1;
    v.weight2 = a.sharedWeight2;
    v.weightScale1 = a.sharedScale1;
    v.weightScale2 = a.sharedScale2;
    v.gmm1Output = a.sharedGmm1;
    v.activation = a.sharedActivation;
    v.activationScale = a.sharedActivationScale;
    v.gmm2Output = a.sharedOutput;
    v.gmm2Contiguous = a.sharedOutput;
    return v;
}
__aicore__ inline Fragment SharedFragment(const Arguments& a, uint32_t group)
{
    const uint32_t row = group * 256U, rows = a.config.bs - row < 256U ? a.config.bs - row : 256U;
    return {
        0, 0, row, rows, 0, uint64_t(group) * (a.sharedIntermediate / 256), uint64_t(group) * (a.config.hidden / 256)};
}
template <class T>
struct SharedOutputBackend {
    const Arguments& a;
    T& timer;
    __aicore__ inline void Arm()
    {
        SetFlag<HardEvent::FIX_S>(EVENT_ID1);
    }
    __aicore__ inline void Complete(bool, uint64_t tile, const TraceMeta& m)
    {
        const auto start = timer.TraceBegin();
        WaitFlag<HardEvent::FIX_S>(EVENT_ID1);
        timer.TraceEnd(TraceEvent::SharedGmm1Drain, start, m);
        WriteGmByPassDCache(ComputeFlag(a.sharedGmm1Done, tile), 1);
    }
};
template <bool First, class T>
__aicore__ inline void RunSharedGmm(const Arguments& a, uint32_t core, T& timer)
{
    auto v = SharedView(a);
    std::conditional_t<First, SharedGmm1, SharedGmm2> gmm;
    SharedOutputBackend<T> backend{a, timer};
    TileCompletion<SharedOutputBackend<T>> pending(backend);
    bool active = false;
    for (uint32_t group = 0; group < (a.config.bs + 255) / 256; ++group) {
        const auto f = SharedFragment(a, group);
        const uint32_t count = First ? a.sharedIntermediate / 256 : a.config.hidden / 256;
        const uint32_t first = FirstOwnedComputeTile(uint64_t(group) * count, RoutedOwnerResidue(First, core), kAic);
        if (first >= count) continue;
        const UbxMegaMoeImpl::BlockJobContext job{core, kAic};
        auto config = UrmaGmmConfig::BuildGmm1ProblemConfig(FragmentShape(v, f), job, kComputeTileM);
        // Both builders have the same ProblemConfig type; shared gate/up stays planar.
        if constexpr (!First) config = UrmaGmmConfig::BuildGmm2ProblemConfig(FragmentShape(v, f), job, kComputeTileM);
        auto addr = First ? Gmm1Addresses(v, f) : Gmm2Addresses(v, f);
        if constexpr (First) {
            addr.aGlobal = reinterpret_cast<GM_ADDR>(a.x + f.rowBegin * a.config.hidden);
            addr.aScaleGlobal = reinterpret_cast<GM_ADDR>(a.xScale + f.rowBegin * (a.config.hidden / 32));
        } else {
            addr.gmm2OutGlobal = reinterpret_cast<GM_ADDR>(a.sharedOutput + f.rowBegin * a.config.hidden * 2);
            if constexpr (g_coreType == AIC)
                WaitComputeFlag(a.sharedActivationDone, group, (a.sharedIntermediate / 256) * ((f.rows + 127) / 128));
        }
        gmm.UpdateProblem(config, addr);
        gmm.Begin();
        active = true;
        for (uint32_t tile = first; tile < count; tile += kAic) {
            TraceMeta meta = FragmentTraceMeta(f, group);
            meta.expert = a.config.world * a.config.localExperts;
            meta.tileN = tile * 256;
            meta.columns = 256;
            const auto start = timer.TraceBegin();
            if constexpr (g_coreType == AIC) gmm.Compute(0, tile * 256, f.rows, 256);
            timer.TraceEnd(First ? TraceEvent::SharedGmm1Tile : TraceEvent::SharedGmm2Tile, start, meta);
            if constexpr (First && g_coreType == AIC) pending.AfterSubmit(true, f.gmm1TileBegin + tile, meta);
        }
    }
    if constexpr (First && g_coreType == AIC) pending.Flush();
    if (active) {
        const auto start = timer.TraceBegin();
        gmm.End(First && g_coreType == AIC);  // W8 has no paired AIV0 context to retire.
        timer.TraceEnd(First ? TraceEvent::SharedGmm1Drain : TraceEvent::SharedGmm2Drain, start);
    }
}

struct SharedWaitScope {
    Timer& timer;
    bool& waiting;
    uint64_t& start;
    const TraceMeta& meta;
    TraceEvent event;
    __aicore__ inline void Before()
    {
        if (waiting) {
            (void)start;
            waiting = false;
        }
    }
};
class SharedActivationState {
public:
    __aicore__ inline SharedActivationState(const Arguments& a, uint32_t worker, Timer& t)
        : a_(a), worker_(worker), timer_(t)
    {
        total_ = SharedTaskCount(a.config.bs, a.sharedIntermediate);
        owned_ = SharedOwnedTasks(total_, worker);
    }
    __aicore__ inline bool Done() const
    {
        return finished_ == owned_;
    }
    template <class BeforeWork>
    __aicore__ inline bool TryOne(BeforeWork& before)
    {
        for (uint32_t j = 0; j < owned_; ++j) {
            const uint32_t local = (cursor_ + j) % owned_;
            if (done_[local / 64] & (uint64_t{1} << (local % 64))) continue;
            const uint32_t task = worker_ + local * kAic;
            const auto assignment = SharedTaskAt(task, a_.sharedIntermediate);
            const uint32_t group = assignment.group, tile = assignment.tile, slice = assignment.slice;
            const auto f = SharedFragment(a_, group);
            if (slice * 128 >= f.rows) {
                Mark(local);
                continue;
            }
            if (!ReadGmByPassDCache(ComputeFlag(a_.sharedGmm1Done, f.gmm1TileBegin + tile))) continue;
            before.Before();  // End a prior idle bar before recording actual shared work.
            // Soft shared activation must never overwrite the low128KiB FIX
            // mailbox, even while no issue has yet been observed by Scalar.
            using HighUbActivation =
                UbxMegaMoeImpl::BlockEpilogueActivationMxQuant<fp8_e4m3fn_t, bfloat16_t, 64U, 256U, false, false, true>;
            constexpr uint32_t highUb = 128U * 1024U, planeBytes = 64U * 256U * 2U;
            static_assert(highUb + 2U * planeBytes <= kAiv1PersistentUbBase);
            HighUbActivation op;
            op.Init({.yGmAddr = reinterpret_cast<GM_ADDR>(a_.sharedActivation),
                     .yScaleGmAddr = reinterpret_cast<GM_ADDR>(a_.sharedActivationScale),
                     .clampLimit = FLT_MAX,
                     .ubOffsetBytes = highUb});
            const uint32_t inter = a_.sharedIntermediate, n = tile * 256U;
            op.UpdateGlobalAddr({static_cast<int64_t>(f.rowBegin * inter),
                                 static_cast<int64_t>(f.rowBegin * (inter / 32U)), 0, 0, 0, 0});
            op.UpdateNextProblem({f.rows, inter, 0, 0});
            GlobalTensor<bfloat16_t> gm;
            gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a_.sharedGmm1));
            LocalTensor<bfloat16_t> gate(TPosition::VECIN, highUb, 64U * 256U);
            LocalTensor<bfloat16_t> up(TPosition::VECIN, highUb + planeBytes, 64U * 256U);
            const uint32_t m = slice * 128U, rows = f.rows - m < 128U ? f.rows - m : 128U;
            auto meta = ComputeTileTraceMeta(FragmentTraceMeta(f, group), m, n, rows, 256U);
            meta.expert = a_.config.world * a_.config.localExperts;
            const auto start = timer_.TraceBegin();
            for (uint32_t sub = 0; sub < rows; sub += 64U) {
                const uint32_t actual = rows - sub < 64U ? rows - sub : 64U;
                const uint64_t input = (f.rowBegin + m + sub) * (2U * inter) + n;
                const DataCopyExtParams copy{static_cast<uint16_t>(actual), 512U, (2U * inter - 256U) * 2U, 0U, 0U};
                SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
                DataCopyPad(gate, gm[input], copy, {false, 0U, 0U, 0U});
                DataCopyPad(up, gm[input + inter], copy, {false, 0U, 0U, 0U});
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                SetCtrlSpr<60, 60>(0);
                op({actual, 256U, 0, 0},
                   {int64_t(m + sub) * inter + n, int64_t(m + sub) * (inter / 32U) + n / 32U, 0, 0, 0, 0});
            }
            // The original M128 task/ready-count contract is unchanged: both
            // sub-slices completed their activation/scale MTE3 stores above.
            AtomicAdd(ComputeFlag(a_.sharedActivationDone, group), 1);
            timer_.TraceEnd(TraceEvent::SharedActivation, start, meta);
            Mark(local);
            cursor_ = (local + 1) % owned_;
            return true;
        }
        return false;
    }

private:
    __aicore__ inline void Mark(uint32_t i)
    {
        done_[i / 64] |= uint64_t{1} << (i % 64);
        ++finished_;
    }
    const Arguments& a_;
    uint32_t worker_;
    Timer& timer_;
    uint64_t done_[kSharedDoneWords]{};  // BS16384/I16384: at most293 tasks per AIV1
    uint32_t total_ = 0, owned_ = 0, finished_ = 0, cursor_ = 0;
};

struct SharedIncoming {
    const Arguments& a;
    uint32_t core;
    bool peer = false, self = false;
    DispatchReceiveObserver dispatch;
    __aicore__ inline void Poll()
    {
        dispatch.Poll();
        if (core < CommunicationWorkers(a.config.world)) ObserveIncomingCompletion(a, core, peer, self);
    }
};

__aicore__ inline void RunActivationShared(Arguments& args, uint32_t worker, Timer& timer)
{
    const uint32_t activationWorkers = ActivationWorkers(args.config.world);
    const uint32_t communicationWorkers = CommunicationWorkers(args.config.world);
    uint32_t nextPackRound = kPackStartupRounds;
    PackBatch pack(args, timer);
    PrepareActivationRoutes(args, worker, nextPackRound, pack, timer);
    timer.CoarseBegin(CoarsePhase::Activation);
    SharedActivationState shared(args, worker + communicationWorkers, timer);
    UrmaActivation epilogue;
    // Match main MegaMoE's unclamped SwiGLU. The epilogue default is zero,
    // which clamps every up value to zero and makes the entire FFN zero.
    epilogue.Init({.yGmAddr = reinterpret_cast<GM_ADDR>(args.activation),
                   .yScaleGmAddr = reinterpret_cast<GM_ADDR>(args.activationScale),
                   .clampLimit = FLT_MAX});
    uint32_t nextGather = worker;
    uint32_t nextActivation = 0U;
    uint32_t activationTile = 0U;

    bool packFinished = false;
    bool activationPrepared = false;
    bool idle = false;
    uint64_t idleStart = 0U;
    TraceEvent idleEvent = TraceEvent::WaitGmm1;
    TraceMeta idleMeta;
    while (nextGather < args.config.fragmentCount || nextActivation < args.config.fragmentCount ||
           nextPackRound < args.config.packRoundCount || !shared.Done()) {
        bool progress = false;
        // Skip groups without this worker's activation tile, but never skip a
        // scale gather merely because its owner has no activation in that group.
        while (nextActivation < args.config.fragmentCount) {
            const auto fragment = LoadFragment(args, nextActivation);
            const uint32_t tileCount = FragmentGmm1Tiles(args, fragment);
            if (!activationPrepared) {
                activationTile = FirstOwnedComputeTile(fragment.gmm1TileBegin, worker, activationWorkers);
                activationPrepared = true;
            }
            if (activationTile >= tileCount) {
                ++nextActivation;
                activationPrepared = false;
                continue;
            }
            if (ReadGmByPassDCache(ComputeFlag(args.gmm1Done, fragment.gmm1TileBegin + activationTile)) != 0) {
                if (idle) {
                    (void)idleStart;
                    idle = false;
                }
                pack.Close();  // Activation reuses DMA event IDs.
                RunReadyActivationTile(args, worker, fragment, nextActivation, activationTile, epilogue, timer);
                activationTile += activationWorkers;
                progress = true;
            }
            break;
        }
        if (!progress && nextGather < args.config.fragmentCount) {
            const auto fragment = LoadFragment(args, nextGather);
            if (AreGroupInputsReady(args, fragment)) {
                if (idle) {
                    (void)idleStart;
                    idle = false;
                }
                pack.Close();  // Scale gather reuses DMA event IDs.
                GatherInputScale(args, fragment, nextGather, timer);
                nextGather += activationWorkers;
                progress = true;
            }
        }
        // Only fall back to Pack when neither activation nor a scale gather
        // ran. Consecutive Pack quotas retain events/cursor; close them only
        // before actual activation/gather work or when all Pack is complete.
        if (!progress && nextPackRound < args.config.packRoundCount) {
            if (idle) {
                (void)idleStart;
                idle = false;
            }
            PackOneRound(args, nextPackRound, worker + communicationWorkers, kAic, pack);
            ++nextPackRound;
            progress = true;
        }
        if (!packFinished && nextPackRound >= args.config.packRoundCount) {
            pack.Close();
            MarkPackWorkerComplete(args);
            packFinished = true;
        }
        if (!progress && !shared.Done()) {
            pack.Close();
            SharedWaitScope before{timer, idle, idleStart, idleMeta, idleEvent};
            progress = shared.TryOne(before);
            if (!progress && nextGather >= args.config.fragmentCount && nextActivation >= args.config.fragmentCount) {
                const auto start = GetSystemCycle();
                while (GetSystemCycle() - start < 64U) {
                }
            }
        }
        if (!progress && (nextGather < args.config.fragmentCount || nextActivation < args.config.fragmentCount)) {
            if (!idle) {
                idleStart = 0U;
                idle = true;
                const bool waitingActivation = nextActivation < args.config.fragmentCount;
                const uint32_t index = waitingActivation ? nextActivation : nextGather;
                idleEvent = waitingActivation ? TraceEvent::WaitGmm1 : TraceEvent::WaitDispatch;
                idleMeta = FragmentTraceMeta(LoadFragment(args, index), index);
            }
            const uint64_t backoff = GetSystemCycle();
            while (GetSystemCycle() - backoff < 64U) {
            }
        }
    }
    if (idle) (void)idleStart;
    pack.Close();
    // Empty compute workloads and <=2 Pack rounds can skip the loop entirely.
    if (!packFinished) MarkPackWorkerComplete(args);
    timer.CoarseEnd(CoarsePhase::Activation);
}

}  // namespace MegaMoeUrma
