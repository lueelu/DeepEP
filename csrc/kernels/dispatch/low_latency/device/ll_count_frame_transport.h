#pragma once
#include "ll_transport.h"
#ifdef DEEPEP_LL_REFERENCE_PROFILE
#include "../common/ll_reference_profile_device.h"
#endif
namespace LlTransport {
// Per-core CQ bookkeeping, independent of receiver publication. A payload
// quiet on the same peer/QP also retires the earlier count request.
class CountRetirement {
    const __gm__ Args* args_ = nullptr;
    uint32_t pending_ = 0U;
public:
    __aicore__ inline void Init(const __gm__ Args* args) { args_ = args; pending_ = 0U; }
    __aicore__ inline void Posted(uint32_t task) { pending_ |= 1U << (task / asc_get_block_num()); }
    __aicore__ inline bool Pending() const { return pending_ != 0U; }
    __aicore__ inline void Completed(uint32_t peer) {
        const uint32_t host = uint32_t(args_->rank) / DispatchDedup::kLlPhysicalRanksPerHost * DispatchDedup::kLlPhysicalRanksPerHost;
        if (peer >= host && peer < host + DispatchDedup::kLlPhysicalRanksPerHost) return;
        const uint32_t task = peer < host ? peer : peer - DispatchDedup::kLlPhysicalRanksPerHost;
        if (task % asc_get_block_num() == asc_get_block_idx()) pending_ &= ~(1U << (task / asc_get_block_num()));
    }
    __aicore__ inline void Finish(__ubuf__ uint8_t* scratch) {
        const uint32_t host = uint32_t(args_->rank) / DispatchDedup::kLlPhysicalRanksPerHost * DispatchDedup::kLlPhysicalRanksPerHost;
        for (uint32_t task = asc_get_block_idx(), bit = 1U; task < uint32_t(args_->rankSize) - DispatchDedup::kLlPhysicalRanksPerHost;
             task += asc_get_block_num(), bit <<= 1U) {
            if (!(pending_ & bit)) continue;
            const uint32_t peer = task < host ? task : task + DispatchDedup::kLlPhysicalRanksPerHost;
            Quiet(peer, scratch);
        }
        pending_ = 0U;
    }
};

// Count TX is isolated in each symmetric bank, outside count/prefix, combine
// controls and the payload region. One disjoint slot per destination rank.
constexpr uint64_t kCountTxBase = 1280ULL * 1024U;
static_assert(kCountTxBase >= 1024U * 1024U + 18432U + 36U * 512U,
              "count TX above existing combine controls");
static_assert(kCountTxBase + (1024U + 256U) * 32U < 2U * 1024U * 1024U, "count TX below payload");
// Remote rank-owned counts only. records includes a fully initialized tail
// record with ready=1 even for zeros. CQ retirement is deferred to the role's
// existing payload quiet or final core completion, never required by SendCnt.
__aicore__ inline void DirectCountFrame(const __gm__ Args* args, uint32_t peer,
                                       uint64_t frameOffset, __ubuf__ uint8_t* records,
                                       uint32_t count, __ubuf__ uint8_t* scratch, uint64_t window = 0U
#ifdef DEEPEP_LL_REFERENCE_PROFILE
                                       , LlReferenceProfile::DeviceProfile* trace = nullptr
#endif
                                       ) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t t0 = trace ? trace->PublishClock() : 0U;
#endif
    const uint32_t bytes = (count + 1U) * 32U;
    Check(args, peer, frameOffset, bytes);
    auto stage = args->peerMems[args->rank] + window + kCountTxBase + uint64_t(peer) * bytes;
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t t1 = trace ? trace->PublishClock() : 0U;
    if (trace) trace->CountBoundary<4, true>();
#endif
    LlCountC::Store(stage, records, bytes, 1U, 0U, true);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t t2 = trace ? trace->PublishClock() : 0U;
    if (trace) trace->CountBoundary<5, true>();
#endif
    asc_sync_notify(PIPE_MTE3, PIPE_S, EVENT_ID7);
    asc_sync_wait(PIPE_MTE3, PIPE_S, EVENT_ID7);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t t3 = trace ? trace->PublishClock() : 0U;
    if (trace) trace->CountBoundary<6, true>();
    if constexpr (LlReferenceProfile::kCountComputeMode >= 3U) {
        auto dst = Address(args,peer,frameOffset);
        const uint64_t t4 = trace ? trace->PublishClock() : 0U;
        if (trace) trace->CountBoundary<7, true>();
        Put(args,peer,dst,stage,bytes,scratch);
        const uint64_t t5 = trace ? trace->PublishClock() : 0U;
        if (trace) {
            trace->CountBoundary<8, true>();
            trace->PublishAdd<4>(t1-t0); trace->PublishAdd<5>(t2-t1);
            trace->PublishAdd<6>(t3-t2); trace->PublishAdd<7>(t4-t3);
            trace->PublishAdd<8>(t5-t4);
        }
    } else
#endif
    Put(args,peer,Address(args,peer,frameOffset),stage,bytes,scratch);
}
// Same DataAsFlag blocks as payload: 480B data followed by eight FP32 ones.
// One copy publishes every intersecting block; the source stays immutable
// until the existing role completion barrier. No body/commit completion wait.
__aicore__ inline void CountFrame(const __gm__ Args* args, uint32_t peer,
                                 uint64_t frameOffset, __ubuf__ uint8_t* records,
                                 uint32_t wireBytes, uint32_t count
#ifdef DEEPEP_LL_REFERENCE_PROFILE
                                 , LlReferenceProfile::DeviceProfile* trace = nullptr
#endif
                                 ) {
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t t0 = trace ? trace->PublishClock() : 0U;
#endif
    Check(args, peer, frameOffset, wireBytes);
    auto dst=Address(args,peer,frameOffset);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t t1 = trace ? trace->PublishClock() : 0U;
    if (trace) trace->CountBoundary<14, true>();
#endif
    LlCountC::Store(dst, records, wireBytes, 1U, 0U, Rdma(args));
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    const uint64_t t2 = trace ? trace->PublishClock() : 0U;
    if (trace) {
        trace->CountBoundary<15, true>();
        trace->CompletePublishAdd<11>(1U); trace->CompletePublishAdd<18>(count);
        trace->PublishAdd<11>(1U); trace->PublishAdd<18>(count);
        trace->PublishAdd<14>(t1-t0); trace->PublishAdd<15>(t2-t1);
        // Retired local wait/commit sections remain zero in profile mode 3.
    }
#endif
}
}
