#pragma once
#include "ll_transport.h"
namespace LlTransport {
// Per-core CQ bookkeeping, independent of receiver publication. A payload
// quiet on the same peer/QP also retires the earlier count request.
class CountRetirement {
    const __gm__ Args* args_ = nullptr;
    uint32_t pending_ = 0U;
public:
    __aicore__ inline void Init(const __gm__ Args* args) { args_ = args; pending_ = 0U; }
    __aicore__ inline void Posted(uint32_t task) { pending_ |= 1U << (task / GetBlockNum()); }
    __aicore__ inline bool Pending() const { return pending_ != 0U; }
    __aicore__ inline void Completed(uint32_t peer) {
        const uint32_t host = uint32_t(args_->rank) / 16U * 16U;
        if (peer >= host && peer < host + 16U) return;
        const uint32_t task = peer < host ? peer : peer - 16U;
        if (task % GetBlockNum() == GetBlockIdx()) pending_ &= ~(1U << (task / GetBlockNum()));
    }
    __aicore__ inline void Finish(LocalTensor<uint8_t> scratch) {
        const uint32_t host = uint32_t(args_->rank) / 16U * 16U;
        for (uint32_t task = GetBlockIdx(), bit = 1U; task < uint32_t(args_->rankSize) - 16U;
             task += GetBlockNum(), bit <<= 1U) {
            if (!(pending_ & bit)) continue;
            const uint32_t peer = task < host ? task : task + 16U;
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
                                       uint64_t frameOffset, LocalTensor<uint8_t> records,
                                       uint32_t count, LocalTensor<uint8_t> scratch, uint64_t window = 0U) {
    const uint32_t bytes = (count + 1U) * 32U;
    Check(args, peer, frameOffset, bytes);
    auto stage = args->peerMems[args->rank] + window + kCountTxBase + uint64_t(peer) * bytes;
    DataCopy(Tensor(stage,true), records, bytes);
    Sync<HardEvent::MTE3_S>(7U);
    Put(args,peer,Address(args,peer,frameOffset),stage,bytes,scratch);
}
// One contiguous body plus one separate 32B commit. The receiver must not read
// body ready/count words until it has acquired the frame commit.
__aicore__ inline void CountFrame(const __gm__ Args* args, uint32_t peer,
                                 uint64_t frameOffset, uint64_t commitOffset,
                                 LocalTensor<uint8_t> records, uint32_t count, LocalTensor<uint8_t> scratch) {
    Check(args, peer, frameOffset, uint64_t(count)*32U);
    Check(args, peer, commitOffset, 32U);
    auto dst=Address(args,peer,frameOffset);
    auto commit=Address(args,peer,commitOffset);
    if (!RemoteWrite(args,peer)) {
        DataCopy(Tensor(dst,Rdma(args)),records,count*32U);
        Sync<HardEvent::MTE3_S>(7U);
        DataCopy(Tensor(commit,Rdma(args)),records,32U);
        return;
    }
    auto stage=Stage(args,count*32U);
    DataCopy(Tensor(stage,true),records,count*32U);
    Sync<HardEvent::MTE3_S>(7U);
    Put(args,peer,dst,stage,count*32U,scratch);
    Quiet(peer,scratch);
    Put(args,peer,commit,stage,32U,scratch);
    Quiet(peer,scratch);
}
}
