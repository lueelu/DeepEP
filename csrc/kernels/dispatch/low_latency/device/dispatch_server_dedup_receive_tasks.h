// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#ifndef DISPATCH_SERVER_DEDUP_RECEIVE_TASKS_H
#define DISPATCH_SERVER_DEDUP_RECEIVE_TASKS_H

#include "kernel_operator.h"
#include "../common/ep_memory_server_dedup_data_layout.h"

namespace DispatchDedup {
using namespace AscendC;

// R05-01/R05-02: stable physical slot positions and a compact pending list.
// The caller supplies the complete, ready prefix in localExpert/source order.
// Both tables remain in UB until this core has finished clearing its slots.
class ServerDedupReceiveTasks {
public:
    __aicore__ inline void Build(LocalTensor<int32_t> prefix, uint32_t rank, uint32_t ranks, uint32_t experts,
                                 uint32_t bs, uint32_t receiver, TPipe* pipe, uint32_t recvCores)
    {
        const uint32_t total = uint32_t(prefix.GetValue(experts - 1U));
        // 交错分配：receiver 核拥有全局行号 {receiver, receiver+recvCores, ...}。
        // 这样每个核同时拿到早到源与晚到源的 token，relay 转发量也随 token 摊平，
        // 不会再整段继承单个源的到达时刻；输出行由首行和步长直接还原。
        begin_ = receiver;
        stride_ = recvCores;
        count_ = begin_ < total ? (total - 1U - begin_) / stride_ + 1U : 0U;
        pendingCount_ = count_;
        if (count_ == 0U) return;

        const uint32_t bytes = uint32_t(DispatchDedupAlign(uint64_t(count_) * sizeof(uint32_t), 32U));
        pipe->InitBuffer(tokenPosBuf_, bytes);
        pipe->InitBuffer(pendingIdxBuf_, bytes);
        tokenPos_ = tokenPosBuf_.Get<uint32_t>();
        pendingIdx_ = pendingIdxBuf_.Get<uint32_t>();

        const uint32_t localExperts = experts / ranks;
        uint32_t record = 0U;
        // 行号递增，二分只需在剩余 record 区间内进行；小批量下约 1µs。
        for (uint32_t j = 0; j < count_; ++j) {
            const uint32_t row = begin_ + j * stride_;
            uint32_t low = record, high = experts;
            while (low < high) {
                const uint32_t middle = low + (high - low) / 2U;
                if (uint32_t(prefix.GetValue(middle)) <= row)
                    low = middle + 1U;
                else
                    high = middle;
            }
            record = low;
            const uint32_t previous = record == 0U ? 0U : uint32_t(prefix.GetValue(record - 1U));
            const uint32_t source = record % ranks;
            const uint32_t localExpert = record / ranks;
            const uint32_t region = ((source + rank) % ranks) * localExperts + localExpert;
            tokenPos_.SetValue(j, region * bs + row - previous);
            pendingIdx_.SetValue(j, j);
        }
    }

    __aicore__ inline uint32_t Begin() const
    {
        return begin_;
    }
    // 本核拥有的全局行号步长；输出行 = Begin() + j * Stride()。
    __aicore__ inline uint32_t Stride() const
    {
        return stride_;
    }
    __aicore__ inline uint32_t Count() const
    {
        return count_;
    }
    __aicore__ inline uint32_t PendingCount() const
    {
        return pendingCount_;
    }
    __aicore__ inline uint32_t Position(uint32_t j)
    {
        return tokenPos_.GetValue(j);
    }
    __aicore__ inline uint32_t PendingIndex(uint32_t p)
    {
        return pendingIdx_.GetValue(p);
    }

    // Call only after every required write for PendingIndex(p) has been issued.
    // Recheck p after removal; output row and cleanup ownership still use j.
    __aicore__ inline void RemovePending(uint32_t p)
    {
        --pendingCount_;
        pendingIdx_.SetValue(p, pendingIdx_.GetValue(pendingCount_));
    }

private:
    uint32_t begin_ = 0U, stride_ = 1U, count_ = 0U, pendingCount_ = 0U;
    TBuf<> tokenPosBuf_, pendingIdxBuf_;
    LocalTensor<uint32_t> tokenPos_, pendingIdx_;
};
}  // namespace DispatchDedup

#endif
