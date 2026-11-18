#include "clock_alignment.h"

#include <cstdint>

#include "kernel_operator.h"
#include "shmem.h"

#include "../dispatch/low_latency/common/comm_args.h"
#include "../dispatch/low_latency/device/ll_dispatch_c_api.h"
#include "../dispatch/low_latency/device/ll_transport.h"

namespace {
using Args = DispatchDedup::CommArgs;
using Protocol = deepep::clock_alignment::Protocol;
using Sample = deepep::clock_alignment::Sample;
using ClockState = deepep::clock_alignment::ClockState;
using SampleStatus = deepep::clock_alignment::SampleStatus;

__aicore__ inline uint64_t Clock()
{
    return static_cast<uint64_t>(AscendC::GetSystemCycle());
}

__aicore__ inline uint64_t Midpoint(uint64_t first, uint64_t last)
{
    return first + (last - first) / 2U;
}

__aicore__ inline bool Difference(uint64_t lhs, uint64_t rhs, int64_t* result)
{
    constexpr uint64_t kMax = uint64_t(0x7fffffffffffffffULL);
    if (lhs >= rhs) {
        const uint64_t amount = lhs - rhs;
        if (amount > kMax) return false;
        *result = static_cast<int64_t>(amount);
        return true;
    }
    const uint64_t amount = rhs - lhs;
    if (amount > kMax) return false;
    *result = -static_cast<int64_t>(amount);
    return true;
}

__aicore__ inline uint64_t Slot(uint64_t base, uint32_t rank)
{
    return base + uint64_t(rank) * deepep::clock_alignment::kSlotBytes;
}

__aicore__ inline void ResetProtocol(__ubuf__ Protocol* protocol, uint64_t magic,
                                     uint64_t sequence, ClockState state)
{
    protocol->magic = magic;
    protocol->sequence = sequence;
    protocol->state = uint64_t(state);
    protocol->t1 = 0U;
    protocol->t2 = 0U;
    protocol->reserved0 = 0U;
    protocol->reserved1 = 0U;
    protocol->reserved2 = 0U;
}

__aicore__ inline void WriteSample(GM_ADDR samples, uint32_t index, __ubuf__ Sample* sample)
{
    LlDispatchC::Store(
        samples + uint64_t(index) * sizeof(Sample), sample, sizeof(Sample), 1U, 0U, 0U, true);
    LlDispatchC::Sync<PIPE_MTE3, PIPE_S>();
}

__aicore__ inline void Fail(GM_ADDR samples, uint32_t index, SampleStatus status)
{
    auto* sample = LlDispatchC::Ub<Sample>(256U);
    sample->t0 = sample->t1 = sample->t2 = sample->t3 = 0U;
    sample->roundTripCycles = 0U;
    sample->offsetCycles = 0;
    sample->sequence = index;
    sample->status = uint32_t(status);
    sample->reserved = 0U;
    WriteSample(samples, index, sample);
}

__aicore__ inline uint64_t Publish(const __gm__ Args* args, uint32_t peer, uint64_t offset,
                                   __ubuf__ Protocol* protocol, __ubuf__ uint8_t* scratch)
{
    LlTransport::Check(args, peer, offset, sizeof(Protocol));
    const uint64_t begin = Clock();
    if (LlTransport::RemoteWrite(args, peer)) {
        // Calibration is isolated from Dispatch. QP0 owns the first 64B of its
        // existing per-core stage and retires it before the next protocol word.
        auto* stage = args->reservedAddresses[1];
        LlDispatchC::Store(stage, protocol, sizeof(Protocol), 1U, 0U, 0U, true);
        LlDispatchC::Sync<PIPE_MTE3, PIPE_S>();
        const uint64_t postedBegin = Clock();
        LlTransport::Put(args, peer, LlTransport::RemoteAddress(args, peer, offset),
                         stage, sizeof(Protocol), scratch, 7U);
        const uint64_t postedEnd = Clock();
        LlTransport::Quiet(peer, scratch, 7U);
        return Midpoint(postedBegin, postedEnd);
    }
    LlDispatchC::Store(
        LlTransport::Address(args, peer, offset), protocol, sizeof(Protocol), 1U, 0U, 0U, true);
    LlDispatchC::Sync<PIPE_MTE3, PIPE_S>();
    return Midpoint(begin, Clock());
}

__aicore__ inline bool Poll(const __gm__ Args* args, uint64_t offset, uint64_t magic,
                            uint64_t sequence, ClockState state, uint64_t timeout_cycles,
                            __ubuf__ Protocol* snapshot, uint64_t* matched)
{
    const uint64_t begin = Clock();
    auto* local = args->peerMems[args->rank] + offset;
    while (true) {
        const uint64_t readBegin = Clock();
        LlDispatchC::Load(snapshot, local, sizeof(Protocol), 1U, 0U, 0U, true);
        LlDispatchC::Sync<PIPE_MTE2, PIPE_S>();
        const uint64_t readEnd = Clock();
        if (snapshot->magic == magic && snapshot->sequence == sequence &&
            snapshot->state == uint64_t(state)) {
            if (matched) *matched = Midpoint(readBegin, readEnd);
            return true;
        }
        if (timeout_cycles && Clock() - begin >= timeout_cycles) return false;
    }
}

__aicore__ inline bool ValidArgs(const __gm__ Args* args, GM_ADDR samples,
                                 uint32_t sample_count, uint32_t warmup_count)
{
    return args && samples && args->rank >= 0 && args->rank < args->rankSize &&
        args->rankSize >= 2 && args->rankSize <= DispatchDedup::DISPATCH_DEDUP_MAX_RANK_SIZE &&
        args->peerMems[args->rank] && sample_count > 0U &&
        sample_count <= deepep::clock_alignment::kMaxSamples &&
        warmup_count <= deepep::clock_alignment::kMaxWarmupSamples;
}
}  // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void deepep_clock_sync_kernel(
    GM_ADDR comm_args, GM_ADDR samples, uint32_t sample_count, uint32_t warmup_count,
    uint64_t magic, uint64_t timeout_cycles)
{
    if ASCEND_IS_AIV {
    } else {
        return;
    }
    if (AscendC::GetBlockIdx() != 0U) return;
    const auto* args = reinterpret_cast<const __gm__ Args*>(comm_args);
    if (!ValidArgs(args, samples, sample_count, warmup_count)) {
        if (samples && sample_count) Fail(samples, 0U, SampleStatus::InvalidConfiguration);
        return;
    }
    set_ffts_base_addr(args->fftsVal);
    auto* protocol = LlDispatchC::Ub<Protocol>(0U);
    auto* snapshot = LlDispatchC::Ub<Protocol>(64U);
    auto* scratch = LlDispatchC::Ub<uint8_t>(512U);
    auto* sample = LlDispatchC::Ub<Sample>(256U);
    const uint32_t total = warmup_count + sample_count;
    const uint32_t rank = uint32_t(args->rank);
    const uint32_t world = uint32_t(args->rankSize);

    for (uint32_t sequence = 0U; sequence < total; ++sequence) {
        if (rank == 0U) {
            for (uint32_t peer = 1U; peer < world; ++peer) {
                ResetProtocol(protocol, magic, sequence, ClockState::Ready);
                Publish(args, peer, Slot(deepep::clock_alignment::kResponseBase, peer), protocol, scratch);

                uint64_t t1 = 0U;
                if (!Poll(args, Slot(deepep::clock_alignment::kRequestBase, peer), magic, sequence,
                          ClockState::Request, timeout_cycles, snapshot, &t1)) {
                    Fail(samples, sequence < warmup_count ? 0U : sequence - warmup_count,
                         SampleStatus::Timeout);
                    return;
                }
                ResetProtocol(protocol, magic, sequence, ClockState::Response);
                protocol->t1 = t1;
                const uint64_t t2 = Publish(
                    args, peer, Slot(deepep::clock_alignment::kResponseBase, peer), protocol, scratch);
                ResetProtocol(protocol, magic, sequence, ClockState::Timestamp);
                protocol->t2 = t2;
                Publish(args, peer, Slot(deepep::clock_alignment::kTimestampBase, peer), protocol, scratch);
                if (!Poll(args, Slot(deepep::clock_alignment::kRequestBase, peer), magic, sequence,
                          ClockState::Ack, timeout_cycles, snapshot, nullptr)) {
                    Fail(samples, sequence < warmup_count ? 0U : sequence - warmup_count,
                         SampleStatus::Timeout);
                    return;
                }
            }
            if (sequence >= warmup_count) {
                const uint32_t index = sequence - warmup_count;
                const uint64_t now = Clock();
                sample->t0 = sample->t1 = sample->t2 = sample->t3 = now;
                sample->roundTripCycles = 0U;
                sample->offsetCycles = 0;
                sample->sequence = index;
                sample->status = uint32_t(SampleStatus::Success);
                sample->reserved = 0U;
                WriteSample(samples, index, sample);
            }
            continue;
        }

        if (!Poll(args, Slot(deepep::clock_alignment::kResponseBase, rank), magic, sequence,
                  ClockState::Ready, timeout_cycles, snapshot, nullptr)) {
            Fail(samples, sequence < warmup_count ? 0U : sequence - warmup_count, SampleStatus::Timeout);
            return;
        }
        ResetProtocol(protocol, magic, sequence, ClockState::Request);
        const uint64_t t0 = Publish(
            args, 0U, Slot(deepep::clock_alignment::kRequestBase, rank), protocol, scratch);
        uint64_t t3 = 0U;
        if (!Poll(args, Slot(deepep::clock_alignment::kResponseBase, rank), magic, sequence,
                  ClockState::Response, timeout_cycles, snapshot, &t3)) {
            Fail(samples, sequence < warmup_count ? 0U : sequence - warmup_count, SampleStatus::Timeout);
            return;
        }
        const uint64_t t1 = snapshot->t1;
        if (!Poll(args, Slot(deepep::clock_alignment::kTimestampBase, rank), magic, sequence,
                  ClockState::Timestamp, timeout_cycles, snapshot, nullptr)) {
            Fail(samples, sequence < warmup_count ? 0U : sequence - warmup_count, SampleStatus::Timeout);
            return;
        }
        const uint64_t t2 = snapshot->t2;
        ResetProtocol(protocol, magic, sequence, ClockState::Ack);
        Publish(args, 0U, Slot(deepep::clock_alignment::kRequestBase, rank), protocol, scratch);

        if (sequence >= warmup_count) {
            const uint32_t index = sequence - warmup_count;
            sample->t0 = t0;
            sample->t1 = t1;
            sample->t2 = t2;
            sample->t3 = t3;
            sample->sequence = index;
            sample->reserved = 0U;
            int64_t first = 0, second = 0;
            if (t3 >= t0 && t2 >= t1 && t3 - t0 >= t2 - t1 &&
                Difference(t1, t0, &first) && Difference(t2, t3, &second)) {
                const uint64_t elapsed = (t3 - t0) - (t2 - t1);
                sample->roundTripCycles = elapsed;
                sample->offsetCycles = (first < 0) != (second < 0)
                    ? (first + second) / 2
                    : first / 2 + second / 2 + (first % 2 + second % 2) / 2;
                sample->status = uint32_t(SampleStatus::Success);
            } else {
                sample->roundTripCycles = 0U;
                sample->offsetCycles = 0;
                sample->status = uint32_t(SampleStatus::InvalidTiming);
            }
            WriteSample(samples, index, sample);
        }
    }
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void deepep_clock_sleep_at_kernel(
    uint64_t local_deadline, GM_ADDR release_cycles)
{
    if ASCEND_IS_AIV {
    } else {
        return;
    }
    if (AscendC::GetBlockIdx() != 0U) return;
    uint64_t now = Clock();
    while (now < local_deadline) now = Clock();
    if (release_cycles) reinterpret_cast<__gm__ uint64_t*>(release_cycles)[0] = now;
}

extern "C" DEEPEP_EXPORT void deepep_clock_sync_kernel_do(
    void* stream, uint8_t* comm_args, uint8_t* samples, uint32_t sample_count,
    uint32_t warmup_sample_count, uint64_t magic, uint64_t timeout_cycles)
{
    deepep_clock_sync_kernel<<<1, DEEPEP_KERNEL_LAUNCH_UB(0), stream>>>(
        comm_args, samples, sample_count, warmup_sample_count, magic, timeout_cycles);
}

extern "C" DEEPEP_EXPORT void deepep_clock_sleep_at_kernel_do(
    void* stream, uint64_t local_deadline, uint8_t* release_cycles)
{
    deepep_clock_sleep_at_kernel<<<1, DEEPEP_KERNEL_LAUNCH_UB(0), stream>>>(
        local_deadline, release_cycles);
}
