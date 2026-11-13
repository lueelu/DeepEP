// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef COMBINE_DEDUP_EP_COMBINE_MEMORY_SERVER_DEDUP_KERNEL_H
#define COMBINE_DEDUP_EP_COMBINE_MEMORY_SERVER_DEDUP_KERNEL_H

#include "kernel_operator.h"
#include "../../kernels/common/core_context.h"
#include "ubx_jit_spec.h"

#include "ep_memory_server_dedup_count_layout.h"
#include "comm_args.h"
#include "ep_memory_server_dedup_buffer_layout.h"
#include "ep_memory_server_dedup_combine_config.h"
#include "combine_server_dedup_partial.h"
#include "combine_stage_wire.h"
#include "combine_server_dedup_source_aggregate.h"
#include "combine_server_dedup_select.h"

namespace UbxCombine {
using namespace AscendC;

// Local stage, readiness, DIRECT/partial return and source-token aggregation.
template <uint32_t Group>
class CombineMemoryServerDedupKernel {
public:
    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR expertOut, GM_ADDR assistInfoForCombine, GM_ADDR sendCounts,
                                GM_ADDR expertScales, GM_ADDR tokenType, GM_ADDR relayReadIndex, GM_ADDR sourceMask,
                                GM_ADDR yOut, uint32_t bs, uint32_t h, uint32_t topK, uint32_t experts, uint8_t dtype,
                                int64_t magic, deepep::CoreContext core, bool enableDedup = true)
    {
        coreContext_ = core;
        coreNum_ = core.count;
        recvCores_ = EpCombineRecvCoreCount(coreNum_);
        if (recvCores_ == 0U) {
            Trap();
            return;
        }
        // Task partitions and control rows retain physical core ordering.

        core_ = CombineLogicalCore(core.index, coreNum_);
        sendCores_ = coreNum_ - recvCores_;
        notifyBegin_ = coreNum_ - UbxDispatch::kLlNotifyCoreNum;
        args_ = reinterpret_cast<const __gm__ UbxDispatch::CommArgs*>(commArgs);
        enableDedup_ = enableDedup;
        h_ = h;
        topK_ = topK;
        bs_ = bs;
        experts_ = experts;
        dtype_ = dtype;
        expertAddr_ = expertOut;
        tableAddr_ = assistInfoForCombine;
        relayAddr_ = relayReadIndex;
        typesAddr_ = tokenType;
        weightsAddr_ = expertScales;
        masksAddr_ = sourceMask;
        outputAddr_ = yOut;
        // Load the device-produced prefix through MTE2 rather than caching a
        // mutable GM scalar across launches. This 32B temporary becomes the
        // independent M2 prepare sum only after the scalar count is consumed.
        pipe_.InitBuffer(arena_, core_ >= sendCores_ ? kEpServerDedupCombineRecvUbBytes : kEpServerDedupCombineUbBytes);
        prepareSum_ = Allocate<float>(32U);
        GlobalTensor<int32_t> prefix;
        prefix.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sendCounts));
        auto tokenCount = prepareSum_.ReinterpretCast<int32_t>();

        DataCopyPad(tokenCount, prefix[experts - 1U], {1U, 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
        Sync<HardEvent::MTE2_S>();

        const int32_t tokens = tokenCount.GetValue(0U);
        const uint32_t localExperts = experts / DEEPEP_UBX_LAUNCH_WORLD(args_);
        const uint32_t copies = topK < localExperts ? topK : localExperts;
        const uint64_t capacity = uint64_t(DEEPEP_UBX_LAUNCH_WORLD(args_)) * bs * copies;
        if (tokens < 0 || uint32_t(tokens) > kEpServerDedupCombineMaxTokenNum || uint64_t(tokens) > capacity) {
            Trap();  // Invalid device data must not become a truncated successful launch.
            return;
        }
        tokens_ = uint32_t(tokens);
        lanes_ = EpCombinePayloadLaneCount(tokens_, h_);
        workspace_ = EpCombineM45Layout(tokens_, bs_, h_, topK_, recvCores_);
        if (lanes_ == 0U || workspace_.inputLanes == 0U) {
            Trap();
            return;
        }

        blocks_ = (h_ * 2U + 479U) / 480U;
        laneBytes_ = blocks_ * 480U;
        resultStride_ = blocks_ * 512U;
        window_ = (uint64_t(magic) % 2U) * UbxDispatch::kEpServerDedupWindowBytes;
        control_ = window_ + COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_CONTROL_OFFSET;
        stageStride_ = resultStride_;
        stageBase_ = window_ + COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET + uint64_t(bs) * topK * resultStride_;
        expert_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(expertOut));
        table_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(assistInfoForCombine));
        types_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenType));
        stage_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(args_->peerMems[DEEPEP_UBX_LAUNCH_RANK(args_)] + stageBase_));

        preparePublish_ = Allocate<float>(256U);
        prepareInput_ = Allocate<float>(kCombinePrepareRowBytes + 256U);
        prepareWork_ = Allocate<float>(512U);
        prepareZero_ = Allocate<float>(kCombinePrepareRowBytes);
        readyInput_ = Allocate<float>(kCombineReadyInputBytes);
        readyWork_ = Allocate<float>(256U);
        readySum_ = Allocate<float>(32U);
        readyZero_ = Allocate<float>(kCombineReadyZeroBytes);
        notifyUb_ = Allocate<float>(kCombineNotifyRows * 32U);
        const uint32_t rowBytes = EpCombineAlign(tokens_ == 0U ? 4U : tokens_ * 4U, 256U);
        typeUb_ = Allocate<int32_t>(rowBytes);
        indexUb_ = Allocate<int32_t>(rowBytes);
        packedUb_ = Allocate<int32_t>(rowBytes);
        maskUb_ = Allocate<uint8_t>(EpCombineAlign((tokens_ + 7U) / 8U + 32U, 256U));
        tableUb_ = Allocate<int32_t>(kEpServerDedupCombineTableRows * 12U);
        payloadUb_ = Allocate<uint16_t>(lanes_ * laneBytes_);
        resultUb_ = Allocate<uint8_t>(lanes_ * resultStride_);
    }

    // 完整combine调度：本地待归约行准备后，发送核处理DIRECT回传，接收核处理ready与归约。
    // 去重路径执行通知→等待→partial回传(M4)→源token加权归约(M5)；非去重走直接聚合(M6)。
    // 下列阶段属于同一个设备kernel，复用UB和完成事件，不能当作独立launch相减计时。
    __aicore__ inline void Process()
    {
        if (enableDedup_) {
            CopyLocalPendingTokens();
        }
        if (core_ >= sendCores_) {
            if (enableDedup_) {
                // Prepare notification state only after this core's stage DMA
                // has retired, so generation setup cannot delay data producers.
                groupReady_.Init(args_, control_, core_, sendCores_, recvCores_, readyInput_, readyWork_, readyZero_,
                                 readySum_, notifyUb_);
                if (core_ >= notifyBegin_) {
                    NotifyServerCard();

                    // Notification has been issued; do not add an MTE3 drain.
                }

                ReturnPartialTokens();

                const uint32_t retiredOutputs = AggregateSourceTokens();

                DrainSharedOutputs(retiredOutputs);

                groupReady_.Retire();

            } else {
                AggregateSourceTokensM6();
            }
        } else {
            ReturnDirectTokens();
        }
        // Retires prepare publication, both zero sources and server notification;
        // no arena/pipe reset or exit may precede completion of their DMA reads.

        PipeBarrier<PIPE_ALL>();
        // All resident control sources and output lanes have retired. Pending
        // return GM slots are independent of M5 UB and are reclaimed only here.

        returnBatch_.Flush(arena_.Get<uint8_t>());
    }

    // SuperKernel sub-task boundary. Retires the member TPipe without the
    // implicit whole-pipe barrier; all business DMA and output retirement has
    // already completed in Process().
    __aicore__ inline void Destroy() { pipe_.DestroyWithoutPipeAll(); }

private:
    __aicore__ inline void AggregateSourceTokensM6()
    {
        ServerDedupSourceAggregate aggregate;
        // Count DMA has retired. Auto uses the same final-stage scratch view as
        // M5, but owns its output events because M4 did not run on this launch.
        aggregate.InitDirectShared(
            arena_.Get<uint8_t>()[kEpServerDedupCombineResidentUbBytes],
            args_->peerMems[DEEPEP_UBX_LAUNCH_RANK(args_)] + window_ + COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET, weightsAddr_,
            outputAddr_, bs_, h_, topK_, dtype_, core_ - sendCores_, recvCores_, workspace_);

        aggregate.Process();

        aggregate.DrainM6();
    }
    __aicore__ inline uint32_t AggregateSourceTokens()
    {
        ServerDedupSourceAggregate aggregate;
        aggregate.Init(arena_.Get<uint8_t>()[kEpServerDedupCombineResidentUbBytes],
                       args_->peerMems[DEEPEP_UBX_LAUNCH_RANK(args_)] + window_ + COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET,
                       masksAddr_, weightsAddr_, outputAddr_, bs_, h_, topK_, dtype_, core_ - sendCores_, recvCores_,
                       workspace_, partialTasks_, enableDedup_);

        return aggregate.Process();
    }
    __aicore__ inline void DrainSharedOutputs(uint32_t retiredOutputs = 0U)
    {
        // M5 already consumed slots repurposed as inputs. Retire the remaining
        // physical slots exactly once, including unused seeds and M5 outputs.
        for (uint32_t lane = 0; lane < workspace_.outputLanes; ++lane)
            if (((retiredOutputs >> lane) & 1U) == 0U) WaitFlag<HardEvent::MTE3_V>(lane);
    }
    template <class T>
    __aicore__ inline LocalTensor<T> Allocate(uint32_t bytes)
    {
        auto result = arena_.Get<uint8_t>()[usedBytes_].template ReinterpretCast<T>();
        usedBytes_ += bytes;
        return result;
    }
    template <HardEvent E>
    __aicore__ inline void Sync()
    {
        const TEventID id = pipe_.FetchEventID(E);
        SetFlag<E>(id);
        WaitFlag<E>(id);
    }
    __aicore__ inline GlobalTensor<float> Control(uint32_t rank, uint64_t offset)
    {
        GlobalTensor<float> result;
        result.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(args_->peerMems[rank] + control_ + offset));
        return result;
    }
    __aicore__ inline void SelectRows(int32_t type)
    {
        selected_ = 0U;
        if (tokens_ != 0U) {
            DataCopyPad(typeUb_, types_, {1U, tokens_ * 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
            Sync<HardEvent::MTE2_V>();

            Compares(maskUb_, typeUb_, type, CMPMODE::EQ, tokens_);
            CreateVecIndex(indexUb_, int32_t(0), tokens_);
            PipeBarrier<PIPE_V>();
            GatherMask(packedUb_, indexUb_, maskUb_.ReinterpretCast<uint32_t>(), true, tokens_, {1U, 1U, 0U, 0U},
                       selected_);
            Sync<HardEvent::V_S>();
        }
    }
    __aicore__ inline void CopyLocalPendingTokens()
    {
        SelectRows(3);

        // Result lanes have no users in this phase. Keep zero gaps in a source
        // sized by packet blocks, not by the unrelated prepare-core row count.
        auto stageZeros = resultUb_.ReinterpretCast<float>();
        Duplicate<float>(stageZeros, 0.0f, blocks_ * 8U);
        Sync<HardEvent::V_MTE3>();
        const uint32_t base = uint32_t(selected_) / coreNum_, extra = uint32_t(selected_) % coreNum_;
        const uint32_t count = base + (core_ < extra ? 1U : 0U);
        const uint32_t start = core_ * base + (core_ < extra ? core_ : extra);

        const uint32_t activeLanes = lanes_;
        for (uint32_t lane = 0; lane < activeLanes; ++lane) SetFlag<HardEvent::MTE3_MTE2>(lane);
        uint32_t cacheBegin = 0U, cacheEnd = 0U;
        for (uint32_t task = 0; task < count; ++task) {
            const uint32_t row = uint32_t(packedUb_.GetValue(start + task));

            CacheTableRows(row, start, task, count, cacheBegin, cacheEnd);

            const uint32_t slot = uint32_t(tableUb_.GetValue((row - cacheBegin) * 3U));
            const uint32_t lane = task % activeLanes;

            WaitFlag<HardEvent::MTE3_MTE2>(lane);

            auto payload = payloadUb_[lane * laneBytes_ / 2U];
            DataCopyPad(payload, expert_[uint64_t(row) * h_], {1U, h_ * 2U, 0U, 0U, 0U}, {false, 0U, 0U, uint16_t(0)});
            SetFlag<HardEvent::MTE2_MTE3>(lane);
            WaitFlag<HardEvent::MTE2_MTE3>(lane);

            StoreStageWire(stage_[uint64_t(slot) * stageStride_ / 2U], payload, stageZeros, h_ * 2U);
            SetFlag<HardEvent::MTE3_MTE2>(lane);
        }

        // Also consumes seeded, unused lane events. All stage stores complete
        // before publication; individual tokens never wait immediately after store.
        for (uint32_t lane = 0; lane < activeLanes; ++lane) WaitFlag<HardEvent::MTE3_MTE2>(lane);

        Duplicate<float>(preparePublish_, 0.0f, 64U);
        Sync<HardEvent::V_S>();
        for (uint32_t j = 0; j < 8U; ++j) preparePublish_.SetValue(j * 8U, 1.0f);
        Sync<HardEvent::S_MTE3>();
        auto done = Control(DEEPEP_UBX_LAUNCH_RANK(args_), COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_PREPARE_DONE_OFFSET);
        DataCopyPad(done[core_ * 8U], preparePublish_, {8U, 32U, 0U, kCombinePrepareRowBytes - 32U, 0U});
    }
    __aicore__ inline void CacheTableRows(uint32_t row, uint32_t start, uint32_t task, uint32_t count,
                                          uint32_t& cacheBegin, uint32_t& cacheEnd)
    {
        if (row >= cacheEnd) {
            cacheBegin = row;
            cacheEnd = row + 1U;
            for (uint32_t next = task + 1U; next < count; ++next) {
                const uint32_t nextRow = uint32_t(packedUb_.GetValue(start + next));
                if (nextRow - cacheBegin >= kEpServerDedupCombineTableRows) break;
                cacheEnd = nextRow + 1U;
            }
            // The old metadata has no users after its scalar offset is read.
            // Payload DMA owns separate lanes, so changing this cache needs
            // no payload drain and preserves the lane sequence across batches.
            DataCopyPad(tableUb_, table_[uint64_t(cacheBegin) * 3U], {1U, (cacheEnd - cacheBegin) * 12U, 0U, 0U, 0U},
                        {false, 0U, 0U, 0});
            Sync<HardEvent::MTE2_S>();
        }
    }
    // The small-table read is issued after type DMA completes, then overlaps
    // the type filter and row-index Vector work. M3 only read this table cache;
    // preserve its original Vector->MTE2 ownership fence before overwriting it.
    __aicore__ inline bool PreloadDirectRoutes()
    {
        if (tokens_ > kEpServerDedupCombineTableRows) return false;

        Sync<HardEvent::V_MTE2>();
        DataCopyPad(tableUb_, table_, {1U, tokens_ * 12U, 0U, 0U, 0U}, {false, 0U, 0U, 0});

        return true;
    }
    __aicore__ inline void WaitDirectRoutePreload()
    {
        Sync<HardEvent::MTE2_V>();
    }
    __aicore__ inline void PrepareDirectRows()
    {
        selected_ = 0U;
        directRoutesCached_ = false;
        if (tokens_ == 0U) return;
        bool preloaded = false;
        if (enableDedup_) {
            DataCopyPad(typeUb_, types_, {1U, tokens_ * 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
            Sync<HardEvent::MTE2_V>();

            preloaded = PreloadDirectRoutes();

            Compares(maskUb_, typeUb_, int32_t(1), CMPMODE::EQ, (tokens_ + 63U) / 64U * 64U);
            CreateVecIndex(indexUb_, int32_t(0), tokens_);
            PipeBarrier<PIPE_V>();
            GatherMask(packedUb_, indexUb_, maskUb_.ReinterpretCast<uint32_t>(), true, tokens_, {1U, 1U, 0U, 0U},
                       selected_);
            Sync<HardEvent::V_S>();

        } else {
            preloaded = PreloadDirectRoutes();
            // Every row is a direct row without the dedup token type table.
            CreateVecIndex(packedUb_, int32_t(0), tokens_);
            selected_ = tokens_;
            Sync<HardEvent::V_S>();
        }

        if (selected_ == 0U) {
            // An eager read must retire even on a type2/type3-only rank.
            if (preloaded) WaitDirectRoutePreload();
            return;
        }
        if (preloaded) {
            // packedUb_ already contains the selected original rows. Gather
            // their ranks directly instead of gathering the whole table and
            // compacting the rank array a second time. No payload lane aliases.
            Muls(indexUb_, packedUb_, int32_t(12), uint32_t(selected_));
            PipeBarrier<PIPE_V>();
            WaitDirectRoutePreload();

            Gather(typeUb_, tableUb_, indexUb_.ReinterpretCast<uint32_t>(), 0U, uint32_t(selected_));

            Sync<HardEvent::V_S>();
        } else {
            const uint32_t chunk = tokens_ < kEpServerDedupCombineTableRows ? tokens_ : kEpServerDedupCombineTableRows;
            CreateVecIndex(typeUb_, int32_t(0), chunk);
            PipeBarrier<PIPE_V>();
            Muls(typeUb_, typeUb_, int32_t(12), chunk);  // Table2 byte offsets
            for (uint32_t begin = 0U; begin < tokens_; begin += chunk) {
                const uint32_t span = tokens_ - begin < chunk ? tokens_ - begin : chunk;

                Sync<HardEvent::V_MTE2>();  // table cache is no longer a Gather source
                DataCopyPad(tableUb_, table_[uint64_t(begin) * 3U], {1U, span * 12U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
                Sync<HardEvent::MTE2_V>();

                Gather(indexUb_[begin], tableUb_, typeUb_.ReinterpretCast<uint32_t>(), 0U, span);
            }

            PipeBarrier<PIPE_V>();
            // Keep two persistent compact arrays: ranks in typeUb_, original rows
            // in packedUb_. indexUb_ becomes the per-server compact output list.
            if (enableDedup_) {
                GatherMask(typeUb_, indexUb_, maskUb_.ReinterpretCast<uint32_t>(), true, tokens_, {1U, 1U, 0U, 0U},
                           selected_);
            } else {
                Muls(typeUb_, indexUb_, int32_t(1), tokens_);
            }
            Sync<HardEvent::V_S>();
        }
        // With one route tile, preserve the immutable Table2 in its existing
        // 6 KiB cache. DIRECT does not use the resident ready checker: its
        // 256B work area can hold this <=64B upper comparison mask instead.
        directRoutesCached_ = tokens_ <= kEpServerDedupCombineTableRows;
        auto upper = tableUb_.ReinterpretCast<uint8_t>();
        if (directRoutesCached_) upper = readyWork_.ReinterpretCast<uint8_t>();
        ServerDedupRankMask(maskUb_, upper, typeUb_, 0, int32_t(DEEPEP_UBX_LAUNCH_WORLD(args_)),
                            uint32_t(selected_));
        uint64_t valid = 0U;
        GatherMask(indexUb_, packedUb_, maskUb_.ReinterpretCast<uint32_t>(), true, uint32_t(selected_),
                   {1U, 1U, 0U, 0U}, valid);
        Sync<HardEvent::V_S>();
        if (valid != selected_) Trap();
    }
    __aicore__ inline uint32_t SelectDirectServer(uint32_t server)
    {
        auto upper = tableUb_.ReinterpretCast<uint8_t>();
        if (directRoutesCached_) upper = readyWork_.ReinterpretCast<uint8_t>();
        ServerDedupRankMask(maskUb_, upper, typeUb_, int32_t(server * Group), int32_t((server + 1U) * Group),
                            uint32_t(selected_));
        uint64_t count = 0U;
        GatherMask(indexUb_, packedUb_, maskUb_.ReinterpretCast<uint32_t>(), true, uint32_t(selected_),
                   {1U, 1U, 0U, 0U}, count);
        Sync<HardEvent::V_S>();
        return uint32_t(count);
    }
    __aicore__ inline void ReturnDirectTokens()
    {
        const uint32_t directCores = sendCores_;
        PrepareDirectRows();

        if (selected_ == 0U) return;

        if (UbxTransport::Rdma(args_)) returnBatch_.Init(args_, resultStride_, coreContext_);
        const uint32_t servers = DEEPEP_UBX_LAUNCH_WORLD(args_) / Group;
        const uint32_t ownServer = uint32_t(DEEPEP_UBX_LAUNCH_RANK(args_)) / Group;
        uint32_t owner = 0U, sendSequence = 0U;
        uint32_t activeLanes = lanes_;

        uint64_t flagMask[2] = {0xffU, 0U};
        for (uint32_t lane = 0; lane < activeLanes; ++lane) {
            // The entire 32B Flag is initialized once and is not touched by Copy.
            Duplicate<float>(resultUb_[lane * resultStride_].ReinterpretCast<float>()[120U], 1.0f, flagMask,
                             uint8_t(blocks_), 1U, 16U);
            SetFlag<HardEvent::V_MTE2>(lane);
            SetFlag<HardEvent::MTE3_V>(lane);
        }
        PipeBarrier<PIPE_V>();

        for (uint32_t step = 0U; step < servers; ++step) {
            const uint32_t server = CombineReturnServer<Group>(servers, ownServer, step);
            const uint32_t count = SelectDirectServer(server);
            const uint32_t first = ServerDedupFirstOwned(core_, owner, directCores);
            owner = (owner + count) % directCores;
            for (uint32_t task = first; task < count; task += directCores) {
                const uint32_t row = uint32_t(indexUb_.GetValue(task));
                // Reuse a complete resident snapshot when available; otherwise
                // reload only owned rows. Persistent rank/row lists stay intact.
                auto route = tableUb_;
                if (directRoutesCached_) {
                    route = tableUb_[row * 3U];

                } else {
                    DataCopyPad(tableUb_, table_[uint64_t(row) * 3U], {1U, 12U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
                    Sync<HardEvent::MTE2_S>();
                }
                const uint32_t source = uint32_t(route.GetValue(0U));
                const uint32_t token = uint32_t(route.GetValue(1U));
                const uint32_t topk = uint32_t(route.GetValue(2U));
                const uint32_t lane = sendSequence++ % activeLanes;

                // Input lifetime ends after Vector has consumed its last block.

                WaitFlag<HardEvent::V_MTE2>(lane);

                auto input = payloadUb_[lane * laneBytes_ / 2U];
                DataCopyPad(input, expert_[uint64_t(row) * h_], {1U, h_ * 2U, 0U, 0U, 0U},
                            {false, 0U, 0U, uint16_t(0)});
                SetFlag<HardEvent::MTE2_V>(lane);

                WaitFlag<HardEvent::MTE2_V>(lane);

                // Output lifetime ends only after the previous send has read it.

                WaitFlag<HardEvent::MTE3_V>(lane);

                auto output = resultUb_[lane * resultStride_];
                auto dstWords = output.ReinterpretCast<int32_t>();
                auto srcWords = input.ReinterpretCast<int32_t>();
                // Same nonquantized vector packing as Dispatch, without its tail.
                // Input UB covers blocks_*480B even though GM reads exactly 2H.
                Copy(dstWords, srcWords, uint64_t(64U), uint8_t(blocks_), {1U, 1U, 16U, 15U});
                Copy(dstWords[64U], srcWords[64U], uint64_t(56U), uint8_t(blocks_), {1U, 1U, 16U, 15U});
                SetFlag<HardEvent::V_MTE2>(lane);
                SetFlag<HardEvent::V_MTE3>(lane);
                WaitFlag<HardEvent::V_MTE3>(lane);

                // One destination owns this output. Its body is overwritten on
                // the next lane use, so SDK scratch need not be restored.
                const uint64_t offset = window_ + COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET +
                                        (uint64_t(token) * topK_ + topk) * resultStride_;

                SendReturnPacket(returnBatch_, args_, source, offset, output, resultStride_);

                SetFlag<HardEvent::MTE3_V>(lane);
            }
            // No server fence: the next scan can overlap pending output DMA.
        }

        // Final retirement includes unused seeded lanes, never a per-token drain.
        for (uint32_t lane = 0; lane < activeLanes; ++lane) {
            WaitFlag<HardEvent::V_MTE2>(lane);
            WaitFlag<HardEvent::MTE3_V>(lane);
        }
    }
    // M4→M5交接需保留尚未完成的MTE3输出槽；只等待当前要复用的输入/向量资源。
    // 修改此处同步必须联查Partial与SourceAggregate的槽位生命周期，避免提前覆盖发送数据。
    __aicore__ inline void ReturnPartialTokens()
    {
        // M1 payload lanes are retired; complete this core's M2 publications
        // before reusing scratch. Independent control buffers remain resident.
        PipeBarrier<PIPE_ALL>();
        // Seed exactly once, including empty M4. M5 inherits these events.
        for (uint32_t lane = 0; lane < workspace_.outputLanes; ++lane) SetFlag<HardEvent::MTE3_V>(lane);
        ServerDedupPartial<Group> partial;
        partial.Init(args_, &pipe_, arena_.Get<uint8_t>()[kEpServerDedupCombineResidentUbBytes], expertAddr_,
                     tableAddr_, relayAddr_, typesAddr_, tokens_, bs_, h_, topK_, experts_, dtype_, window_, workspace_,
                     &returnBatch_, coreContext_);

        partial.SetReady(&groupReady_);
        partialTasks_ = partial.Process();

        // Retire only input and Vector users before scalar metadata reuse.
        // M4 output slots and MTE3_V events stay live for M5's per-slot waits.
        Sync<HardEvent::MTE2_S>();
        Sync<HardEvent::V_S>();
    }
    __aicore__ inline void WaitFlags(GlobalTensor<float> gm, LocalTensor<float> input, LocalTensor<float> work,
                                     LocalTensor<float> sum, uint32_t flags, bool incoming, bool dense = false)
    {
        const uint32_t bytes = dense ? flags * 4U : flags * 32U;
        while (true) {
            // The previous reduction has completed before MTE2 overwrites input.
            Sync<HardEvent::V_MTE2>();
            DataCopyPad(input, gm, {1U, bytes, 0U, 0U, 0U}, {false, 0U, 0U, 0.0f});
            Sync<HardEvent::MTE2_V>();

            if (dense)
                ReduceSum<float>(sum, input, work, int32_t(flags));
            else
                ReduceSum<float>(sum, input, work, 1U, flags, 1U);
            Sync<HardEvent::V_S>();
            // Dense rows contain zero padding after each producer rank's cores.
            const bool ready = sum.GetValue(0U) == float(dense ? Group * coreNum_ : flags);

            if (ready) {
                return;
            }
        }
    }
    __aicore__ inline void NotifyServerCard()
    {
        auto done = Control(DEEPEP_UBX_LAUNCH_RANK(args_), COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_PREPARE_DONE_OFFSET +
                                             uint64_t(core_ - notifyBegin_) * kCombinePrepareRowBytes);
        WaitFlags(done, prepareInput_, prepareWork_, prepareSum_, coreNum_, false);

        Duplicate<float>(prepareZero_, 0.0f, coreNum_ * 8U);
        Duplicate<float>(notifyUb_, 0.0f, recvCores_ * 8U);
        Sync<HardEvent::V_S>();
        for (uint32_t r = 0; r < recvCores_; ++r) notifyUb_.SetValue(r * 8U, 1.0f);
        Sync<HardEvent::V_MTE3>();
        Sync<HardEvent::S_MTE3>();
        DataCopyPad(done, prepareZero_, {1U, coreNum_ * 32U, 0U, 0U, 0U});

        // Eight notification cores cover all peers in the logical group.
        // Ready rows retain 512-byte capacity, independent of active group size.
        const uint32_t groupBase = uint32_t(DEEPEP_UBX_LAUNCH_RANK(args_)) / Group * Group;
        for (uint32_t local = core_ - notifyBegin_; local < Group; local += 8U) {
            // Keep publication, polling and clearing on the same default
            // L2 view. This ready table never crosses a physical host.
            if (!UbxTransport::SameHost(args_, groupBase + local) || !args_->peerMems[groupBase + local]) {
                Trap();
                return;
            }
            DataCopyPad(Control(groupBase + local, COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET +
                                                       (uint32_t(DEEPEP_UBX_LAUNCH_RANK(args_)) % Group) * 32U),
                        notifyUb_, {uint16_t(recvCores_), 32U, 0U, 480U, 0U});
        }
    }
    __aicore__ inline void WaitServerReady()
    {
        auto ready = Control(DEEPEP_UBX_LAUNCH_RANK(args_), COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET +
                                              uint64_t(core_ - sendCores_) * 512U);
        WaitFlags(ready, readyInput_, readyWork_, readySum_, Group, true);
        const uint32_t resetBytes = 512U;

        Duplicate<float>(readyZero_, 0.0f, resetBytes / 4U);
        Sync<HardEvent::V_MTE3>();
        DataCopyPad(ready, readyZero_, {1U, resetBytes, 0U, 0U, 0U});
    }

    CombineGroupReady<Group> groupReady_;

    EpCombineM45WorkspaceLayout workspace_{};
    ReturnBatch returnBatch_;
    deepep::CoreContext coreContext_{};
    uint32_t partialTasks_ = 0U;
    uint32_t coreNum_ = 0, sendCores_ = 0, recvCores_ = 0, notifyBegin_ = 0;
    TPipe pipe_;
    TBuf<> arena_;
    const __gm__ UbxDispatch::CommArgs* args_ = nullptr;
    GM_ADDR expertAddr_ = nullptr;
    GM_ADDR typesAddr_ = nullptr;
    GM_ADDR tableAddr_ = nullptr;
    GM_ADDR relayAddr_ = nullptr;
    GM_ADDR weightsAddr_ = nullptr;
    GM_ADDR masksAddr_ = nullptr;
    GM_ADDR outputAddr_ = nullptr;
    bool enableDedup_ = false;
    uint32_t bs_ = 0, experts_ = 0;
    uint8_t dtype_ = 0;
    bool directRoutesCached_ = false;
    uint32_t core_ = 0, h_ = 0, topK_ = 0, blocks_ = 0, resultStride_ = 0, tokens_ = 0, lanes_ = 0, laneBytes_ = 0,
             usedBytes_ = 0;
    uint64_t selected_ = 0, window_ = 0, control_ = 0, stageBase_ = 0, stageStride_ = 0;
    GlobalTensor<uint16_t> expert_, stage_;
    GlobalTensor<int32_t> table_, types_;
    LocalTensor<int32_t> typeUb_, indexUb_, packedUb_, tableUb_;
    LocalTensor<uint8_t> maskUb_, resultUb_;
    LocalTensor<uint16_t> payloadUb_;
    LocalTensor<float> preparePublish_, prepareInput_, prepareWork_, prepareSum_, prepareZero_;
    LocalTensor<float> readyInput_, readyWork_, readySum_, readyZero_, notifyUb_;
};
}  // namespace UbxCombine
#endif
