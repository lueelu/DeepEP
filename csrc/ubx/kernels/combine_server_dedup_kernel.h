// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef COMBINE_DEDUP_EP_COMBINE_MEMORY_SERVER_DEDUP_KERNEL_H
#define COMBINE_DEDUP_EP_COMBINE_MEMORY_SERVER_DEDUP_KERNEL_H

#include <kernel_operator.h>

#include <ubx/kernels/ep_memory_server_dedup_count_layout.h>
#include <ubx/kernels/comm_args.h>
#include <ubx/kernels/ep_memory_server_dedup_buffer_layout.h>
#include <ubx/kernels/ep_memory_server_dedup_combine_config.h>
#include <ubx/kernels/combine_server_dedup_partial.h>
#include <ubx/kernels/combine_stage_wire.h>
#include <ubx/kernels/combine_return_schedule.h>
#include <ubx/kernels/combine_server_dedup_source_aggregate.h>
#include <ubx/kernels/combine_server_dedup_select.h>

namespace UbxCombine {
using namespace AscendC;

// Member PUTs, DIRECT/partial returns and final per-source-token aggregation.
template <uint32_t Group, uint32_t UbBytes>
class CombineMemoryServerDedupKernel {
    static_assert(UbBytes > 0 && UbBytes <= kEpServerDedupCombineMaxUbBytes && UbBytes % 32U == 0,
                  "invalid Combine UB arena size");

public:
    __aicore__ inline void Init(GM_ADDR commArgs, GM_ADDR expertOut, GM_ADDR assistInfoForCombine, GM_ADDR sendCounts,
                                GM_ADDR expertScales, GM_ADDR tokenType, GM_ADDR relayReadIndex, GM_ADDR sourceMask,
                                GM_ADDR yOut, uint32_t bs, uint32_t h, uint32_t topK,
                                uint32_t experts, uint8_t dtype, int64_t magic, bool enableDedup = true)
    {
        coreNum_ = GetBlockNum();
        recvCores_ = EpCombineRecvCoreCount(coreNum_);
        if (recvCores_ == 0U) {
            Trap();
            return;
        }
        // Task partitions retain physical core ordering.

        core_ = CombineLogicalCore(GetBlockIdx(), coreNum_);
        sendCores_ = coreNum_ - recvCores_;
        args_ = reinterpret_cast<const __gm__ UbxDispatch::CommArgs*>(commArgs);
        enableDedup_ = enableDedup;
        h_ = h;
        topK_ = topK;
        bs_ = bs;
        dtype_ = dtype;
        expertAddr_ = expertOut;
        tableAddr_ = assistInfoForCombine;
        relayAddr_ = relayReadIndex;
        typesAddr_ = tokenType;
        weightsAddr_ = expertScales;
        masksAddr_ = sourceMask;
        outputAddr_ = yOut;
        // Load the device-produced prefix through MTE2 rather than caching a
        // mutable GM scalar across launches.
        pipe_.InitBuffer(arena_, UbBytes);
        auto tokenCount = Allocate<int32_t>(32U);
        GlobalTensor<int32_t> prefix;
        prefix.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sendCounts));

        DataCopyPad(tokenCount, prefix[experts - 1U], {1U, 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
        Sync<HardEvent::MTE2_S>();

        const int32_t tokens = tokenCount.GetValue(0U);
        const uint32_t localExperts = experts / uint32_t(args_->rankSize);
        const uint32_t copies = topK < localExperts ? topK : localExperts;
        const uint64_t capacity = uint64_t(args_->rankSize) * bs * copies;
        if (tokens < 0 || uint64_t(tokens) > capacity) {
            Trap();  // Invalid device data must not become a truncated successful launch.
            return;
        }
        totalRows_ = uint32_t(tokens);
        const auto plan = PlanPutRows(totalRows_, bs_, h_, topK_, recvCores_, UbBytes);
        tokens_ = plan.rows;
        lanes_ = plan.lanes;
        workspace_ = plan.receive;
        if (lanes_ == 0U || workspace_.inputLanes == 0U) {
            Trap();
            return;
        }

        blocks_ = (h_ * 2U + 479U) / 480U;
        laneBytes_ = blocks_ * 480U;
        resultStride_ = blocks_ * 512U;
        window_ = UbxDispatch::DispatchServerDedupWindowOffset(magic);
        stageBase_ = window_ + COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET + uint64_t(bs) * topK * resultStride_;
        // An exact-sized Dispatch may leave this rank with no expert storage.
        if (totalRows_ != 0U) expert_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(expertOut));
        table_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(assistInfoForCombine));
        types_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenType));
        directUpperMask_ = Allocate<uint8_t>(256U);
        returnBatch_.SetWorkspace(Allocate<uint8_t>(EpCombineRdmaBytes(h_)));
        const uint32_t rowBytes = EpCombineAlign(tokens_ == 0U ? 4U : tokens_ * 4U, 256U);
        typeUb_ = Allocate<int32_t>(rowBytes);
        indexUb_ = Allocate<int32_t>(rowBytes);
        packedUb_ = Allocate<int32_t>(rowBytes);
        maskUb_ = Allocate<uint8_t>(EpCombineAlign((tokens_ + 7U) / 8U + 32U, 256U));
        tableUb_ = Allocate<int32_t>(kEpServerDedupCombineTableRows * 12U);
        payloadUb_ = Allocate<uint16_t>(lanes_ * laneBytes_);
        resultUb_ = Allocate<uint8_t>(lanes_ * resultStride_);
    }

    __aicore__ inline void Process()
    {
        const uint32_t tile = tokens_;
        if (enableDedup_) {
            if (tile == totalRows_) PushMembers();
            else ForEachTile<RowPhase::Member>(tile);
        }
        // Keep NIC staging slots live across tiles until retirement.
        returnBatch_.Init(args_, resultStride_);
        if (core_ < sendCores_) {
            if (tile == totalRows_) ReturnDirectTokens();
            else ForEachTile<RowPhase::Direct>(tile);
        } else if (enableDedup_) {
            if (tile == totalRows_) ReturnPartialTokens(0U);
            else ForEachTile<RowPhase::Partial>(tile);
            DrainSharedOutputs(AggregateSourceTokens());
        } else {
            AggregateSourceTokensM6();
        }
        PipeBarrier<PIPE_ALL>();
        returnBatch_.Flush();
    }

private:
    enum class RowPhase { Member, Direct, Partial };

    template <RowPhase Phase>
    __aicore__ inline void ForEachTile(uint32_t tile)
    {
        for (uint32_t first = 0U; first < totalRows_;) {
            tokens_ = totalRows_ - first < tile ? totalRows_ - first : tile;
            const uint64_t row = first;
            if constexpr (Phase == RowPhase::Partial) {
                ReturnPartialTokens(first);
                if (first + tokens_ < totalRows_) DrainSharedOutputs();
            } else {
                expert_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(expertAddr_) + row * h_);
                table_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tableAddr_) + row * 3U);
                types_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(typesAddr_) + row);
                if constexpr (Phase == RowPhase::Member) PushMembers();
                else ReturnDirectTokens();
            }
            first += tokens_;
        }
    }

    __aicore__ inline void AggregateSourceTokensM6()
    {
        ServerDedupSourceAggregate aggregate;
        // Count DMA has retired. Direct mode uses the same final-stage scratch view as
        // M5, but owns its output events because M4 did not run on this launch.
        aggregate.InitDirectShared(
            arena_.Get<uint8_t>()[workspace_.residentBytes],
            args_->peerMems[args_->rank] + window_ + COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET, weightsAddr_,
            outputAddr_, bs_, h_, topK_, dtype_, core_ - sendCores_, recvCores_, workspace_);

        aggregate.Process();

        aggregate.DrainM6();
    }
    __aicore__ inline uint32_t AggregateSourceTokens()
    {
        ServerDedupSourceAggregate aggregate;
        aggregate.Init(arena_.Get<uint8_t>()[workspace_.residentBytes],
                       args_->peerMems[args_->rank] + window_ + COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET,
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
    __aicore__ inline void SelectRows(int32_t type)
    {
        selected_ = 0U;
        if (tokens_ != 0U) {
            DataCopyPad(typeUb_, types_, {1U, tokens_ * 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
            Sync<HardEvent::MTE2_V>();

            Compares(maskUb_, typeUb_, type, CMPMODE::EQ, EpCombineAlign(tokens_, 64U));
            CreateVecIndex(indexUb_, int32_t(0), tokens_);
            PipeBarrier<PIPE_V>();
            GatherMask(packedUb_, indexUb_, maskUb_.ReinterpretCast<uint32_t>(), true, tokens_, {1U, 1U, 0U, 0U},
                       selected_);
            Sync<HardEvent::V_S>();
        }
    }
    __aicore__ inline void PushMembers()
    {
        SelectRows(int32_t(deep_ep::EpServerDedupTokenType::Terminal));
        const uint32_t base = uint32_t(selected_) / coreNum_, extra = uint32_t(selected_) % coreNum_;
        const uint32_t count = base + (core_ < extra);
        const uint32_t start = core_ * base + (core_ < extra ? core_ : extra);
        if (!count) return;
        uint64_t mask[2] = {0xffU, 0U};
        for (uint32_t lane = 0; lane < lanes_; ++lane)
            Duplicate<float>(resultUb_[lane * resultStride_].ReinterpretCast<float>()[120U],
                             1.0f, mask, uint8_t(blocks_), 1U, 16U);
        Sync<HardEvent::V_MTE2>();
        Sync<HardEvent::V_MTE3>();
        for (uint32_t lane = 0; lane < lanes_; ++lane) SetFlag<HardEvent::MTE3_MTE2>(lane);
        uint32_t cacheBegin = 0U, cacheEnd = 0U;
        for (uint32_t task = 0; task < count; ++task) {
            const uint32_t row = uint32_t(packedUb_.GetValue(start + task));
            CacheTableRows(row, start, task, count, cacheBegin, cacheEnd);
            const uint32_t index = (row - cacheBegin) * 3U;
            const uint32_t owner = uint32_t(tableUb_.GetValue(index + 1U));
            const uint32_t slot = uint32_t(tableUb_.GetValue(index + 2U));
            if (owner >= uint32_t(args_->rankSize) || owner / Group != uint32_t(args_->rank) / Group ||
                slot >= uint64_t(args_->rankSize) * bs_ * topK_ || slot % topK_ == 0U ||
                !UbxTransport::SameHost(args_, owner) || !args_->peerMems[owner]) {
                Trap();
                return;
            }
            const uint64_t offset = stageBase_ + uint64_t(slot) * resultStride_;
            UbxTransport::Check(args_, owner, offset, resultStride_);
            const uint32_t lane = task % lanes_;
            WaitFlag<HardEvent::MTE3_MTE2>(lane);
            auto packet = resultUb_[lane * resultStride_];
            LoadDirectWire(packet.ReinterpretCast<uint16_t>(), expert_[uint64_t(row) * h_], h_ * 2U);
            UbxTransport::Sync<HardEvent::MTE2_MTE3>(lane);
            DataCopy(UbxTransport::Tensor(args_->peerMems[owner] + offset, true), packet, resultStride_);
            SetFlag<HardEvent::MTE3_MTE2>(lane);
        }
        for (uint32_t lane = 0; lane < lanes_; ++lane) WaitFlag<HardEvent::MTE3_MTE2>(lane);
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

            Compares(maskUb_, typeUb_, int32_t(deep_ep::EpServerDedupTokenType::Direct), CMPMODE::EQ, EpCombineAlign(tokens_, 64U));
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
        // Preserve the single-tile route table by placing its comparison mask in separate UB.
        directRoutesCached_ = tokens_ <= kEpServerDedupCombineTableRows;
        auto upper = tableUb_.ReinterpretCast<uint8_t>();
        if (directRoutesCached_) upper = directUpperMask_;
        ServerDedupRankMask(maskUb_, upper, typeUb_, 0, int32_t(args_->rankSize), uint32_t(selected_));
        uint64_t valid = 0U;
        GatherMask(indexUb_, packedUb_, maskUb_.ReinterpretCast<uint32_t>(), true, uint32_t(selected_),
                   {1U, 1U, 0U, 0U}, valid);
        Sync<HardEvent::V_S>();
        if (valid != selected_) Trap();
    }
    __aicore__ inline uint32_t SelectDirectServer(uint32_t server)
    {
        auto upper = tableUb_.ReinterpretCast<uint8_t>();
        if (directRoutesCached_) upper = directUpperMask_;
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

        const uint32_t servers = uint32_t(args_->rankSize) / Group;
        const uint32_t ownServer = uint32_t(args_->rank) / Group;
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

        const uint32_t split = servers / 2U;
        const uint32_t ownLength = ownServer < split ? split : servers - split;
        const uint32_t ownStep = (ownLength - 1U) * (servers >= 16U ? 2U : 1U);
        for (uint32_t visit = 0U; visit < servers; ++visit) {
            // Defer the local group without an extra selection pass.
            const uint32_t step = visit + 1U == servers ? ownStep : visit + uint32_t(visit >= ownStep);
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
                if (token >= bs_ || topk >= topK_) { Trap(); return; }
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

                const uint64_t offset = window_ + COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET +
                                        (uint64_t(token) * topK_ + topk) * resultStride_;

                SendReturnPacket(returnBatch_, args_, source, offset, output, resultStride_);

                SetFlag<HardEvent::MTE3_V>(lane);
            }
            // The next server scan can overlap pending output DMA.
        }

        // Final retirement includes unused seeded lanes, never a per-token drain.
        for (uint32_t lane = 0; lane < activeLanes; ++lane) {
            WaitFlag<HardEvent::V_MTE2>(lane);
            WaitFlag<HardEvent::MTE3_V>(lane);
        }
    }
    // M4→M5交接需保留尚未完成的MTE3输出槽；只等待当前要复用的输入/向量资源。
    // 修改此处同步必须联查Partial与SourceAggregate的槽位生命周期，避免提前覆盖发送数据。
    __aicore__ inline void ReturnPartialTokens(uint32_t first)
    {
        // Finish member PUTs before reusing their scratch for partial reduction.
        PipeBarrier<PIPE_ALL>();
        // Seed exactly once, including empty M4. M5 inherits these events.
        for (uint32_t lane = 0; lane < workspace_.outputLanes; ++lane) SetFlag<HardEvent::MTE3_V>(lane);
        // An empty expert rank still consumes returns for its source tokens.
        // Leave the events for final aggregation without forming an expert address.
        if (tokens_ == 0U) return;
        ServerDedupPartial<Group> partial;
        partial.Init(args_, &pipe_, arena_.Get<uint8_t>()[workspace_.residentBytes],
                     expertAddr_ + uint64_t(first) * h_ * 2U,
                     tableAddr_ + uint64_t(first) * 12U, relayAddr_ + uint64_t(first) * topK_ * 8U,
                     typesAddr_ + uint64_t(first) * 4U, tokens_, bs_, h_, topK_, dtype_, window_, workspace_,
                     &returnBatch_);

        partialTasks_ = partial.Process();

        // Retire only input and Vector users before scalar metadata reuse.
        // M4 output slots and MTE3_V events stay live for M5's per-slot waits.
        Sync<HardEvent::MTE2_S>();
        Sync<HardEvent::V_S>();
    }
    EpCombineM45WorkspaceLayout workspace_{};
    ReturnBatch returnBatch_;
    uint32_t partialTasks_ = 0U;
    uint32_t coreNum_ = 0, sendCores_ = 0, recvCores_ = 0;
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
    uint32_t bs_ = 0, totalRows_ = 0;
    uint8_t dtype_ = 0;
    bool directRoutesCached_ = false;
    uint32_t core_ = 0, h_ = 0, topK_ = 0, blocks_ = 0, resultStride_ = 0, tokens_ = 0, lanes_ = 0, laneBytes_ = 0,
             usedBytes_ = 0;
    uint64_t selected_ = 0, window_ = 0, stageBase_ = 0;
    GlobalTensor<uint16_t> expert_;
    GlobalTensor<int32_t> table_, types_;
    LocalTensor<int32_t> typeUb_, indexUb_, packedUb_, tableUb_;
    LocalTensor<uint8_t> maskUb_, resultUb_;
    LocalTensor<uint16_t> payloadUb_;
    LocalTensor<uint8_t> directUpperMask_;
};
}  // namespace UbxCombine
#endif
