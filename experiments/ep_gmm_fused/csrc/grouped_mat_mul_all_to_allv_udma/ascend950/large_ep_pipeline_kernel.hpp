/* Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef DEEPEP_LARGE_EP_PIPELINE_KERNEL_HPP
#define DEEPEP_LARGE_EP_PIPELINE_KERNEL_HPP
#include "large_ep_pipeline_launch.hpp"
#include "large_ep_pipeline_sync.hpp"
#include "large_ep_chunk_schedule.hpp"
#include "peer_aggregate_kernel.hpp"
namespace deepep::ep_gmm_fused::LargeEpPipeline {
template <bool WholeRound>
class Kernel {
public:
    using Config = LargeEpN128Config;
    using Params = LargeEpN128Params;
    using Policy = Catlass::Gemm::MmadPingpong<Catlass::Arch::Ascend950, false, false, 1, false, 2, 2, 2, 2>;
    using Block = Catlass::Gemm::Block::BlockMmadTla<Policy, typename Config::L1TileShape, typename Config::L0TileShape,
                                                     bfloat16_t, bfloat16_t, bfloat16_t, void, PeerAggregate::TileCopy>;
    using Scheduler = PeerAggregate::RoundSchedule<16U>;
    using SegmentedOutput = PeerAggregate::SegmentedOutputFor<Scheduler::kPeerCapacity>;
    using OutputSegment = PeerAggregate::OutputSegment;
    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(const Params& params)
    {
        if constexpr (CORE_TYPE == AscendC::AIC)
            RunAic(params);
        else
            RunAiv(params);
    }

private:
    CATLASS_DEVICE void RunAic(const Params& pipeline)
    {
        const auto& params = pipeline.base;
        const uint32_t chunks = pipeline.chunks;
        const uint32_t expertsPerChunk = CeilDiv(params.problemShape.localExpertNum(), chunks);
        constexpr uint32_t m0 = Config::kM0;
        constexpr uint32_t n0 = Config::kN0;
        constexpr uint32_t k0 = Config::kK0;
        const uint32_t groupsPerStage = params.stageRows / m0;
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = AscendC::GetBlockNum();
        const uint32_t rankSize = params.problemShape.rankSize();
        const uint32_t columns = params.problemShape.n();
        const uint32_t nTiles = CeilDiv(columns, n0);
        // Host validates the complete AIC/AIV implementation as one unit.
        Scheduler scheduler{params.problemShape, groupsPerStage, m0};

        AscendC::GlobalTensor<bfloat16_t> gmA;
        AscendC::GlobalTensor<bfloat16_t> gmB;
        gmA.SetGlobalBuffer(params.ptrA);
        gmB.SetGlobalBuffer(params.ptrB);
        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Catlass::Arch::PositionGM{});
        uint32_t nextTask = core;
        constexpr uint32_t stages = Config::kWorkspaceStages;
        const uint64_t stageElements = static_cast<uint64_t>(params.stageRows) * rankSize * columns;
        for (uint32_t round = 0U; scheduler.HasRound(round); ++round) {
            scheduler.UpdateCommContext(round);
            const uint32_t stageIdx = round % stages;
            if (round >= stages) {
                // Same event id/direction as Base::RunTargetOrderAic. AIV only
                // releases this stage after remote readers and local copies join.
                SyncState::WaitReleased(stageIdx);
            }
            auto stage = reinterpret_cast<__gm__ bfloat16_t*>(params.ptrSymmetric) + stageIdx * stageElements;
            {
                Catlass::Arch::Resource<Catlass::Arch::Ascend950> computeResource;
                Block block(computeResource);
                uint32_t expert = scheduler.FirstExpert();
                for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
                    const uint32_t expertEnd = Min(params.problemShape.localExpertNum(), (chunk + 1) * expertsPerChunk);
                    for (; expert < expertEnd && scheduler.HasExpert(expert); ++expert) {
                        scheduler.UpdateLocalExpertContext(expert);
                        const auto& cache = scheduler.expertCache;
                        const uint64_t weightOffset = static_cast<uint64_t>(expert) * params.problemShape.k() * columns;
                        auto tensorB = tla::MakeTensor(gmB[weightOffset], params.layoutB, Catlass::Arch::PositionGM{});
                        for (uint32_t cursor = 0; cursor < rankSize;) {
                            const auto run = scheduler.NextContiguousRun(cursor, rankSize);
                            cursor = run.end;
                            const uint32_t rows = run.rows;
                            if (rows == 0U) continue;
                            // Within one source-contiguous run, B blocks of q or q+1 rows
                            // cover the input exactly and never exceed the compiled M tile.
                            const uint32_t mBlocks = 1U + (rows - 1U) / m0;
                            const uint32_t baseRows = rows / mBlocks;
                            const uint32_t extraBlocks = rows - baseRows * mBlocks;
                            const uint32_t taskCount = mBlocks * nTiles;
                            uint32_t task = nextTask;
                            for (; task < taskCount; task += cores) {
                                const uint32_t mBlock = task / nTiles;
                                const uint32_t row = mBlock * baseRows + Min(mBlock, extraBlocks);
                                const uint32_t column = (task % nTiles) * n0;
                                const uint32_t tileRows = baseRows + static_cast<uint32_t>(mBlock < extraBlocks);
                                const uint32_t tileColumns = Min(n0, columns - column);
                                const auto shape = Catlass::GemmCoord{tileRows, tileColumns, params.problemShape.k()};
                                auto tileA = tla::GetTile(tensorA, tla::MakeCoord(run.inputOffset + row, 0U),
                                                          tla::MakeShape(tileRows, params.problemShape.k()));
                                auto tileB = tla::GetTile(tensorB, tla::MakeCoord(0U, column),
                                                          tla::MakeShape(params.problemShape.k(), tileColumns));
                                SegmentedOutput output;
                                output.stage = stage;
                                output.columns = columns;
                                output.columnOffset = column;
                                output.tileColumns = tileColumns;
                                output.segmentCount = 0;
                                uint32_t peerOffset = 0;
                                for (uint32_t peer = run.begin; peer < run.end && peerOffset < row + tileRows; ++peer) {
                                    const uint32_t peerLimit = peerOffset + cache.tokens[peer];
                                    const uint32_t begin = Max(row, peerOffset);
                                    const uint32_t end = Min(row + tileRows, peerLimit);
                                    if (begin < end) {
                                        output.segments[output.segmentCount++] = OutputSegment{
                                            begin - row,
                                            peer * params.stageRows + cache.rowOffset[peer] + begin - peerOffset,
                                            end - begin};
                                    }
                                    peerOffset = peerLimit;
                                }
                                block(tileA, tileB, output, shape);
                            }
                            // Carry the global round-robin task assignment across
                            // runs/groups/experts/rounds, including zero-token groups.
                            nextTask = task - taskCount;
                        }
                    }
                    // PIPE_FIX orders the notification after all segmented stores
                    // already issued by this core, without draining the next tile.
                    if constexpr (WholeRound) {
                        Catlass::Arch::CrossCoreFlag stored(stageIdx);
                        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(stored);
                    } else {
                        SyncState::PublishChunk(stageIdx);
                    }
                }
                scheduler.FinishCommContext();
                // The block destructor drains FIX_M and the operand events before
                // leaving the round. Recreate the same block in the next round
                // so no L0C event token is lost or reused before the segmented store.
            }
        }
    }
    CATLASS_DEVICE void RunAiv(const Params& pipeline)
    {
        const auto& params = pipeline.base;
        const uint32_t chunks = pipeline.chunks;
        const uint32_t core = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        const uint32_t subcore = AscendC::GetSubBlockIdx();
        const uint32_t rank = params.problemShape.rankIdx();
        const uint32_t ranks = params.problemShape.rankSize();
        const uint32_t columns = params.problemShape.n();
        const uint32_t selfCores = Max(1U, Min(params.selfCopyCores, 16U));
        const bool notifier = subcore == 0;
        const bool coordinator = subcore == 1 && core == 31;
        // 15 remote owners, up to16 self workers, and one coordinator.
        const bool remoteOwner = subcore == 1 && core < ranks - 1;
        const uint32_t selfWorker = core - (ranks - 1);
        const bool selfOwner = subcore == 1 && core >= ranks - 1 && core < ranks - 1 + selfCores;
        const bool copyOwner = remoteOwner || selfOwner;
        const uint32_t peer = remoteOwner ? (core < rank ? core : core + 1) : rank;
        ChunkCommSchedule<16> schedule(params.problemShape, params.stageRows / 256U, chunks);
        auto cursor = schedule.MakePeerCursor(peer);
        const uint32_t loops = schedule.GetControlLoops();
        SyncState sync(params.syncMmadFinish);
        // The same initialization barrier as the original kernel also orders
        // all control resets before any remote reader can inspect a ready word.
        if (notifier && !WholeRound) {
            for (uint32_t stage = 0; stage < 2; ++stage)
                aclshmemx_signal_op(sync.Core(core, stage), 0, ACLSHMEM_SIGNAL_SET, rank);
        }
        if (coordinator) {
            for (uint32_t stage = 0; stage < 2; ++stage) {
                aclshmemx_signal_op(sync.Ready(stage), 0, ACLSHMEM_SIGNAL_SET, rank);
                aclshmemx_signal_op(sync.Complete(stage), 0, ACLSHMEM_SIGNAL_SET, rank);
            }
        }
        aclshmemx_barrier_all_vec();
        Catlass::Arch::Resource<Catlass::Arch::Ascend950> resource;
        typename Config::BlockComm copy(resource, params.blockCommParams);
        AscendC::GlobalTensor<bfloat16_t> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.ptrC));
        const auto sourceLayout = Catlass::layout::RowMajor(params.stageRows, columns);
        const auto& outputLayout = params.layoutTagComm;
        uint32_t expected[2] = {0, 0};
        for (uint32_t round = 0; round < loops; ++round) {
            const uint32_t stage = round % 2;
            const uint32_t sequenceBase = round * chunks;
            if constexpr (WholeRound) {
                Catlass::Arch::CrossCoreFlag stored(stage);
                Catlass::Arch::CrossCoreWaitFlag(stored);
                AscendC::SyncAll<true>();
                if (coordinator) {
                    aclshmemx_signal_op(sync.Ready(stage), sequenceBase + 1, ACLSHMEM_SIGNAL_SET, rank);
                }
            } else if (notifier) {
                for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
                    SyncState::WaitChunk(stage);
                    aclshmemx_signal_op(sync.Core(core, stage), sequenceBase + chunk + 1, ACLSHMEM_SIGNAL_SET, rank);
                }
            } else if (coordinator) {
                for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
                    const int32_t sequence = sequenceBase + chunk + 1;
                    for (uint32_t producer = 0; producer < 32; ++producer)
                        aclshmem_signal_wait_until(sync.Core(producer, stage), ACLSHMEM_CMP_GE, sequence);
                    aclshmemx_signal_op(sync.Ready(stage), sequence, ACLSHMEM_SIGNAL_SET, rank);
                }
            }
            if (coordinator) {
                expected[stage] += schedule.GetExpectedReaders(round);
                aclshmem_signal_wait_until(sync.Complete(stage), ACLSHMEM_CMP_GE, expected[stage]);
            } else if (copyOwner) {
                cursor.BeginRound(round);
                const bool hasInput = cursor.HasInput();
                AscendC::GlobalTensor<bfloat16_t> source;
                source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(params.ptrSymmetric) +
                                       (static_cast<uint64_t>(stage) * ranks + rank) * params.stageRows * columns);
                auto remoteReady = static_cast<__gm__ int32_t*>(aclshmem_ptr(sync.Ready(stage), peer));
                while (cursor.HasInput()) {
                    const uint32_t next = cursor.NextNonEmptyChunk();
                    const int32_t ready =
                        aclshmem_signal_wait_until(remoteReady, ACLSHMEM_CMP_GE, sequenceBase + next + 1);
                    // Self workers use identical chunk boundaries, because
                    // different ready snapshots would otherwise repartition
                    // the same rows differently across workers.
                    const uint32_t readyPrefix =
                        selfOwner ? next + 1 : Min(chunks, static_cast<uint32_t>(ready) - sequenceBase);
                    const auto transfer = cursor.AdvanceThroughChunk(readyPrefix);
                    uint32_t offset = 0;
                    uint32_t rows = transfer.rows;
                    if (selfOwner) {
                        const uint32_t perWorker = CeilDiv(rows, selfCores);
                        offset = selfWorker * perWorker;
                        rows = Min(perWorker, offset < rows ? rows - offset : 0U);
                    }
                    if (rows != 0) {
                        const Catlass::MatrixCoord shape{rows, columns};
                        const Catlass::MatrixCoord srcOffset{transfer.stageRow + offset, 0};
                        const Catlass::MatrixCoord dstOffset{transfer.outputRow + offset, 0};
                        copy.InitBlockLoop();
                        copy(source[sourceLayout.GetOffset(srcOffset)], sourceLayout.GetTileLayout(shape),
                             output[outputLayout.GetOffset(dstOffset)], outputLayout.GetTileLayout(shape), shape, peer);
                        copy.FinalizeBlockLoop();
                    }
                }
                if (hasInput && (remoteOwner || selfWorker == 0)) {
                    aclshmemx_mte_quiet();
                    aclshmemx_signal_op(sync.Complete(stage), 1, ACLSHMEM_SIGNAL_ADD, peer);
                }
            }
            // Joins every local self worker as well as coordinator's remote
            // completion wait. Both subblocks must release each paired AIC.
            AscendC::SyncAll<true>();
            if (round + 2 < loops) SyncState::Release(stage);
        }
    }
};
}  // namespace deepep::ep_gmm_fused::LargeEpPipeline
#endif
