/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See csrc/catccos_overlay/LICENSE.
 */
#ifndef DEEPEP_EP_GMM_LARGE_EP_CHUNK_SCHEDULE_HPP
#define DEEPEP_EP_GMM_LARGE_EP_CHUNK_SCHEDULE_HPP

#include "catlass/catlass.hpp"
#include "catccos/dgemm/alltoallv_allgather_problem_shape.hpp"

namespace deepep::ep_gmm_fused::LargeEpPipeline {

// Rows are compact within one destination's stage slice. Group counts only
// select the physical round; their padding must never become transfer rows.
struct ChunkTransfer {
    uint32_t stageRow;
    uint32_t outputRow;
    uint32_t rows;
};

// One cursor belongs to one source peer. BeginRound must be called in physical
// round order, starting with zero. Expert position and partially consumed
// expert groups survive round boundaries; each ready advance scans forwards.
class PeerChunkCursor {
public:
    using ProblemShape = Catccos::DGemm::AllToAllVAllGatherProblemShape;

    CATLASS_DEVICE PeerChunkCursor() = default;

    CATLASS_DEVICE PeerChunkCursor(const ProblemShape& shape, uint32_t sourcePeer, uint32_t outputBase,
                                   uint32_t groupsPerRound, uint32_t chunkCount, uint32_t incomingGroups,
                                   uint32_t blockM = 256U)
        : problemShape_(shape),
          sourcePeer_(sourcePeer),
          outputBase_(outputBase),
          groupsPerRound_(groupsPerRound),
          chunkCount_(chunkCount),
          expertCount_(shape.localExpertNum()),
          expertsPerChunk_(CeilDiv(shape.localExpertNum(), chunkCount)),
          blockM_(blockM),
          unconsumedGroups_(incomingGroups)
    {
    }

    CATLASS_DEVICE void BeginRound(uint32_t round)
    {
        round_ = round;
        remainingRoundGroups_ = Min(groupsPerRound_, unconsumedGroups_);
        stageRow_ = 0U;
        nextChunk_ = 0U;
    }

    CATLASS_DEVICE bool HasInput() const
    {
        return remainingRoundGroups_ != 0U;
    }
    CATLASS_DEVICE uint32_t GetRound() const
    {
        return round_;
    }
    CATLASS_DEVICE uint32_t GetChunkCount() const
    {
        return chunkCount_;
    }
    CATLASS_DEVICE uint32_t GetConsumedRows() const
    {
        return consumedRows_;
    }
    CATLASS_DEVICE uint32_t GetStageRows() const
    {
        return stageRow_;
    }

    // Empty expert bands need no ready wait. Reading the immutable routing
    // counts is independent of the output-data ready protocol.
    CATLASS_DEVICE uint32_t NextNonEmptyChunk()
    {
        if (!HasInput()) return chunkCount_;
        LoadNonEmptyExpert();
        if (expert_ >= expertCount_) return chunkCount_;
        return expert_ / expertsPerChunk_;
    }

    // endChunk is an exclusive, monotonically increasing ready-prefix length.
    // A single call may consume several adjacent ready chunks into one copy.
    CATLASS_DEVICE ChunkTransfer AdvanceThroughChunk(uint32_t endChunk)
    {
        const uint32_t stageBegin = stageRow_;
        const uint32_t outputBegin = outputBase_ + consumedRows_;
        endChunk = Min(endChunk, chunkCount_);
        if (endChunk <= nextChunk_) return {stageBegin, outputBegin, 0U};
        const uint32_t expertEnd = Min(expertCount_, endChunk * expertsPerChunk_);

        while (expert_ < expertEnd && remainingRoundGroups_ != 0U) {
            LoadNonEmptyExpert();
            if (expert_ >= expertEnd) break;
            const uint32_t availableGroups = expertGroups_ - consumedExpertGroups_;
            const uint32_t takeGroups = Min(remainingRoundGroups_, availableGroups);
            const uint32_t startRow = consumedExpertGroups_ * blockM_;
            const uint32_t rows = Min(takeGroups * blockM_, expertTokens_ - startRow);
            stageRow_ += rows;
            consumedRows_ += rows;
            remainingRoundGroups_ -= takeGroups;
            unconsumedGroups_ -= takeGroups;
            if (takeGroups == availableGroups) {
                ++expert_;
                consumedExpertGroups_ = 0U;
                expertLoaded_ = false;
            } else {
                // The physical round ended inside this expert. Resume its
                // remaining groups in the same expert band next round.
                consumedExpertGroups_ += takeGroups;
            }
        }
        nextChunk_ = endChunk;
        return {stageBegin, outputBegin, stageRow_ - stageBegin};
    }

private:
    CATLASS_DEVICE void LoadNonEmptyExpert()
    {
        while (!expertLoaded_ && expert_ < expertCount_) {
            expertTokens_ = problemShape_.localTokensPerExpert(sourcePeer_ * expertCount_ + expert_);
            expertGroups_ = CeilDiv(expertTokens_, blockM_);
            if (expertGroups_ != 0U) {
                expertLoaded_ = true;
                return;
            }
            ++expert_;
        }
    }

    ProblemShape problemShape_;
    uint32_t sourcePeer_{0U};
    uint32_t outputBase_{0U};
    uint32_t groupsPerRound_{0U};
    uint32_t chunkCount_{0U};
    uint32_t expertCount_{0U};
    uint32_t expertsPerChunk_{0U};
    uint32_t blockM_{0U};
    uint32_t unconsumedGroups_{0U};
    uint32_t round_{0U};
    uint32_t remainingRoundGroups_{0U};
    uint32_t nextChunk_{0U};
    uint32_t expert_{0U};
    uint32_t expertTokens_{0U};
    uint32_t expertGroups_{0U};
    uint32_t consumedExpertGroups_{0U};
    uint32_t consumedRows_{0U};
    uint32_t stageRow_{0U};
    bool expertLoaded_{false};
};

// Rank-local routing metadata. The host selects this implementation only for
// TP1 and rankSize <= PeerCapacity. Both tables are scanned once per instance,
// not once for each chunk. No task/offset table is materialized in GM.
template <uint32_t PeerCapacity = 16U>
class ChunkCommSchedule {
public:
    using ProblemShape = Catccos::DGemm::AllToAllVAllGatherProblemShape;
    static constexpr uint32_t kPeerCapacity = PeerCapacity;

    CATLASS_DEVICE ChunkCommSchedule(const ProblemShape& shape, uint32_t groupsPerRound, uint32_t chunkCount,
                                     uint32_t blockM = 256U)
        : problemShape_(shape),
          groupsPerRound_(groupsPerRound),
          chunkCount_(chunkCount),
          blockM_(blockM),
          rankCount_(shape.rankSize()),
          expertCount_(shape.localExpertNum())
    {
        uint32_t inputRows = 0U;
        uint32_t maxGroups = 0U;
        for (uint32_t peer = 0U; peer < rankCount_; ++peer) {
            inputRowOffset_[peer] = inputRows;
            uint32_t inputGroups = 0U;
            uint32_t outputGroups = 0U;
            for (uint32_t expert = 0U; expert < expertCount_; ++expert) {
                const uint32_t incoming = shape.localTokensPerExpert(peer * expertCount_ + expert);
                const uint32_t outgoing = shape.globalTokensPerLocalExpert(peer * expertCount_ + expert);
                inputGroups += CeilDiv(incoming, blockM_);
                outputGroups += CeilDiv(outgoing, blockM_);
                inputRows += incoming;
            }
            inputGroupNum_[peer] = inputGroups;
            outputGroupNum_[peer] = outputGroups;
            maxGroups = Max(maxGroups, Max(inputGroups, outputGroups));
        }
        commLoops_ = CeilDiv(maxGroups, groupsPerRound_);
    }

    CATLASS_DEVICE uint32_t GetCommLoops() const
    {
        return commLoops_;
    }
    // A fully empty rank still consumes the agreed first-round notifications.
    CATLASS_DEVICE uint32_t GetControlLoops() const
    {
        return Max(1U, commLoops_);
    }
    CATLASS_DEVICE uint32_t GetChunkCount() const
    {
        return chunkCount_;
    }
    CATLASS_DEVICE uint32_t InputRowOffset(uint32_t peer) const
    {
        return inputRowOffset_[peer];
    }
    CATLASS_DEVICE uint32_t InputGroups(uint32_t peer) const
    {
        return inputGroupNum_[peer];
    }

    CATLASS_DEVICE bool HasPeerInput(uint32_t sourcePeer, uint32_t round) const
    {
        return static_cast<uint64_t>(round) * groupsPerRound_ < inputGroupNum_[sourcePeer];
    }

    // Count source-stage readers, including self only when it has real rows.
    // Caller adds this count to the cumulative expected[round % 2] value.
    CATLASS_DEVICE uint32_t GetExpectedReaders(uint32_t round) const
    {
        const uint64_t groupBegin = static_cast<uint64_t>(round) * groupsPerRound_;
        uint32_t readers = 0U;
        for (uint32_t peer = 0U; peer < rankCount_; ++peer) {
            readers += static_cast<uint32_t>(groupBegin < outputGroupNum_[peer]);
        }
        return readers;
    }

    CATLASS_DEVICE PeerChunkCursor MakePeerCursor(uint32_t sourcePeer) const
    {
        return PeerChunkCursor(problemShape_, sourcePeer, inputRowOffset_[sourcePeer], groupsPerRound_, chunkCount_,
                               inputGroupNum_[sourcePeer], blockM_);
    }

private:
    ProblemShape problemShape_;
    const uint32_t groupsPerRound_;
    const uint32_t chunkCount_;
    const uint32_t blockM_;
    const uint32_t rankCount_;
    const uint32_t expertCount_;
    uint32_t inputGroupNum_[kPeerCapacity];
    uint32_t inputRowOffset_[kPeerCapacity];
    uint32_t outputGroupNum_[kPeerCapacity];
    uint32_t commLoops_{0U};
};

}  // namespace deepep::ep_gmm_fused::LargeEpPipeline
#endif  // DEEPEP_EP_GMM_LARGE_EP_CHUNK_SCHEDULE_HPP
