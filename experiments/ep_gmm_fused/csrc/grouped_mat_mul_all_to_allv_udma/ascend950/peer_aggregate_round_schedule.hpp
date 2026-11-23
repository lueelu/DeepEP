/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See csrc/catccos_overlay/LICENSE.
 */
#ifndef DEEPEP_EP_GMM_PEER_AGGREGATE_ROUND_SCHEDULE_HPP
#define DEEPEP_EP_GMM_PEER_AGGREGATE_ROUND_SCHEDULE_HPP

namespace deepep::ep_gmm_fused::PeerAggregate {

// Stream the same expert/peer M-block slices as the AIV scheduler. Round zero
// captures A expert bases while discovering outgoing rounds. Subsequent rounds
// resume per-peer cursors; no route eligibility prepass or full GM task table.
template <uint32_t PeerCapacity = 16U, uint32_t ExpertCapacity = 64U>
class RoundSchedule {
public:
    using ProblemShape = Catccos::DGemm::AllToAllVAllGatherProblemShape;
    static constexpr uint32_t kPeerCapacity = PeerCapacity;
    static constexpr uint32_t kExpertCapacity = ExpertCapacity;

    struct ExpertCache {
        uint32_t tokens[kPeerCapacity];
        uint32_t outputOffset[kPeerCapacity];
        uint32_t rowOffset[kPeerCapacity];
    };

    struct ContiguousRun {
        uint32_t begin;
        uint32_t end;
        uint32_t rows;
        uint32_t inputOffset;
    };

    CATLASS_DEVICE RoundSchedule(const ProblemShape& shape, uint32_t groupsPerStage, uint32_t m0)
        : problemShape(shape),
          rankCount(shape.rankSize()),
          expertCount(shape.localExpertNum()),
          groupsPerRound(Max(1U, groupsPerStage)),
          blockM(m0)
    {
        uint32_t maxInputGroups = 0U;
        for (uint32_t peer = 0; peer < shape.epSize(); ++peer) {
            uint32_t groups = 0U;
            for (uint32_t expert = 0; expert < expertCount; ++expert) {
                groups += CeilDiv(shape.localTokensPerExpert(peer * expertCount + expert), blockM);
            }
            maxInputGroups = Max(maxInputGroups, groups);
        }
        commLoops = CeilDiv(maxInputGroups, groupsPerRound);
    }

    CATLASS_DEVICE bool HasRound(uint32_t round) const
    {
        return round == 0U || round < commLoops;
    }
    CATLASS_DEVICE uint32_t GetCommLoops() const
    {
        return commLoops;
    }
    CATLASS_DEVICE uint32_t FirstExpert() const
    {
        return firstExpert;
    }
    CATLASS_DEVICE bool HasExpert(uint32_t expert) const
    {
        return expert < expertCount && (commIdx == 0U || activePeers != 0U);
    }

    CATLASS_DEVICE void UpdateCommContext(uint32_t round)
    {
        commIdx = round;
        activePeers = 0U;
        if (round == 0U) {
            firstExpert = 0U;
            for (uint32_t peer = 0; peer < rankCount; ++peer) {
                totalGroups[peer] = 0U;
                nextExpert[peer] = expertCount;
                consumedGroups[peer] = 0U;
                expertCache.tokens[peer] = 0U;
                expertCache.rowOffset[peer] = 0U;
            }
        } else {
            firstExpert = expertCount;
            for (uint32_t peer = 0; peer < rankCount; ++peer) {
                remainingGroups[peer] = Min(groupsPerRound, totalGroups[peer]);
                expertCache.tokens[peer] = 0U;
                expertCache.rowOffset[peer] = 0U;
                if (remainingGroups[peer] != 0U) {
                    firstExpert = Min(firstExpert, nextExpert[peer]);
                    ++activePeers;
                }
            }
        }
    }

    CATLASS_DEVICE void UpdateLocalExpertContext(uint32_t expert)
    {
        if (commIdx == 0U) {
            UpdateFirstRoundExpert(expert);
        } else {
            UpdateLaterRoundExpert(expert);
        }
    }

    CATLASS_DEVICE void UpdateFirstRoundExpert(uint32_t expert)
    {
        expertInputBase[expert] = inputOffset;
        uint32_t source = inputOffset;
        completeExpert = true;
        for (uint32_t peer = 0; peer < rankCount; ++peer) {
            const uint32_t tokens = problemShape.globalTokensPerLocalExpert(peer * expertCount + expert);
            const uint32_t before = totalGroups[peer];
            const uint32_t after = before + CeilDiv(tokens, blockM);
            totalGroups[peer] = after;
            uint32_t rows = tokens;
            if (after > groupsPerRound) {
                rows = 0U;
                if (before <= groupsPerRound) {
                    // First overflow only: preserve the earliest unconsumed
                    // expert and its block offset. Full slices need no cursor
                    // writes. Exact capacity followed by zero experts is valid.
                    const uint32_t take = groupsPerRound - before;
                    rows = take * blockM;
                    nextExpert[peer] = expert;
                    consumedGroups[peer] = take;
                }
            }
            expertCache.rowOffset[peer] += expertCache.tokens[peer];
            expertCache.tokens[peer] = rows;
            expertCache.outputOffset[peer] = source;
            source += tokens;
            completeExpert = completeExpert && rows == tokens;
        }
        // Always finish the first full expert scan: outgoing rounds and source
        // bases are still needed after this round's output capacity is full.
        inputOffset = source;
    }

    CATLASS_DEVICE void UpdateLaterRoundExpert(uint32_t expert)
    {
        uint32_t source = expertInputBase[expert];
        completeExpert = true;
        for (uint32_t peer = 0; peer < rankCount; ++peer) {
            const uint32_t tokens = problemShape.globalTokensPerLocalExpert(peer * expertCount + expert);
            const uint32_t groups = CeilDiv(tokens, blockM);
            expertCache.rowOffset[peer] += expertCache.tokens[peer];
            uint32_t startRow = 0U;
            uint32_t rows = 0U;
            if (remainingGroups[peer] != 0U && expert >= nextExpert[peer]) {
                const uint32_t skipped = consumedGroups[peer];
                const uint32_t left = groups - skipped;
                startRow = skipped * blockM;
                uint32_t take;
                if (left <= remainingGroups[peer]) {
                    take = left;
                    rows = tokens - startRow;
                    nextExpert[peer] = expert + 1U;
                    consumedGroups[peer] = 0U;
                } else {
                    take = remainingGroups[peer];
                    rows = take * blockM;
                    nextExpert[peer] = expert;
                    consumedGroups[peer] = skipped + take;
                }
                remainingGroups[peer] -= take;
                totalGroups[peer] -= take;
                if (remainingGroups[peer] == 0U) --activePeers;
            }
            expertCache.tokens[peer] = rows;
            expertCache.outputOffset[peer] = source + startRow;
            source += tokens;
            completeExpert = completeExpert && rows == tokens;
        }
    }

    CATLASS_DEVICE void FinishCommContext()
    {
        if (commIdx == 0U) {
            uint32_t maxOutputGroups = 0U;
            for (uint32_t peer = 0; peer < rankCount; ++peer) {
                const uint32_t groups = totalGroups[peer];
                maxOutputGroups = Max(maxOutputGroups, groups);
                // After discovery this array becomes the unconsumed block count.
                totalGroups[peer] = groups - Min(groups, groupsPerRound);
            }
            commLoops = Max(commLoops, CeilDiv(maxOutputGroups, groupsPerRound));
        }
    }

    CATLASS_DEVICE ContiguousRun NextContiguousRun(uint32_t begin, uint32_t limit) const
    {
        if (completeExpert) {
            uint32_t rows = 0U;
            for (uint32_t peer = begin; peer < limit; ++peer) rows += expertCache.tokens[peer];
            return {begin, limit, rows, expertCache.outputOffset[begin]};
        }
        while (begin < limit && expertCache.tokens[begin] == 0U) ++begin;
        if (begin == limit) return {limit, limit, 0U, 0U};
        uint32_t rows = expertCache.tokens[begin];
        const uint32_t offset = expertCache.outputOffset[begin];
        uint32_t end = begin + 1U;
        for (; end < limit; ++end) {
            const uint32_t nextRows = expertCache.tokens[end];
            if (nextRows == 0U) continue;
            if (expertCache.outputOffset[end] != offset + rows) break;
            rows += nextRows;
        }
        return {begin, end, rows, offset};
    }

    ExpertCache expertCache;

private:
    ProblemShape problemShape;
    const uint32_t rankCount;
    const uint32_t expertCount;
    const uint32_t groupsPerRound;
    const uint32_t blockM;
    uint32_t expertInputBase[kExpertCapacity];
    uint32_t nextExpert[kPeerCapacity];
    uint32_t consumedGroups[kPeerCapacity];
    uint32_t remainingGroups[kPeerCapacity];
    uint32_t totalGroups[kPeerCapacity];
    uint32_t inputOffset{0U};
    uint32_t commLoops{0U};
    uint32_t commIdx{0U};
    uint32_t firstExpert{0U};
    uint32_t activePeers{0U};
    bool completeExpert{false};
};

}  // namespace deepep::ep_gmm_fused::PeerAggregate
#endif
