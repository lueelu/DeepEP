// Copyright (c) 2026, Lu Lu
// Modified by huangxiaolan 2026

#pragma once

// Compact source->(received row, final row) pairs; preserve source readiness.
// Remote: SIMT builds RO WQEs against published ordinary GM virtual addresses on private QP0.
// Local: MTE ping-pong, never interpreted as a remote address.
template <bool Profile, bool Local>
__aicore__ inline void ForwardPeer(DeepepSecondaryState& s, const DispatchTiling& t,
                                   TBuf<QuePosition::VECCALC>& metadata, TBuf<QuePosition::VECCALC>& staging,
                                   __ubuf__ uint8_t* wqe)
{
    if constexpr (!Local) {
        if (s.core + 1U >= s.lanes) return;
    }
    const uint32_t lane = Local ? s.rank % 8U : (s.rank % 8U + s.core + 1U) % s.lanes;
    const uint32_t peer = s.rank / 8U * 8U + lane;
    auto* plan = reinterpret_cast<__gm__ uint8_t*>(t.peer_table) + t.forward_plan_offset;
    auto* pairs = reinterpret_cast<__gm__ uint32_t*>(plan + t.forward_header_bytes);
    auto prefix = metadata.GetWithOffset<uint32_t>(1024U, 4096U);
    NativeLoad(prefix, reinterpret_cast<__gm__ uint32_t*>(plan) + lane * (s.world + 1U), s.world + 1U);
    auto records = metadata.Get<uint32_t>();
    auto indices = metadata.GetWithOffset<uint32_t>(kFmSimtMaxWqes, kFmSimtIndicesOffset);
    uint32_t seen[8]{};
    if constexpr (Local) {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
    uint32_t stage = 0U;
    uint64_t issued = 0U, batches = 0U;
    while (true) {
        bool pending = false;
        uint32_t selected = s.world, first = 0U, last = 0U;
        for (uint32_t source = 0; source < s.world; ++source) {
            const uint32_t bit = 1U << (source % 32U);
            if (seen[source / 32U] & bit) continue;
            uint32_t begin = prefix.GetValue(source), end = prefix.GetValue(source + 1U);
            if constexpr (Local) {
                const uint32_t count = end - begin, worker = s.core - ascend_deepep::scale_clos::LocalFirst(s.world);
                end = begin + uint64_t(count) * (worker + 1U) / ascend_deepep::scale_clos::LocalWorkers(s.world);
                begin += uint64_t(count) * worker / ascend_deepep::scale_clos::LocalWorkers(s.world);
            }
            if (begin == end) {
                seen[source / 32U] |= bit;
                continue;
            }
            pending = true;
            if (!IsSourceReadyForForwarding(source, GetGroupRankIndex(s.rank, s.world), s.magic, 2U, s.sync)) continue;
            selected = source;
            first = begin;
            last = end;
            break;
        }
        if (!pending) break;
        if (selected == s.world) continue;
        for (uint32_t at = first; at < last;) {
            const uint32_t count = last - at < 128U ? last - at : 128U;
            NativeLoad(records, pairs + uint64_t(at) * 2U, count * 2U);
            if constexpr (!Local) {
                // Same configurable power-of-two DB granularity as Clos (1..64),
                // not a quiet per batch. WQE loop has no data-dependent branch.
                for (uint32_t row = 0; row < count;) {
                    const uint32_t n = count - row < s.rotation_wqes ? count - row : s.rotation_wqes;
                    IssueFmRows(s, peer, wqe, records, indices, row, n);
                    row += n;
                    issued += n;
                    ++batches;
                }
            } else {
                for (uint32_t row = 0; row < count; row += 4U) {
                    const uint32_t n = count - row < 4U ? count - row : 4U;
                    uint32_t source_slot[4];
                    const TEventID event = stage ? EVENT_ID1 : EVENT_ID0;
                    auto ub =
                        staging.GetWithOffset<uint8_t>(kDuplicateHiddenStageBytes, stage * kDuplicateHiddenStageBytes);
                    WaitFlag<HardEvent::MTE3_MTE2>(event);
                    for (uint32_t i = 0; i < n; ++i) {
                        source_slot[i] = i;
                        const uint32_t primary = records.GetValue((row + i) * 2U);
                        for (uint32_t j = 0; j < i; ++j)
                            if (records.GetValue((row + j) * 2U) == primary) {
                                source_slot[i] = source_slot[j];
                                break;
                            }
                        if (source_slot[i] == i)
                            IssueGmBytesToUb(ub[i * s.row_bytes], s.hidden + uint64_t(primary) * s.row_bytes,
                                             s.row_bytes);
                    }
                    SetFlag<HardEvent::MTE2_MTE3>(event);
                    WaitFlag<HardEvent::MTE2_MTE3>(event);
                    for (uint32_t i = 0; i < n; ++i)
                        IssueUbBytesToGm(s.hidden + uint64_t(records.GetValue((row + i) * 2U + 1U)) * s.row_bytes,
                                         ub[source_slot[i] * s.row_bytes], s.row_bytes);
                    SetFlag<HardEvent::MTE3_MTE2>(event);
                    stage ^= 1U;
                }
            }
            at += count;
        }
        seen[selected / 32U] |= 1U << (selected % 32U);
    }
    if constexpr (Local) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        aclshmemx_mte_quiet();
    } else {
        const uint64_t profile = t.profile + uint64_t(s.core) * 256U;
        DispatchProfileMark<Profile>(profile, 8U);
        if (issued) FinishFmRows(s, peer, wqe);
        DispatchProfileMark<Profile>(profile, 9U);
        if (issued) aclshmemx_udma_qp_quiet(peer, 0U);
        DispatchProfileMark<Profile>(profile, 10U);
        if constexpr (Profile) {
            auto* record = reinterpret_cast<__gm__ uint64_t*>(profile);
            record[11] = issued;
            record[12] = batches;
            record[13] = peer;
        }
    }
}
