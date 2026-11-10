import numpy as np
import pytest

from deep_ep_experimental.megamoe.urma_timer import TIMER_NUMEL, communication_bandwidth, summarize_bandwidth


def test_dispatch_send_to_receive_scopes_include_receive_only_peers():
    raw = np.zeros((28, 3, 8, 16), dtype=np.int64)
    # Rank0 EP8: peers1..3 INTRA,4..7 CROSS. All incoming edges are known.
    for owner in range(7):
        raw[owner, 2, 1, 5:7] = [2, 1]

    def edge(peer, first, last, send_bytes, received_at, recv_bytes):
        row = raw[peer - 1, 2, 1]
        if send_bytes:
            row[:7] = [first, last, last - first, send_bytes, 1, 2, 1]
        row[7:10] = [received_at, recv_bytes, 1 if recv_bytes else 0]

    edge(1, 100, 1000, 100, 500, 100)  # Rx may precede last outgoing submission.
    edge(2, 120, 300, 200, 900, 300)
    edge(3, 130, 300, 300, 0, 0)  # no incoming: exclude from receive end
    edge(4, 0, 0, 0, 1200, 400)  # incoming-only peer must affect CROSS end
    edge(5, 150, 400, 100, 0, 0)
    edge(6, 70, 250, 200, 800, 200)
    report = communication_bandwidth(raw, 0, world=8)
    rows = {r["communication_class"]: r for r in report["classes"] if r["phase"] == "Dispatch"}
    assert rows["INTRA"]["envelope_ticks"] == 800
    assert rows["INTRA"]["payload_bytes"] == 600
    assert rows["INTRA"]["received_payload_bytes"] == 400
    assert rows["INTRA"]["completion_status"] == "peer_dispatch_received"
    assert rows["CROSS"]["envelope_ticks"] == 1130
    assert rows["CROSS"]["payload_bytes"] == 300
    assert rows["CROSS"]["receive_peers"] == [4, 6]
    assert rows["CROSS"]["endpoint_semantics"] == "outgoing_first_to_incoming_last"
    raw[3, 2, 1, 7] = 0
    cross = next(
        r
        for r in communication_bandwidth(raw, 0, world=8)["classes"]
        if r["phase"] == "Dispatch" and r["communication_class"] == "CROSS"
    )
    assert cross["bandwidth_gbps"] is None and cross["missing_completion_peers"] == [4]


def timer_with_records():
    value = np.zeros((28, 3, 8, 16), dtype=np.int64)
    # rank0 owner0 -> rank1 INTRA; owner3 -> rank4 CROSS; owner15 -> self.
    value[0, 2, 1, :7] = [100, 200, 100, 1000, 1, 0, 1]
    value[15, 2, 1, :7] = [150, 300, 150, 2000, 1, 0, 1]
    value[3, 2, 1, :7] = [120, 520, 200, 8000, 2, 0, 1]
    value[3, 2, 5, :7] = [600, 1400, 500, 16000, 3, 0, 1]
    return value.reshape(-1)


def test_bandwidth_uses_envelope_and_separates_stages_and_classes():
    report = communication_bandwidth(timer_with_records(), 0, clock_mhz=1000)
    rows = {(r["phase"], r["communication_class"]): r for r in report["classes"]}
    intra = rows["Dispatch", "INTRA"]
    assert intra["payload_bytes"] == 3000
    assert intra["envelope_ticks"] == 200
    assert intra["bandwidth_gbps"] == 15
    assert intra["peers"] == [1, 0]
    assert intra["includes_self_copy"]
    assert report["includes_self_copy_in_intra"]
    assert rows["Dispatch", "CROSS"]["bandwidth_gbps"] == 20
    assert not rows["Dispatch", "CROSS"]["includes_self_copy"]
    assert rows["Combine", "CROSS"]["bandwidth_gbps"] == 20
    assert rows["Combine", "INTRA"]["bandwidth_gbps"] is None


def test_raw_ticks_work_without_a_frequency_assumption():
    report = communication_bandwidth(timer_with_records(), 0, clock_mhz=None)
    assert all(row["bandwidth_gbps"] is None for row in report["classes"])
    assert report["classes"][0]["envelope_ticks"] == 200


def test_submission_only_timers_require_outbound_completion_for_bandwidth():
    raw = timer_with_records().reshape(28, 3, 8, 16)
    raw[..., 5][raw[..., 4] > 0] = 1
    raw[3, 2, 5, 7] = 2000
    report = communication_bandwidth(raw, 0)
    rows = {(row["phase"], row["communication_class"]): row for row in report["classes"]}
    dispatch = rows["Dispatch", "CROSS"]
    assert dispatch["bandwidth_gbps"] is None and dispatch["envelope_ticks"] is None
    assert dispatch["last_tick"] is None and dispatch["envelope_us"] is None
    assert dispatch["completion_status"] == "not_recorded" and dispatch["missing_completion_peers"] == [4]
    assert dispatch["submission_envelope_ticks"] == 400 and dispatch["submission_sum_ticks"] == 200
    combine = rows["Combine", "CROSS"]
    assert combine["completion_status"] == "outbound_complete"
    assert combine["first_tick"] == 600 and combine["last_tick"] == 2000
    assert combine["envelope_ticks"] == 1400 and combine["bandwidth_gbps"] == pytest.approx(16000 / 1400)
    assert combine["submission_envelope_ticks"] == 800 and combine["submission_sum_ticks"] == 500


def test_a_class_requires_completion_of_every_payload_peer():
    raw = np.zeros((28, 3, 8, 16), dtype=np.int64)
    raw[0, 2, 5, :8] = [100, 200, 100, 1000, 1, 1, 1, 900]
    raw[1, 2, 5, :8] = [200, 300, 100, 2000, 1, 1, 1, 0]
    row = communication_bandwidth(raw, 0)["classes"][2]
    assert row["payload_bytes"] == 3000 and row["bandwidth_gbps"] is None
    assert row["missing_completion_peers"] == [2]
    raw[1, 2, 5, 7] = 1100
    row = communication_bandwidth(raw, 0)["classes"][2]
    assert row["envelope_ticks"] == 1000 and row["bandwidth_gbps"] == 3


@pytest.mark.parametrize("mode,completion,match", [(2, 900, "semantics"), (1, 199, "completion precedes")])
def test_reject_unknown_timer_semantics_and_completion_before_submission(mode, completion, match):
    raw = np.zeros((28, 3, 8, 16), dtype=np.int64)
    raw[0, 2, 5, :8] = [100, 200, 100, 1000, 1, mode, 1, completion]
    with pytest.raises(ValueError, match=match):
        communication_bandwidth(raw, 0)


def test_summary_identifies_ranks_with_payload_but_no_completion():
    reports = []
    for rank in range(4):
        raw = np.zeros((28, 3, 8, 16), dtype=np.int64)
        raw[0, 2, 1, :8] = [100, 200, 100, 1000, 1, 1, 1, 0]
        reports.append(communication_bandwidth(raw, rank, world=4))
    row = summarize_bandwidth(reports)[0]
    assert row["mean_gbps"] is None and row["payload_ranks"] == 4
    assert row["missing_completion_ranks"] == list(range(4))


def test_reject_incomplete_timer_schema():
    raw = timer_with_records().reshape(28, 3, 8, 16)
    raw[3, 2, 1, 6] = 0
    with pytest.raises(ValueError, match="schema"):
        communication_bandwidth(raw, 0)


@pytest.mark.parametrize("raw", [np.zeros(12, dtype=np.int64), np.zeros(TIMER_NUMEL)])
def test_reject_incompatible_timer_buffer(raw):
    with pytest.raises(ValueError):
        communication_bandwidth(raw, 0)


@pytest.mark.parametrize("bs", [2048, 4096, 8192])
def test_direct_self_pack_and_return_do_not_count_as_transport(bs):
    reports = []
    hidden, topk = 3584, 8
    peer_rows = bs * topk // 16
    for rank in range(16):
        raw = np.zeros((28, 3, 8, 16), dtype=np.int64)
        for block in range(28):
            for subblock in range(2):
                worker = 2 * block + subblock
                pack_rows = len(range(worker, bs * topk, 56))
                reduce_rows = len(range(worker, bs, 56))
                raw[block, subblock + 1, 0, :7] = [
                    1,
                    101,
                    100,
                    pack_rows * (hidden + hidden // 32),
                    1,
                    0,
                    1,
                ]
                raw[block, subblock + 1, 7, :7] = [900, 1100, 200, reduce_rows * hidden * 2, 1, 0, 1]
        for owner in range(15):
            raw[owner, 2, 1, :7] = [200, 400, 200, peer_rows * (hidden + hidden // 32), 4, 0, 1]
            raw[owner, 2, 5, :7] = [500, 800, 300, peer_rows * hidden * 2, 4, 0, 1]
        report = communication_bandwidth(raw, rank)
        assert not report["includes_self_copy_in_intra"]
        for record in report["classes"]:
            peers = 3 if record["communication_class"] == "INTRA" else 12
            row_bytes = hidden + hidden // 32 if record["phase"] == "Dispatch" else hidden * 2
            assert record["payload_bytes"] == peers * peer_rows * row_bytes
            assert len(record["peers"]) == peers and rank not in record["peers"]
            assert not record["includes_self_copy"]
        assert raw[:, :, 0, 3].sum() == bs * topk * (hidden + hidden // 32)
        assert raw[:, :, 7, 3].sum() == bs * hidden * 2
        reports.append(report)
    summary = summarize_bandwidth(reports)
    assert all(not row["includes_self_copy"] and row["nonempty_ranks"] == 16 for row in summary)


def test_summary_retains_self_copy_label_for_older_timer_payloads():
    reports = [communication_bandwidth(timer_with_records(), rank) for rank in range(16)]
    summary = {(row["phase"], row["communication_class"]): row for row in summarize_bandwidth(reports)}
    assert summary["Dispatch", "INTRA"]["includes_self_copy"]
    assert not summary["Dispatch", "CROSS"]["includes_self_copy"]
    assert not summary["Combine", "INTRA"]["includes_self_copy"]


@pytest.mark.parametrize("world", [4, 8, 16])
def test_active_ep_transport_workers_and_empty_cross_group(world):
    records = []
    for rank in range(world):
        raw = np.zeros((28, 3, 8, 16), dtype=np.int64)
        for worker in range(world - 1):
            raw[worker, 2, 1, :7] = [100, 300, 200, 1000, 1, 0, 1]
            raw[worker, 2, 5, :7] = [400, 800, 400, 2000, 1, 0, 1]
        report = communication_bandwidth(raw, rank, world=world)
        rows = {(r["phase"], r["communication_class"]): r for r in report["classes"]}
        assert rows["Dispatch", "INTRA"]["payload_bytes"] == 3000
        assert rows["Dispatch", "CROSS"]["payload_bytes"] == (world - 4) * 1000
        assert rows["Combine", "CROSS"]["payload_bytes"] == (world - 4) * 2000
        assert not report["includes_self_copy_in_intra"]
        records.append(report)
    summary = summarize_bandwidth(records)
    for row in summary:
        assert row["nonempty_ranks"] == (0 if world == 4 and row["communication_class"] == "CROSS" else world)
