"""Bandwidth numerator and profiler-boundary tests; no torch/NPU dependency."""

import csv
import contextlib
import copy
import io
import json
import shutil
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks/clos"))
from stage_metrics import byte_ledger, file_identity, parse_profile, route_counts, summarize, write_json
from bench_v2_bandwidth import arguments, check_idle, local_world_size, report, run_case, suite_contract, worker_command
from bandwidth_multinode import Peer, collection_manifest, library_identity, safe_path, validate_peers
from stage_metrics import L2_FLUSH_RANGE, PROFILE_SAMPLE_RANGE


class StageMetricsTest(unittest.TestCase):
    def setUp(self):
        # rank 0: two experts on self, two on rank 1; rank 1: only sends to self.
        self.counts = route_counts([[[0, 1], [2, 3], [-1, -1]], [[2, -1], [-1, -1], [-1, -1]]], 2)

    def ledger(self, rank=0, dtype="bf16", op="dispatch"):
        return byte_ledger(self.counts, rank, tokens=3, hidden=128, topk=2, dtype=dtype, idx_bytes=8, op=op)

    def test_dedup_sparse_metadata_and_self_exclusion(self):
        self.assertEqual(self.counts["rows"], [[1, 1], [0, 1]])
        value = self.ledger()
        self.assertEqual(value["remote_x_bytes"], 256)
        self.assertEqual(value["remote_src_idx_bytes"], 4)
        self.assertEqual(value["remote_topk_idx_bytes"], 16)
        self.assertEqual(value["remote_weight_bytes"], 8)
        self.assertEqual(value["remote_so_bytes"], 8)
        self.assertEqual(value["remote_application_bytes"], 292)
        self.assertEqual(value["algorithm_bytes"], 1536)
        self.assertEqual(value["valid_algorithm_bytes"], 1024)
        self.assertIsNone(value["wire_bytes"])

    def test_fp8_scales_and_cached_omissions(self):
        value = self.ledger(dtype="fp8", op="cached-dispatch")
        self.assertEqual(value["remote_payload_bytes"], 132)
        self.assertEqual(value["remote_application_bytes"], 140)
        for key in ("remote_src_idx_bytes", "remote_topk_idx_bytes", "remote_weight_bytes"):
            self.assertEqual(value[key], 0)

    def test_official_numerator_uses_receive_rows_including_self(self):
        # Rank 0 receives one self row, though it sends two rows in total.
        value = self.ledger()
        self.assertEqual(value["official_logical_rows"], 1)
        self.assertEqual(value["official_logical_bytes_per_row"], 128 * 2 + 2 * (8 + 4))
        self.assertEqual(value["official_logical_bytes"], 280)
        # Rank 1 receives two rows: one remote and one self.
        value = self.ledger(rank=1)
        self.assertEqual(value["official_logical_bytes"], 560)
        self.assertEqual(value["remote_payload_bytes"], 0)

    def test_official_cached_counts_logical_metadata_and_combine_payload_only(self):
        fresh = self.ledger(rank=1, dtype="fp8")
        cached = self.ledger(rank=1, dtype="fp8", op="cached-dispatch")
        self.assertEqual(fresh["official_logical_bytes"], 2 * (128 + 4 + 2 * 12))
        self.assertEqual(cached["official_logical_bytes"], fresh["official_logical_bytes"])
        for op in ("combine_unweighted", "combine_weighted"):
            value = self.ledger(rank=1, dtype="fp8", op=op)
            self.assertEqual(value["official_logical_bytes"], 2 * 128 * 2)
            self.assertEqual(value["remote_x_bytes"], 128 * 2)

    def test_combine_transposes_direction_and_remains_bf16(self):
        value = self.ledger(rank=1, dtype="fp8", op="combine_weighted")
        self.assertEqual(value["remote_tx_rows"], 1)
        self.assertEqual(value["remote_rx_rows"], 0)
        self.assertEqual(value["remote_x_bytes"], 256)
        self.assertEqual(value["remote_scale_bytes"], 0)
        self.assertEqual(value["remote_weight_bytes"], 8)
        self.assertEqual(self.ledger(op="combine_unweighted")["remote_payload_bytes"], 0)

    def test_invalid_routes_and_unknown_ops_rejected(self):
        with self.assertRaises(ValueError):
            route_counts([[[4]]], 4)
        with self.assertRaises(ValueError):
            self.ledger(op="other")

    def test_resource_preflight_fails_closed(self):
        text = "| 0 | Ascend950DT | OK | 380 |\n| 1 | Ascend950DT | Warning | 380 |\n"
        text += "| No running processes found in NPU 0 |\n| No running processes found in NPU 1 |\n"
        check_idle(text, 0, 2)
        for invalid in (text.replace("Warning", "Alarm"), text.replace("in NPU 1", "in NPU 2"), ""):
            with self.assertRaises(RuntimeError):
                check_idle(invalid, 0, 2)

    def make_profile(self, root, sequence, repeat=2, device=0, flush=False):
        events = []

        def add_flow(name, device_start, host_start, duration):
            identity = len(events)
            events.extend(
                [
                    dict(name="torch_to_npu", cat="async_npu", ph="s", id=identity, pid=1, tid=10, ts=host_start),
                    dict(name="torch_to_npu", cat="async_npu", ph="f", id=identity, pid=2, tid=20, ts=device_start),
                    dict(
                        name=name,
                        ph="X",
                        pid=2,
                        tid=20,
                        ts=device_start,
                        dur=duration,
                        args={"Task Type": "KERNEL_AIVEC"},
                    ),
                ]
            )

        (root / "profiler_info_1.json").write_text("{}")
        with (root / "kernel_details.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f, fieldnames=["Name", "Start Time(us)", "Duration(us)", "Device_id", "Step Id", "Block Num"]
            )
            writer.writeheader()
            for i in range(repeat):
                if flush:
                    # Host range deliberately overlaps an unrelated API kernel
                    # in device time. Only explicit flow ownership is valid.
                    events.append(dict(name=L2_FLUSH_RANGE, ph="X", ts=i * 100 + 20, dur=5, pid=1, tid=10))
                    add_flow("zero", i * 100 + 1, i * 100 + 21, 2.0000625)
                    writer.writerow(
                        dict(
                            Name="zero",
                            **{
                                "Start Time(us)": str(i * 100 + 1),
                                "Duration(us)": "2",
                                "Device_id": str(device),
                                "Step Id": "1",
                                "Block Num": "64",
                            },
                        )
                    )
                for j, name in enumerate(sequence):
                    if flush:
                        add_flow(name, i * 100 + j * 10 + 10, i * 100 + j * 10 + 30, 2.1250625)
                    writer.writerow(
                        dict(
                            Name=name,
                            **{
                                "Start Time(us)": str(i * 100 + j * 10 + (10 if flush else 0)),
                                "Duration(us)": "2.125",
                                "Device_id": str(device),
                                "Step Id": "1",
                                "Block Num": "32",
                            },
                        )
                    )
        (root / "trace_view.json").write_text(json.dumps(events if flush else {}))

    def make_guarded_profile(self, root, sequence, flush=False, device=0):
        self.make_profile(root, sequence, repeat=4, flush=flush, device=device)
        path = root / "trace_view.json"
        events = json.loads(path.read_text()) if flush else []
        # Deliberately offset Host and device clocks; timestamp slicing on Host
        # ranges would lose all measured kernels.
        for event in events:
            if event.get("pid") == 1:
                event["ts"] += 10000
        for ordinal, sample in enumerate((-1, 0, 1, 2)):
            for edge, offset in (("start", -5), ("end", 80)):
                host, device = 10000 + ordinal * 100 + offset, ordinal * 100 + offset
                identity = f"marker-{sample}-{edge}"
                events.extend(
                    [
                        dict(name=f"{PROFILE_SAMPLE_RANGE}{sample}/{edge}", ph="X", pid=1, tid=10, ts=host, dur=3),
                        dict(name="torch_to_npu", cat="async_npu", ph="s", id=identity, pid=1, tid=10, ts=host + 1),
                        dict(name="torch_to_npu", cat="async_npu", ph="f", id=identity, pid=2, tid=20, ts=device),
                        dict(name="EVENT RECORD", ph="X", pid=2, tid=20, ts=device, dur=0.01),
                    ]
                )
        path.write_text(json.dumps(events))

    def test_guarded_capture_excludes_clipped_edges_and_preserves_fixed_ids(self):
        sequences = [
            ("cached-dispatch", False, "v2", ["cached_notify_dispatch_kernel", "v1_dispatch_udma_kernel"]),
            ("dispatch", False, "v2", ["dispatch_layout_kernel", "notify_dispatch_kernel", "v1_dispatch_udma_kernel"]),
            (
                "combine_weighted",
                False,
                "v2",
                ["cached_notify_combine_kernel", "v1_combine_udma_comm_kernel", "v1_combine_udma_reduce_kernel"],
            ),
            (
                "combine_unweighted",
                True,
                "v2",
                [
                    "cached_notify_combine_kernel",
                    "v1_combine_udma_early_kernel",
                    "aggregate_kernel",
                    "v1_combine_udma_remaining_kernel",
                    "v1_combine_udma_reduce_kernel",
                ],
            ),
            ("combine_weighted", False, "v1", ["cached_notify_combine_kernel", "v1_combine_udma_kernel"]),
        ]
        for op, expanded, api, sequence in sequences:
            for flush in (False, True):
                with self.subTest(op=op, flush=flush, api=api), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    self.make_guarded_profile(root, sequence, flush)
                    # Reproduce the real failure shape: capture begins with an
                    # orphan communication task and a blank Step Id.
                    path = root / "kernel_details.csv"
                    with path.open(encoding="utf-8-sig") as f:
                        rows = list(csv.DictReader(f))
                    rows = rows[1:]
                    rows[0]["Step Id"] = ""
                    with path.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=rows[0])
                        writer.writeheader()
                        writer.writerows(rows)
                    profile = parse_profile(
                        root, op, expanded, 2, 0, 32, api=api, flush_l2_cache=flush, sample_markers=True
                    )
                    self.assertEqual(profile["sampling"]["sample_ids"], [0, 1])
                    self.assertEqual(profile["kernel_sequence"], sequence)
                    self.assertEqual(len(profile["samples"]), 2)
                    self.assertTrue(all(100 <= float(k["start_us"]) < 280 for k in profile["kernels"]))
                    self.assertEqual(len(profile["l2_flush_kernels"]), 2 if flush else 0)
                    with self.assertRaises(ValueError):
                        parse_profile(root, op, expanded, 2, 0, 32, api=api, flush_l2_cache=flush)

    def test_shmem_alignment_excludes_only_known_combine_barriers_outside_samples(self):
        barrier = "_Z24barrier_on_stream_kernelim"
        for api in ("v1", "v2"):
            sequence = (
                ["v1_combine_udma_kernel"]
                if api == "v1"
                else ["v1_combine_udma_comm_kernel", "v1_combine_udma_reduce_kernel"]
            )
            for damage in (None, "unknown", "device", "step", "crossing"):
                with self.subTest(api=api, damage=damage), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    self.make_guarded_profile(root, sequence)
                    baseline = parse_profile(root, "combine_unweighted", False, 2, 0, 32, api=api, sample_markers=True)
                    path = root / "kernel_details.csv"
                    with path.open(encoding="utf-8-sig") as f:
                        rows = list(csv.DictReader(f))
                    gap = dict(rows[0], Name=barrier)
                    gap["Start Time(us)"] = "185"
                    if damage == "unknown":
                        gap["Name"] = "unexpected_kernel"
                    if damage == "device":
                        gap["Device_id"] = "1"
                    if damage == "step":
                        gap["Step Id"] = "2"
                    if damage == "crossing":
                        gap["Duration(us)"] = "20"
                    rows.append(gap)
                    with path.open("w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=rows[0])
                        writer.writeheader()
                        writer.writerows(rows)
                    if damage:
                        with self.assertRaises(ValueError):
                            parse_profile(root, "combine_unweighted", False, 2, 0, 32, api=api, sample_markers=True)
                    else:
                        result = parse_profile(
                            root, "combine_unweighted", False, 2, 0, 32, api=api, sample_markers=True
                        )
                        self.assertEqual(result["samples"], baseline["samples"])
                        self.assertEqual(result["kernels"], baseline["kernels"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_guarded_profile(root, [barrier, "v1_combine_udma_comm_kernel", "v1_combine_udma_reduce_kernel"])
            result = parse_profile(root, "combine_unweighted", False, 2, 0, 32, sample_markers=True)
            self.assertIn(barrier, result["kernel_sequence"])
            self.assertEqual(float(result["samples"][0]["other_us"]), 2.125)

    def test_guarded_capture_rejects_missing_markers_and_interior_damage(self):
        for damage in ("marker", "flow", "duplicate", "interior", "shift", "overlap"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.make_guarded_profile(root, ["cached_notify_dispatch_kernel", "v1_dispatch_udma_kernel"])
                path = root / "trace_view.json"
                events = json.loads(path.read_text())
                if damage == "marker":
                    events = [e for e in events if e.get("name") != PROFILE_SAMPLE_RANGE + "0/start"]
                elif damage == "flow":
                    events = [e for e in events if not (e.get("id") == "marker-0-start" and e.get("ph") == "f")]
                elif damage == "duplicate":
                    events += [e for e in events if e.get("name") == PROFILE_SAMPLE_RANGE + "0/start"]
                elif damage == "overlap":
                    for e in events:
                        if e.get("name") == "EVENT RECORD" and e["ts"] == 180:
                            e["name"] = "WRONG TASK"
                else:
                    csvpath = root / "kernel_details.csv"
                    with csvpath.open(encoding="utf-8-sig") as f:
                        rows = list(csv.DictReader(f))
                    if damage == "interior":
                        del rows[2]  # formal sample 0's notify
                    else:
                        # Counts remain correct but one stage moves to sample 1.
                        rows[3]["Start Time(us)"] = "205"
                    with csvpath.open("w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=rows[0])
                        writer.writeheader()
                        writer.writerows(rows)
                path.write_text(json.dumps(events))
                with self.assertRaises(ValueError):
                    parse_profile(root, "cached-dispatch", False, 2, 0, 32, sample_markers=True)

    def test_flush_exclusion_preserves_api_zero_and_all_api_times(self):
        sequence = [
            "zero",
            "cached_notify_combine_kernel",
            "v1_combine_udma_comm_kernel",
            "v1_combine_udma_reduce_kernel",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_profile(root, sequence)
            baseline = parse_profile(root, "combine_unweighted", False, 2, 0, 32)
            self.make_profile(root, sequence, flush=True)
            result = parse_profile(root, "combine_unweighted", False, 2, 0, 32, flush_l2_cache=True)
            self.assertEqual(result["samples"], baseline["samples"])
            self.assertEqual(result["kernel_sequence"], sequence)
            self.assertEqual(len(result["l2_flush_kernels"]), 2)
            self.assertEqual(result["l2_flush_kernels"][0][0]["duration_us"], "2")
            trace = root / "trace_view.json"
            events = json.loads(trace.read_text())
            for invalid in (
                events[:1],
                [dict(e, dur=25) if e["name"] == L2_FLUSH_RANGE else e for e in events],
                [dict(e, ts=e["ts"] + 3) if e["name"] == L2_FLUSH_RANGE else e for e in events],
                [e for e in events if e["ph"] != "f"],
                [dict(e, tid=99) if e["ph"] == "s" else e for e in events],
            ):
                trace.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    parse_profile(root, "combine_unweighted", False, 2, 0, 32, flush_l2_cache=True)

    def test_profile_preserves_repeated_auxiliary_kernels_and_gaps(self):
        sequence = [
            "zero",
            "aggregate_kernel",
            "zero",
            "cached_notify_combine_kernel",
            "v1_combine_udma_comm_kernel",
            "v1_combine_udma_reduce_kernel",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_profile(root, sequence)
            result = parse_profile(root, "combine_unweighted", True, 2, 0, 32)
            sample = result["samples"][0]
            self.assertEqual(sample["other_us"], "4.250")
            self.assertEqual(sample["device_sum_us"], "12.750")
            self.assertEqual(sample["device_span_us"], "52.125")
            self.assertEqual(sample["combine_stage_us"], "4.250")
            self.assertEqual(result["absent_stages"], ["layout", "expand"])
            with self.assertRaises(ValueError):
                parse_profile(root, "combine_unweighted", True, 3, 0, 32)
            with self.assertRaises(ValueError):
                parse_profile(root, "combine_unweighted", True, 2, 0, 64)
            with self.assertRaises(ValueError):
                parse_profile(root, "combine_unweighted", False, 2, 0, 32)

    def test_fixed_header_combine_without_notify(self):
        for api in ("v1", "v2"):
            for expanded in (False, True):
                for flush in (False, True):
                    with self.subTest(api=api, expanded=expanded, flush=flush), tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp)
                        sequence = ["aggregate_kernel"] if expanded else []
                        sequence += (
                            ["v1_combine_udma_kernel"]
                            if api == "v1"
                            else ["v1_combine_udma_comm_kernel", "v1_combine_udma_reduce_kernel"]
                        )
                        self.make_profile(root, sequence, flush=flush)
                        result = parse_profile(
                            root, "combine_weighted", expanded, 2, 0, 32, api=api, flush_l2_cache=flush
                        )
                        self.assertEqual(result["kernel_sequence"], sequence)
                        for sample in result["samples"]:
                            self.assertEqual(float(sample["notify_us"]), 0)
                            self.assertGreater(float(sample["fused_combine_us" if api == "v1" else "comm_us"]), 0)

    def test_partial_notify_and_early_without_notify_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequence = ["cached_notify_combine_kernel", "v1_combine_udma_comm_kernel", "v1_combine_udma_reduce_kernel"]
            self.make_profile(root, sequence)
            path = root / "kernel_details.csv"
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            # Only one sample lost notify: never silently turn a missing stage into zero.
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows[1:])
            with self.assertRaises(ValueError):
                parse_profile(root, "combine_unweighted", False, 2, 0, 32)
            self.make_profile(
                root,
                [
                    "v1_combine_udma_early_kernel",
                    "aggregate_kernel",
                    "v1_combine_udma_remaining_kernel",
                    "v1_combine_udma_reduce_kernel",
                ],
            )
            with self.assertRaises(ValueError):
                parse_profile(root, "combine_unweighted", True, 2, 0, 32)

    def test_missing_reduce_is_error_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_profile(root, ["cached_notify_combine_kernel", "v1_combine_udma_comm_kernel"])
            with self.assertRaises(ValueError):
                parse_profile(root, "combine_unweighted", False, 2, 0, 32)

    def test_early_submission_counts_once_and_preserves_completion_boundary(self):
        sequence = [
            "zero",
            "cached_notify_combine_kernel",
            "v1_combine_udma_early_kernel",
            "aggregate_kernel",
            "v1_combine_udma_remaining_kernel",
            "v1_combine_udma_reduce_kernel",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_profile(root, sequence)
            sample = parse_profile(root, "combine_unweighted", True, 2, 0, 32)["samples"][0]
            self.assertEqual(sample["early_submit_us"], "2.125")
            self.assertEqual(sample["late_comm_us"], "2.125")
            self.assertEqual(sample["comm_us"], "4.250")
            self.assertEqual(sample["combine_stage_us"], "6.375")
            self.assertEqual(sample["device_sum_us"], "12.750")
            self.assertEqual(sample["combine_stage_span_us"], "22.125")
            for bad in [
                sequence[1:],
                sequence[:2] + sequence[3:],
                sequence[:2] + [sequence[3], sequence[2]] + sequence[4:],
            ]:
                if bad == sequence[1:]:
                    bad = bad + ["v1_combine_udma_comm_kernel"]
                self.make_profile(root, bad)
                with self.assertRaises(ValueError):
                    parse_profile(root, "combine_unweighted", True, 2, 0, 32)

    def test_out_of_order_stage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_profile(
                root, ["cached_notify_combine_kernel", "v1_combine_udma_reduce_kernel", "v1_combine_udma_comm_kernel"]
            )
            with self.assertRaises(ValueError):
                parse_profile(root, "combine_unweighted", False, 2, 0, 32)

    def test_global_throughput_uses_total_bytes_and_per_sample_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_profile(root, ["cached_notify_dispatch_kernel", "v1_dispatch_udma_kernel"])
            profile = parse_profile(root, "cached-dispatch", False, 2, 0, 32)
            records = []
            for rank in (0, 1):
                records.append(
                    dict(
                        rank=rank,
                        operators={
                            "cached-dispatch": dict(
                                profile=profile,
                                caller_us=[10 + rank, 20 + rank],
                                bytes=self.ledger(rank=rank, op="cached-dispatch"),
                            )
                        },
                    )
                )
            summary = summarize(records)
            self.assertEqual(summary[-1]["caller_us"], 16)
            self.assertEqual(summary[-1]["remote_payload_bytes"], 256)
            self.assertAlmostEqual(summary[-1]["remote_payload_GBps"], 256 / 2.125 / 1000)


class MultinodeTest(unittest.TestCase):
    def args(self, node=0):
        return arguments(
            [
                "--ep-size",
                "16",
                "--nnodes",
                "2",
                "--node-rank",
                str(node),
                "--master-addr",
                "192.0.2.10",
                "--run-id",
                "ep16_cpu_test",
            ]
        )

    def test_ep8_command_unchanged_and_ep16_rank_mapping(self):
        case = dict(name="bf16_expand0_aiv32", dtype="bf16", expand=0, aiv=32)
        single = worker_command(arguments([]), case, Path("out"))
        self.assertIn("--standalone", single)
        self.assertIn("--nproc-per-node=8", single)
        for node in (0, 1):
            args = self.args(node)
            command = worker_command(args, case, Path("out"))
            self.assertEqual(local_world_size(args), 8)
            self.assertIn("--nnodes=2", command)
            self.assertIn("--nproc-per-node=8", command)
            self.assertIn(f"--node-rank={node}", command)
            self.assertIn("--master-port=29501", command)
            self.assertNotIn("--standalone", command)
            parsed = arguments(command[command.index("--worker") :])
            self.assertEqual(parsed.node_rank, node)
            self.assertEqual(parsed.ep_size, 16)
        self.assertEqual(suite_contract(self.args(0)), suite_contract(self.args(1)))

    def test_invalid_topologies_and_endpoints_rejected(self):
        cases = [
            ["--ep-size", "16", "--node-rank", "1"],
            ["--nnodes", "2"],
            ["--node-rank", "1"],
            ["--ep-size", "16", "--nnodes", "2"],
            ["--ep-size", "16", "--nnodes", "2", "--master-addr", "127.0.0.1", "--run-id", "x"],
            ["--ep-size", "16", "--nnodes", "2", "--master-addr", "192.0.2.10", "--run-id", "../x"],
            ["--master-port", "65535"],
            ["--collection-timeout", "0"],
        ]
        for argv in cases:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(argv)

    def test_shared_contract_rejects_mixed_runs_shapes_and_sources(self):
        hello = dict(node_rank=0, contract=suite_contract(self.args()), source={"sha": "a"}, topk_idx_bytes=8)
        other = copy.deepcopy(hello)
        other["node_rank"] = 1
        validate_peers(hello, other)
        for key, value in [("run_id", "other_run"), ("num_tokens", 1), ("dtypes", ["bf16"])]:
            bad = copy.deepcopy(other)
            bad["contract"][key] = value
            with self.assertRaises(ValueError):
                validate_peers(hello, bad)
        for key, value in [("source", {"sha": "b"}), ("node_rank", 0), ("topk_idx_bytes", 4)]:
            bad = dict(other, **{key: value})
            with self.assertRaises(ValueError):
                validate_peers(hello, bad)

    def pair(self):
        left, right = socket.socketpair()
        peers = Peer(left, 0, timeout=2), Peer(right, 1, timeout=2)
        self.addCleanup(peers[0].close)
        self.addCleanup(peers[1].close)
        return peers

    def parallel(self, functions):
        values, errors = {}, {}

        def run(index, function):
            try:
                values[index] = function()
            except Exception as exc:
                errors[index] = exc

        threads = [threading.Thread(target=run, args=(i, f), daemon=True) for i, f in enumerate(functions)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive(), "CPU protocol test hung")
        return values, errors

    def test_busy_or_failed_preflight_reaches_both_nodes(self):
        peers = self.pair()
        _, errors = self.parallel(
            [lambda: peers[0].require_both("preflight"), lambda: peers[1].require_both("preflight", "NPU 3 busy")]
        )
        self.assertEqual(set(errors), {0, 1})
        self.assertTrue(all("NPU 3 busy" in str(e) for e in errors.values()))

    def test_disconnect_and_stage_mismatch_stop_protocol(self):
        peers = self.pair()
        _, errors = self.parallel([lambda: peers[0].exchange("caseA", {}), lambda: peers[1].exchange("caseB", {})])
        self.assertEqual(set(errors), {0, 1})
        peers[1].close()
        with self.assertRaises((ConnectionError, OSError)):
            peers[0].receive_json()

    def test_file_transfer_is_streamed_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "source", Path(tmp) / "collected"
            source.mkdir()
            (source / "raw.bin").write_bytes(bytes(range(256)) * 8200)
            (source / "empty.log").write_bytes(b"")
            manifest = [
                dict(relative=p.name, **{k: file_identity(p)[k] for k in ("size", "sha256")})
                for p in sorted(source.iterdir())
            ]
            peers = self.pair()
            _, errors = self.parallel(
                [
                    lambda: peers[0].transfer(source, target, None, 5),
                    lambda: peers[1].transfer(source, target, manifest, 5),
                ]
            )
            self.assertEqual(errors, {})
            for item in manifest:
                self.assertEqual(file_identity(target / item["relative"])["sha256"], item["sha256"])
            self.assertTrue(json.loads((target / "collection.json").read_text())["passed"])

    def test_transfer_corruption_and_unsafe_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("../escape", "/abs", "a/../b", "a\\b", "a:b", "a//b"):
                with self.assertRaises(ValueError):
                    safe_path(root, name)
            source, target = root / "src", root / "dst"
            source.mkdir()
            (source / "file").write_bytes(b"abc")
            peers = self.pair()
            manifest = [dict(relative="file", size=3, sha256="0" * 64)]
            _, errors = self.parallel(
                [
                    lambda: peers[0].transfer(source, target, None, 5),
                    lambda: peers[1].transfer(source, target, manifest, 5),
                ]
            )
            self.assertEqual(set(errors), {0, 1})
            self.assertFalse((target / "file").exists())
            self.assertFalse((target / "collection.json").exists())

    def test_library_identity_compares_bytes_not_install_path(self):
        base = dict(
            torch="2.8",
            torch_npu="2.8",
            python_module=dict(size=2, sha256="a"),
            libraries=[dict(path="/one/libshmem.so", size=4, sha256="b")],
        )
        other = copy.deepcopy(base)
        other["libraries"][0]["path"] = "/different/libshmem.so"
        self.assertEqual(library_identity(base), library_identity(other))
        other["libraries"][0]["sha256"] = "c"
        self.assertNotEqual(library_identity(base), library_identity(other))

    def test_peer_failure_or_local_timeout_terminates_owned_child(self):
        case = dict(name="bf16_expand0_aiv32", dtype="bf16", expand=0, aiv=32)
        for failure in ("peer", "timeout", "disconnect"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                args = self.args()
                args.case_timeout = 1
                child = mock.Mock(pid=123456, returncode=-15)
                child.poll.return_value = None
                peer = mock.Mock()
                peer.exchange.return_value = dict(code=1 if failure == "peer" else None, timed_out=False)
                if failure == "disconnect":
                    peer.exchange.side_effect = ConnectionError("lost peer")
                clock = [0, 2] if failure == "timeout" else [0, 0.1]
                with (
                    mock.patch("bench_v2_bandwidth.subprocess.Popen", return_value=child),
                    mock.patch("bench_v2_bandwidth.stop_child") as stop,
                    mock.patch("bench_v2_bandwidth.time.monotonic", side_effect=clock),
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaises((RuntimeError, ConnectionError)),
                ):
                    run_case(args, case, Path(tmp), {}, peer)
                stop.assert_called_once_with(child)
                status = json.loads((Path(tmp) / "launcher_exit.json").read_text())
                self.assertTrue(status["suite_aborted"])

    def test_two_node_report_replays_all_16_raw_rank_records(self):
        self.check_two_node_report()

    def test_v1_two_node_report_replays_fused_results_and_kernel_details(self):
        self.check_two_node_report(api="v1")

    def test_v1_expanded_fp8_two_node_report(self):
        self.check_two_node_report(api="v1", expanded=True, dtype="fp8")

    def test_single_node_ep16_reports_both_apis(self):
        for api in ("v1", "v2"):
            with self.subTest(api=api):
                self.check_two_node_report(api=api, expanded=True, dtype="fp8", single=True)

    def test_eight_node_ep64_report_replays_all_raw_rank_records(self):
        self.check_two_node_report(expanded=True, dtype="fp8", ep_size=64, nnodes=8)

    def test_automatic_ep32_report_replays_notify_reduce_and_policy(self):
        self.check_two_node_report(ep_size=32, nnodes=2, auto_policy=True)

    def test_effective_config_survives_report_and_rejects_rank_mismatch(self):
        self.check_two_node_report(single=True, effective=True)

    def test_flush_report_replay_both_apis(self):
        for api in ("v1", "v2"):
            with self.subTest(api=api):
                self.check_two_node_report(api=api, expanded=True, ep_size=4, nnodes=2, flush=True)

    def test_guarded_report_replay_both_apis_and_flush_modes(self):
        for api in ("v1", "v2"):
            for flush in (False, True):
                with self.subTest(api=api, flush=flush):
                    self.check_two_node_report(api=api, expanded=True, ep_size=4, nnodes=2, flush=flush, guarded=True)

    def check_two_node_report(
        self,
        api="v2",
        expanded=False,
        dtype="bf16",
        single=False,
        ep_size=16,
        nnodes=None,
        auto_policy=False,
        effective=False,
        flush=False,
        guarded=False,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "node0"
            nnodes = nnodes or (1 if single else 2)
            node_paths = [root] + [Path(tmp) / "different_install" / f"node{n}" for n in range(1, nnodes)]
            case = dict(name=f"{dtype}_expand{int(expanded)}_aiv32", dtype=dtype, expand=int(expanded), aiv=32)
            flags = ["--ep-size", str(ep_size), "--nnodes", str(nnodes)]
            if nnodes > 1:
                flags += ["--master-addr", "192.0.2.1", "--run-id", "cpu_fixture"]
            args = arguments(flags)
            # Keep coverage for reports generated by the old manual entrypoint.
            args.udma_auto_policy = auto_policy
            args.peers_per_worker, args.wqes_per_visit = 4, 8
            # Historical reports predate the switch entirely, not merely False.
            if not flush:
                del args.flush_l2_cache
                del args.combine_bias
            local_size = local_world_size(args)
            if api == "v1":
                args.api = "v1"
            args.num_tokens, args.hidden, args.num_topk, args.iterations = 1, 128, 1, 2
            if guarded:
                args.profile_sample_markers = True
            plan_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            plan = dict(format_version=2, api=api, arguments=plan_args, cases=[case], topk_idx_bytes=8)
            write_json(root / "plan.json", plan)
            for node in range(1, nnodes):
                write_json(
                    root / ("peer_plan.json" if node == 1 else f"peer_plan_node{node}.json"),
                    dict(arguments=dict(plan_args, node_rank=node)),
                )
            routes = [[[(rank + 1) % ep_size * 32]] for rank in range(ep_size)]
            counts = route_counts(routes, 32)
            identity = dict(
                torch="2.8",
                torch_npu="2.8",
                python_module=dict(size=2, sha256="module"),
                libraries=[dict(path="/lib/libshmem.so", size=1, sha256="same")],
            )
            ops = ("dispatch", "cached-dispatch", "combine_unweighted", "combine_weighted")
            for node_path in node_paths:
                path = node_path / case["name"]
                for name, data in [
                    ("launcher_exit.json", {"code": 0}),
                    ("command.json", {}),
                    ("resources.before.txt", {}),
                ]:
                    write_json(path / name, data)
                (path / "run.log").write_text("complete")
            for rank in range(ep_size):
                path = node_paths[rank // local_size] / case["name"]
                rank_dir = path / f"rank{rank}"
                write_json(rank_dir / "input_routes.json", routes[rank])
                for name in ("exit.json", "precision.json", "process.json"):
                    write_json(rank_dir / name, dict(code=0, passed=True))
                config = dict(
                    plan_args,
                    num_aiv=32,
                    dtype=dtype,
                    expand=[int(expanded)],
                    node_rank=rank // local_size,
                    output=str(path),
                )
                if effective:
                    config["effective_udma_config"] = dict(
                        udma_auto_policy=False,
                        udma_schedule="peer",
                        udma_peers_per_worker=4,
                        udma_workers_per_peer=1,
                        udma_qps_per_worker=1,
                        udma_wqes_per_visit=8,
                        udma_dispatch_peer_order="rank",
                        udma_combine_peer_order="rank",
                        udma_ranks_per_group=None,
                        udma_peers_per_group_per_worker=None,
                    )
                if auto_policy:
                    config["resolved_udma_policy"] = dict(
                        version="clos-v1",
                        world_size=ep_size,
                        peers_per_worker=4,
                        workers_per_peer=1,
                        qps_per_worker=1,
                        wqes_per_visit=32,
                    )
                operators = {}
                for op in ops:
                    profile_path = rank_dir / op / "profiler"
                    profile_path.mkdir(parents=True)
                    sequence = (
                        ["dispatch_layout_kernel", "notify_dispatch_kernel", "v1_dispatch_udma_kernel"]
                        if op == "dispatch"
                        else ["cached_notify_dispatch_kernel", "v1_dispatch_udma_kernel"]
                        if op == "cached-dispatch"
                        else [
                            "cached_notify_combine_kernel",
                            "v1_combine_udma_comm_kernel",
                            "v1_combine_udma_reduce_kernel",
                        ]
                    )
                    if api == "v1" and op.startswith("combine"):
                        sequence = ["cached_notify_combine_kernel", "v1_combine_udma_kernel"]
                    if expanded:
                        sequence = (
                            ["aggregate_kernel"] + sequence
                            if op.startswith("combine")
                            else sequence + ["expand_kernel"]
                        )
                    fixture = StageMetricsTest().make_guarded_profile if guarded else StageMetricsTest().make_profile
                    fixture(profile_path, sequence, device=rank % local_size, flush=flush)
                    operators[op] = dict(
                        profile=parse_profile(
                            profile_path,
                            op,
                            expanded,
                            2,
                            rank % local_size,
                            32,
                            api=api,
                            flush_l2_cache=flush,
                            sample_markers=guarded,
                        ),
                        caller_us=[10, 12],
                        bytes=byte_ledger(counts, rank, tokens=1, hidden=128, topk=1, dtype=dtype, idx_bytes=8, op=op),
                    )
                    write_json(rank_dir / op / "measurement.json", operators[op])
                write_json(
                    rank_dir / "result.json",
                    dict(
                        rank=rank,
                        device=rank % local_size,
                        config=config,
                        precision_passed=True,
                        destroyed=True,
                        route_counts=counts,
                        identity=identity,
                        input_routes=file_identity(rank_dir / "input_routes.json"),
                        operators=operators,
                    ),
                )
            target = root / case["name"]
            for node in range(1, nnodes):
                source = node_paths[node] / case["name"]
                manifest = collection_manifest(source, range(node * local_size, (node + 1) * local_size))
                target = root / case["name"] / f"peer_node{node}"
                for item in manifest:
                    destination = target / item["relative"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / item["relative"], destination)
                write_json(target / "collection.json", dict(passed=True, files=manifest))
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                report(root)
            result = json.loads((root / "report.json").read_text())
            self.assertTrue(result["passed"])
            self.assertEqual(result["profile_sample_markers"], guarded)
            self.assertEqual(result["flush_l2_cache"], flush)
            self.assertEqual(result["l2_flush_bytes"], 256_000_000 if flush else 0)
            self.assertEqual(result["combine_bias"], "none" if flush else "two")
            for row in result["summary"]:
                self.assertEqual(row["bias_count"], 2 if not flush and row["op"] == "combine_weighted" else 0)
            self.assertEqual(result["api"], api)
            self.assertEqual(len(result["summary"]), (ep_size + 1) * 4)
            if auto_policy:
                self.assertEqual(result["resolved_udma_policies"][case["name"]], config["resolved_udma_policy"])
                self.assertIn("GBps_per_rank / comm_us / notify_us / reduce_us", printed.getvalue())
                combined = next(r for r in result["summary"] if r["rank"] == "all" and r["op"] == "combine_unweighted")
                self.assertGreater(combined["notify_us"], 0)
                self.assertGreater(combined["reduce_us"], 0)
                self.assertAlmostEqual(
                    combined["official_logical_GBps"], combined["official_logical_bytes"] / combined["comm_us"] / 1000
                )
            row = next(r for r in result["summary"] if r["rank"] == "all" and r["op"] == "dispatch")
            payload = 132 if dtype == "fp8" else 256
            self.assertEqual(row["remote_payload_bytes"], ep_size * payload)
            self.assertAlmostEqual(row["remote_payload_GBps"], ep_size * payload / 2.125 / 1000)
            self.assertEqual(row["official_logical_bytes"], ep_size * (payload + 12))
            self.assertAlmostEqual(row["official_logical_GBps"], ep_size * (payload + 12) / 2.125 / 1000)
            self.assertEqual((root / "summary.csv").read_bytes()[:3], b"\xef\xbb\xbf")
            with (root / "kernels.csv").open(encoding="utf-8-sig", newline="") as stream:
                kernels = list(csv.DictReader(stream))
            self.assertEqual(len(kernels), ep_size * 2 * ((9 if api == "v1" else 11) + (4 if expanded else 0)))
            if api == "v1":
                fused = next(r for r in result["summary"] if r["rank"] == "all" and r["op"] == "combine_unweighted")
                self.assertIsNone(fused["comm_us"])
                self.assertIsNone(fused["reduce_us"])
                self.assertIsNone(fused["official_logical_GBps"])
                self.assertAlmostEqual(fused["logical_fused_combine_GBps"], ep_size * 256 / 2.125 / 1000)
            if effective:
                self.assertEqual(result["effective_udma_configs"][case["name"]], config["effective_udma_config"])
                record_path = root / case["name"] / "rank1" / "result.json"
                original = json.loads(record_path.read_text())
                changed = copy.deepcopy(original)
                changed["config"]["effective_udma_config"]["udma_wqes_per_visit"] = 16
                write_json(record_path, changed)
                with self.assertRaisesRegex(ValueError, "different effective UDMA"):
                    report(root)
                write_json(record_path, original)
            (target / f"rank{ep_size - 1}" / "exit.json").unlink()
            with self.assertRaises(FileNotFoundError):
                report(root)


if __name__ == "__main__":
    unittest.main()
