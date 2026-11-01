"""CPU-only sweep planning, failure propagation, collection and ranking checks."""

from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks/clos"))
import bench_udma_sweep as sweep
import bench_v2_bandwidth as shared
from bandwidth_multinode import PeerGroup
from stage_metrics import file_identity, write_json


class SweepTest(unittest.TestCase):
    def plan(self, extra=()):
        options, base = sweep.arguments(list(extra))
        return options, base, sweep.make_plan(options, base)

    def test_full_default_grid_and_unique_jobs(self):
        _, _, plan = self.plan()
        self.assertEqual(plan["candidates"], 2368)
        self.assertEqual(len(plan["jobs"]), len({j["name"] for j in plan["jobs"]}))
        configs = [json.dumps(j["config"], sort_keys=True) for j in plan["jobs"]]
        self.assertEqual(len(configs), len(set(configs)))
        token = [j for j in plan["jobs"] if j["config"]["udma_schedule"] == "token"]
        self.assertEqual(len(token), 64)
        self.assertTrue(
            all(
                j["config"]["dispatch_peer_order"] == "rank"
                and j["config"]["combine_peer_order"] == "rank"
                and j["config"]["ranks_per_group"] is None
                for j in token
            )
        )

    def test_ep64_filters_illegal_workers_and_divisors(self):
        _, _, plan = self.plan(
            [
                "--ep-size",
                "64",
                "--nnodes",
                "8",
                "--node-rank",
                "0",
                "--master-addr",
                "192.0.2.1",
                "--run-id",
                "check",
                "--group-sizes",
                "8",
                "3",
            ]
        )
        self.assertTrue(plan["skipped"])
        for job in plan["jobs"]:
            config = job["config"]
            if config["udma_schedule"] == "peer":
                self.assertIn(config["peers_per_worker"], (2, 4, 8))
                self.assertIn(config["ranks_per_group"], (None, 8))

    def test_group_round_robin_grid_and_worker_command(self):
        _, base, plan = self.plan(
            [
                "--ep-size",
                "64",
                "--nnodes",
                "8",
                "--node-rank",
                "0",
                "--master-addr",
                "192.0.2.1",
                "--run-id",
                "rr",
                "--apis",
                "v2",
                "--schedules",
                "peer",
                "--dispatch-peer-orders",
                "rank",
                "group_round_robin",
                "--combine-peer-orders",
                "rank",
                "group_round_robin",
                "--paired-peer-orders",
                "--group-sizes",
                "8",
                "16",
                "--group-peer-counts",
                "1",
                "2",
                "4",
                "--peers-per-worker",
                "4",
                "--wqes-per-visit",
                "8",
                "--aiv",
                "64",
                "--dtypes",
                "bf16",
                "--expand",
                "0",
                "1",
            ]
        )
        self.assertEqual(len(plan["jobs"]), 14)
        self.assertFalse(plan["skipped"])
        for job in plan["jobs"]:
            args = sweep.job_arguments(base, job, Path("out"), plan["sha256"])
            case = dict(name="case", dtype=args.dtypes[0], aiv=args.aiv[0], expand=args.expand[0])
            command = shared.worker_command(args, case, Path("out"))
            replay = shared.arguments(command[command.index("--worker") :])
            self.assertEqual(replay.peers_per_group_per_worker, job["config"]["peers_per_group_per_worker"])
            self.assertEqual(replay.dispatch_peer_order, replay.combine_peer_order)

    def test_multi_qp_grid_and_command(self):
        _, base, plan = self.plan(
            [
                "--ep-size",
                "16",
                "--apis",
                "v2",
                "--schedules",
                "peer",
                "--dispatch-peer-orders",
                "group_round_robin",
                "--combine-peer-orders",
                "group_round_robin",
                "--group-sizes",
                "16",
                "--group-peer-counts",
                "1",
                "--workers-per-peer",
                "1",
                "2",
                "--peers-per-worker",
                "1",
                "--wqes-per-visit",
                "8",
                "--aiv",
                "64",
                "--dtypes",
                "bf16",
                "--expand",
                "0",
            ]
        )
        self.assertEqual(len(plan["jobs"]), 2)
        self.assertFalse(plan["skipped"])
        for job in plan["jobs"]:
            args = sweep.job_arguments(base, job, Path("out"), plan["sha256"])
            case = dict(name="case", dtype="bf16", aiv=64, expand=0)
            command = shared.worker_command(args, case, Path("out"))
            replay = shared.arguments(command[command.index("--worker") :])
            self.assertEqual(replay.workers_per_peer, job["config"]["workers_per_peer"])

    def test_two_qps_per_core_grid_and_command(self):
        _, base, plan = self.plan(
            [
                "--ep-size",
                "16",
                "--apis",
                "v2",
                "--schedules",
                "peer",
                "--dispatch-peer-orders",
                "rank",
                "--combine-peer-orders",
                "rank",
                "--workers-per-peer",
                "1",
                "2",
                "--qps-per-worker",
                "1",
                "2",
                "--peers-per-worker",
                "1",
                "--wqes-per-visit",
                "8",
                "--aiv",
                "32",
                "64",
                "--dtypes",
                "bf16",
                "--expand",
                "0",
            ]
        )
        self.assertEqual(len(plan["jobs"]), 6)
        self.assertEqual(len(plan["skipped"]), 2)
        for job in plan["jobs"]:
            args = sweep.job_arguments(base, job, Path("out"), plan["sha256"])
            case = dict(name="case", dtype="bf16", aiv=args.aiv[0], expand=0)
            command = shared.worker_command(args, case, Path("out"))
            replay = shared.arguments(command[command.index("--worker") :])
            self.assertEqual(replay.qps_per_worker, job["config"]["qps_per_worker"])
            self.assertEqual(replay.workers_per_peer, job["config"]["workers_per_peer"])
        _, _, oversized = self.plan(
            [
                "--ep-size",
                "32",
                "--nnodes",
                "2",
                "--master-addr",
                "192.0.2.1",
                "--run-id",
                "dual",
                "--apis",
                "v2",
                "--schedules",
                "peer",
                "--dispatch-peer-orders",
                "rank",
                "--combine-peer-orders",
                "rank",
                "--peers-per-worker",
                "1",
                "--qps-per-worker",
                "2",
                "--wqes-per-visit",
                "8",
                "--aiv",
                "64",
                "--dtypes",
                "bf16",
                "--expand",
                "0",
            ]
        )
        self.assertFalse(oversized["jobs"])
        self.assertEqual(len(oversized["skipped"]), 1)
        for options in (
            ["--qps-per-worker", "2", "--workers-per-peer", "2"],
            ["--qps-per-worker", "2", "--udma-schedule", "token"],
        ):
            with self.assertRaises(SystemExit):
                shared.arguments(options)

    def test_nodes_have_identical_plan_and_torchrun_contract(self):
        digests = []
        for node in range(8):
            _, base, plan = self.plan(
                [
                    "--ep-size",
                    "64",
                    "--nnodes",
                    "8",
                    "--node-rank",
                    str(node),
                    "--master-addr",
                    "192.0.2.1",
                    "--run-id",
                    "ep64",
                    "--output",
                    f"node{node}_different_path",
                ]
            )
            digests.append(plan["sha256"])
            job = next(j for j in plan["jobs"] if j["config"]["udma_schedule"] == "token")
            args = sweep.job_arguments(base, job, Path("out"), plan["sha256"])
            case = dict(name="case", dtype=args.dtypes[0], aiv=args.aiv[0], expand=args.expand[0])
            command = shared.worker_command(args, case, Path("out"))
            replay = shared.arguments(command[command.index("--worker") :])
            self.assertIn("--nproc-per-node=8", command)
            self.assertIn("--nnodes=8", command)
            self.assertEqual(replay.udma_schedule, "token")
            self.assertEqual(replay.node_rank, node)
            self.assertEqual(replay.sweep_plan_sha256, plan["sha256"])
        self.assertEqual(len(set(digests)), 1)

    def test_repeats_shuffle_order_but_keep_candidate_ids(self):
        _, _, plan = self.plan(
            [
                "--repeats",
                "3",
                "--apis",
                "v1",
                "--schedules",
                "peer",
                "--dispatch-peer-orders",
                "rank",
                "--combine-peer-orders",
                "rank",
            ]
        )
        orders = [[j["candidate"] for j in plan["jobs"] if j["repeat"] == r] for r in (1, 2, 3)]
        self.assertNotEqual(orders[0], orders[1])
        self.assertEqual(sorted(orders[0]), sorted(orders[2]))
        self.assertEqual(len(plan["jobs"]), plan["candidates"] * 3)

    def test_cli_rejects_ambiguous_options(self):
        for flags in (["--peers-per-worker", "1", "1"], ["--repeats", "0"], ["--udma-schedule", "token"], ["--worker"]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    sweep.arguments(flags)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                shared.arguments(["--udma-schedule", "token", "--ranks-per-group", "8"])

    def test_rank_directory_is_global_and_not_fixed_at_eight_devices(self):
        for ep, nnodes in ((16, 1), (16, 2), (64, 4), (64, 8)):
            plan = dict(arguments=dict(ep_size=ep, nnodes=nnodes))
            for rank in range(ep):
                node = rank // (ep // nnodes)
                expected = (
                    Path("job") / f"rank{rank}" if node == 0 else Path("job") / f"peer_node{node}" / f"rank{rank}"
                )
                self.assertEqual(shared.rank_directory(Path("job"), rank, plan), expected)

    def result_rows(self):
        _, _, plan = self.plan(
            [
                "--apis",
                "v1",
                "--schedules",
                "peer",
                "--aiv",
                "32",
                "--dtypes",
                "bf16",
                "--expand",
                "0",
                "--dispatch-peer-orders",
                "rank",
                "--combine-peer-orders",
                "rank",
                "--repeats",
                "2",
            ]
        )
        config = plan["jobs"][0]["config"]
        return [
            {
                **config,
                "rank": "all",
                "op": "combine_unweighted",
                "candidate": candidate,
                "repeat": repeat,
                "caller_us": value,
                "device_span_us": value / 2,
                "comm_us": None,
                "fused_combine_us": value / 3,
                "official_logical_GBps": None,
                "logical_fused_combine_GBps": 1000 / value,
            }
            for candidate, repeat, value in ((0, 1, 30), (0, 2, 10), (1, 1, 15), (1, 2, 17), (2, 1, 1))
        ]

    def test_ranking_uses_all_repeats_and_excludes_null_or_incomplete(self):
        rows = self.result_rows()
        ranking = sweep.rankings(rows, 2)
        caller = [r for r in ranking if r["metric"] == "caller_us"]
        self.assertEqual([r["candidate"] for r in caller], [1, 0])
        self.assertEqual([r["median_us"] for r in caller], [16, 20])
        self.assertFalse(any(r["metric"] == "comm_us" for r in ranking))
        self.assertTrue(all(r["median_official_logical_GBps"] is None for r in ranking))
        # Different workloads must never compete in the same ranking.
        changed = [dict(r, expand=1, caller_us=r["caller_us"] * 2) for r in rows]
        self.assertEqual(
            sum(r["metric"] == "caller_us" and r["position"] == 1 for r in sweep.rankings(rows + changed, 2)), 2
        )

    def test_incremental_failure_summary_preserves_rows_and_utf8bom(self):
        _, _, plan = self.plan(["--repeats", "2"])
        state = dict(status="failed", completed=[dict(job="ok")], failed="bad")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sweep.publish(root, plan, state, self.result_rows())
            report = json.loads((root / "report.json").read_text())
            self.assertFalse(report["complete"])
            self.assertEqual(report["completed"], 1)
            self.assertEqual(report["failed"], "bad")
            self.assertTrue((root / "summary.csv").read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertTrue((root / "best.csv").is_file())

    def test_completed_job_binds_configuration_source_and_report(self):
        _, base, plan = self.plan(["--apis", "v2", "--dtypes", "bf16", "--expand", "0", "--aiv", "32"])
        job = plan["jobs"][0]
        plan["source"] = dict(sources=[("benchmarks/test.py", 1, "sha")])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = sweep.job_arguments(base, job, root, plan["sha256"])
            child = args.output
            config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            child_plan = dict(arguments=config, api="v2", cases=[dict(name="case")], source_identity=plan["source"])
            write_json(child / "plan.json", child_plan)
            identity = dict(
                torch="cpu_fixture",
                torch_npu="cpu_fixture",
                python_module=dict(size=1, sha256="module"),
                libraries=[dict(path="/lib/test.so", size=1, sha256="lib")],
            )
            write_json(child / "case/rank0/result.json", dict(identity=identity))
            # dtype/aiv also occur in the child summary; merging must not raise
            # duplicate-key errors or lose the effective configuration.
            write_json(
                child / "report.json",
                dict(
                    passed=True,
                    summary=[
                        dict(rank="all", op="dispatch", dtype="bf16", aiv=32, expanded=0, caller_us=10, comm_us=5)
                    ],
                ),
            )
            rows, _, artifact = sweep.review_job(root, plan, job)
            self.assertEqual(rows[0]["udma_schedule"], job["config"]["udma_schedule"])
            self.assertEqual(rows[0]["aiv"], 32)
            self.assertGreater(artifact["size"], 0)
            child_plan["arguments"]["wqes_per_visit"] = 999
            write_json(child / "plan.json", child_plan)
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                sweep.review_job(root, plan, job)

    def test_remote_failure_stops_only_owned_child_no_retry(self):
        child, peer = Mock(), Mock()
        child.poll.return_value = None
        child.returncode = -15
        peer.exchange.return_value = dict(code=1, timed_out=False)
        args = shared.arguments([])
        case = dict(name="test", dtype="bf16", expand=0, aiv=32)
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(shared.subprocess, "Popen", return_value=child) as launch,
        ):
            with patch.object(shared, "stop_child") as stop:
                with self.assertRaises(RuntimeError):
                    shared.run_case(args, case, Path(tmp), {}, peer)
                launch.assert_called_once()
                stop.assert_called_once_with(child)
                self.assertTrue(json.loads((Path(tmp) / "launcher_exit.json").read_text())["suite_aborted"])


class GroupTest(unittest.TestCase):
    def test_eight_node_socket_consensus_failure_and_raw_collection(self):
        with socket.socket() as port_socket:
            port_socket.bind(("127.0.0.1", 0))
            port = port_socket.getsockname()[1]
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / f"node{node}" for node in range(8)]
            for node, root in enumerate(roots):
                root.mkdir()
                (root / "raw.bin").write_bytes(bytes([node]) * (10000 + node))

            def participant(node):
                group = PeerGroup.connect("127.0.0.1", port, node, 10, 8)
                try:
                    self.assertEqual(group.exchange_all("identity", node), list(range(8)))
                    state = group.exchange("poll", dict(code=1 if node == 7 else None, timed_out=False))
                    self.assertEqual(state, dict(code=1, timed_out=False))
                    with self.assertRaises(RuntimeError):
                        group.require_both("gate", "occupied" if node == 3 else None)
                    identity = file_identity(roots[node] / "raw.bin")
                    manifest = [dict(relative="raw.bin", size=identity["size"], sha256=identity["sha256"])]
                    group.collect(roots[node], manifest, 10)
                finally:
                    group.close()

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(participant, node) for node in range(8)]
                for future in futures:
                    future.result(timeout=30)
            for node in range(1, 8):
                collected = roots[0] / f"peer_node{node}"
                shared.verify_collection(collected)
                self.assertEqual((collected / "raw.bin").read_bytes(), (roots[node] / "raw.bin").read_bytes())


if __name__ == "__main__":
    unittest.main()
