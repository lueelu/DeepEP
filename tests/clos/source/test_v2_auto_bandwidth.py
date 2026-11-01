"""Automatic benchmark launch/summary contracts, without NPU execution."""

import contextlib
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks/clos"))
import bench_v2_auto_bandwidth as auto
import bench_v2_bandwidth as v2


class AutoBandwidthTest(unittest.TestCase):
    def test_defaults_and_worker_replay(self):
        case = dict(name="bf16_expand0_aiv32", dtype="bf16", expand=0, aiv=32)
        for ep in (4, 8, 16, 32):
            for node in range(2 if ep == 32 else 1):
                flags = ["--ep-size", str(ep), "--node-rank", str(node)]
                if ep == 32:
                    flags += ["--master-addr", "192.0.2.1", "--run-id", "auto_test"]
                args = auto.arguments(flags)
                self.assertEqual((args.aiv, args.dtypes, args.expand), ([32], ["bf16"], [0]))
                self.assertEqual(args.route_mode, "uniform")
                self.assertEqual(args.nnodes, 2 if ep == 32 else 1)
                self.assertEqual(v2.v2_tuning_options(args), {"udma_auto_policy": True})
                command = v2.worker_command(args, case, Path("out"), entrypoint=auto.__file__)
                self.assertIn(f"--nproc-per-node={min(ep, 16)}", command)
                self.assertIn(str(Path(auto.__file__).resolve()), command)
                self.assertNotIn("--peers-per-worker", command)
                self.assertNotIn("--workers-per-peer", command)
                replay = auto.arguments(command[command.index("--worker") :])
                self.assertEqual(v2.suite_contract(args), v2.suite_contract(replay))
                self.assertTrue(replay.udma_auto_policy)

    def test_user_core_count_and_rejected_manual_tuning(self):
        self.assertEqual(auto.arguments(["--aiv", "20"]).aiv, [20])
        for flags in (
            ["--ep-size", "2"],
            ["--ep-size", "32"],
            ["--aiv", "21"],
            ["--workers-per-peer", "1"],
            ["--wqes-per-visit", "8"],
            ["--udma-schedule", "token"],
        ):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                auto.arguments(flags)
        manual = v2.arguments([])
        self.assertFalse(manual.udma_auto_policy)
        self.assertEqual(
            v2.v2_tuning_options(manual),
            dict(
                udma_auto_policy=False,
                udma_peers_per_worker=1,
                udma_workers_per_peer=1,
                udma_qps_per_worker=1,
                udma_wqes_per_visit=8,
            ),
        )

    def test_summary_keeps_notify_reduce_outside_bandwidth(self):
        row = dict(
            rank="all",
            case="bf16_expand0_aiv32",
            op="combine_unweighted",
            official_logical_GBps=3200,
            comm_us=10,
            notify_us=3,
            reduce_us=7,
            layout_us=0,
            aggregate_us=0,
            expand_us=0,
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            v2.print_auto_summary([dict(row, rank=0), row], 8, {row["case"]: {"version": "clos-v1"}})
        self.assertIn("combine_unweighted / 400.000 / 10.000 / 3.000 / 7.000", output.getvalue())
        self.assertEqual(output.getvalue().count("combine_unweighted"), 1)


if __name__ == "__main__":
    unittest.main()
