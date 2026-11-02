"""V1 benchmark public call/event contracts and honest fused timer accounting."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks/clos"))
import bench_v1_bandwidth as v1
import bench_v2_bandwidth as v2
from bandwidth_multinode import validate_peers
from stage_metrics import byte_ledger, parse_profile, route_counts, summarize
import test_stage_metrics as fixtures


class V1BandwidthTest(unittest.TestCase):
    def test_same_matrix_and_ep16_entrypoint(self):
        first, second = v1.arguments([]), v2.arguments([])
        for key in (
            "ep_size",
            "aiv",
            "dtypes",
            "expand",
            "num_tokens",
            "hidden",
            "num_topk",
            "experts_per_rank",
            "iterations",
            "warmup",
            "udma_auto_policy",
        ):
            self.assertEqual(getattr(first, key), getattr(second, key))
        case = dict(name="bf16_expand0_aiv32", dtype="bf16", expand=0, aiv=32)
        for node in (0, 1):
            args = v1.arguments(
                [
                    "--ep-size",
                    "16",
                    "--nnodes",
                    "2",
                    "--node-rank",
                    str(node),
                    "--master-addr",
                    "192.0.2.1",
                    "--run-id",
                    "v1_ep16",
                ]
            )
            command = v2.worker_command(args, case, Path("out"), entrypoint=v1.__file__)
            self.assertIn(str(Path(v1.__file__).resolve()), command)
            self.assertNotIn(str(Path(v2.__file__).resolve()), command)
            self.assertIn("--nproc-per-node=8", command)
            self.assertIn("--master-port=29501", command)
            replay = v1.arguments(command[command.index("--worker") :])
            self.assertEqual(replay.api, "v1")
            self.assertEqual(replay.node_rank, node)

    def test_single_node_ep16_launch_and_idle_gate(self):
        case = dict(name="bf16_expand0_aiv32", dtype="bf16", expand=0, aiv=32)
        for module in (v1, v2):
            args = module.arguments(["--ep-size", "16"])
            command = v2.worker_command(args, case, Path("out"), entrypoint=module.__file__)
            self.assertIn("--standalone", command)
            self.assertIn("--nproc-per-node=16", command)
            self.assertNotIn("--nnodes=2", command)
            replay = module.arguments(command[command.index("--worker") :])
            self.assertEqual((replay.ep_size, replay.nnodes, replay.node_rank), (16, 1, 0))
            self.assertEqual(v2.local_world_size(replay), 16)
        rows = [f"| {i} | Ascend950 | OK |\nNo running processes found in NPU {i}" for i in range(16)]
        v2.check_idle("\n".join(rows), 0, 16)
        for text in (
            "\n".join(rows[:8]),
            "\n".join(rows).replace("NPU 15", "NPU missing"),
            "\n".join(rows).replace("| 15 | Ascend950 | OK", "| 15 | Ascend950 | Alarm"),
        ):
            with self.assertRaises(RuntimeError):
                v2.check_idle(text, 0, 16)

    def test_mixed_v1_v2_nodes_rejected_before_launch(self):
        hello = dict(node_rank=0, contract=v2.suite_contract(v1.arguments([])), source={}, topk_idx_bytes=8)
        other = dict(hello, node_rank=1, contract=v2.suite_contract(v2.arguments([])))
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            validate_peers(hello, other)

    def operations(self, expanded):
        buffer = Mock()
        buffer.get_dispatch_layout.return_value = ("counts", None, "experts", "membership", "layout_event")
        buffer.dispatch.return_value = ("recv", "recv_idx", "recv_weights", [3, 4], "handle", "dispatch_event")
        buffer.expand.return_value = ("expanded", "expanded_weights", "expand_handle", "expand_event")
        buffer.aggregate.return_value = ("aggregated", "aggregated_values", "aggregate_event")
        buffer.combine.return_value = ("combined", "weights_out", "combine_event")
        return v1.V1Operations(buffer, "config", "idx", "weights", 512, expanded), buffer

    def test_fresh_includes_layout_and_explicit_event_chain(self):
        op, buffer = self.operations(True)
        result = op.dispatch("input")
        self.assertEqual([c[0] for c in buffer.mock_calls], ["get_dispatch_layout", "dispatch", "expand"])
        self.assertEqual(buffer.dispatch.call_args.kwargs["previous_event"], "layout_event")
        self.assertEqual(buffer.expand.call_args.kwargs["previous_event"], "dispatch_event")
        self.assertEqual(buffer.expand.call_args.kwargs["expert_alignment"], 1)
        self.assertIs(buffer.dispatch.call_args.kwargs["zero_copy"], False)
        self.assertIs(buffer.dispatch.call_args.kwargs["allocate_on_comm_stream"], False)
        self.assertEqual(result[-1], "expand_event")
        self.assertEqual(result[0], buffer.dispatch.return_value)
        self.assertEqual(result[2], buffer.get_dispatch_layout.return_value)

    def test_cached_reuses_both_handles_and_omits_topk_layout(self):
        op, buffer = self.operations(True)
        result = op.dispatch("changed_payload", ("saved_dispatch", "saved_expand"))
        buffer.get_dispatch_layout.assert_not_called()
        for fn in (buffer.dispatch, buffer.expand):
            self.assertNotIn("topk_idx", fn.call_args.kwargs)
            self.assertNotIn("recv_topk_idx", fn.call_args.kwargs)
            self.assertNotIn("topk_weights", fn.call_args.kwargs)
        self.assertEqual(buffer.dispatch.call_args.kwargs["handle"], "saved_dispatch")
        self.assertEqual(buffer.expand.call_args.kwargs["handle"], "saved_expand")
        self.assertEqual(result[-1], "expand_event")

    def test_expanded_weights_are_values_not_payload_multipliers(self):
        op, buffer = self.operations(True)
        result = op.combine("expert_input", ("dh", "eh"), "expert_weights", ("bias0", "bias1"))
        aggregate = buffer.aggregate.call_args
        self.assertEqual(aggregate.kwargs["expanded_values"], "expert_weights")
        self.assertNotIn("topk_weights", aggregate.kwargs)
        combine = buffer.combine.call_args
        self.assertEqual(combine.args, ("aggregated", "dh"))
        self.assertEqual(combine.kwargs["topk_weights"], "aggregated_values")
        self.assertEqual(combine.kwargs["previous_event"], "aggregate_event")
        self.assertEqual(combine.kwargs["bias"], ("bias0", "bias1"))
        self.assertEqual(result[0], buffer.aggregate.return_value)
        self.assertEqual(result[-1], "combine_event")

    def test_unexpanded_calls_public_combine_without_extra_kernels(self):
        op, buffer = self.operations(False)
        op.dispatch("input")
        buffer.expand.assert_not_called()
        result = op.combine("rank_rows", ("dh", None))
        buffer.aggregate.assert_not_called()
        self.assertEqual(buffer.combine.call_args.args, ("rank_rows", "dh"))
        self.assertIsNone(buffer.combine.call_args.kwargs["topk_weights"])
        self.assertIsNone(buffer.combine.call_args.kwargs["bias"])
        self.assertEqual(result[-1], "combine_event")

    def profile(self, path, expanded=False, sequence=None):
        sequence = sequence or (["aggregate_kernel"] if expanded else []) + [
            "cached_notify_combine_kernel",
            "v1_combine_udma_kernel",
        ]
        fixtures.StageMetricsTest().make_profile(path, sequence)
        return parse_profile(path, "combine_unweighted", expanded, 2, 0, 32, api="v1")

    def test_fused_timer_is_not_pure_comm_or_zero_reduce(self):
        for expanded in (False, True):
            with self.subTest(expanded=expanded), tempfile.TemporaryDirectory() as tmp:
                result = self.profile(Path(tmp), expanded)
                sample = result["samples"][0]
                for key in ("comm_us", "reduce_us", "combine_stage_us", "combine_stage_span_us", "late_comm_us"):
                    self.assertIsNone(sample[key])
                self.assertEqual(sample["fused_combine_us"], "2.125")
                self.assertEqual(sample["combine_with_reduce_us"], "4.250" if expanded else "2.125")
                self.assertEqual(result["unseparated_stages"], ["comm", "reduce"])
                self.assertNotIn("reduce", result["absent_stages"])
                self.assertEqual(len(result["kernels"]), 6 if expanded else 4)

    def test_fused_summary_global_denominator_and_null_bandwidth(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = self.profile(Path(tmp), True)
            counts = route_counts([[[0, 1]], [[0, 1]]], 1)
            records = [
                dict(
                    rank=r,
                    operators={
                        "combine_unweighted": dict(
                            profile=profile,
                            caller_us=[10 + r, 20 + r],
                            bytes=byte_ledger(
                                counts,
                                r,
                                tokens=1,
                                hidden=128,
                                topk=2,
                                dtype="bf16",
                                idx_bytes=8,
                                op="combine_unweighted",
                            ),
                        )
                    },
                )
                for r in range(2)
            ]
            rows = summarize(records)
            for row in rows:
                for key in (
                    "official_logical_GBps",
                    "remote_payload_GBps",
                    "remote_application_GBps",
                    "official_logical_combine_stage_GBps",
                ):
                    self.assertIsNone(row[key])
            self.assertAlmostEqual(rows[-1]["logical_fused_combine_GBps"], 1024 / 2.125 / 1000)
            self.assertAlmostEqual(rows[-1]["logical_combine_with_reduce_GBps"], 1024 / 4.250 / 1000)
            self.assertEqual(rows[-1]["device_sum_us"], 6.375)

    def test_missing_or_mixed_fused_kernels_fail_closed(self):
        bad_sequences = [
            ["cached_notify_combine_kernel"],
            ["cached_notify_combine_kernel", "v1_combine_udma_comm_kernel", "v1_combine_udma_reduce_kernel"],
            ["cached_notify_combine_kernel", "v1_combine_udma_kernel", "v1_combine_udma_reduce_kernel"],
            ["v1_combine_udma_kernel", "cached_notify_combine_kernel"],
            ["cached_notify_combine_kernel", "v1_combine_udma_kernel", "v1_combine_udma_kernel"],
        ]
        for sequence in bad_sequences:
            with self.subTest(sequence=sequence), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    self.profile(Path(tmp), sequence=sequence)

    def test_fresh_layout_is_required_and_cached_layout_forbidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq = ["dispatch_layout_kernel", "notify_dispatch_kernel", "v1_dispatch_udma_kernel", "expand_kernel"]
            fixtures.StageMetricsTest().make_profile(root, seq)
            profile = parse_profile(root, "dispatch", True, 2, 0, 32, api="v1")
            self.assertEqual(profile["samples"][0]["device_sum_us"], "8.500")
            with self.assertRaises(ValueError):
                parse_profile(root, "cached-dispatch", True, 2, 0, 32, api="v1")
            fixtures.StageMetricsTest().make_profile(root, seq[1:])
            with self.assertRaises(ValueError):
                parse_profile(root, "dispatch", True, 2, 0, 32, api="v1")


if __name__ == "__main__":
    unittest.main()
