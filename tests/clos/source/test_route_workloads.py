"""CPU checks for input routes and workload isolation; no torch/NPU required."""

from pathlib import Path
import ast
import random
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks/clos"))
import route_workloads as routing
import bench_v1_bandwidth as v1
import bench_v2_bandwidth as v2


class RouteWorkloadsTest(unittest.TestCase):
    def test_actual_v1_v2_route_adapters_share_routes_and_ignore_expand(self):
        # Execute the production adapters without importing torch_npu. Tensor
        # storage is the only stub; route generation and caching are real.
        class Tensor:
            def __init__(self, rows):
                self.rows = rows

            def reshape(self, *shape):
                return self

            def tolist(self):
                return self.rows

            def __getitem__(self, index):
                return Tensor(self.rows[index])

        def load(filename, names):
            tree = ast.parse((Path(__file__).parent / filename).read_text(encoding="utf8"))
            nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
            scope = dict(
                torch=SimpleNamespace(Tensor=Tensor, tensor=lambda rows, dtype: Tensor(rows)),
                topk_idx_t="int64",
                route_options=routing.options,
                generate_routes=routing.generate,
            )
            exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), scope)
            return scope

        first = load("test_v1_intranode.py", {"_make_topk_idx"})
        second = load("test_v2_elastic_v1_kernel.py", {"_topk", "_routes", "_rank_num_tokens"})
        args = SimpleNamespace(
            ep_size=8, experts_per_rank=32, num_experts=256, num_tokens=129, num_topk=8, seed=71, uneven_tokens=False
        )
        for mode in ("uniform", "rank_hotspot", "local_group"):
            args.route_mode, args.route_group_size = mode, 4
            for rank in (0, 7):
                expected = first["_make_topk_idx"](rank, args).tolist()
                for expand in ("bf16", "expand", "fp8"):
                    actual = second["_topk"](rank, 129, 8, 256, 8, expand, args).tolist()
                    self.assertEqual(actual, expected)
                    self.assertEqual(second["_routes"](rank, 17, 8, 256, 8, expand, args), expected[17])
        args.route_mask_rate = 1.0
        self.assertEqual(first["_make_topk_idx"](0, args).tolist(), [[-1] * 8] * 129)

    def test_uniform_not_forced_and_independent_random_state(self):
        random.seed(53)
        state = random.getstate()
        rows = routing.generate(0, 2048, 8, 32, 8)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(rows, routing.generate(0, 2048, 8, 32, 8))
        self.assertNotEqual(rows, routing.generate(1, 2048, 8, 32, 8))
        self.assertNotEqual(rows, routing.generate(0, 2048, 8, 32, 8, seed=7))
        self.assertTrue(all(len(set(row)) == 8 and min(row) >= 0 for row in rows))
        self.assertGreater(sum(row[0] // 32 != row[1] // 32 for row in rows), 1500)
        counts = routing.statistics([rows] * 8, 32)
        self.assertLess(counts["recv_max_over_mean"], 1.08)
        self.assertGreater(len(counts["contribution_histogram"]), 2)

    def test_score_sampler_matches_dense_uniform_score_distribution(self):
        sparse_rng, dense_rng = random.Random(11), random.Random(12)
        sparse = [routing._sample(sparse_rng, 32, 6, 8, 1.5) for _ in range(5000)]
        dense = [
            sorted(range(32), key=lambda e: dense_rng.random() * (1.5 if e < 8 else 1), reverse=True)[:6]
            for _ in range(5000)
        ]
        for rows in (sparse, dense):
            self.assertTrue(all(len(set(row)) == 6 for row in rows))
        self.assertAlmostEqual(
            sum(e < 8 for row in sparse for e in row) / len(sparse),
            sum(e < 8 for row in dense for e in row) / len(dense),
            delta=0.08,
        )

    def test_rank_hotspot_actual_deduplicated_ratio(self):
        for target in (2.0, 4.0):
            rows = routing.generate(1, 8192, 8, 32, 8, route_mode="rank_hotspot", route_skew=target, route_hotspot=3)
            stats = routing.statistics([rows] * 8, 32)
            self.assertAlmostEqual(stats["rank_over_other_mean"][3], target, delta=0.3)
            self.assertGreater(stats["contribution_histogram"].get("2", 0), 0)

    def test_source_local_groups(self):
        rows = [
            routing.generate(r, 2048, 8, 8, 8, route_mode="local_group", route_group_size=4, route_local_fraction=0.8)
            for r in range(8)
        ]
        self.assertAlmostEqual(routing.statistics(rows, 8, 4)["local_group_row_fraction"], 0.8, delta=0.025)

    def test_mask_and_full_topk(self):
        full = routing.generate(0, 20, 2, 3, 6)
        self.assertTrue(all(sorted(row) == list(range(6)) for row in full))
        empty = routing.generate(0, 20, 2, 3, 6, route_mask_rate=1)
        stats = routing.statistics([empty, empty], 3)
        self.assertEqual(stats["valid_expert_hits"], 0)
        self.assertIsNone(stats["recv_max_over_mean"])
        self.assertIsNone(stats["single_contribution_fraction"])
        with self.assertRaisesRegex(ValueError, "unreachable"):
            routing.generate(0, 20, 2, 3, 6, route_mode="rank_hotspot", route_skew=4)

    def test_independent_statistics(self):
        routes = [[[0, 1, 4], [4, -1, -1]], [[2, 3, 4]]]
        stats = routing.statistics(routes, 4)
        self.assertEqual(stats["rows"], [[1, 2], [1, 1]])
        self.assertEqual(stats["recv_rows"], [2, 3])
        self.assertEqual(stats["contribution_histogram"], {"2": 2, "1": 3})
        self.assertEqual(stats["fanout_histogram"], {"2": 2, "1": 1})
        self.assertEqual(stats["valid_expert_hits"], 7)
        with self.assertRaises(ValueError):
            routing.statistics([[[0, 0]]], 4)

    def test_cli_roundtrip_for_both_apis(self):
        for module in (v1, v2):
            args = module.arguments(
                [
                    "--route-mode",
                    "local_group",
                    "--route-group-size",
                    "4",
                    "--route-skew",
                    "4",
                    "--route-hotspot",
                    "1",
                    "--route-mask-rate",
                    "0.1",
                ]
            )
            case = dict(name="test", aiv=64, dtype="bf16", expand=1)
            command = v2.worker_command(args, case, Path("out"), entrypoint=module.__file__)
            replay = module.arguments(command[command.index("--worker") :])
            self.assertEqual(routing.options(args), routing.options(replay))
            for flags in (
                ["--route-mode", "group_hotspot"],
                ["--route-group-size", "3"],
                ["--route-mask-rate", "nan"],
                ["--route-skew", "inf"],
            ):
                with self.assertRaises(SystemExit):
                    module.arguments(flags)


if __name__ == "__main__":
    unittest.main()
