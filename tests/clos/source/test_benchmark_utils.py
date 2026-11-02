# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU-only checks for benchmark statistics and route generation."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "benchmarks/clos"
sys.path.insert(0, str(BENCHMARK_DIR))

from bench_utils import (  # noqa: E402
    L2_CACHE_FLUSH_BYTES,
    benchmark_operator,
    destination_count,
    dispatch_layout_reference,
    flush_l2_cache,
    make_topk_idx,
    percentile,
    summarize_times,
)
from bench_ep import parse_args  # noqa: E402


class BenchmarkUtilsTest(unittest.TestCase):
    def test_multinode_udma_benchmark_arguments(self):
        argv = [
            "bench_ep.py",
            "--num-processes",
            "4",
            "--num-experts",
            "256",
            "--transport",
            "udma",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "1"}, clear=False),
        ):
            args = parse_args()
        self.assertEqual(args.ep_size, 8)
        self.assertEqual(args.num_local_ranks, 4)
        self.assertEqual(args.node_rank, 1)
        self.assertEqual(args.num_experts, 256)
        self.assertEqual(args.transport, "udma")

    def test_explicit_multinode_benchmark_arguments(self):
        argv = [
            "bench_ep.py",
            "--ep-size",
            "64",
            "--nproc-per-node",
            "8",
            "--node-rank",
            "7",
            "--master-addr",
            "10.0.0.1",
            "--master-port",
            "8899",
            "--num-experts",
            "384",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, {}, clear=True):
            args = parse_args()
        self.assertEqual(args.ep_size, 64)
        self.assertEqual(args.num_local_ranks, 8)
        self.assertEqual(args.node_rank, 7)
        self.assertEqual(args.master_addr, "10.0.0.1")
        self.assertEqual(args.master_port, 8899)
        self.assertEqual(args.shmem_ip_port, "tcp://10.0.0.1:8900")

    def test_single_node_ep_size_remains_compatible(self):
        argv = ["bench_ep.py", "--ep-size", "8", "--num-experts", "256"]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, {}, clear=True):
            args = parse_args()
        self.assertEqual(args.ep_size, 8)
        self.assertEqual(args.num_local_ranks, 8)
        self.assertEqual(args.node_rank, 0)

    def test_endpoint_environment_is_resolved_by_benchmark_cli(self):
        # Environment configuration remains an explicit choice of the caller;
        # Buffer itself no longer reads this variable.
        environment = {"ASCEND_DEEPEP_SHMEM_IP_PORT": "tcp://127.0.0.1:30000"}
        for extra, expected in (
            ([], environment["ASCEND_DEEPEP_SHMEM_IP_PORT"]),
            (["--shmem-ip-port", "tcp://127.0.0.1:30001"], "tcp://127.0.0.1:30001"),
        ):
            argv = ["bench_ep.py", "--ep-size", "8", "--num-experts", "256", *extra]
            with (
                self.subTest(extra=extra),
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(os.environ, environment, clear=True),
            ):
                args = parse_args()
                self.assertEqual(args.shmem_ip_port, expected)
                self.assertEqual(os.environ["ASCEND_DEEPEP_SHMEM_IP_PORT"], environment["ASCEND_DEEPEP_SHMEM_IP_PORT"])

    def test_num_experts_must_divide_global_ep_size(self):
        argv = [
            "bench_ep.py",
            "--num-processes",
            "4",
            "--num-experts",
            "12",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0"}, clear=False),
            mock.patch.object(sys, "stderr"),
            self.assertRaises(SystemExit),
        ):
            parse_args()

    def test_l2_cache_flush_configuration(self):
        self.assertEqual(L2_CACHE_FLUSH_BYTES, 256_000_000)
        flush_l2_cache(None, None)

    def test_percentile_and_summary(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        result = summarize_times(
            "dispatch",
            [1.0, 2.0, 3.0],
            num_ranks=2,
            num_tokens_per_rank=100,
            logical_bytes_per_rank=1000,
        )
        self.assertEqual(result["mean_us"], 2000.0)
        self.assertEqual(result["aggregate_tokens_per_second"], 100000.0)
        self.assertEqual(result["per_rank_logical_payload_gbps"], 0.0005)
        self.assertNotIn("logical_payload_gbps", result)

    def test_balanced_routes_match_layout_reference(self):
        routes = make_topk_idx(0, 4, 16, 2, 16, "balanced")
        rank_counts, expert_counts, rank_mask = dispatch_layout_reference(routes, 4, 16)
        self.assertEqual(routes.dtype, torch.int32)
        self.assertEqual(rank_counts.tolist(), [8, 8, 8, 8])
        self.assertEqual(int(expert_counts.sum()), 32)
        self.assertTrue(torch.all(rank_mask.sum(dim=1) == 2))
        self.assertEqual(destination_count(4, 2, "balanced"), 2)
        self.assertEqual(destination_count(4, 2, "local"), 1)

    def test_random_routes_match_deepep_shape_and_are_repeatable(self):
        first = make_topk_idx(1, 4, 16, 8, 256, "random", seed=2026)
        second = make_topk_idx(1, 4, 16, 8, 256, "random", seed=2026)
        self.assertEqual(tuple(first.shape), (16, 8))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.all(first >= 0))
        self.assertTrue(torch.all(first < 256))
        self.assertTrue(all(row.unique().numel() == 8 for row in first))

    def test_benchmark_operator_drops_first_measured_sample(self):
        class FakeEvent:
            next_index = 0

            def __init__(self, enable_timing=False):
                self.index = FakeEvent.next_index
                FakeEvent.next_index += 1

            def record(self, stream=None):
                return None

            def elapsed_time(self, end):
                return [99.0, 2.0, 4.0][self.index]

        class FakeNpu:
            Event = FakeEvent

            @staticmethod
            def synchronize():
                return None

        class FakeBuffer:
            group_size = 2

            @staticmethod
            def get_comm_stream():
                return object()

            @staticmethod
            def barrier():
                raise AssertionError("DeepEP sampling must not add barriers")

        calls = []
        with mock.patch.object(torch, "npu", FakeNpu(), create=True):
            result = benchmark_operator(
                "dispatch",
                lambda: calls.append(None),
                buffer=FakeBuffer(),
                num_warmups=1,
                num_iterations=3,
                num_tokens_per_rank=10,
                flush_l2=False,
            )
        self.assertEqual(len(calls), 4)
        self.assertEqual(result["iterations"], 2)
        self.assertEqual(result["mean_us"], 3000.0)


if __name__ == "__main__":
    unittest.main()
