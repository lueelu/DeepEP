"""CPU tests for the duplicate-server golden notify metadata builder."""

import unittest

import torch

from deep_ep.backends.clos.dup_server_golden import (
    DupServerDispatchMeta,
    build_dup_server_golden_notify,
    build_dup_server_golden_source_notify,
)


def _segment(
    meta: DupServerDispatchMeta,
    local_expert: int,
    src_rank: int,
) -> torch.Tensor:
    segment = local_expert * meta.world_size + src_rank
    begin = int(meta.recv_segment_offset[segment].item())
    end = int(meta.recv_segment_offset[segment + 1].item())
    return meta.recv_route_position[begin:end]


def _gateway_segment(
    meta: DupServerDispatchMeta,
    local_expert: int,
    src_rank: int,
) -> torch.Tensor:
    segment = local_expert * meta.world_size + src_rank
    begin = int(meta.recv_segment_offset[segment].item())
    end = int(meta.recv_segment_offset[segment + 1].item())
    return meta.recv_gateway_lane[begin:end]


class DupServerGoldenNotifyTest(unittest.TestCase):
    def test_source_prepare_outputs_are_stable_and_communication_ready(self):
        topk_idx = torch.tensor(
            [
                [0, 1, 16, 17],
                [2, 18, -1, -1],
                [0, 8, 9, -1],
                [-1, -1, -1, -1],
            ],
            dtype=torch.int64,
        )
        topk_weights = torch.arange(16, dtype=torch.float32).reshape(4, 4)

        meta = build_dup_server_golden_source_notify(
            topk_idx,
            num_experts=32,
            topk_weights=topk_weights,
            world_size=16,
            num_max_tokens_per_rank=6,
        )

        self.assertEqual(meta.send_token_count.tolist(), [3, 2])
        self.assertEqual(meta.send_token_ids[0].tolist(), [0, 1, 2, -1, -1, -1])
        self.assertEqual(meta.send_token_ids[1].tolist(), [0, 1, -1, -1, -1, -1])
        self.assertEqual(meta.send_gateway_lane[0].tolist(), [0, 0, 0, 255, 255, 255])
        self.assertEqual(meta.send_route_count.tolist(), [6, 3])
        self.assertEqual(meta.packed_route_position[:9].tolist(), [0, 8, 1, 4, 9, 10, 2, 3, 5])
        self.assertEqual(meta.packed_gateway_lane[:9].tolist(), [0] * 9)
        self.assertEqual(meta.source_server_mask.tolist(), [3, 3, 1, 0])
        self.assertEqual(meta.send_expert_prefix[:, 0].tolist(), [0, 0])
        self.assertEqual(meta.send_expert_prefix[:, -1].tolist(), [6, 3])
        self.assertTrue(
            torch.equal(
                meta.packed_topk_weights[0, :6],
                topk_weights.reshape(-1)[torch.tensor([0, 8, 1, 4, 9, 10])],
            )
        )

    def test_first_hit_is_rejected(self):
        routes = torch.full((16, 1, 4), -1, dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "gateway_policy"):
            build_dup_server_golden_notify(
                routes,
                rank=0,
                num_experts=32,
                gateway_policy="first_hit",
            )
        with self.assertRaisesRegex(ValueError, "gateway_policy"):
            build_dup_server_golden_source_notify(
                routes[0],
                num_experts=32,
                world_size=16,
                gateway_policy="first_hit",
            )

    def test_hand_written_server_dedup_and_destination_segments(self):
        world_size = 16
        num_tokens = 4
        num_topk = 4
        num_experts = 32
        routes = torch.full(
            (world_size, num_tokens, num_topk),
            -1,
            dtype=torch.int64,
        )
        routes[0] = torch.tensor(
            [
                [0, 1, 16, 17],
                [2, 18, -1, -1],
                [0, 8, 9, -1],
                [-1, -1, -1, -1],
            ],
            dtype=torch.int64,
        )
        routes[1, 0] = torch.tensor([1, 0, -1, -1])
        routes[1, 1] = torch.tensor([0, -1, -1, -1])
        routes[8, 0] = torch.tensor([0, 1, -1, -1])
        routes[8, 2] = torch.tensor([1, -1, -1, -1])

        meta = build_dup_server_golden_notify(
            routes,
            rank=0,
            num_experts=num_experts,
            num_max_tokens_per_rank=6,
            num_tokens_per_chunk=2,
        )

        self.assertEqual(meta.send_token_count.tolist(), [3, 2])
        self.assertEqual(meta.send_token_ids[0].tolist(), [0, 1, 2, -1, -1, -1])
        self.assertEqual(meta.send_token_ids[1].tolist(), [0, 1, -1, -1, -1, -1])
        self.assertEqual(meta.num_recv_tokens, 9)
        self.assertEqual(meta.num_recv_tokens_per_expert, [5, 4])
        self.assertEqual(meta.recv_expert_offset.tolist(), [0, 5, 9])
        self.assertEqual(meta.recv_segment_offset.numel(), 2 * world_size + 1)

        self.assertEqual(_segment(meta, 0, 0).tolist(), [0, 8])
        self.assertEqual(_segment(meta, 0, 1).tolist(), [1, 4])
        self.assertEqual(_segment(meta, 0, 8).tolist(), [0])
        self.assertEqual(_segment(meta, 1, 0).tolist(), [1])
        self.assertEqual(_segment(meta, 1, 1).tolist(), [0])
        self.assertEqual(_segment(meta, 1, 8).tolist(), [1, 8])
        self.assertEqual(_segment(meta, 0, 15).numel(), 0)
        self.assertEqual(meta.recv_src_rank.tolist(), [0, 0, 1, 1, 8, 0, 1, 8, 8])
        self.assertEqual(tuple(meta.recv_chunk_token_end.shape), (world_size, 3))
        self.assertEqual(meta.recv_chunk_token_end[0].tolist(), [2, 4, 4])
        self.assertEqual(meta.recv_chunk_token_end[1].tolist(), [2, 4, 4])
        self.assertEqual(meta.recv_chunk_token_end[8].tolist(), [3, 4, 4])
        self.assertEqual(meta.recv_chunk_token_end[15].tolist(), [4, 4, 4])

        copied = meta.to("cpu")
        self.assertTrue(torch.equal(copied.send_token_ids, meta.send_token_ids))
        self.assertTrue(torch.equal(copied.recv_route_position, meta.recv_route_position))
        self.assertTrue(torch.equal(copied.recv_src_rank, meta.recv_src_rank))
        self.assertTrue(torch.equal(copied.recv_chunk_token_end, meta.recv_chunk_token_end))
        self.assertTrue(torch.equal(copied.recv_expert_offset, meta.recv_expert_offset))

    def test_every_valid_route_and_server_membership_is_preserved(self):
        world_size = 16
        num_tokens = 11
        num_topk = 4
        num_experts = 64
        ranks_per_server = 8
        routes = torch.empty((world_size, num_tokens, num_topk), dtype=torch.int64)
        for src_rank in range(world_size):
            for token_id in range(num_tokens):
                for topk_slot in range(num_topk):
                    if (src_rank + token_id + topk_slot) % 9 == 0:
                        expert = -1
                    else:
                        expert = (src_rank * 7 + token_id * 5 + topk_slot * 13) % num_experts
                    routes[src_rank, token_id, topk_slot] = expert

        recovered_routes = []
        experts_per_rank = num_experts // world_size
        experts_per_server = experts_per_rank * ranks_per_server
        for rank in range(world_size):
            meta = build_dup_server_golden_notify(
                routes,
                rank=rank,
                num_experts=num_experts,
                num_max_tokens_per_rank=num_tokens + 3,
                ranks_per_server=ranks_per_server,
            )
            self.assertEqual(meta.send_token_count.dtype, torch.int32)
            self.assertEqual(meta.send_token_ids.dtype, torch.int32)
            self.assertEqual(meta.send_gateway_lane.dtype, torch.uint8)
            self.assertEqual(meta.recv_route_position.dtype, torch.int32)
            self.assertEqual(meta.recv_gateway_lane.dtype, torch.uint8)
            self.assertEqual(meta.recv_src_rank.dtype, torch.int32)
            self.assertEqual(meta.recv_segment_offset.dtype, torch.int32)
            self.assertEqual(meta.recv_chunk_token_end.dtype, torch.int32)
            self.assertEqual(meta.recv_expert_offset.dtype, torch.int32)
            self.assertEqual(meta.recv_src_rank.numel(), meta.num_recv_tokens)
            self.assertEqual(meta.recv_gateway_lane.numel(), meta.num_recv_tokens)
            self.assertTrue(bool(torch.all(meta.recv_segment_offset[1:] >= meta.recv_segment_offset[:-1])))

            expected_server_tokens = []
            for dst_server in range(meta.num_servers):
                token_ids = []
                for token_id in range(num_tokens):
                    experts = routes[rank, token_id]
                    belongs = experts.ge(dst_server * experts_per_server) & experts.lt(
                        (dst_server + 1) * experts_per_server
                    )
                    if bool(belongs.any()):
                        token_ids.append(token_id)
                expected_server_tokens.append(token_ids)

            for dst_server, expected in enumerate(expected_server_tokens):
                count = int(meta.send_token_count[dst_server].item())
                self.assertEqual(
                    meta.send_token_ids[dst_server, :count].tolist(),
                    expected,
                )
                self.assertTrue(bool(meta.send_token_ids[dst_server, count:].eq(-1).all()))

            for local_expert in range(meta.num_local_experts):
                global_expert = rank * experts_per_rank + local_expert
                expected_expert_count = 0
                for src_rank in range(world_size):
                    expected_positions = (
                        torch.nonzero(
                            routes[src_rank].reshape(-1).eq(global_expert),
                            as_tuple=False,
                        )
                        .flatten()
                        .to(torch.int32)
                    )
                    actual_positions = _segment(meta, local_expert, src_rank)
                    self.assertTrue(torch.equal(actual_positions, expected_positions))
                    expected_expert_count += int(expected_positions.numel())
                    for route_position in actual_positions.tolist():
                        recovered_routes.append(
                            (
                                src_rank,
                                route_position // num_topk,
                                route_position % num_topk,
                                global_expert,
                            )
                        )
                self.assertEqual(
                    meta.num_recv_tokens_per_expert[local_expert],
                    expected_expert_count,
                )

        expected_routes = []
        for src_rank in range(world_size):
            for token_id in range(num_tokens):
                for topk_slot in range(num_topk):
                    expert = int(routes[src_rank, token_id, topk_slot].item())
                    if expert >= 0:
                        expected_routes.append((src_rank, token_id, topk_slot, expert))
        self.assertCountEqual(recovered_routes, expected_routes)

    def test_rejects_invalid_layouts_and_routes(self):
        routes = torch.zeros((16, 3, 2), dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "rank must be"):
            build_dup_server_golden_notify(routes, 16, 32)
        with self.assertRaisesRegex(ValueError, "no smaller than N"):
            build_dup_server_golden_notify(
                routes,
                0,
                32,
                num_max_tokens_per_rank=2,
            )
        with self.assertRaisesRegex(ValueError, "divisible by ranks_per_server"):
            build_dup_server_golden_notify(
                torch.zeros((15, 3, 2), dtype=torch.int64),
                0,
                30,
            )
        with self.assertRaisesRegex(ValueError, "num_tokens_per_chunk"):
            build_dup_server_golden_notify(
                routes,
                0,
                32,
                num_tokens_per_chunk=0,
            )
        with self.assertRaisesRegex(ValueError, "gateway_policy"):
            build_dup_server_golden_notify(
                routes,
                0,
                32,
                gateway_policy="unknown",
            )

        routes[4, 1, 1] = 32
        with self.assertRaisesRegex(ValueError, "invalid expert ID"):
            build_dup_server_golden_notify(routes, 0, 32)


if __name__ == "__main__":
    unittest.main()


# python -m unittest test.test_dup_server_golden_notify
