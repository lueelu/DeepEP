# Copyright (c) 2026, Lu Lu

"""CPU specification tests for the two-host MegaMoE wire contract.

The model represents messages and dependency events, not device instructions,
cache coherence, SHMEM completion, or NPU scheduling. It compares hierarchical
transport with direct routing and uses the existing BF16 oracle for arithmetic.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
import random
import re

import numpy as np
import pytest

from deep_ep_experimental.megamoe.reference import bf16, quantize_input


WORLD = 32
HOST_SIZE = 16
EXPERTS_PER_RANK = 32


def host(rank):
    return rank // HOST_SIZE


def pair(rank):
    return (1 - host(rank)) * HOST_SIZE + rank % HOST_SIZE


def routing(bs, topk, pattern):
    ids = np.empty((WORLD, bs, topk), dtype=np.int32)
    for source in range(WORLD):
        for token in range(bs):
            for slot in range(topk):
                remote = pattern in ("remote", "duplicate-rank") or (
                    pattern == "mixed" and (slot + token + source) % 3 == 0
                )
                target_host = 1 - host(source) if remote else host(source)
                local_rank = (source + 7) % HOST_SIZE if pattern == "duplicate-rank" else (slot + token) % HOST_SIZE
                ids[source, token, slot] = (target_host * HOST_SIZE + local_rank) * EXPERTS_PER_RANK + slot
    return ids


def routes(ids):
    for source, token, slot in np.ndindex(ids.shape):
        expert = int(ids[source, token, slot])
        yield (source, token, slot), expert // EXPERTS_PER_RANK, expert


def input_records(bs, seed):
    rng = np.random.default_rng(seed)
    data, scales = quantize_input(rng.normal(size=(WORLD, bs, 32)).astype(np.float32))
    return {
        (source, token): data[source, token].tobytes() + scales[source, token].tobytes()
        for source in range(WORLD)
        for token in range(bs)
    }


def hierarchical_dispatch(ids, records):
    """Stage every actual token once; original routes alone select consumers."""
    transfer_keys = {(source, token, 1 - host(source)) for source in range(WORLD) for token in range(ids.shape[1])}
    relayed = {(pair(source), token): records[source, token] for source, token, _ in transfer_keys}
    consumers = {}
    for key, owner, expert in routes(ids):
        source, token, _ = key
        payload = records[source, token] if host(owner) == host(source) else relayed[pair(source), token]
        consumers[owner, expert, key] = payload
    return transfer_keys, consumers


@dataclass
class PersistentGatewayMemory:
    # Flags deliberately survive calls, including calls with different BS/routes.
    epochs: dict = field(default_factory=dict)
    rows: dict = field(default_factory=dict)


class CombineMessages:
    """An event-level model; all rows are opaque BF16 bytes until source reduce."""

    def __init__(self, ids, sequence, memory=None):
        self.ids = ids
        self.topk = ids.shape[2]
        self.sequence = sequence
        self.memory = memory if memory is not None else PersistentGatewayMemory()
        self.owners = {key: owner for key, owner, _ in routes(ids)}
        self.pending = {key for key, owner in self.owners.items() if host(key[0]) != host(owner)}
        self.inbox = {}
        self.outstanding = []
        self.puts = []
        self.produced = set()

    def produce(self, key, bf16_row):
        assert key not in self.produced
        self.produced.add(key)
        source, token, slot = key
        owner = self.owners[key]
        raw = np.asarray(bf16_row, dtype=np.float32).view(np.uint32)
        assert not np.any(raw & 0xFFFF)  # raw result is already BF16 rounded
        wire = (raw >> 16).astype(np.uint16).tobytes()
        if host(owner) == host(source):
            self.inbox[key] = wire
        else:
            destination = pair(source)
            assert host(destination) == host(owner)
            address = destination, token * self.topk + slot
            self.memory.rows[address] = wire
            self.memory.epochs[address] = self.sequence

    def poll(self):
        # Set intersection expresses readiness independently of the kernel's
        # per-token bit masks and sequential flag scanner.
        available = defaultdict(list)
        for source, token, slot in self.pending:
            row = token * self.topk + slot
            if self.memory.epochs.get((pair(source), row)) == self.sequence:
                available[source].append(row)
        posted = 0
        for source, rows in available.items():
            for _, group in groupby(enumerate(sorted(rows)), key=lambda item: item[1] - item[0]):
                run = tuple(row for _, row in group)
                keys = tuple((source, row // self.topk, row % self.topk) for row in run)
                payload = b"".join(self.memory.rows[pair(source), row] for row in run)
                self.outstanding.append((keys, payload))
                self.puts.append((pair(source), source, run))
                self.pending.difference_update(keys)
                posted += len(run)
        return posted

    def network_progress(self, rng):
        # Distinct runs may arrive in any order. A commit must follow completion
        # of all of them; poll/Put submission itself never fills the inbox.
        rng.shuffle(self.outstanding)
        for keys, payload in self.outstanding:
            width = len(payload) // len(keys)
            for index, key in enumerate(keys):
                assert key not in self.inbox
                self.inbox[key] = payload[index * width : (index + 1) * width]
        self.outstanding.clear()

    def source_results(self, weights, shared=None):
        assert not self.pending and not self.outstanding
        assert set(self.inbox) == set(self.owners)
        width = len(next(iter(self.inbox.values()))) // 2
        result = np.empty((*self.ids.shape[:2], width), np.float32)
        for source in range(WORLD):
            for token in range(self.ids.shape[1]):
                accumulator = None
                for slot in range(self.topk):
                    raw = np.frombuffer(self.inbox[source, token, slot], np.uint16).astype(np.uint32)
                    row = (raw << 16).view(np.float32)
                    weighted = np.multiply(row, weights[source, token, slot], dtype=np.float32)
                    accumulator = weighted if accumulator is None else np.add(accumulator, weighted, dtype=np.float32)
                if shared is not None:
                    accumulator = np.add(accumulator, shared[source, token], dtype=np.float32)
                result[source, token] = bf16(accumulator)
        return result


@pytest.mark.parametrize("bs", [1, 2, 7, 31, 32])
@pytest.mark.parametrize("topk", [1, 8, 32])
@pytest.mark.parametrize("pattern", ["local", "remote", "mixed", "duplicate-rank"])
def test_hierarchical_messages_match_direct_routing(bs, topk, pattern):
    ids = routing(bs, topk, pattern)
    for token_ids in ids.reshape(-1, topk):
        assert len(set(token_ids)) == topk  # duplicate ranks still have distinct experts
    records = input_records(bs, seed=bs + topk)
    transfers, consumers = hierarchical_dispatch(ids, records)
    direct = {(owner, expert, key): records[key[:2]] for key, owner, expert in routes(ids)}
    assert consumers == direct
    expected_tokens = {(source, token, 1 - host(source)) for source in range(WORLD) for token in range(bs)}
    assert transfers == expected_tokens
    assert len(transfers) == WORLD * bs
    # Extra staging rows must not create extra expert invocations. In particular,
    # all-local routing still transfers payload but has no cross-host consumer.
    remote_consumers = {key for key in consumers if host(key[0]) != host(key[2][0])}
    assert remote_consumers == {
        (owner, expert, key) for key, owner, expert in routes(ids) if host(owner) != host(key[0])
    }
    if pattern == "local":
        assert not remote_consumers
    if pattern == "duplicate-rank":
        assert len(transfers) == WORLD * bs
        assert all(len(set(token_ids // EXPERTS_PER_RANK)) == 1 for token_ids in ids.reshape(-1, topk))

    rng = np.random.default_rng(100 + bs + topk)
    contributions = bf16(rng.normal(size=(*ids.shape, 4)).astype(np.float32))
    weights = rng.uniform(-0.75, 1.25, size=ids.shape).astype(np.float32)
    shared = bf16(rng.normal(size=(*ids.shape[:2], 4)).astype(np.float32))
    oracle = np.multiply(contributions, weights[..., None], dtype=np.float32)
    expected = bf16(np.add(np.add.accumulate(oracle, axis=2, dtype=np.float32)[:, :, -1], shared, dtype=np.float32))

    model = CombineMessages(ids, sequence=137)
    arrivals = list(model.owners)
    scheduler = random.Random(bs * 1000 + topk)
    scheduler.shuffle(arrivals)
    for index, key in enumerate(arrivals):
        model.produce(key, contributions[key])
        if index % 37 == 0:
            model.poll()
            model.network_progress(scheduler)
    model.poll()
    model.network_progress(scheduler)
    assert not model.pending
    # Wire contents must equal the original GMM2 BF16 row, before weighting.
    for key, wire in model.inbox.items():
        raw = (contributions[key].view(np.uint32) >> 16).astype(np.uint16).tobytes()
        assert wire == raw
    counts = Counter((source, row // topk, row % topk) for _, source, run in model.puts for row in run)
    assert all(count == 1 for count in counts.values())
    assert set(counts) == {key for key, owner in model.owners.items() if host(key[0]) != host(owner)}
    assert all(destination == pair(gateway) for gateway, destination, _ in model.puts)
    put_count = len(model.puts)
    np.testing.assert_array_equal(model.source_results(weights, shared), expected)
    assert len(model.puts) == put_count  # shared expert accumulation is entirely local


def test_late_rows_do_not_block_ready_runs_or_merge_over_holes():
    ids = routing(1, 8, "remote")
    model = CombineMessages(ids, 4)
    value = bf16(np.array([0.25, -2.0], np.float32))
    for slot in (0, 1, 3, 4, 6, 7):
        model.produce((5, 0, slot), value)
    assert model.poll() == 6
    assert [run for _, source, run in model.puts if source == 5] == [(0, 1), (3, 4), (6, 7)]
    assert not model.inbox  # enqueued payload is not yet a completed receive
    assert model.poll() == 0
    model.network_progress(random.Random(4))
    assert set(model.inbox) == {(5, 0, slot) for slot in (0, 1, 3, 4, 6, 7)}
    for slot in (5, 2):
        model.produce((5, 0, slot), value)
        assert model.poll() == 1
    assert [run for _, source, run in model.puts if source == 5][-2:] == [(5,), (2,)]


def test_token_31_and_topk_31_preserve_dispatch_and_combine_payloads():
    # Only the final token's final top-k slot crosses hosts. Dispatch and Combine
    # must preserve its source/token/slot identity at both supported boundaries.
    ids = routing(32, 32, "local")
    for source in range(WORLD):
        ids[source, 31, 31] = pair(source) * EXPERTS_PER_RANK + 31
    records = input_records(32, seed=3131)
    transfers, consumers = hierarchical_dispatch(ids, records)
    assert transfers == {(source, token, 1 - host(source)) for source in range(WORLD) for token in range(32)}
    assert consumers == {(owner, expert, key): records[key[:2]] for key, owner, expert in routes(ids)}
    assert {key[2] for key in consumers if host(key[0]) != host(key[2][0])} == {
        (source, 31, 31) for source in range(WORLD)
    }

    model = CombineMessages(ids, sequence=3131)
    expected_remote = {(source, 31, 31) for source in range(WORLD)}
    assert model.pending == expected_remote
    assert model.poll() == 0
    last_rows = {}
    # Delay source rank31's final contribution while all other gateways progress.
    for source in range(WORLD - 1):
        row = bf16(np.array([source + 0.25, -source - 0.5], np.float32))
        last_rows[source] = row
        model.produce((source, 31, 31), row)
    assert model.poll() == WORLD - 1
    assert model.pending == {(31, 31, 31)}
    assert all(run == (1023,) for _, _, run in model.puts)
    model.network_progress(random.Random(3131))
    assert (31, 31, 31) not in model.inbox
    assert model.poll() == 0
    last_rows[31] = bf16(np.array([31.25, -31.5], np.float32))
    model.produce((31, 31, 31), last_rows[31])
    assert model.poll() == 1
    model.network_progress(random.Random(31))
    assert not model.pending
    assert len(model.puts) == WORLD
    for source, row in last_rows.items():
        assert model.inbox[source, 31, 31] == (row.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    assert model.poll() == 0  # the high-bit row must not be retransmitted


def test_fp32_source_accumulation_keeps_topk_order():
    ids = routing(1, 8, "mixed")
    model = CombineMessages(ids, 9)
    rows = bf16(np.array([2**24, 1, -(2**24), 0, 0, 0, 0, 0], np.float32))
    for key in reversed(list(model.owners)):
        model.produce(key, rows[key[2] : key[2] + 1])
        model.poll()
    model.network_progress(random.Random(3))
    output = model.source_results(np.ones_like(ids, dtype=np.float32))
    np.testing.assert_array_equal(output, 0)
    regrouped = np.float32(np.float32(rows[0] + rows[2]) + rows[1])
    assert regrouped == 1  # arrival-order or host-local reduction changes the result


def test_repeated_calls_ignore_old_epoch_after_legacy_marker_wrap():
    memory = PersistentGatewayMemory()
    scheduler = random.Random(83)
    calls = [
        (32, 32, 1, "remote"),
        (1, 32, 128, "local"),
        (7, 8, 129, "mixed"),
        (2, 32, 256, "remote"),
        (32, 32, 257, "duplicate-rank"),
    ]
    assert ((calls[0][2] & 127) | 128) == ((calls[-1][2] & 127) | 128)
    for bs, topk, sequence, pattern in calls:
        ids = routing(bs, topk, pattern)
        model = CombineMessages(ids, sequence, memory)
        assert model.poll() == 0  # stale flags/rows from prior shapes cannot satisfy this call
        value = bf16(np.array([sequence, -sequence, bs], np.float32))
        arrivals = list(model.owners)
        scheduler.shuffle(arrivals)
        for key in arrivals:
            model.produce(key, value)
        model.poll()
        model.network_progress(scheduler)
        output = model.source_results(np.full(ids.shape, 1 / topk, np.float32))
        np.testing.assert_array_equal(output, np.broadcast_to(value, output.shape))


def lifecycle_dependencies(*, backend, whole_host=True, wait_before_send=False, remote_rows=1, tail_quiet=True):
    """Separate remote visibility from local CQ retirement.

    HNS's ordered terminal WQE publishes remote visibility before the sender
    needs to retire its CQ. XSCALE retains quiet -> commit -> quiet. ``post``
    includes the empty final drain, which must still be followed by a commit.
    """
    assert backend in ("HNS_1825", "XSCALE")
    graph = {}
    for rank in range(WORLD):
        graph["produce", rank] = set()
    for machine in (0, 1):
        graph["host", machine] = {("produce", rank) for rank in range(WORLD) if host(rank) == machine}
    for rank in range(WORLD):
        finish = ("host", host(rank)) if whole_host else ("produce", rank)
        graph["post", rank] = {finish}
        data_visible = set()
        if remote_rows:
            graph["data-visible", rank] = {("post", rank)}
            data_visible.add(("data-visible", rank))
        if backend == "HNS_1825":
            graph["commit", rank] = {("post", rank)}
            # Automatic SO orders all prior data before the terminal flag at GM.
            graph["commit-visible", rank] = {("commit", rank)} | data_visible
        else:
            graph["quiet", rank] = {("post", rank)} | data_visible
            graph["commit", rank] = {("quiet", rank)}
            graph["commit-visible", rank] = {("commit", rank)}
        graph["commit-complete", rank] = {("commit-visible", rank)}
        outgoing = "commit" if backend == "HNS_1825" else "commit-complete"
        graph["receive", rank] = {(outgoing, rank), ("commit-visible", pair(rank))}
        graph["unpermute", rank] = {("receive", rank)}
        retired = ("unpermute", rank)
        if backend == "HNS_1825" and tail_quiet:
            graph["tail-quiet", rank] = {("receive", rank), ("commit-complete", rank)}
            retired = ("tail-quiet", rank)
        graph["reuse", rank] = {retired, ("unpermute", rank), ("host", host(rank))}
        if wait_before_send:
            graph["post", rank].add(("receive", rank))
    return graph


def ancestors(graph, event):
    result = set()
    pending = list(graph[event])
    while pending:
        dependency = pending.pop()
        if dependency not in result:
            result.add(dependency)
            pending.extend(graph[dependency])
    return result


@pytest.mark.parametrize("backend", ["HNS_1825", "XSCALE"])
@pytest.mark.parametrize("remote_rows", [0, 1])
@pytest.mark.parametrize("late_rank", [0, 15, 16, 31])
def test_completion_progress_keeps_single_commit_after_host_barrier(backend, remote_rows, late_rank):
    graph = lifecycle_dependencies(backend=backend, remote_rows=remote_rows)
    for rank in range(WORLD):
        graph["owner-ready", rank] = set()
        # Completion can post payloads before unrelated host producers finish.
        # These routes only need the next neighbour; empty routes need no rows.
        producer = host(rank) * 16 + (rank + 1) % 16
        graph["post", rank] = {("owner-ready", rank)}
        if remote_rows:
            graph["post", rank].add(("produce", producer))
        # The sole commit still closes the entire host's old-reader lifetime.
        phase = "commit" if backend == "HNS_1825" else "quiet"
        graph[phase, rank].add(("host", host(rank)))
    all_producers = {("produce", rank) for rank in range(WORLD)}
    for rank in range(WORLD):
        assert all_producers <= ancestors(graph, ("reuse", rank))
        assert ("host", host(rank)) not in ancestors(graph, ("post", rank))
        assert ("host", host(rank)) in ancestors(graph, ("commit", rank))
    finished = set()
    scheduler = random.Random(late_rank)
    while len(finished) < len(graph):
        runnable = [event for event, deps in graph.items() if event not in finished and deps <= finished]
        assert runnable, "completion progress and host commit created a cyclic wait"
        ordinary = [event for event in runnable if event != ("produce", late_rank)]
        event = scheduler.choice(ordinary or runnable)
        if event[0] == "reuse":
            assert all_producers <= finished
        finished.add(event)


@pytest.mark.parametrize("late_rank", [0, 15, 16, 31])
@pytest.mark.parametrize("late_stage", ["produce", "commit", "commit-complete"])
@pytest.mark.parametrize("backend", ["HNS_1825", "XSCALE"])
def test_host_finish_and_commit_close_buffer_lifetime_without_wait_cycle(late_rank, late_stage, backend):
    graph = lifecycle_dependencies(backend=backend)
    finished = set()
    scheduler = random.Random(late_rank)
    while len(finished) < len(graph):
        runnable = [event for event, deps in graph.items() if event not in finished and deps <= finished]
        assert runnable, "outbound-before-inbound protocol must remain acyclic"
        ordinary = [event for event in runnable if event != (late_stage, late_rank)]
        event = scheduler.choice(ordinary or runnable)
        if event[0] == "reuse":
            assert {("produce", rank) for rank in range(WORLD)} <= finished
            assert ("commit-complete", event[1]) in finished
        if event[0] == "receive":
            assert ("commit", event[1]) in finished
            assert ("commit-visible", pair(event[1])) in finished
            if backend == "XSCALE":
                assert ("commit-complete", event[1]) in finished
        finished.add(event)
    for rank in range(WORLD):
        assert {("produce", peer) for peer in range(WORLD)} <= ancestors(graph, ("reuse", rank))
        if backend == "HNS_1825":
            assert ("commit-complete", rank) not in ancestors(graph, ("unpermute", rank))
            assert ("tail-quiet", rank) in ancestors(graph, ("reuse", rank))
            assert ("quiet", rank) not in graph


@pytest.mark.parametrize("backend", ["HNS_1825", "XSCALE"])
def test_lifecycle_requires_even_zero_contribution_commit_and_full_host_finish(backend):
    # A rank with no expected remote rows still closes the remote host's reads of
    # its old quantization/relay/rail buffers. Omitting its commit strands its peer.
    graph = lifecycle_dependencies(backend=backend, remote_rows=0)
    assert not any(event[0] == "data-visible" for event in graph)
    assert ("commit", 16) in ancestors(graph, ("reuse", 0))
    assert ("commit-complete", 0) in ancestors(graph, ("reuse", 0))
    early = lifecycle_dependencies(backend=backend, whole_host=False, remote_rows=0)
    assert ("produce", 17) not in ancestors(early, ("reuse", 0))
    assert ("produce", 17) in ancestors(graph, ("reuse", 0))
    cyclic = lifecycle_dependencies(backend=backend, wait_before_send=True, remote_rows=0)
    assert ("receive", 0) in ancestors(cyclic, ("receive", 0))


def test_hns_incoming_commit_does_not_retire_the_outgoing_source():
    graph = lifecycle_dependencies(backend="HNS_1825")
    assert ("commit-complete", 0) not in ancestors(graph, ("receive", 0))
    assert ("commit-complete", 0) in ancestors(graph, ("reuse", 0))
    missing_retirement = lifecycle_dependencies(backend="HNS_1825", tail_quiet=False)
    assert ("commit-complete", 0) not in ancestors(missing_retirement, ("reuse", 0))


@pytest.mark.parametrize("backend", ["HNS_1825", "XSCALE"])
def test_next_call_cannot_overwrite_sequence_or_payload_before_local_and_remote_closure(backend):
    graph = {}
    # Alternating empty and nonempty Combine traffic models shape/route changes.
    sequences = (127, 128, 129, 256, 257)
    for index, sequence in enumerate(sequences):
        call = lifecycle_dependencies(backend=backend, remote_rows=index % 2)
        for event, deps in call.items():
            graph[sequence, *event] = {(sequence, *dependency) for dependency in deps}
        if index:
            for rank in range(WORLD):
                # The public API's dedicated stream serializes kernel calls.
                graph[sequence, "produce", rank].add((sequences[index - 1], "reuse", rank))
    for previous, sequence in zip(sequences, sequences[1:]):
        for rank in (0, 15, 16, 31):
            before_reuse = ancestors(graph, (sequence, "produce", rank))
            assert (previous, "commit-complete", rank) in before_reuse
            assert (previous, "commit-visible", pair(rank)) in before_reuse
            assert {(previous, "produce", peer) for peer in range(WORLD)} <= before_reuse
            if backend == "HNS_1825":
                assert (previous, "tail-quiet", rank) in before_reuse


def test_every_expert_source_pair_uses_only_local_mte_then_paired_rdma():
    for source in range(WORLD):
        assert pair(pair(source)) == source
        assert pair(source) % HOST_SIZE == source % HOST_SIZE
        for owner in range(WORLD):
            mte_destination = source if host(owner) == host(source) else pair(source)
            assert host(mte_destination) == host(owner)
            if mte_destination != source:
                assert pair(mte_destination) == source


def a8w8_gmm2_assignments(tile_count, aic_count, start_block):
    """RunGmm2Generic/Gmm2AicMmadGeneric's cyclic tile ownership.

    Empty ranges model HandleWaveProblemWithoutWork: those AICs advance their
    cursor without reaching NotifyTokenCombineSliceReady.
    """
    return {
        aic: tuple(range((aic + aic_count - start_block) % aic_count, tile_count, aic_count))
        for aic in range(aic_count)
    }


class FullRowSliceCompletion:
    """Opaque tile payloads plus FIX completion and one notification per AIC.

    This models the dependency contract of NotifyTokenCombineSliceReady and
    SendHierarchicalCombineSlice, not Cube arithmetic or hardware atomics.
    """

    def __init__(self, assignments, coordinates, required_producers):
        self.assignments = assignments
        self.coordinates = coordinates
        self.required_producers = required_producers
        self.visible = {}
        self.notified = set()

    def complete_fix(self, aic, tile):
        assert tile in self.assignments[aic]
        assert tile not in self.visible
        self.visible[tile] = self.coordinates[tile]

    def notify(self, aic):
        assert self.assignments[aic], "idle AIC must not increment the ready counter"
        assert aic not in self.notified, "one notification per producer, not per tile"
        assert set(self.assignments[aic]) <= self.visible.keys(), "FIX writes must finish before notifying"
        self.notified.add(aic)

    def ready(self):
        return len(self.notified) == self.required_producers

    def send_rows(self):
        assert self.ready()
        return set(self.visible.values())


@pytest.mark.parametrize("aic_count", [24, 32, 36])
@pytest.mark.parametrize("rows", [1, 32, 255, 256, 257, 1024])
@pytest.mark.parametrize("width", [1024, 1088, 8192])
def test_a8w8_full_row_ready_counts_only_producers_after_every_n_tile(aic_count, rows, width):
    row_groups = tuple(range(0, rows, 256))
    column_tiles = tuple(range(0, width, 256))
    required = {(m, n) for m in row_groups for n in column_tiles}
    producer_count = min(((rows + 255) // 256) * ((width + 255) // 256), aic_count)
    for start_block in range(aic_count):
        assignments = a8w8_gmm2_assignments(len(required), aic_count, start_block)
        participants = {aic for aic, tiles in assignments.items() if tiles}
        assert len(participants) == producer_count
        ownership = Counter(tile for tiles in assignments.values() for tile in tiles)
        assert ownership == Counter({tile: 1 for tile in range(len(required))})
        # Independent inverse check: advancing around the physical AIC ring once
        # per tile must recover exactly the owners found by the strided loops.
        for aic, tiles in assignments.items():
            assert all((tile + start_block) % aic_count == aic for tile in tiles)

        scheduler = random.Random(aic_count * 100000 + rows * 100 + width + start_block)
        coordinates = sorted(required)
        # Producer counting depends on coverage, not CATLASS's tile swizzle.
        # Exercise arbitrary bijections from scheduled IDs to M/N coordinates.
        scheduler.shuffle(coordinates)
        model = FullRowSliceCompletion(assignments, coordinates, producer_count)
        delayed = coordinates.index((row_groups[-1], column_tiles[-1]))
        delayed_owner = next(aic for aic, tiles in assignments.items() if delayed in tiles)
        completions = [(aic, tile) for aic, tiles in assignments.items() for tile in tiles if tile != delayed]
        scheduler.shuffle(completions)
        for aic, tile in completions:
            model.complete_fix(aic, tile)
            if aic != delayed_owner and aic not in model.notified and set(assignments[aic]) <= model.visible.keys():
                model.notify(aic)
            assert not model.ready()
        for idle in set(assignments) - participants:
            with pytest.raises(AssertionError, match="idle AIC"):
                model.notify(idle)
        assert len(model.notified) == producer_count - 1
        assert set(model.visible.values()) == required - {(row_groups[-1], column_tiles[-1])}
        with pytest.raises(AssertionError, match="FIX writes"):
            model.notify(delayed_owner)
        with pytest.raises(AssertionError):
            model.send_rows()

        model.complete_fix(delayed_owner, delayed)
        assert not model.ready()  # tile visibility alone does not publish the counter
        model.notify(delayed_owner)
        assert model.ready()
        assert model.send_rows() == required
        for m in row_groups:
            arrived_columns = {n for row, n in model.send_rows() if row == m}
            assert arrived_columns == set(column_tiles)
            # H1088 ends in a 64-column N tile. That producer is required even
            # though the tile is not full; the valid full-row payload ends at H.
            valid_columns = [column for n in arrived_columns for column in range(n, min(n + 256, width))]
            assert Counter(valid_columns) == Counter(range(width))
            assert max(valid_columns) == width - 1
        with pytest.raises(AssertionError, match="one notification"):
            model.notify(delayed_owner)


def a8w8_qp_phase_dependencies(*, backend, combine_owner=1, input_barrier=True):
    """Phase/resource model shared by every rank, with a fixed AIV1 Dispatch owner."""
    assert backend in ("HNS_1825", "XSCALE")
    dispatch_owner = 1
    graph = {
        "quantized-input": set(),
        "packed-metadata": set(),
        "dispatch-meta-put": {"packed-metadata"},
        "dispatch-put": {"quantized-input", "dispatch-meta-put"},
        "incoming-metadata": set(),
        "incoming-payload-commit": {"incoming-metadata"},
        # An independent receive leader observes the incoming flag. It never
        # waits for this rank's outgoing CQ and never borrows the owner's UB.
        "dispatch-receive": {"incoming-metadata"},
        "input-finish-aiv0": set(),
        "all-core-input-barrier": {"dispatch-receive", "input-finish-aiv0", "input-finish-aiv1"},
        "input-count-ready": {"all-core-input-barrier"},
        "gateway-broadcast": {"dispatch-payload-receive"},
        "other-local-gateway-broadcasts": set(),
        "gateway-payload-acquire": {"input-count-ready", "gateway-broadcast", "other-local-gateway-broadcasts"},
        "aic-fix-next-wave-aiv0": {"all-core-input-barrier"},
        # The same AIV finishes Dispatch before entering Combine, even without
        # the barrier that additionally closes every other core's input work.
        "combine-progress": {"all-core-input-barrier"} if input_barrier else {"input-finish-aiv1"},
        "local-producers-finished": {"combine-progress", "aic-fix-next-wave-aiv0", "gateway-payload-acquire"},
        "host-finish-barrier": {"local-producers-finished"},
        "combine-final-drain": {"host-finish-barrier"},
        "incoming-combine-commit": set(),
        "output-aiv-barrier": {"combine-receive"},
        "unpermute": {"output-aiv-barrier"},
    }
    if backend == "HNS_1825":
        graph.update(
            {
                "dispatch-commit": {"dispatch-put"},
                "dispatch-payload-receive": {"dispatch-commit", "incoming-payload-commit"},
                "input-finish-aiv1": {"gateway-broadcast"},
                "combine-commit": {"combine-final-drain"},
                "combine-receive": {"combine-commit", "incoming-combine-commit"},
                "tail-quiet": {"output-aiv-barrier"},
                "return": {"tail-quiet", "unpermute"},
            }
        )
    else:
        graph.update(
            {
                "dispatch-meta-quiet": {"dispatch-meta-put"},
                "dispatch-meta-commit": {"dispatch-meta-quiet"},
                "dispatch-meta-commit-quiet": {"dispatch-meta-commit"},
                "dispatch-put": {"quantized-input", "dispatch-meta-commit-quiet"},
                "dispatch-payload-quiet": {"dispatch-put"},
                "dispatch-commit": {"dispatch-payload-quiet"},
                "dispatch-commit-quiet": {"dispatch-commit"},
                "dispatch-payload-receive": {"dispatch-commit-quiet", "incoming-payload-commit"},
                "input-finish-aiv1": {"gateway-broadcast"},
                "combine-payload-quiet": {"combine-final-drain"},
                "combine-commit": {"combine-payload-quiet"},
                "combine-commit-quiet": {"combine-commit"},
                "combine-receive": {"combine-commit-quiet", "incoming-combine-commit"},
                "return": {"unpermute"},
            }
        )
    actions = {}
    for event in (
        "dispatch-meta-put",
        "dispatch-meta-quiet",
        "dispatch-meta-commit",
        "dispatch-meta-commit-quiet",
        "dispatch-put",
        "dispatch-payload-quiet",
        "dispatch-commit",
        "dispatch-commit-quiet",
        "dispatch-payload-receive",
        "combine-progress",
        "combine-final-drain",
        "combine-payload-quiet",
        "combine-commit",
        "combine-commit-quiet",
        "tail-quiet",
    ):
        if event not in graph:
            continue
        owner = dispatch_owner if event.startswith("dispatch-") else combine_owner
        actions[event] = owner, {"qp1", "cq1", ("ub-tail", owner), ("event7", owner)}
    if backend == "HNS_1825":
        # During this receive wait the same QP owner may reclaim ready CQEs.
        actions["combine-receive"] = (
            combine_owner,
            {"qp1", "cq1", ("ub-tail", combine_owner), ("event7", combine_owner)},
        )
    actions["aic-fix-next-wave-aiv0"] = "aic", {("ub-tail", 0)}
    return graph, actions


def unordered_resource_conflicts(graph, actions):
    events = list(actions)
    return {
        (left, right)
        for index, left in enumerate(events)
        for right in events[index + 1 :]
        if actions[left][1] & actions[right][1]
        and left not in ancestors(graph, right)
        and right not in ancestors(graph, left)
    }


@pytest.mark.parametrize("backend", ["HNS_1825", "XSCALE"])
def test_a8w8_fixed_qp_owner_and_aiv1_scratch_allow_next_wave_fix_overlap(backend):
    graph, actions = a8w8_qp_phase_dependencies(backend=backend)
    assert all(actions[event][0] == 1 for event in actions if event.startswith("dispatch-"))
    assert all(actions[event][0] == 1 for event in actions if event.startswith("combine-"))
    dispatch_finished = "dispatch-commit" if backend == "HNS_1825" else "dispatch-commit-quiet"
    assert dispatch_finished in ancestors(graph, "combine-progress")
    assert dispatch_finished not in ancestors(graph, "dispatch-receive")
    assert "incoming-payload-commit" not in ancestors(graph, "dispatch-receive")
    assert "incoming-payload-commit" in ancestors(graph, "input-count-ready")
    assert dispatch_finished in ancestors(graph, "gateway-broadcast")
    assert "input-count-ready" not in ancestors(graph, "gateway-broadcast")
    assert "all-core-input-barrier" not in ancestors(graph, "gateway-broadcast")
    assert "incoming-payload-commit" in ancestors(graph, "gateway-payload-acquire")
    assert "input-count-ready" in ancestors(graph, "gateway-payload-acquire")
    assert "all-core-input-barrier" in ancestors(graph, "combine-progress")
    assert "host-finish-barrier" in ancestors(graph, "combine-final-drain")
    assert "combine-commit-quiet" not in ancestors(graph, "combine-progress")
    assert not unordered_resource_conflicts(graph, actions)
    if backend == "HNS_1825":
        assert "tail-quiet" in ancestors(graph, "return")
        assert "unpermute" not in ancestors(graph, "tail-quiet")
        assert "unpermute" in ancestors(graph, "return")
        assert "output-aiv-barrier" in ancestors(graph, "tail-quiet")
        assert "tail-quiet" not in ancestors(graph, "unpermute")
        assert "dispatch-payload-quiet" not in graph and "combine-payload-quiet" not in graph
    # FIX may continue on AIV0's UB while the communication owner progresses.
    # Neither action must wait for the other, and their resources must be disjoint.
    assert "aic-fix-next-wave-aiv0" not in ancestors(graph, "combine-progress")
    assert "combine-progress" not in ancestors(graph, "aic-fix-next-wave-aiv0")
    assert actions["aic-fix-next-wave-aiv0"][1].isdisjoint(actions["combine-progress"][1])

    missing_barrier = a8w8_qp_phase_dependencies(backend=backend, input_barrier=False)
    assert not unordered_resource_conflicts(*missing_barrier)
    assert dispatch_finished in ancestors(missing_barrier[0], "combine-progress")
    assert "input-finish-aiv0" not in ancestors(missing_barrier[0], "combine-progress")
    assert "input-finish-aiv0" in ancestors(graph, "combine-progress")
    aiv0_owner = a8w8_qp_phase_dependencies(backend=backend, combine_owner=0)
    assert ("combine-progress", "aic-fix-next-wave-aiv0") in unordered_resource_conflicts(*aiv0_owner)


def cpp_function_body(path, signature):
    """Inspect actual entrypoint order in addition to the dependency models.

    The real transport helper is compiled by test_roce_transport.py. These
    guards catch either kernel forgetting to call its final retirement helper;
    they do not establish NIC/cache visibility or execute device barriers.
    """
    source = path.read_text(encoding="utf-8")
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)
    start = source.index("{", source.index(signature))
    depth = 1
    for end in range(start + 1, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[start + 1 : end]
    raise AssertionError(f"Unclosed function body in {path.name}: {signature}")


def test_real_kernel_entrypoint_overlaps_owner_retirement_after_output_ready():
    kernels = Path(__file__).resolve().parents[1] / "csrc_ep32/kernels/mega_moe"
    body = cpp_function_body(kernels / "mega_moe_arch35.hpp", "::ProcessWave(Derived& derived)")
    expected = (
        "CrossRankSyncInHierarchicalHost(",
        "DrainHierarchicalReducedCombine(",
        "RocePublishCommit(",
        "RoceWaitCommit(",
        "RoceFinishOutstanding(",
        "UnpermuteTokens<",
        "Stage::COMPLETE",
    )
    positions = []
    for token in expected:
        positions.append(body.index(token, positions[-1] + 1 if positions else 0))
    assert positions == sorted(positions)
    assert body.count("RoceFinishOutstanding(") == 1
    tail_begin = body.index("Stage::UNPERMUTE")
    tail = body[tail_begin : body.index("Stage::COMPLETE")]
    assert "IsHierarchicalCombineOwner(params_.tilingData, aivJob_.jobIndex)" in tail
    assert "HierarchicalCombineLane(params_.tilingData, aivJob_.jobIndex) + 1U" in tail
    assert "expectedRows" not in tail and "sentRows" not in tail
    assert "} else {" in tail[tail.index("RoceFinishOutstanding(") : tail.index("UnpermuteTokens<")]
    assert "SyncAll<true>()" in body[body.index("RoceWaitCommit(") : tail_begin]
    assert "SyncAll" not in tail, "The owner must not gate other AIVs behind its final quiet"
    assert positions[-2] < body.index("timer_.End(AscendTimer::KERNEL_TIMING)")
    assert positions[-2] < body.index("timer_.FlushDynamicCounts()")


def test_a8w8_entrypoint_keeps_local_transport_only():
    kernels = Path(__file__).resolve().parents[1] / "csrc_ep32/kernels/mega_moe"
    body = cpp_function_body(kernels / "mega_moe_wave_a8w8.hpp", "::Process()")
    assert "CrossRankSyncInWorldSize(" in body
    assert "Roce" not in body and "Hierarchical" not in body
