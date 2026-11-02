"""Cache eviction lifecycle and benchmark launch contracts, without NPU."""

import contextlib
import ast
import os
import tempfile
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks/clos"))
from stage_metrics import (
    L2CacheFlush,
    L2_CACHE_FLUSH_BYTES,
    ProfileSamples,
    PROFILE_SAMPLE_RANGE,
    PROFILE_GUARD_ITERS,
)
import bench_v1_bandwidth as v1
import bench_v2_bandwidth as v2


class BandwidthCacheTest(unittest.TestCase):
    def test_profile_loop_keeps_legacy_counts_and_marks_only_active_samples(self):
        # Load only the real helper so this orchestration regression needs no
        # torch_npu import. Device execution itself still needs an NPU test.
        source = Path(__file__).with_name("test_v1_intranode.py").read_text(encoding="utf-8")
        function = next(
            n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_profile_operator"
        )
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
            type_ignores=[],
        )
        for guarded in (False, True):
            with self.subTest(guarded=guarded), tempfile.TemporaryDirectory() as tmp:
                profiler = Mock()
                prof = Mock()
                profiler.profile.return_value = prof
                consume, invoke = Mock(), Mock()
                env = dict(
                    os=os,
                    torch_npu=SimpleNamespace(profiler=profiler, npu=Mock()),
                    _consume_profile_result=consume,
                    _ProfileComplete=RuntimeError,
                )
                exec(compile(ast.fix_missing_locations(module), "<profile_helper>", "exec"), env)
                ids = []

                def sample(index, callback):
                    ids.append(index)
                    callback()

                args = SimpleNamespace(profile_dir=tmp, profile_warmup=3, profile_iters=30)
                options = dict(sample_callback=sample, guard_iters=PROFILE_GUARD_ITERS) if guarded else {}
                with self.assertRaises(RuntimeError):
                    env["_profile_operator"](1, args, "cached-dispatch", invoke, **options)
                self.assertEqual(invoke.call_count, 63 + (4 if guarded else 0))
                self.assertEqual(consume.call_count, invoke.call_count)
                self.assertEqual(ids, list(range(-2, 32)) if guarded else [])
                self.assertEqual(prof.step.call_count, 2)
                prof.stop.assert_called_once()

    def test_sample_markers_bracket_completion_without_host_sync(self):
        calls = []

        @contextlib.contextmanager
        def annotation(name):
            calls.append(name)
            yield

        def event():
            return SimpleNamespace(record=lambda: calls.append("record"))

        torch = SimpleNamespace(
            npu=SimpleNamespace(Event=event),
            autograd=SimpleNamespace(profiler=SimpleNamespace(record_function=annotation)),
        )
        sampler = ProfileSamples(torch)
        sampler(0, lambda: calls.extend(["api", "completion_wait"]))
        self.assertEqual(
            calls,
            [
                PROFILE_SAMPLE_RANGE + "0/start",
                "record",
                "api",
                "completion_wait",
                PROFILE_SAMPLE_RANGE + "0/end",
                "record",
            ],
        )
        self.assertEqual(len(sampler.events), 2)

    def test_parse_failure_is_exchanged_by_failing_and_successful_ranks(self):
        for local_failure in (False, True):
            with self.subTest(local_failure=local_failure), tempfile.TemporaryDirectory() as tmp:
                gathered = []

                def exchange(statuses, local):
                    gathered.append(local)
                    statuses[:] = [local, dict(rank=8, error="missing notify")]

                dist = SimpleNamespace(get_rank=lambda: 0, get_world_size=lambda: 2, all_gather_object=exchange)
                with patch.object(
                    v2,
                    "parse_profile",
                    side_effect=ValueError("missing notify") if local_failure else None,
                    return_value={"samples": []},
                ):
                    with self.assertRaisesRegex(RuntimeError, "rank 8: missing notify"):
                        v2.parse_profile_collective(dist, Path(tmp) / "error.json", "profile")
                self.assertEqual(bool(gathered[0]["error"]), local_failure)
                self.assertEqual((Path(tmp) / "error.json").exists(), local_failure)

    def test_parse_success_returns_profile_after_exchange(self):
        dist = SimpleNamespace(
            get_rank=lambda: 0,
            get_world_size=lambda: 1,
            all_gather_object=lambda statuses, local: statuses.__setitem__(0, local),
        )
        with patch.object(v2, "parse_profile", return_value={"samples": [0]}):
            self.assertEqual(v2.parse_profile_collective(dist, None, "profile"), {"samples": [0]})

    def test_reuses_allocation_and_completes_flush_before_api(self):
        calls = []
        stream = SimpleNamespace(
            wait_stream=lambda s: calls.append(("wait", s)), synchronize=lambda: calls.append("sync")
        )
        cache = SimpleNamespace(zero_=lambda: calls.append("zero"))

        @contextlib.contextmanager
        def annotation(name):
            calls.append("range_start")
            yield
            calls.append("range_end")

        torch = SimpleNamespace(
            npu=SimpleNamespace(
                Stream=lambda: stream, stream=lambda s: contextlib.nullcontext(), current_stream=lambda: "caller"
            ),
            empty=Mock(return_value=cache),
            int32="int32",
            autograd=SimpleNamespace(profiler=SimpleNamespace(record_function=annotation)),
        )
        flush = L2CacheFlush(torch, SimpleNamespace(get_comm_stream=lambda: "comm"), True)
        fn = flush.wrap(lambda: calls.append("api"))
        fn()
        fn()
        self.assertEqual(
            calls, [("wait", "caller"), ("wait", "comm"), "sync", "range_start", "zero", "sync", "range_end", "api"] * 2
        )
        torch.empty.assert_called_once_with(L2_CACHE_FLUSH_BYTES // 4, dtype="int32", device="npu")

    def test_disabled_flush_needs_no_npu_allocation_or_synchronization(self):
        flush = L2CacheFlush(None, None, False)
        flush()
        self.assertEqual(flush.wrap(lambda: 42)(), 42)

    def test_all_entrypoints_and_worker_replay(self):
        case = dict(name="bf16_expand0_aiv32", dtype="bf16", expand=0, aiv=32)
        for entry in (v1, v2):
            for enabled in (False, True):
                with self.subTest(entry=entry.__name__, enabled=enabled):
                    flags = [] if enabled else ["--no-flush-l2-cache", "--combine-bias", "two"]
                    args = entry.arguments(flags)
                    self.assertEqual(args.flush_l2_cache, enabled)
                    self.assertEqual(args.combine_bias, "none" if enabled else "two")
                    command = v2.worker_command(args, case, Path("out"), entrypoint=entry.__file__)
                    replay = entry.arguments(command[command.index("--worker") :])
                    self.assertEqual(replay.flush_l2_cache, enabled)
                    self.assertEqual(replay.combine_bias, args.combine_bias)


if __name__ == "__main__":
    unittest.main()
