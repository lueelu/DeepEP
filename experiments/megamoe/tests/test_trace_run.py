# Copyright (c) 2026, Lu Lu

"""Exercise trace orchestration without CANN, including ranks without timers."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from deep_ep_experimental.megamoe import api, benchmark, validation
from deep_ep_experimental.megamoe.timer import breakdown


@pytest.mark.parametrize("rank", [0, 1, 16])
@pytest.mark.parametrize("warmup,iters", [(2, 1), (0, 5)])
@pytest.mark.parametrize("mode", ["trace", "perf"])
@pytest.mark.parametrize("method", ["legacy", "shmem"])
def test_profile_and_trace_run_one_warmup_and_continuous_batch(
    tmp_path, monkeypatch, rank, warmup, iters, mode, method
):
    output = tmp_path / "output"
    config = dict(
        schema_version=1,
        model=dict(hidden=32, intermediate_hidden=32, num_experts=32, num_topk=1, num_shared_experts=0),
        bs_values=[2],
        warmup=warmup,
        iters=iters,
        seed=0,
        alignment=dict(rows=2, cols=2, repeats=1),
    )
    config_path = tmp_path / "case.json"
    config_path.write_text(json.dumps(config))
    (tmp_path / "_build_info.json").write_text("{}")
    monkeypatch.setattr(benchmark, "files", lambda _: tmp_path)
    monkeypatch.setenv("LOCAL_RANK", str(rank % 16))
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "16")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--mode",
            mode,
            "--perf-method",
            method,
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
            "--input",
            "bf16",
            "--timer-ranks",
            "0,16",
        ],
    )
    torch = MagicMock()
    torch.npu.get_device_properties.return_value.L2_cache_size = 128
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", torch.distributed)
    torch.distributed.init_process_group.side_effect = AssertionError("benchmark must not initialize HCCL")
    torch.distributed.all_reduce.side_effect = AssertionError("benchmark must not insert AllReduce")
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())  # No profiler API available.
    group = SimpleNamespace(
        rank=lambda: rank,
        size=lambda: 32,
        all_gather=lambda value, stage: [value] * 32 if stage.startswith("profiling BS=") else [value],
        barrier=MagicMock(),
        close=MagicMock(),
        abort=MagicMock(),
    )
    monkeypatch.setattr(api.StoreGroup, "from_env", lambda: group)

    def check(group, stage, error, signature=None):
        assert error is None, f"{stage}: {error}"

    monkeypatch.setattr(validation, "collective_check", check)
    buffer = MagicMock(timer_enabled=True, workspace_bytes=64, symmetric_buffer_bytes=128, num_max_tokens_per_rank=2)
    buffer._roce_config = dict(combine_reduce_version=4, combine_qps=16, dispatch_qps=8, dispatch_workers=14)
    buffer._domains.configs = 16
    buffer._plan.effective_wave = 6
    monkeypatch.setattr(benchmark, "loaded_libraries", lambda buffer: {})
    raw = np.array([11, 22, 33, 44], dtype="<i8")
    timer = buffer.allocate_timer.return_value
    timer.cpu.return_value.numpy.return_value = raw
    buffer_options = []
    monkeypatch.setattr(api, "SymmBuffer", lambda *a, **kw: buffer_options.append(kw) or buffer)
    monkeypatch.setattr(api, "transform_weights_for_mega_moe", lambda *a, **kw: (object(), object()))
    calls = []
    monkeypatch.setattr(api, "fp8_fp4_mega_moe", lambda *a, **kw: calls.append(kw["timer"]))

    def collect(call, group, torch, config, prefix, directory, bs, **batch_options):
        assert mode == "perf", "trace started performance profiler"
        benchmark.enqueue_batch(call, torch, prefix, config["alignment"], config["iters"], **batch_options)
        return [dict(duration_us=i + 1) for i in range(iters)]

    monkeypatch.setattr(benchmark, "collect_profile", collect)
    monkeypatch.setattr(benchmark, "accuracy", lambda *a, **kw: pytest.fail("trace started accuracy check"))
    monkeypatch.setattr(breakdown, "compact_events", lambda *a: [])
    monkeypatch.setattr(breakdown, "rank_breakdown", lambda events, rank: dict(rank=rank))
    monkeypatch.setattr(breakdown, "write_breakdown", lambda *a: None)

    # --mode trace must enable timers itself; no --timer flag is passed.
    benchmark.main()
    assert set(buffer_options[0]) == {"num_max_tokens_per_rank", "ranks_per_node", "num_max_configs", *config["model"]}
    manifest = json.loads((output / f"rank_{rank}" / "manifest.json").read_text())
    assert manifest["compiled_roce_config"] == buffer._roce_config
    assert manifest["implementation"] == "dispatch_qp8_combine_qp16"

    selected = mode == "trace" and rank in (0, 16)
    assert len(calls) == warmup + iters
    assert all(value is None for value in calls[:-1])
    assert calls[-1] is (timer if selected else None)
    assert buffer.allocate_timer.call_count == int(selected)
    timer_path = output / "bs_2" / "timer" / f"rank_{rank}_npu_time.bin"
    assert timer_path.exists() is selected
    if selected:
        assert timer_path.read_bytes() == raw.tobytes()
    for name in (
        ("profiling.json", "kernel_samples.json", "profiler", "accuracy.json")
        if mode == "trace"
        else ("accuracy.json",)
    ):
        assert not list(output.rglob(name))
    if mode == "perf":
        report = json.loads((output / f"rank_{rank}" / "bs_2" / "profiling.json").read_text())
        assert report["implementation"] == "dispatch_qp8_combine_qp16"
        assert report["iteration_sync"] == ("none" if method == "legacy" else "shmem_world_on_stream")
        assert report["alignment_scope"] == ("once_before_batch" if method == "legacy" else "before_each_iteration")
        assert report["perf_method"] == method
        assert report["recorded_rounds"] == iters
        assert len(report["per_rank_rounds_us"]) == 32
        assert report["timer_pass"] is False
    manifest = json.loads((output / f"rank_{rank}" / "manifest.json").read_text())
    assert manifest["mode"] == mode
    assert manifest["iteration_sync"] == ("none" if method == "legacy" else "shmem_world_on_stream")
    assert manifest["alignment_scope"] == ("once_before_batch" if method == "legacy" else "before_each_iteration")
    assert manifest["perf_method"] == method
    assert torch.exp.call_count == (1 + config["alignment"]["repeats"] if method == "legacy" else 0)
    assert buffer.benchmark_barrier.call_count == (1 + iters if method == "shmem" else 0)
    assert buffer.benchmark_sleep.call_count == (1 + iters if method == "shmem" else 0)
    measurement = json.loads((output / f"rank_{rank}" / "bs_2" / "measurement.json").read_text())
    assert measurement["l2_flush_bytes"] == (256 if method == "shmem" else 0)
    assert measurement["alignment"] == (None if method == "shmem" else config["alignment"])
    torch.distributed.init_process_group.assert_not_called()
    torch.distributed.all_reduce.assert_not_called()
    torch.distributed.destroy_process_group.assert_not_called()
    buffer.destroy.assert_called_once()
    group.close.assert_called_once()
    group.abort.assert_not_called()


def test_loaded_library_hashes_follow_process_maps(tmp_path):
    import hashlib
    from deep_ep_experimental.megamoe import benchmark

    extension = tmp_path / "_C.so"
    loaded = tmp_path / "libmegamoe device.so"
    for path in (extension, loaded, tmp_path / "libshmem-unused.so"):
        path.write_bytes(path.name.encode())
    maps = tmp_path / "maps"
    maps.write_text(f"0000-1000 r-xp 00000 00:01 123 {loaded}\n" * 2)
    assert benchmark._library_hashes(extension, maps) == {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (extension, loaded)
    }
