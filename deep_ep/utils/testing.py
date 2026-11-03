# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""DeepEP-compatible benchmark helpers implemented with Ascend NPU events."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


L2_CACHE_FLUSH_BYTES = 256_000_000


def flush_l2_cache(enabled: bool = True) -> None:
    """Flush device L2 by overwriting 256 MB, matching DeepEP's helper."""

    if enabled:
        torch.empty(
            L2_CACHE_FLUSH_BYTES // 4,
            dtype=torch.int32,
            device="npu",
        ).zero_()


def bench(
    fn: Callable[[], Any],
    num_warmups: int = 50,
    num_tests: int = 50,
    post_fn: Optional[Callable[[], Any]] = None,
    flush_l2: bool = True,
) -> Tuple[float, float, float]:
    """Benchmark ``fn`` with the DeepEP ``bench`` signature and return seconds."""

    if num_warmups < 0:
        raise ValueError("num_warmups must be greater than or equal to zero")
    if num_tests < 2:
        raise ValueError("num_tests must be at least two")

    torch.npu.synchronize()
    for _ in range(num_warmups):
        fn()

    start_events = [torch.npu.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.npu.Event(enable_timing=True) for _ in range(num_tests)]
    for index in range(num_tests):
        flush_l2_cache(flush_l2)
        start_events[index].record()
        fn()
        end_events[index].record()
        if post_fn is not None:
            post_fn()

    torch.npu.synchronize()
    times = [start.elapsed_time(end) / 1e3 for start, end in zip(start_events, end_events)][1:]
    return sum(times) / len(times), min(times), max(times)


class empty_suppress:
    """No-op context manager matching DeepEP's testing helper."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None


class suppress_stdout_stderr:
    """Temporarily suppress Python and native stdout/stderr output."""

    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        self.stdout_fd = sys.stdout.fileno()
        self.stderr_fd = sys.stderr.fileno()
        self.saved_stdout_fd = os.dup(self.stdout_fd)
        self.saved_stderr_fd = os.dup(self.stderr_fd)
        os.dup2(self.outnull_file.fileno(), self.stdout_fd)
        os.dup2(self.errnull_file.fileno(), self.stderr_fd)
        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        os.dup2(self.saved_stdout_fd, self.stdout_fd)
        os.dup2(self.saved_stderr_fd, self.stderr_fd)
        os.close(self.saved_stdout_fd)
        os.close(self.saved_stderr_fd)
        self.outnull_file.close()
        self.errnull_file.close()
        return None


def _trace_events(trace_path: str):
    profile_data = json.loads(Path(trace_path).read_text())
    if isinstance(profile_data, dict):
        return profile_data.get("traceEvents", [])
    if isinstance(profile_data, list):
        return profile_data
    raise RuntimeError("unsupported torch_npu Chrome trace format")


def _is_npu_kernel_event(event: Any) -> bool:
    if not isinstance(event, dict) or event.get("ph") != "X":
        return False
    if not isinstance(event.get("dur"), (int, float)):
        return False
    args = event.get("args") or {}
    task_type = str(args.get("Task Type", "")).upper()
    category = str(event.get("cat", "")).upper()
    return task_type.startswith("KERNEL") or "NPU" in category


def bench_kineto(
    fn: Callable[[], Any],
    kernel_names: Union[str, tuple],
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: Optional[str] = None,
    flush_l2: bool = True,
    barrier_comm_profiling: bool = False,
    num_kernels_per_period: int = 1,
    barrier: Optional[Callable[[], Any]] = None,
):
    """Profile named NPU kernels with DeepEP's ``bench_kineto`` API.

    Kernel durations are returned in seconds. A tuple of names produces a list
    of results, matching DeepEP; one string produces one scalar result.
    """

    if not isinstance(kernel_names, (str, tuple)):
        raise TypeError("kernel_names must be a string or tuple of strings")
    names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    if not names or not all(isinstance(name, str) for name in names):
        raise TypeError("kernel_names must contain non-empty strings")
    if any(not name for name in names):
        raise ValueError("kernel names must not be empty")
    if num_tests <= 0:
        raise ValueError("num_tests must be greater than zero")
    if num_kernels_per_period <= 0:
        raise ValueError("num_kernels_per_period must be greater than zero")

    # Retain DeepEP's profiler-conflict escape hatch for drop-in test scripts.
    if int(os.environ.get("EP_USE_NVIDIA_TOOLS", 0)):
        return [1] * len(names) if isinstance(kernel_names, tuple) else 1

    fn()
    torch.npu.synchronize()
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    temporary_trace_dir = None
    if trace_path is None:
        temporary_trace_dir = tempfile.TemporaryDirectory()
        profile_path = str(Path(temporary_trace_dir.name) / "trace.json")
    else:
        profile_path = str(Path(trace_path).expanduser().resolve())

    barrier_comm_profiling = barrier_comm_profiling and int(os.environ.get("EP_DISABLE_BARRIER_PROFILING", 0)) == 0
    barrier_tensor = (
        torch.ones(1, dtype=torch.float32, device="npu") if barrier_comm_profiling and barrier is None else None
    )
    with suppress():
        profiler = torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU])
        with profiler:
            for _ in range(num_tests):
                flush_l2_cache(flush_l2)
                if barrier_comm_profiling:
                    if barrier is None:
                        dist.all_reduce(barrier_tensor)
                    else:
                        barrier()
                fn()
            torch.npu.synchronize()
        profiler.export_chrome_trace(profile_path)

    events = [event for event in _trace_events(profile_path) if _is_npu_kernel_event(event)]
    if temporary_trace_dir is not None:
        temporary_trace_dir.cleanup()

    kernel_durations = []
    for name in names:
        matching = sorted(
            (event for event in events if name in str(event.get("name", ""))),
            key=lambda event: event.get("ts", 0),
        )
        durations = [float(event["dur"]) / 1e6 for event in matching]
        if not durations:
            kernel_durations.append(0)
            continue
        if num_kernels_per_period > 1:
            if len(durations) % num_kernels_per_period:
                raise AssertionError(
                    f"kernel {name!r} count {len(durations)} is not divisible by "
                    f"num_kernels_per_period={num_kernels_per_period}"
                )
            periods = len(durations) // num_kernels_per_period
            kernel_durations.append(
                [sum(durations[index::num_kernels_per_period]) / periods for index in range(num_kernels_per_period)]
            )
        else:
            kernel_durations.append(sum(durations) / len(durations))
    return kernel_durations if isinstance(kernel_names, tuple) else kernel_durations[0]


__all__ = [
    "L2_CACHE_FLUSH_BYTES",
    "bench",
    "bench_kineto",
    "empty_suppress",
    "flush_l2_cache",
    "suppress_stdout_stderr",
]
