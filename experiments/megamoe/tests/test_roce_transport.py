# Copyright (c) 2026, Lu Lu

"""Compile the real transport helper against asynchronous SHMEM/event doubles.

These checks cover API use and protocol sequencing, not NPU/NIC visibility.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("backend", ["XSCALE", "HNS_1825", None], ids=["xscale", "hns-1825", "mte-only"])
def test_roce_transport_resource_and_completion_contract(tmp_path, backend):
    compile_and_run(tmp_path, backend, "roce_transport_test.cpp")


def test_hns_transport_host_pass_uses_compatible_signal_api(tmp_path):
    compile_and_run(tmp_path, "HNS_1825", "roce_host_pass_test.cpp")


@pytest.mark.parametrize("backend", ["XSCALE", "HNS_1825"])
def test_reduction_queues_complete_independently_and_preserve_sources(tmp_path, backend):
    compile_and_run(tmp_path, backend, "roce_multi_qp_test.cpp")


@pytest.mark.parametrize("backend", ["XSCALE", "HNS_1825"])
def test_hierarchical_dispatch_full_batch_and_completion_contract(tmp_path, backend):
    compile_and_run(
        tmp_path,
        backend,
        "hierarchical_dispatch_test.cpp",
        stub_directories=("dispatch_stubs", "roce_stubs", "layout_stubs"),
        run_timeout=30,  # BS256 also compares the full production heap and untouched regions.
    )


@pytest.mark.parametrize("backend", ["XSCALE", "HNS_1825"])
@pytest.mark.parametrize("timer", [False, True], ids=["timer-off", "timer-on"])
def test_rdma_timer_preserves_compute_and_transport(tmp_path, backend, timer):
    compile_and_run(tmp_path, backend, "rdma_timer_test.cpp", timer)


@pytest.mark.parametrize("timer", [False, True], ids=["timer-off", "timer-on"])
def test_startup_timer_closes_at_first_work_without_extra_barriers(tmp_path, timer):
    compile_and_run(tmp_path, None, "startup_timer_test.cpp", timer)


@pytest.mark.parametrize("timer", [False, True], ids=["timer-off", "timer-on"])
def test_combine_progress_timer_preserves_state_and_excludes_paused_work(tmp_path, timer):
    compile_and_run(tmp_path, None, "combine_progress_timer_test.cpp", timer)


@pytest.mark.parametrize("timer", [False, True], ids=["timer-off", "timer-on"])
def test_startup_detail_timer_preserves_spans_and_does_not_add_barriers(tmp_path, timer):
    compile_and_run(tmp_path, None, "startup_detail_timer_test.cpp", timer)


def compile_and_run(tmp_path, backend, source_name, timer=False, stub_directories=("roce_stubs",), run_timeout=10):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Transport helper compilation requires g++ or clang++")
    source = Path(__file__).with_name(source_name)
    binary = tmp_path / "roce_transport_test"
    command = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror"]
    if backend is not None:
        command.append(f"-DACLSHMEMI_RDMA_K_BACKEND_{backend}=1")
    if timer:
        command.append("-DDEEPEP_MEGAMOE_TIMER=1")
    command += [f"-I{Path(__file__).with_name(directory)}" for directory in stub_directories]
    command += [str(source), "-o", str(binary)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=run_timeout)
    assert result.returncode == 0, result.stderr
