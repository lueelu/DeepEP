"""Check benchmark precision, failure cleanup and profiler stage accounting."""

import csv
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import struct
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/clos"))
try:
    spec = importlib.util.spec_from_file_location("dispatch_stages", ROOT / "benchmarks/clos/bench_dispatch_stages.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
finally:
    sys.path.pop(0)


def test_cached_weight_generations_use_one_fp32_addition():
    def fp32(value):
        return struct.unpack("f", struct.pack("f", value))[0]

    class Weight:
        additions = []

        def __init__(self, value):
            self.value = fp32(value)

        def __add__(self, offset):
            self.additions.append(offset)
            return Weight(self.value + offset)

    # An FP32 witness where rounding after each addition changes the answer.
    w = Weight(5 * 2**-25)
    assert ((w + 1) + 1).value != (w + 2).value
    for generation in range(2):
        Weight.additions.clear()
        result = bench.offset_weights(w, generation + 1)
        assert Weight.additions == [generation + 1]
        assert result.value == fp32(w.value + (generation + 1))
    Weight.additions.clear()
    assert bench.offset_weights(None, 2) is None
    assert not Weight.additions


def test_weight_input_and_reference_keep_zero_tolerance():
    tree = ast.parse((ROOT / "benchmarks/clos/bench_dispatch_stages.py").read_text())
    assertions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "assert_close"
    ]
    assert assertions
    for call in assertions:
        tolerances = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg in ("rtol", "atol")}
        assert tolerances == {"rtol": 0, "atol": 0}
    (cached,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "cached_w" for t in node.targets)
    ]
    assert ast.unparse(cached.value.body) == "offset_weights(dw, generation + 1)"


@pytest.mark.parametrize("failure", [None, "precision", "agreement"])
def test_benchmark_session_only_destroys_after_successful_agreement(monkeypatch, failure):
    calls = []

    class Buffer:
        def destroy(self):
            calls.append("destroy")

    class Group:
        def agree(self, value):
            calls.append(value)
            if failure == "agreement":
                raise RuntimeError("peer failed")

    envs = (
        *bench.OVERLAP_ENVS,
        *bench.METADATA_OUTPUT_ENVS,
        bench.TRACE_ENV,
        "DEEPEP_CLOS_COMBINE_DEVICE_TIMING",
        bench.METADATA_ENV,
        bench.METADATA_TILE_ENV,
    )
    for name in envs:
        monkeypatch.setenv(name, "1")

    def work():
        with bench.benchmark_session(Buffer(), Group()):
            calls.append("work")
            if failure == "precision":
                raise AssertionError("original precision error")

    if failure:
        with pytest.raises((AssertionError, RuntimeError), match="original precision error|peer failed"):
            work()
    else:
        work()
    assert (
        calls
        == {
            None: ["work", "ready_to_destroy", "destroy"],
            "precision": ["work"],
            "agreement": ["work", "ready_to_destroy"],
        }[failure]
    )
    assert os.environ[bench.OVERLAP_ENVS[0]] == "1"
    assert all(os.environ[name] == "0" for name in envs if name != bench.OVERLAP_ENVS[0])


def test_failed_worker_preserves_traceback_without_collective_or_destructors(tmp_path):
    # A real Python subprocess is essential: mocking os._exit would unwind
    # frames and run exactly the destructors this failure path must avoid.
    program = r"""
import atexit
import sys
sys.path.insert(0, sys.argv[1])
from bench_dispatch_stages import benchmark_session, run_worker
atexit.register(lambda: print("UNEXPECTED_ATEXIT"))
class Buffer:
    def destroy(self):
        raise RuntimeError("UNEXPECTED_DESTROY")
    def __del__(self):
        print("UNEXPECTED_DESTRUCTOR", flush=True)
class Group:
    def agree(self, value):
        raise RuntimeError("UNEXPECTED_COLLECTIVE")
def work():
    buffer = Buffer()
    print("worker diagnostic before failure")
    with benchmark_session(buffer, Group()):
        raise AssertionError("original precision mismatch")
    print("UNEXPECTED_SUCCESS")
run_worker(work)
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(ROOT / "benchmarks/clos")],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 1
    assert "Traceback (most recent call last)" in result.stderr
    assert "AssertionError: original precision mismatch" in result.stderr
    assert "worker diagnostic before failure" in result.stdout
    assert "UNEXPECTED_" not in result.stdout + result.stderr


@pytest.mark.parametrize("packed", [False, True])
def test_cached_weighted_profiler_accounts_for_plan_rebuild(tmp_path, packed):
    notify = "cached_notify_dispatch_kernel"
    names = ["cached_notify_combine_kernel", notify]
    if not packed:
        names.append("cached_dispatch_plan_kernel")
    names.append("direct_dispatch_udma_kernel")
    costs = {
        "cached_notify_combine_kernel": 999,
        notify: 20,
        "cached_dispatch_plan_kernel": 30,
        "direct_dispatch_udma_kernel": 40,
    }
    with (tmp_path / "kernel_details.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Start Time(us)", "Duration(us)"])
        for i, name in enumerate(names * 6):
            writer.writerow([name, i * 1000, costs[name]])
    result = bench.capture_stages(tmp_path, 2, True, cached_weighted=True, cached_pack=packed)
    assert result["notify_total_us"] == [20 if packed else 50] * 2
    assert result["direct_dispatch_udma_kernel"] == [40, 40]
    with pytest.raises(RuntimeError, match="sequence"):
        bench.capture_stages(tmp_path, 2, True, cached_weighted=True, cached_pack=not packed)


def fixture_trace():
    fields = [
        "version",
        "rank",
        "core",
        "cached",
        "whole_channel",
        "entry_cycles",
        *bench.TIMELINE,
        "src_cycles",
        "topk_cycles",
        "self_cycles",
        "list_cycles",
    ]
    rows = []
    for core in range(8):
        rows.append([1, 3, core, 1, 1, 123456, *range(1000, 12000, 1000), 1500, 2500, 3000, 0])
    return fields, rows


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("simt_layout", [False, True])
def test_capture_excludes_alignment_and_guards(tmp_path, cached, corrupt, simt_layout):
    names = ["cached_notify_combine_kernel"]
    if not cached:
        names.append("dispatch_layout_simt_kernel" if simt_layout else "dispatch_layout_kernel")
    names += ["cached_notify_dispatch_kernel" if cached else "notify_dispatch_kernel", "direct_dispatch_udma_kernel"]
    rows = []
    for sample in range(7):  # 2 guards + 3 samples + 2 guards
        for name in names:
            rows.append({"Name": name, "Start Time(us)": len(rows) * 10, "Duration(us)": sample + 1})
    if corrupt:
        rows.pop(2)
    with (tmp_path / "kernel_details.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if corrupt:
        with pytest.raises(RuntimeError):
            bench.capture_stages(tmp_path, 3, cached)
    else:
        result = bench.capture_stages(tmp_path, 3, cached)
        assert "cached_notify_combine_kernel" not in result
        assert result["direct_dispatch_udma_kernel"] == [3, 4, 5]


@pytest.mark.parametrize("independent_notify", [False, True])
@pytest.mark.parametrize("archived_names", [False, True])
def test_combine_profiler_excludes_alignment_and_checks_sequence(tmp_path, independent_notify, archived_names):
    import csv

    sys.path.insert(0, str(ROOT / "benchmarks/clos"))
    try:
        spec = importlib.util.spec_from_file_location(
            "combine_queue", ROOT / "benchmarks/clos/combine_queue_diagnostics.py"
        )
        bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench)
    finally:
        sys.path.pop(0)
    names = ["cached_notify_combine_kernel", "direct_combine_udma_comm_kernel", "direct_combine_udma_reduce_kernel"]
    sequence = ([names[0]] + names) if independent_notify else names
    durations = [999, 7, 20, 30] if independent_notify else [999, 20, 30]
    path = tmp_path / "kernel_details.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Start Time(us)", "Duration(us)"])
        for i, name in enumerate(sequence * 6):
            writer.writerow(
                [name.replace("direct_", "v1_", 1) if archived_names else name, i * 100, durations[i % len(sequence)]]
            )
    result = bench.capture_combine_stages(tmp_path, 2)
    expected = {names[1]: [20, 20], names[2]: [30, 30], "alignment_notify_us": [999, 999]}
    if independent_notify:
        expected[names[0]] = [7, 7]
    assert result == expected
    with pytest.raises(RuntimeError, match="sequence"):
        bench.capture_combine_stages(tmp_path, 3)
