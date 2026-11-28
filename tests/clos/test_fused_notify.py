"""CPU source-body oracle and strict profiling-accounting checks."""

import csv
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("fused_bench", ROOT / "benchmarks/clos/bench_fused_notify.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def host_fields():
    return [
        "api_entry",
        "native_entry",
        "notify_submit_begin",
        "notify_submitted",
        "count_wait_begin",
        "total_observed",
        "counts_observed",
        "output_ready",
        "comm_submit_begin",
        "comm_submitted",
        "native_return",
        "api_return",
        "complete",
        "fused",
        "host_prep",
        "deferred_plan",
        "mapped_counts",
    ]


@pytest.mark.parametrize("mode", list(bench.MODES))
def test_host_intervals_are_same_call_and_native_flags_are_checked(mode):
    fields = host_fields()
    row = [1000 * (i + 1) for i in range(13)] + list(bench.MODES[mode]) + [1]
    result = bench.host_durations([row], fields, 1, mode)
    assert result["api_to_notify_submit"] == [3.0]
    assert result["post_launch_prepare"] == [1.0]
    assert result["wait_total"] == [1.0]
    assert result["wait_remaining_counts"] == [1.0]
    assert result["comm_submit_to_complete"] == [3.0]
    assert result["api_complete_diagnostic"] == [12.0]
    with pytest.raises(RuntimeError, match="expected"):
        bench.host_durations([row], fields, 2, mode)
    with pytest.raises(RuntimeError, match="schema"):
        bench.host_durations([row[:-1]], fields, 1, mode)
    wrong = row.copy()
    wrong[13] = 1 - wrong[13]
    with pytest.raises(RuntimeError, match="flags"):
        bench.host_durations([wrong], fields, 1, mode)
    for timestamp in (0, 999999):
        wrong = row.copy()
        wrong[5] = timestamp
        with pytest.raises(RuntimeError, match="timestamp"):
            bench.host_durations([wrong], fields, 1, mode)


@pytest.mark.parametrize("fused,simt", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("guards", [0, 2])
def test_profile_sequence_excludes_outer_barrier(tmp_path, fused, simt, expanded, guards):
    names = ["cached_notify_combine_kernel"]
    if not fused:
        names.append("dispatch_layout_simt_kernel" if simt else "dispatch_layout_kernel")
    names += ["notify_dispatch_kernel", "direct_dispatch_udma_kernel"]
    if expanded:
        names.append("expand_kernel")
    path = tmp_path / "kernel_details.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Name", "Start Time(us)", "Duration(us)"])
        writer.writeheader()
        for i, name in enumerate(names * (3 + 2 * guards)):
            measured = guards <= i // len(names) < guards + 3
            writer.writerow(
                {
                    "Name": name,
                    "Start Time(us)": i * 1000,
                    "Duration(us)": 900 if "combine" in name else (10 if measured else 777),
                }
            )
    result = bench.parse_capture(tmp_path, fused, 3, expanded, guards, simt_layout=simt)
    assert result["notify_dispatch_kernel"] == [10] * 3
    assert ("dispatch_layout_kernel" in result) != fused
    with pytest.raises(RuntimeError, match="sequence"):
        bench.parse_capture(tmp_path, fused, 2, expanded, guards, simt_layout=simt)
    if simt:
        with pytest.raises(RuntimeError, match="sequence"):
            bench.parse_capture(tmp_path, False, 3, expanded, guards)
    if fused:
        with pytest.raises(RuntimeError, match="sequence"):
            bench.parse_capture(tmp_path, False, 3, expanded, guards)


def vf_body(name):
    src = (ROOT / "csrc/direct/kernels/notify/notify_fused_layout.h").read_text()
    start = src.index("inline void " + name)
    brace = src.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (src[end] == "{") - (src[end] == "}")
        end += 1
    return src[start:end]


def test_deferred_plan_production_builders_and_host_clock(tmp_path):
    from tests.utils.cpp import cpp_functions

    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++17 compiler required for deferred plan CPU oracle")
    source = ROOT / "csrc/direct"
    plan = (source / "kernels/notify/notify_dispatch_plan.h").read_text()
    fused = (source / "kernels/notify/notify_fused_layout.h").read_text()
    functions = plan[plan.index("constexpr uint32_t kPlanCountBytes") : plan.index("#if defined")]
    functions += cpp_functions(
        plan, "CountDispatchChunksVf", "PrefixDispatchChunksVf", "ScatterDispatchPlanVf", "BuildDispatchPlan"
    )
    functions += cpp_functions(
        fused, "DecodeFusedRoutesVf", "CountFusedExpertsVf", "StoreFusedExpertsVf", "BuildFusedDispatchPlan"
    )
    harness = (
        Path(__file__).with_name("fused_pipeline_oracle.cpp").read_text().replace("// PRODUCTION_FUNCTIONS", functions)
    )
    path = tmp_path / "pipeline.cpp"
    path.write_text(harness)
    executable = tmp_path / ("pipeline.exe" if os.name == "nt" else "pipeline")
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-I", str(source), str(path), "-o", str(executable)], check=True, timeout=180
    )
    try:
        subprocess.run([str(executable)], check=True, timeout=60)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            pytest.skip("compiled, but Windows application control blocked oracle execution (4551)")
        raise


def test_actual_vf_bodies_against_histogram_oracle(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++ compiler required for production VF body CPU emulation")
    functions = "\n".join(
        vf_body(n)
        for n in ("DecodeFusedRoutesVf", "CountFusedExpertsVf", "StoreFusedExpertsVf", "ReduceFusedPartialsVf")
    )
    program = (
        r"""
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#define __ubuf__
#define __gm__
using topk_idx_t = int64_t;
struct { unsigned x; } threadIdx;
inline void asc_threadfence() {}
namespace __asc_simt_vf { inline void __trap() { std::abort(); } }
"""
        + functions
        + r"""
int main() {
  for (unsigned world : {2u,4u,8u,16u,128u,256u})
  for (unsigned experts : {256u,512u})
  for (unsigned shards : {1u,7u,32u})
  for (unsigned k : {1u,8u,10u}) {
    std::vector<int32_t> hist(shards*experts,0), expected(experts,0), actual(experts,-1);
    for (unsigned tokens : {0u,1u,65u,128u,129u}) {
      std::vector<topk_idx_t> idx(tokens*k);
      std::vector<uint8_t> routes(tokens*world,99), want(tokens*world,0);
      for (unsigned i=0; i<idx.size(); ++i) {
        idx[i] = i%11 == 0 ? topk_idx_t(-1) : topk_idx_t(i%3 == 0 ? 0 : (i*17)%experts);
        if (idx[i]>=0) { ++expected[idx[i]]; want[(i/k)*world+idx[i]/(experts/world)]=1; }
      }
      for (threadIdx.x=0;threadIdx.x<128;++threadIdx.x)
        DecodeFusedRoutesVf(idx.data(),routes.data(),tokens,k,experts,world);
      assert(routes==want);
      for (threadIdx.x=0;threadIdx.x<128;++threadIdx.x)
        CountFusedExpertsVf(idx.data(),hist.data(),tokens,k,experts,experts,shards);
    }
    for (threadIdx.x=0;threadIdx.x<128;++threadIdx.x)
      StoreFusedExpertsVf(hist.data(),actual.data(),experts,shards);
    assert(actual==expected);
    std::fill(actual.begin(),actual.end(),-1);
    for (threadIdx.x=0;threadIdx.x<128;++threadIdx.x)
      ReduceFusedPartialsVf(hist.data(),actual.data(),shards,experts);
    assert(actual==expected);
  }
}
"""
    )
    source = tmp_path / "oracle.cpp"
    source.write_text(program)
    executable = tmp_path / ("oracle.exe" if os.name == "nt" else "oracle")
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(executable)], check=True, timeout=180)
    try:
        subprocess.run([str(executable)], check=True, timeout=30)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            pytest.skip("compiled, but Windows application control blocked harness execution (4551)")
        raise
