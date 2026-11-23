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


@pytest.mark.parametrize(
    "hosts,expected_group,known",
    [
        (["a"] * 8, 8, True),
        (["a"] * 4 + ["b"] * 4, 4, True),
        ([None] * 8, 8, False),
    ],
)
def test_store_topology_runs_real_auto_policy_and_restores_collective(hosts, expected_group, known):
    from tests.api.test_clos_compatibility import run_host

    run_host(f"""
        import importlib.util
        from unittest.mock import patch
        from pathlib import Path
        from deep_ep import _machine
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        dist = ModuleType('torch.distributed')
        dist.ProcessGroup = type('ProcessGroup', (), {{}})
        dist.get_world_size = lambda g: 2
        dist.all_gather_object = lambda rows, record, group: rows.__setitem__(slice(None), [record] * 2)
        sys.modules['torch.distributed'] = torch.distributed = dist
        spec = importlib.util.spec_from_file_location('fused', 'benchmarks/clos/bench_fused_notify.py')
        fused = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fused)
        records = []
        def gather(record):
            records.append(record)
            return {hosts!r}
        group = SimpleNamespace(gather=gather)
        policy = resolve_udma_policy(True, 8, 'udma', None, None, None, None, None)
        original = _machine.gather_records
        # Reproduce the reported failure before installing the adapter.
        try: policy.bind_group(group)
        except TypeError as e: assert 'EP ProcessGroup or MPI' in str(e)
        else: raise AssertionError('unsupported StoreGroup unexpectedly accepted')
        boot = '11111111-1111-1111-1111-111111111111'
        with fused.store_topology_collective(group), patch.object(Path, 'read_text', return_value=boot):
            bound = policy.bind_group(group)
            assert records == [boot]  # actual host probe, not a guessed rank count
            assert bound.ranks_per_group == {expected_group} and bound.topology_known is {known}
            assert _machine.gather_records(dist.ProcessGroup(), 'foreign') == ['foreign', 'foreign']
        assert _machine.gather_records is original
        class InitFailed(Exception): pass
        try:
            with fused.store_topology_collective(group): raise InitFailed
        except InitFailed: pass
        assert _machine.gather_records is original
    """)


@pytest.mark.parametrize("mode_name", list(bench.MODES))
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("tokens", [0, 129])
def test_public_dispatch_routes_layout_and_preserves_eight_channels(mode_name, cached, tokens):
    from tests.api.test_clos_compatibility import run_host

    mode, host, plan = bench.MODES[mode_name]
    run_host(f"""
        import os
        from contextlib import nullcontext
        from deep_ep.backends.direct import elastic as m
        from deep_ep.backends.direct.udma_policy import resolve_udma_policy
        class Submitted(Exception): pass
        class Tensor:
            def __init__(self, shape, dtype="torch.bfloat16", device="npu"):
                self.shape, self.dtype, self.device = shape, dtype, device
            def size(self, dim): return self.shape[dim]
            def element_size(self): return 2
        torch.bool = 'bool'
        allocations, layouts, native_allocations = [], [], []
        def empty(shape, **kw):
            allocations.append(tuple(shape))
            return Tensor(shape, **kw)
        torch.empty = empty
        os.environ['DEEPEP_CLOS_FUSED_LAYOUT_NOTIFY'] = '{mode}'
        os.environ['DEEPEP_CLOS_NOTIFY_HOST_PREP'] = '{host}'
        os.environ['DEEPEP_CLOS_NOTIFY_DEFER_PLAN'] = '{plan}'
        os.environ['DEEPEP_CLOS_DISPATCH_HOST_TIMING'] = '0'
        for name in ('_validate_dispatch_scope', '_validate_dispatch_tensors'):
            setattr(m, name, lambda *a: None)
        m._get_allocation_stream_id = lambda _: 0
        m._try_record_event = lambda _: None
        m._current_npu_stream = lambda: 'caller'
        m._use_stream_after_event = lambda *a: nullcontext()
        m._use_comm_stream = lambda *a: nullcontext()
        m.load_native_module = lambda: SimpleNamespace(fused_layout_notify_version=1, fused_notify_pipeline_version=1)
        x, topk = Tensor(({tokens},2816)), Tensor(({tokens},8))
        mask = Tensor(({tokens},8))
        def layout(*a):
            layouts.append(a)
            return Tensor((8,)), None, Tensor((256,)), mask
        def allocate(*a):
            assert a == (topk, 256, 0)
            native_allocations.append(a)
            return Tensor((8,)), None, Tensor((256,)), mask
        def submit(*a):
            assert a[12] == 8
            assert a[5].shape == ({tokens},8)
            if not {cached}:
                assert a[4].shape == (8,) and a[6].shape == (256,)
            else:
                assert a[4] is None and a[6] is None
            raise Submitted
        policy = resolve_udma_policy(True, 8, 'udma', None, None, None, None, None)
        obj = SimpleNamespace(_ensure_alive=lambda: None, _validate_handle=lambda _: None,
            use_fp8_dispatch=False, num_max_tokens_per_rank=129, num_ranks=8, hidden=2816, num_topk=8,
            get_buffer_size_hint=lambda *a: 0, group=None, allow_hybrid_mode=True,
            num_bytes=1, num_device_aiv=64, _udma_auto_policy=policy, transport='udma',
            _communication_stream='comm', _use_single_source_combine=False,
            _native=SimpleNamespace(get_dispatch_layout=layout, allocate_dispatch_layout=allocate,
                intranode_dispatch_with_plan=submit, intranode_dispatch_with_handle=submit))
        handle = SimpleNamespace(num_sms=8, num_experts=256, num_max_tokens_per_rank=129,
            expert_alignment=1, do_expand=False, topk_idx=topk, num_recv_tokens=0,
            _dispatch_layout=(object(),object(),None,None,mask,None), _dispatch_plan=object())
        try:
            m.ElasticBuffer.dispatch(obj, x, handle=handle if {cached} else None,
                topk_idx=None if {cached} else topk, num_experts=256, num_sms=8, do_expand=False)
        except Submitted: pass
        else: raise AssertionError('dispatch never reached native submission')
        assert len(layouts) == int(not {cached} and {mode} == 0)
        assert allocations == ([(8,), (256,), ({tokens},8)] if not {cached} and {mode} and not {host} else [])
        assert len(native_allocations) == int(not {cached} and {host})
    """)


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


def test_sweep_is_forward_reverse_and_resets_all_options(monkeypatch):
    for env in (bench.ENV, bench.HOST_PREP_ENV, bench.DEFER_PLAN_ENV, bench.SIMT_LAYOUT_ENV):
        monkeypatch.setenv(env, "stale")
    for sweep in (False, True):
        order = bench.mode_order(sweep)
        assert order == order[::-1]
        assert len(order) == (10 if sweep else 4)
        for name in order:
            bench.set_mode(name)
            assert (
                tuple(int(os.environ[e]) for e in (bench.ENV, bench.HOST_PREP_ENV, bench.DEFER_PLAN_ENV))
                == bench.MODES[name]
            )
            assert os.environ[bench.SIMT_LAYOUT_ENV] == "0"
    assert bench.mode_order(False, True) == ["baseline", "simt", "simt", "baseline"]
    bench.set_mode("simt")
    assert os.environ[bench.SIMT_LAYOUT_ENV] == "1"
    assert os.environ[bench.ENV] == "0"


@pytest.mark.parametrize("fused,simt", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("guards", [0, 2])
def test_profile_sequence_excludes_outer_barrier(tmp_path, fused, simt, expanded, guards):
    names = ["cached_notify_combine_kernel"]
    if not fused:
        names.append("dispatch_layout_simt_kernel" if simt else "dispatch_layout_kernel")
    names += ["notify_dispatch_kernel", "v1_dispatch_udma_kernel"]
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


@pytest.mark.parametrize("world", [2, 4, 8, 16, 32, 64, 128, 256, 1024])
@pytest.mark.parametrize("k", [1, 8, 10, 128])
@pytest.mark.parametrize("experts", [256, 512, 4096])
def test_ub_budget_and_tile_tail(world, k, experts):
    # Check the planned worst-case footprints, not any hardware performance.
    ub_bytes = 224 * 1024 - 8192 - 64
    topk_offset, histogram_offset = 106496, 139264
    shards = min(32, (ub_bytes - histogram_offset) // (experts * 4))
    assert shards >= 1
    tile = min(128, 65536 // world, 32768 // (k * 8)) // 8 * 8
    assert tile > 0
    assert 4096 + tile * world <= 69632
    assert 69632 + world * ((tile + 7) // 8) * 4 <= topk_offset
    assert topk_offset + tile * k * 8 <= histogram_offset
    assert histogram_offset + shards * experts * 4 <= ub_bytes
