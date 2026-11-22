"""Validate diagnostic accounting independently of device clocks/profiler runtime."""

import csv
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests.utils.cpp import cpp_functions

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/clos"))
try:
    spec = importlib.util.spec_from_file_location("dispatch_stages", ROOT / "benchmarks/clos/bench_dispatch_stages.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
finally:
    sys.path.pop(0)


def test_post_submit_slots_and_private_ub(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    source = ROOT / "csrc/direct"
    metadata = (source / "kernels/dispatch/udma_dispatch_metadata.h").read_text()
    callback = metadata[
        metadata.index("template <class Resource, class Trace>\nstruct DispatchOverlapWork") : metadata.rindex("#endif")
    ]
    slots = cpp_functions((source / "kernels/common/v2_handle.h").read_text(), "V2HandleSlotsVf")
    metadata_signature = cpp_functions(metadata, "DispatchMetadata").split("{", 1)[0]
    program = r"""
#include <cassert>
#include <cstdint>
#include <vector>
#define __aicore__
#define __simt_vf__
#define __launch_bounds__(n)
#define __gm__
#define ASCEND_DEEPEP_ARCH 3510
#define ASCEND_DEEPEP_DEVICE_SCHEDULE
#include "common/types.h"
#include "common/platform.h"
#include "common/udma_schedule.h"
using namespace ascend_deepep;
using namespace ascend_deepep::udma;
using GM_ADDR = uint8_t*;
constexpr int EVENT_ID6=6;
struct { unsigned x; } threadIdx;
struct dim3 { unsigned x; explicit dim3(unsigned v):x(v){} };
void asc_threadfence() {}
namespace AscendC {
enum class HardEvent { V_S };
bool pending=false;
template<HardEvent> void SetFlag(int) { assert(!pending); pending=true; }
template<HardEvent> void WaitFlag(int) { assert(pending); pending=false; }
}
template<auto Fn, class... Args> void asc_vf_call(dim3 n, Args... args) {
    for(threadIdx.x=0;threadIdx.x<n.x;++threadIdx.x)Fn(args...);
}
struct Resource { unsigned calls=0, offset=0; };
struct Trace {
    static constexpr int SlotsCycles=1;
    unsigned slots=0;
    uint64_t Now() { return 1; }
    void Add(int field, uint64_t) { assert(field==SlotsCycles); ++slots; }
};
// SLOTS
// METADATA_SIGNATURE
{
    assert(copy_self && !publish_metadata);
    assert(stage_bytes + scratch_offset == platform::kKernelUbBytes);
    assert(scratch_offset % 32 == 0 && stage_bytes >= 32768);
    ++resource.calls;
    resource.offset=scratch_offset;
}
// CALLBACK
int main() {
    static_assert(DispatchLiveSendUbBytes(8,1)==12832);
    static_assert(DispatchLiveSendUbBytes(64,2)==93216);
    constexpr unsigned T=65, K=3, W=8, C=8, capacity=128, rank=3;
    std::vector<topk_idx_t> idx(T*K);
    std::vector<int32_t> head(T*W,-1), prefix(W*C,0), output(T*K,-99), expected(T*K,-1);
    unsigned counts[W]{};
    for(unsigned t=0;t<T;++t) {
        // Two routes to one peer; an invalid slot; all tokens use peer 2.
        idx[t*K]=4; idx[t*K+1]=5; idx[t*K+2]=-1;
        const unsigned ch=t/((T+C-1)/C);
        head[t*W+2]=t%((T+C-1)/C);
        ++counts[2];
        for(unsigned p=0;p<W;++p)prefix[p*C+ch]=counts[p];
        expected[t*K]=rank*capacity+t;
    }
    DispatchTilingData td{};
    td.rank=rank; td.world_size=W; td.num_max_tokens_per_rank=capacity*W;
    td.num_local_experts=2; td.num_topk=K; td.num_tokens=T; td.num_channels=C;
    td.v2_dst_slots=reinterpret_cast<uint64_t>(output.data());
    for(unsigned mask=0;mask<4;++mask) {
        td.dispatch_self_after_submit=mask&1; td.dispatch_early_slots=(mask>>1)&1;
        std::fill(output.begin(),output.end(),-99);
        unsigned copies=0,slots=0;
        for(unsigned core=0;core<8;++core) {
            Resource resource; Trace trace;
            DispatchOverlapWork<Resource,Trace> work{resource,trace,nullptr,nullptr,
                reinterpret_cast<GM_ADDR>(idx.data()),nullptr,nullptr,nullptr,
                reinterpret_cast<GM_ADDR>(prefix.data()),nullptr,reinterpret_cast<GM_ADDR>(head.data()),
                nullptr,nullptr,td,core,8};
            work(DispatchLiveSendUbBytes(8,1));
            copies+=resource.calls; slots+=trace.slots;
            assert(!AscendC::pending);
        }
        assert(copies==((mask&1)?8U:0U));
        assert(slots==((mask&2)?8U:0U));
        if(mask&2) assert(output==expected);
        else for(auto v:output)assert(v==-99);
    }
}
"""
    path = tmp_path / "overlap.cpp"
    path.write_text(
        program.replace("// SLOTS", slots)
        .replace("// METADATA_SIGNATURE", metadata_signature)
        .replace("// CALLBACK", callback)
    )
    executable = tmp_path / ("overlap.exe" if os.name == "nt" else "overlap")
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-I", str(source), str(path), "-o", str(executable)], check=True, timeout=180
    )
    try:
        subprocess.run([str(executable)], check=True, timeout=60)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            pytest.skip("compiled, but Windows application control blocked execution (4551)")
        raise


@pytest.mark.parametrize("drop_peer_trace", [False, True])
def test_dispatch_calls_compile_against_actual_helper_signatures(tmp_path, drop_peer_trace):
    """Compile both routing branches; no NPU SDK or executable launch required."""
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required for dispatch call-signature regression")
    kernels = ROOT / "csrc/direct/kernels/dispatch"
    declarations = []
    for filename, name in (
        ("udma_dispatch_metadata.h", "DispatchMetadata"),
        ("udma_dispatch_schedule.h", "DispatchRemote"),
        ("udma_token_dispatch_schedule.h", "DispatchTokenRemote"),
    ):
        definition = cpp_functions((kernels / filename).read_text(), name)
        declarations.append(definition.split("{", 1)[0] + " {}")
    source = (kernels / "dispatch_udma.cpp").read_text()
    start = source.index("    ascend_deepep::udma::DispatchMetadata(")
    calls = source[start : source.index("    const int32_t actual_recv_tokens", start)]
    if drop_peer_trace:
        assert "tiling, core_idx, core_num, trace, overlap);" in calls
        calls = calls.replace("tiling, core_idx, core_num, trace, overlap);", "tiling, core_idx, core_num);")
    program = r"""
#include <cstdint>
#define __aicore__
using GM_ADDR = uint8_t*;
namespace ascend_deepep::udma {
struct DispatchTilingData { bool udma_token_schedule; bool dispatch_self_after_submit = false; };
struct NoDispatchOverlapWork {};
template<class R,class T> struct DispatchOverlapWork {
    R& resource; T& trace;
    GM_ADDR x,scales,topk,weights,plan,ranks,channels,workspace,head,recv_x,recv_scales;
    const DispatchTilingData& td; uint32_t core,cores;
};
// DECLARATIONS
}
struct EnabledTrace { enum { MetadataDone }; void Mark(int) {} };
struct DisabledTrace { enum { MetadataDone }; void Mark(int) {} };
template<class Trace> void check(bool token) {
    int resource = 0;
    GM_ADDR x{}, dispatch_scales{}, topk_idx{}, topk_weights{}, dispatch_plan{}, rank_prefix_matrix{},
        channel_prefix_matrix{}, symmetric_buffer{}, send_head{}, recv_x{}, recv_x_scales{};
    ascend_deepep::udma::DispatchTilingData tiling{token};
    uint32_t core_idx = 0, core_num = 8, kDispatchStageBytes = 65536;
    Trace trace;
    // CALLS
}
template void check<EnabledTrace>(bool);
template void check<DisabledTrace>(bool);
"""
    path = tmp_path / "dispatch_calls.cpp"
    path.write_text(program.replace("// DECLARATIONS", "\n".join(declarations)).replace("// CALLS", calls))
    result = subprocess.run(
        [compiler, "-std=c++17", "-c", str(path), "-o", str(tmp_path / "dispatch_calls.o")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if drop_peer_trace:
        assert result.returncode != 0
        assert "DispatchRemote" in result.stderr
    else:
        assert result.returncode == 0, result.stderr


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


def test_channel_comparison_keeps_notify_comm_and_api_separate():
    rows = []
    for turn, (mode, comm) in enumerate(((0, 100), (1, 80), (1, 82), (0, 102))):
        rows.append(
            dict(
                op="fresh",
                cores=8,
                channel=0,
                turn=turn,
                metadata_requested=mode,
                policy={"visit": 32},
                payload_bytes_per_rank=1000000,
                stats_us={
                    "notify_dispatch_kernel": {"mean": 20 + mode * 10},
                    "v1_dispatch_udma_kernel": {"mean": comm},
                    "api_wall_us": {"mean": 150 - mode * 5},
                },
            )
        )
    (result,) = bench.channel_comparison(rows)
    assert result["pairs"] == 2
    assert result["notify_on_us"] == 30
    assert result["comm_on_us"] == 81
    assert result["api_on_us"] == 145  # Not the sum of separately measured stages.
    assert result["comm_pair_delta_min_us"] == result["comm_pair_delta_max_us"] == -20
    assert result["on_gbps"] == pytest.approx(1000 / 81)
    rows[-1]["policy"] = {"visit": 16}
    with pytest.raises(ValueError, match="unmatched"):
        bench.channel_comparison(rows)


def test_zeng_dispatch_override_preserves_metadata_and_auto_policy():
    from tests.api.test_clos_compatibility import run_host

    run_host("""
        sys.path.insert(0, 'benchmarks/clos')
        from bench_dispatch_stages import DispatchBenchmarkPolicy
        from deep_ep.backends.direct.udma_policy import UdmaPolicy
        base = UdmaPolicy(8, 2, 1, 2, 32, ranks_per_group=8, topology_known=True)
        wrapper = DispatchBenchmarkPolicy(base)
        automatic = wrapper.for_launch(8, 5632, 'dispatch', bf16=True)
        assert automatic.qps_per_worker == 2
        wrapper.variant = 'zeng'
        plan = wrapper.for_launch(8, 5632, 'dispatch', bf16=True)
        assert (plan.peer_order, plan.peers_per_worker, plan.workers_per_peer,
                plan.qps_per_worker, plan.wqes_per_visit, plan.self_copy, plan.weight_mode) == (0,2,1,1,32,0,0)
        assert plan.send_workers == 4
        wrapper.variant = 'auto'
        assert wrapper.for_launch(8, 5632, 'dispatch', bf16=True) == automatic
        wrapper.variant = 'zeng'
        for cores, width, op in [(16,5632,'dispatch'), (8,8192,'dispatch'), (8,5632,'weighted')]:
            try:
                wrapper.for_launch(cores, width, op, bf16=True)
            except ValueError:
                pass
            else:
                raise AssertionError('unsupported fixture must not silently use the Zeng label')
    """)


def test_channel_comparison_separates_tiles_and_payload_plans():
    rows = []
    for plan in ("auto", "zeng"):
        for tile in (32, 128):
            for mode in (0, 1, 1, 0):
                rows.append(
                    dict(
                        op="fresh",
                        cores=8,
                        channel=0,
                        metadata_requested=mode,
                        metadata_tile_requested=tile,
                        dispatch_plan=plan,
                        policy={"plan": plan},
                        payload_bytes_per_rank=1000000,
                        stats_us={
                            name: {"mean": 100 - mode * 10}
                            for name in ("notify_dispatch_kernel", "v1_dispatch_udma_kernel", "api_wall_us")
                        },
                    )
                )
    result = bench.channel_comparison(rows)
    assert len(result) == 4
    assert {(r["dispatch_plan"], r["metadata_tile_requested"]) for r in result} == {
        (plan, tile) for plan in ("auto", "zeng") for tile in (32, 128)
    }
    assert all(r["pairs"] == 2 for r in result)


def test_overlap_comparison_separates_cores_and_rejects_changed_policy():
    rows = []
    for cores in (8, 32):
        for mode in ("off", "all", "all", "off"):
            rows.append(
                dict(
                    op="fresh",
                    cores=cores,
                    channel=1,
                    metadata_requested=1,
                    metadata_tile_requested=128,
                    dispatch_plan="auto",
                    overlap_mode=mode,
                    overlap_effective=15 if mode == "all" else 0,
                    policy={"cores": cores},
                    payload_bytes_per_rank=1000000,
                    stats_us={
                        name: {"mean": 80 if mode == "all" else 100}
                        for name in ("notify_dispatch_kernel", "v1_dispatch_udma_kernel", "api_wall_us")
                    },
                )
            )
    result = bench.overlap_comparison(rows)
    assert len(result) == 2
    assert {r["cores"] for r in result} == {8, 32}
    assert all(r["pairs"] == 2 and r["effective_masks"] == [15] for r in result)
    assert all(r["api_time_change_pct"] == pytest.approx(-20) for r in result)
    rows[1]["policy"] = {"changed": True}
    with pytest.raises(ValueError, match="unmatched"):
        bench.overlap_comparison(rows)


def test_trace_intervals_do_not_add_overlapping_metadata_twice():
    fields, rows = fixture_trace()
    decoded = bench.decode_trace(rows, fields, 3, 8, True, True, 1000)
    for row in decoded:
        intervals = row["intervals"]
        assert all(intervals[name + "_interval_us"] == 1 for name in bench.TIMELINE)
        assert intervals["src_cycles_us"] == 1.5
        assert intervals["topk_cycles_us"] == 2.5
        assert sum(intervals[name + "_interval_us"] for name in bench.TIMELINE) == 11


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 0),
        ("rank", 0),
        ("core", 7),
        ("cached", 0),
        ("whole_channel", 0),
        ("entry_cycles", 0),
        ("quiet_done", 1),
        ("init_done", -1),
    ],
)
def test_reject_wrong_launch_or_incomplete_trace(field, value):
    fields, rows = fixture_trace()
    rows[0][fields.index(field)] = value
    with pytest.raises(ValueError):
        bench.decode_trace(rows, fields, 3, 8, True, True, 1000)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("corrupt", [False, True])
def test_capture_excludes_alignment_and_guards(tmp_path, cached, corrupt):
    names = ["cached_notify_combine_kernel"]
    if not cached:
        names.append("dispatch_layout_kernel")
    names += ["cached_notify_dispatch_kernel" if cached else "notify_dispatch_kernel", "v1_dispatch_udma_kernel"]
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
        assert result["v1_dispatch_udma_kernel"] == [3, 4, 5]
