"""Production sender template compilation and diagnostic export invariants."""

import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from tests.utils.cpp import cpp_functions

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "csrc/direct/kernels"


def test_sender_templates_compile_with_and_without_diagnostics(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    shared = (KERNELS / "common/udma_shared_sender.h").read_text()
    trace = (KERNELS / "dispatch/dispatch_device_trace.h").read_text()
    trace = "\n".join(line for line in trace.splitlines() if not line.startswith(("#include", "#pragma")))
    shared = shared[shared.index("struct PeerQueue") : shared.index("using Sender =")]
    senders = []
    for kind in ("dispatch", "combine"):
        source = (KERNELS / kind / f"udma_{kind}_sender.h").read_text()
        start = source.index("__simt_vf__")
        senders.append(source[start : source.index(f"using {kind.capitalize()}Sender =")])
    source = r"""
#include <cstdint>
#define __aicore__
#define __simt_vf__
#define __gm__
#define __ubuf__
#define __launch_bounds__(n)
#define ASCEND_DEEPEP_ARCH 3510
#include "common/types.h"
#include "common/udma_schedule.h"
#include "common/dispatch_output.h"
using GM_ADDR = uint8_t*;
constexpr int EVENT_ID5=5, EVENT_ID6=6, EVENT_ID7=7, PIPE_MTE3=3;
struct { unsigned x; } threadIdx;
struct dim3 { explicit dim3(unsigned) {} };
namespace __asc_simt_vf { void __trap(); }
void trap();
namespace AscendC {
enum class TPosition { VECOUT };
enum class HardEvent { S_V, V_MTE3, V_S, MTE3_S, S_MTE3, S_MTE2, MTE2_S };
template<HardEvent> void SetFlag(int);
template<HardEvent> void WaitFlag(int);
uint64_t GetSystemCycle();
template<class T> struct LocalTensor {
    struct { uint8_t logicPos; uint64_t bufferAddr; } address_;
    void SetValue(unsigned,T); void* GetPhyAddr(); LocalTensor operator[](uint64_t);
};
template<class T> struct GlobalTensor { void SetGlobalBuffer(T*); T GetValue(uint64_t); GlobalTensor operator[](uint64_t); };
struct DataCopyExtParams { uint32_t blocks, bytes, src_stride, dst_stride, reserved; };
template<class T> struct DataCopyPadExtParams { bool pad; uint8_t left,right; T value; };
template<class T> void DataCopyPad(GlobalTensor<T>,LocalTensor<T>,DataCopyExtParams);
template<class T> void DataCopyPad(LocalTensor<T>,GlobalTensor<T>,DataCopyExtParams,DataCopyPadExtParams<T>);
template<class... Args> void printf(const char*,Args...);
}
template<auto Fn,class... Args> void asc_vf_call(dim3,Args... args) { Fn(args...); }
struct aclshmemx_udma_op_config_t { uint32_t a,b,c,d; };
struct aclshmemi_udma_queue_state_t { uint32_t sq_head,sq_tail,cqe_cnt,cq_tail; };
struct aclshmemi_sqe_ctx_t { uint64_t words[6]; };
struct aclshmemi_sge_ctx_t { uint64_t words[2]; };
struct aclshmemi_udma_wq_ctx_t { uint64_t state_addr,buf_addr; uint32_t depth,wqe_size; };
struct Info { unsigned qp_num; };
enum class aclshmemi_udma_opcode_t { UDMA_OP_WRITE };
template<aclshmemi_udma_opcode_t> constexpr uint32_t get_wqe_bb_cnt() { return 1; }
namespace shm { constexpr unsigned UDMA_SQ_BASKBLK_CNT=1024; }
constexpr unsigned ACLSHMEM_UDMA_DATA_MOVER_WQE_SIZE=64;
Info* aclshmemi_udma_qp_info_fetch();
uint32_t aclshmemi_udma_compute_slot(uint32_t,uint32_t);
aclshmemi_udma_wq_ctx_t* aclshmemi_udma_get_qp_ctx(Info*,uint32_t,uint32_t);
void* aclshmem_ptr(void*,uint32_t);
void* aclshmemi_udma_get_mem_info(Info*,uint32_t,uint32_t);
template<class T,aclshmemi_udma_opcode_t Op,class P,const aclshmemx_udma_op_config_t& C>
void aclshmemi_udma_fill_sqe_base_ctx(P,uint8_t*,void*,uint32_t);
bool aclshmemi_udma_sq_has_credit(aclshmemi_udma_queue_state_t*,uint32_t);
void aclshmemi_udma_poll_cq_when_sq_overflow(aclshmemi_udma_queue_state_t*,uint32_t,uint32_t,uint32_t);
void aclshmemi_udma_flush_aggregate_wqes(uint8_t*,aclshmemi_udma_wq_ctx_t*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
void aclshmemi_udma_copy_wqe_from_ub(uint8_t*,AscendC::LocalTensor<uint8_t>&,uint32_t,int);
void aclshmemi_udma_post_send_update_info(uint32_t,aclshmemi_udma_wq_ctx_t*,aclshmemi_udma_queue_state_t*);
template<class T,int Pipe,const aclshmemx_udma_op_config_t& C>
void aclshmemx_udma_qp_put_nbi(T*,T*,uint64_t*,uint32_t,uint32_t,uint32_t,int);
void aclshmemx_udma_qp_quiet(uint32_t,uint32_t);
// TRACE
namespace ascend_deepep::udma {
constexpr aclshmemx_udma_op_config_t kData{}, kLast{}, kDone{};
constexpr unsigned kRequestsOffset=32*64,kHeadersOffset=kRequestsOffset+32*32;
constexpr unsigned kPressureScratchOffset=kHeadersOffset+64;

uint64_t* Signal(GM_ADDR,uint64_t,uint32_t);
// SENDERS
template class SenderT<8,1>;
template class SenderT<8,2,DispatchDeviceTrace>;
template class DispatchSenderT<8,1>;
template class DispatchSenderT<64,2,DispatchDeviceTrace>;
template class CombineSenderT<8,1>;
template class CombineSenderT<64,2,DispatchDeviceTrace>;
}
"""
    path = tmp_path / "senders.cpp"
    path.write_text(source.replace("// TRACE", trace).replace("// SENDERS", shared + "\n" + "\n".join(senders)))
    result = subprocess.run(
        [compiler, "-std=c++17", "-c", "-I", str(ROOT / "csrc/direct"), str(path), "-o", str(tmp_path / "senders.o")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr


def test_device_fields_match_native_export_and_fit_row():
    source = (KERNELS / "dispatch/dispatch_device_trace.h").read_text()
    fields = source.split("enum Field {", 1)[1].split("Fields", 1)[0]
    fields = [item.strip() for item in fields.split(",") if item.strip()]
    native = (ROOT / "csrc/direct/bindings/pybind.cpp").read_text()
    names = re.findall(r'"([a-z_]+)"', native.split('m.attr("dispatch_device_timing_fields")', 1)[1].split("};", 1)[0])
    assert len(fields) == len(names) <= 64
    assert names[fields.index("CreditBlockedCycles")] == "credit_blocked_cycles"
    assert names[fields.index("Operation")] == "operation"
    assert names[fields.index("WqeSubmitCycles")] == "wqe_submit_cycles"


def test_combine_host_trace_owns_tensor_and_rejects_unsupported_paths():
    body = cpp_functions((ROOT / "csrc/direct/ops/combine.cpp").read_text(), "V1Combine")
    assert "(void)combine_trace;" in body
    assert "currentStreamCaptureStatusMayInitCtx()" in body
    assert "!shmem.TokenSchedule()" in body and "!pipeline_chunk_tokens && !expanded_x.has_value()" in body
    assert "CombineDeviceTimingSlot().reset()" in body


def test_combine_routing_accepts_traced_and_existing_callers(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    source = (KERNELS / "combine/udma_combine_schedule.h").read_text()
    impl = cpp_functions(source, "CombineRemoteImpl").split("{", 1)[0] + " {}"
    wrapper = source[
        source.index("template <uint32_t Phase = 0, class Resource, class Trace>") : source.index("#endif")
    ]
    program = r"""
#include <cstdint>
#define __aicore__
using GM_ADDR = uint8_t*;
struct CombineTilingData { unsigned udma_qps_per_worker, udma_peer_order; };
struct DisabledDispatchDeviceTrace { explicit DisabledDispatchDeviceTrace(uint64_t) {} };
struct EnabledTrace { explicit EnabledTrace(uint64_t) {} };
constexpr unsigned kGroupRoundRobinOrder=1, kGroupPeerCapacity=8, kMaxPeers=256;
// FUNCTIONS
void test() {
    int resource=0;
    GM_ADDR p{};
    CombineTilingData td{};
    EnabledTrace trace(0);
    CombineRemote<0>(resource,p,p,p,p,p,p,td,0,trace);
    CombineRemote<0>(resource,p,p,p,p,p,p,td,0);
    CombineRemote<1>(resource,p,p,p,p,p,p,td,0);
    CombineRemote<3>(resource,p,p,p,p,p,p,td,0);
}
"""
    path = tmp_path / "combine_calls.cpp"
    path.write_text(program.replace("// FUNCTIONS", impl + "\n" + wrapper))
    result = subprocess.run(
        [compiler, "-std=c++17", "-c", str(path), "-o", str(tmp_path / "calls.o")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr


def test_combine_profiler_excludes_alignment_and_checks_sequence(tmp_path):
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
    names = ["cached_notify_combine_kernel", "v1_combine_udma_comm_kernel", "v1_combine_udma_reduce_kernel"]
    path = tmp_path / "kernel_details.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Start Time(us)", "Duration(us)"])
        for i, name in enumerate(names * 6):
            writer.writerow([name, i * 100, [999, 20, 30][i % 3]])
    result = bench.capture_combine_stages(tmp_path, 2)
    assert result == {names[1]: [20, 20], names[2]: [30, 30]}
    with pytest.raises(RuntimeError, match="sequence"):
        bench.capture_combine_stages(tmp_path, 3)


def test_decoder_distinguishes_queue_counters_and_rejects_invalid_data():
    sys.path.insert(0, str(ROOT / "benchmarks/clos"))
    try:
        spec = importlib.util.spec_from_file_location("queue_bench", ROOT / "benchmarks/clos/bench_dispatch_stages.py")
        bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench)
    finally:
        sys.path.pop(0)
    native = (ROOT / "csrc/direct/bindings/pybind.cpp").read_text()
    names = re.findall(r'"([a-z_]+)"', native.split('m.attr("dispatch_device_timing_fields")', 1)[1].split("};", 1)[0])
    row = dict.fromkeys(names, 0)
    row.update(
        version=2,
        rank=1,
        core=0,
        entry_cycles=100,
        payload_peers=2,
        credit_checks=4,
        credit_blocked=1,
        credit_cycles=400,
        credit_blocked_cycles=300,
        flushes=2,
        short_flushes=1,
        wqes=48,
        metadata_wqes=16,
        metadata_flushes=1,
        sq_used_sum=800,
        cqe_requested=3,
        credit_checkpoints=1,
        credit_quiet_calls=1,
        final_quiet_cycles=250,
        final_quiet_calls=2,
    )
    row.update({name: (i + 1) * 1000 for i, name in enumerate(bench.TIMELINE)})
    row["incoming_begin"] = row["published"]
    (decoded,) = bench.decode_trace([[row[k] for k in names]], names, 1, 1, False, False, 1000)
    assert decoded["queue"]["mean_batch_wqes"] == 24
    assert decoded["queue"]["blocked_admission_fraction"] == 0.25
    assert decoded["intervals"]["credit_blocked_cycles_us"] == 0.3
    assert decoded["intervals"]["final_quiet_cycles_us"] == 0.25
    assert decoded["raw"]["cqe_requested"] == 3
    row["credit_blocked"] = 5
    with pytest.raises(ValueError, match="inconsistent queue"):
        bench.decode_trace([[row[k] for k in names]], names, 1, 1, False, False, 1000)
