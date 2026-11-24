"""Compile-time credit/ownership tests using the production sender methods.

These simulate the pinned SDK's SQ/CQ bookkeeping, not UDMA hardware ordering.
constexpr runs the checks even on hosts that cannot execute compiled binaries.
"""

from pathlib import Path
import shutil
import subprocess

import pytest

from tests.utils.cpp import cpp_functions


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("corrupt_sink", [False, True])
def test_physical_sq_credit_and_final_protocol(tmp_path, corrupt_sink):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++17 compiler required")
    source = (ROOT / "csrc/direct/kernels/common/udma_shared_sender.h").read_text()
    methods = cpp_functions(source, "EnsureCredit", "Admit", "SubmitBatch", "Publish", "Quiet")
    # Replace only address-space casts with the model's pointer lookup. Keep
    # production decisions, SDK calls, arguments and CQ accounting unchanged.
    methods = methods.replace(
        "reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(peers_[p].state)", "&states[peers_[p].state]"
    ).replace("reinterpret_cast<GM_ADDR>(local_base_)", "local_base_")
    methods = methods.replace(
        "reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(peers_[last_].state)", "&states[peers_[last_].state]"
    )
    methods = methods.replace(
        "reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kPressureScratchOffset)", "pressure"
    ).replace("reinterpret_cast<__ubuf__ uint64_t*>(scratch_)", "scratch")
    methods = (
        methods.replace("reinterpret_cast<uint64_t>(scratch_)", "scratch_")
        .replace("reinterpret_cast<uint64_t>(scratch_ + first * kWqeBytes)", "(scratch_ + first * kWqeBytes)")
        .replace(
            "reinterpret_cast<__gm__ uint8_t*>(ctx->buf_addr + uint64_t(index) * kWqeBytes)",
            "(ctx->buf_addr + uint64_t(index) * kWqeBytes)",
        )
        .replace("reinterpret_cast<__gm__ uint8_t*>(ctx->buf_addr)", "ctx->buf_addr")
    )
    if corrupt_sink:
        methods = methods.replace("one + kQpCount + qp", "one + qp")
    program = (Path(__file__).with_name("sparse_cqe_credit_model.cpp")).read_text()
    program = program.replace("// PRODUCTION_METHODS", methods)
    path = tmp_path / "sparse.cpp"
    path.write_text(program)
    result = subprocess.run(
        [compiler, "-std=c++17", "-c", str(path), "-o", str(tmp_path / "sparse.o")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if corrupt_sink:
        assert result.returncode != 0 and "static assertion" in result.stderr, result.stderr
    else:
        assert result.returncode == 0, result.stderr


def test_sender_layout_and_coverage():
    source = (ROOT / "csrc/direct/kernels/common/udma_shared_sender.h").read_text()
    submit = cpp_functions(source, "SubmitBatch")
    assert "aclshmemi_udma_flush_aggregate_wqes(" not in source
    assert "aclshmemi_udma_post_send_update_info(head + count, ctx, state)" in submit
    assert "cqe_cnt++" not in source and "sq_tail =" not in submit
    assert "marker_count > kQpCount" in source
    assert "count && qp_base_ + Qps > kQpCount" in source
    for kind in ("dispatch", "combine"):
        sender = (ROOT / f"csrc/direct/kernels/{kind}/udma_{kind}_sender.h").read_text()
        assert "peer * 2U * 8U" in sender
        assert "Base::SubmitBatch(" in sender
    # SIMT metadata is outside the UDMA final-SO protocol: preserve its barrier.
    for name in ("udma_dispatch_schedule.h", "udma_token_dispatch_schedule.h"):
        schedule = (ROOT / "csrc/direct/kernels/dispatch" / name).read_text()
        assert schedule.index("sender.Quiet()") < schedule.index("aclshmemx_sync_vec_all()")


def test_combine_fast_wqe_equivalence(tmp_path):
    """Compile-time byte equality, including split weight pairs and QP header slots."""
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++17 compiler required")
    text = (ROOT / "csrc/direct/kernels/combine/udma_combine_sender.h").read_text()
    functions = []
    for name in ("BuildCombineWqes", "BuildCombineWqesFast"):
        body = cpp_functions(text, name)
        end = body.rfind(")", 0, body.index("{"))
        body = (body[:end] + ", uint32_t tid" + body[end:]).replace("threadIdx.x", "tid")
        body = body.replace(
            "reinterpret_cast<__ubuf__ int32_t*>(span[0])[position / fields * index_step]",
            "index_at(span[0], position / fields * index_step)",
        ).replace("reinterpret_cast<__ubuf__ int32_t*>(span[0])[index]", "index_at(span[0], index)")
        functions.append(body)
    source = r"""
#include <cstdint>
#define __simt_vf__ constexpr
#define __launch_bounds__(n)
#define __ubuf__
#define __gm__
namespace __asc_simt_vf { void __trap(); }
constexpr int32_t index_at(uint64_t seed, unsigned i) { return (seed+i*7)%1024; }
// FUNCTIONS
template<bool Weighted> constexpr bool check(unsigned count,unsigned width,unsigned start_field) {
    uint64_t expected[256]{}, actual[256]{}, spans[128]{}, headers[64*16]{}, bases[64]{};
    for(unsigned i=0;i<1024;++i) headers[i]=0x100000000ULL+i;
    for(unsigned p=0;p<64;++p) bases[p]=0x200000000ULL+p*123456;
    unsigned n=0;
    for(unsigned begin=0;begin<count;begin+=width,++n) {
        unsigned len=count-begin<width?count-begin:width;
        spans[n*4]=n*13;
        spans[n*4+1]=(uint64_t(len)<<32)|(17+n*19);
        spans[n*4+2]=(uint64_t(start_field)<<32)|((n*11)%64);
        spans[n*4+3]=begin;
    }
    for(unsigned tid=0;tid<32;++tid) {
        BuildCombineWqes(expected,spans,headers,bases,10000,20000,64,128,5632,32,
                         5664,64,1024,Weighted?2:1,n,count,1,1,0,0,0,tid);
        BuildCombineWqesFast<Weighted>(actual,spans,headers,bases,10000,20000,64,128,5632,32,
                                      5664,64,1024,n,count,tid);
    }
    for(unsigned i=0;i<256;++i) if(expected[i]!=actual[i])return false;
    return true;
}
template<bool Weighted> constexpr bool matrix(unsigned width) {
    for(unsigned count=0;count<=32;++count)
        for(unsigned field=0;field<(Weighted?2U:1U);++field)
            if(!check<Weighted>(count,width,field))return false;
    return true;
}
static_assert(matrix<false>(1) && matrix<true>(1));
static_assert(matrix<false>(3) && matrix<true>(3));
static_assert(matrix<false>(16) && matrix<true>(16));
static_assert(matrix<false>(32) && matrix<true>(32));
"""
    path = tmp_path / "combine_fast_wqes.cpp"
    path.write_text(source.replace("// FUNCTIONS", "\n".join(functions)))
    result = subprocess.run(
        [compiler, "-std=c++17", "-c", str(path), "-o", str(tmp_path / "combine_fast.o")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr


def test_payload_wqe_headers_and_addresses(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++17 compiler required")
    functions = []
    for kind in ("dispatch", "combine"):
        text = (ROOT / f"csrc/direct/kernels/{kind}/udma_{kind}_sender.h").read_text()
        body = cpp_functions(text, f"Build{kind.capitalize()}Wqes")
        end = body.rfind(")", 0, body.index("{"))
        body = (body[:end] + ", uint32_t tid" + body[end:]).replace("threadIdx.x", "tid")
        body = body.replace("reinterpret_cast<const __gm__ int32_t*>(span[0])[index]", "index_at(index)")
        body = body.replace("reinterpret_cast<const __ubuf__ int32_t*>(span[0])[index]", "index_at(index)")
        body = body.replace(
            "reinterpret_cast<__ubuf__ int32_t*>(span[0])[position / fields * index_step]",
            "index_at(position / fields * index_step)",
        ).replace("reinterpret_cast<__gm__ int32_t*>(single_slots)[row]", "slot_at(row)")
        functions.append(body)
    source = r"""
#include <cstdint>
#define __simt_vf__ constexpr
#define __launch_bounds__(n)
#define __ubuf__
#define __gm__
constexpr unsigned kOutputRecordWords=8;
namespace __asc_simt_vf { void __trap(); } // a trap makes constant evaluation fail
constexpr int32_t index_at(unsigned i) { return (i*7+3)%32; }
constexpr int32_t slot_at(unsigned i) { return i%2 ? -1 : int(i+7); }
// FUNCTIONS
constexpr bool check(bool combine,unsigned fields,unsigned count,bool early_single) {
    uint64_t wqes[32*8]{}, spans[8]{}, headers[4*2*8]{}, bases[4*kOutputRecordWords]{};
    for(unsigned i=0;i<64;++i)headers[i]=1000+i;
    for(unsigned p=0;p<4;++p) {
        if(combine)bases[p]=100000+p*100000;
        else { bases[p*8]=100000+p*100000; bases[p*8+1]=500000+p*100000; bases[p*8+2]=128; }
    }
    unsigned split=count/2;
    spans[1]=(uint64_t(split)<<32)|5; spans[2]=1;
    spans[5]=(uint64_t(count-split)<<32)|17; spans[6]=3; spans[7]=split;
    for(unsigned tid=0;tid<32;++tid) {
        if(combine)BuildCombineWqes(wqes,spans,headers,bases,10000,20000,64,128,5632,32,
            5632,32,32,fields,2,count,1,1,30000,early_single,128,tid);
        else BuildDispatchWqes(wqes,spans,headers,bases,10000,20000,5632,32,32,fields,2,count,tid);
    }
    for(unsigned i=0;i<count;++i) {
        unsigned p=i<split?1:3, position=i<split?i:i-split;
        unsigned row=(i<split?5:17)+position/fields, token=index_at(position/fields);
        bool extra=position%fields!=0;
        unsigned bytes=extra?32:5632;
        for(unsigned h=0;h<5;++h)if(wqes[i*8+h]!=headers[p*16+h])return false;
        uint64_t src=(extra?20000:10000)+uint64_t(combine?row:token)*bytes;
        if(combine && early_single && !extra && slot_at(row)>=0)src=30000+uint64_t(slot_at(row))*bytes;
        uint64_t dst=combine ? bases[p]+(extra?128:64)+uint64_t(token)*bytes : bases[p*8+extra]+uint64_t(row)*bytes;
        if(wqes[i*8+5]!=dst || wqes[i*8+6]!=bytes || wqes[i*8+7]!=src)return false;
    }
    for(unsigned i=count*8;i<32*8;++i)if(wqes[i])return false;
    return true;
}
constexpr bool matrix() {
    for(unsigned count=0;count<=32;++count)for(unsigned fields=1;fields<=2;++fields)
        if(!check(false,fields,count,false) || !check(true,fields,count,false) || !check(true,fields,count,true))return false;
    return true;
}
static_assert(matrix());
"""
    path = tmp_path / "payload_wqes.cpp"
    path.write_text(source.replace("// FUNCTIONS", "\n".join(functions)))
    result = subprocess.run(
        [compiler, "-std=c++17", "-c", str(path), "-o", str(tmp_path / "payload.o")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("corrupt_retention", [False, True])
@pytest.mark.parametrize("corrupt_cached_refill", [False, True])
def test_dispatch_index_windows_and_pending_batch_lifetime(tmp_path, corrupt_retention, corrupt_cached_refill):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++17 compiler required")
    sender = (ROOT / "csrc/direct/kernels/dispatch/udma_dispatch_sender.h").read_text()
    schedule = (ROOT / "csrc/direct/kernels/dispatch/udma_dispatch_schedule.h").read_text()
    helpers = cpp_functions(schedule, "ReuseDispatchIndexWindow", "DispatchIndexLowerBound")
    methods = cpp_functions(sender, "PutRowsImpl", "PreserveIndices", "BuildDispatchWqes")
    methods = methods.replace("reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kRequestsOffset)", "this->spans.data()")
    methods = methods.replace(
        "reinterpret_cast<__ubuf__ int32_t*>(scratch_ + kSavedIndicesOffset)", "memory.data() + saved_base"
    )
    methods = methods.replace("reinterpret_cast<const __ubuf__ int32_t*>(span[0])", "(memory.data() + span[0])")
    methods = methods.replace("reinterpret_cast<const __gm__ int32_t*>(span[0])", "(memory.data() + span[0])")
    methods = methods.replace("reinterpret_cast<uint64_t>(saved + saved_indices_)", "saved_base + saved_indices_")
    methods = methods.replace(
        "uint32_t span_count, uint32_t count)", "uint32_t span_count, uint32_t count, uint32_t tid)"
    )
    methods = methods.replace("threadIdx.x", "tid")
    if corrupt_retention:
        methods = methods.replace("saved[saved_indices_ + t] = indices[t]", "saved[saved_indices_ + t] = 0")
    program = Path(__file__).with_name("dispatch_index_oracle.cpp").read_text()
    begin = schedule.index("while (remaining)", schedule.index("DispatchRemoteImpl("))
    body = schedule.index("{", begin)
    depth = 1
    end = body + 1
    while depth:
        depth += (schedule[end] == "{") - (schedule[end] == "}")
        end += 1
    loop = schedule[begin:end].replace("reinterpret_cast<const __ubuf__ int32_t*>(lists.GetPhyAddr())", "0U")
    if corrupt_cached_refill:
        loop = loop.replace("sender.PreserveIndices();", "")
    program = program.replace("// CACHED_SENDER_LOOP", loop)
    path = tmp_path / "dispatch_indices.cpp"
    path.write_text(program.replace("// WINDOW_HELPERS", helpers).replace("// SENDER_METHODS", methods))
    result = subprocess.run(
        [compiler, "-std=c++17", "-c", str(path), "-o", str(tmp_path / "indices.o")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if corrupt_retention or corrupt_cached_refill:
        assert result.returncode != 0 and "static assertion" in result.stderr, result.stderr
    else:
        assert result.returncode == 0, result.stderr
