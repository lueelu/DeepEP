"""Compile/evaluate the real packing and WQE bodies without a device SDK."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.utils.cpp import cpp_functions

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("device_parse", [False, True])
def test_output_publication_preserves_epoch_and_rejects_invalid_capacity(tmp_path, device_parse):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    functions = cpp_functions(
        (ROOT / "csrc/direct/kernels/dispatch/udma_dispatch_output.h").read_text(), "PublishDispatchOutput"
    )
    source = r"""
#include <array>
#include <cassert>
#include <vector>
#include "common/dispatch_output.h"
#define __simt_vf__
#define __launch_bounds__(n)
#define __gm__
using namespace ascend_deepep::udma;
struct { unsigned x; } threadIdx;
std::array<uint64_t,128> control;
std::vector<unsigned> writes;
bool trapped=false;
void asc_threadfence() {}
namespace __asc_simt_vf { void __trap() { trapped=true; } }
namespace simt {
void aclshmemx_instance_ctx_set(unsigned) {}
void aclshmem_uint64_p(uint64_t* addr,uint64_t value,unsigned peer) {
    assert(peer==0); writes.push_back(unsigned(addr-control.data())); *addr=value;
}
}
// FUNCTIONS
static_assert(ValidMetadataOutput(0,0,4));
static_assert(!ValidMetadataOutput(0,1,4));
static_assert(!ValidMetadataOutput(32,1,0));
static_assert(!ValidMetadataOutput(UINT64_MAX-3,1,4));
static_assert(ValidMetadataOutput(16,UINT32_MAX,UINT32_MAX));
int main() {
    for(unsigned cached:{0U,1U}) for(unsigned direct:{0U,1U}) {
        control.fill(77); writes.clear(); trapped=false;
        PublishDispatchOutput(reinterpret_cast<uint8_t*>(control.data()),0,1,9,1024,0,3,3,16,0,0,
            direct&&!cached?2048:0,direct?4096:0,direct&&!cached?4:0,direct?32:0,direct?8192:0,direct?64:0);
        assert(!trapped && writes.back()==64+kOutputEpochWord);
        assert(writes.size()==(direct?9:6));
        assert(control[64+kOutputSrcWord]==(direct?(cached?0:2048):77));
        assert(control[64+kOutputWeightWord]==(direct?4096:77));
        assert(control[64+kOutputTopkWord]==(direct?8192:77));
    }
    writes.clear(); trapped=false;
    PublishDispatchOutput(reinterpret_cast<uint8_t*>(control.data()),0,1,10,1024,0,3,3,16,0,0,0,4096,4,32,8192,64);
    assert(trapped && writes.empty());
    // A retained output cannot publish a null or overflowing top-k tensor.
    for (uint64_t topk : {uint64_t(0), UINT64_MAX-16}) {
        writes.clear(); trapped=false;
        PublishDispatchOutput(reinterpret_cast<uint8_t*>(control.data()),0,1,11,1024,0,3,3,16,0,0,
            2048,4096,4,32,topk,64);
        assert(trapped && writes.empty());
    }
    // Reusing a slot for cached omit must clear the previous top-k address.
    writes.clear(); trapped=false;
    PublishDispatchOutput(reinterpret_cast<uint8_t*>(control.data()),0,1,12,1024,0,3,3,16,0,0,
        0,4096,0,32,0,0);
    assert(!trapped && control[64+kOutputTopkWord]==0 && writes.back()==64+kOutputEpochWord);
}
"""
    path = tmp_path / "publication.cpp"
    path.write_text(source.replace("// FUNCTIONS", functions))
    binary = tmp_path / ("publication.exe" if os.name == "nt" else "publication")
    # Exercise the device preprocessor branch as well: its helper must remain
    # constexpr for both the compile-time oracle and the SIMT caller contract.
    flags = ["-D__CCE_AICORE__", "-D__aicore__="] if device_parse else []
    result = subprocess.run(
        [compiler, "-std=c++17", "-O2", *flags, "-I", str(ROOT / "csrc/direct"), str(path), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    try:
        subprocess.run([str(binary)], check=True, timeout=30)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            pytest.skip("compiled, but Windows application control blocked execution (4551)")
        raise


def vf_as_constexpr(source, name):
    body = cpp_functions(source, name)
    # SIMT lanes run independently; inject the lane id as a constexpr argument.
    end = body.rfind(")", 0, body.index("{"))
    return (body[:end] + ", uint32_t tid" + body[end:]).replace("threadIdx.x", "tid")


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("corrupt", [False, True])
def test_channel_output_uses_remote_tensor_va_and_queue_lane(tmp_path, bits, corrupt):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    functions = cpp_functions(
        (ROOT / "csrc/direct/kernels/dispatch/udma_dispatch_sender.h").read_text(), "PutChannelOutput"
    )
    # Keep production addressing/guards; replace only UB and GM pointer casts
    # with a constexpr address ledger so this runs even where Windows blocks EXEs.
    functions = functions.replace(
        "reinterpret_cast<__ubuf__ uint64_t*>(scratch_ + kDispatchBasesOffset)", "records.data()"
    ).replace("reinterpret_cast<__gm__ uint8_t*>(source)", "source")
    if corrupt:
        functions = functions.replace("output[field] + row * width", "output[field] + (row + 1) * width")
    source = r"""
#include <array>
#include "common/dispatch_output.h"
#define __aicore__ constexpr
using namespace ascend_deepep::udma;
using topk_idx_t = INDEX_TYPE;
struct Base {
    unsigned lane=0, peer=0, bytes=0, calls=0;
    uint64_t destination=0, source=0;
    bool trapped=false;
    constexpr void trap() { trapped=true; }
    constexpr unsigned QueuePeer(unsigned p) { return lane*4+p; }
    constexpr void PutAbsolute(unsigned p,uint64_t dst,uint64_t src,unsigned n) {
        ++calls; peer=p; destination=dst; source=src; bytes=n;
    }
};
struct Sender : Base {
    struct { unsigned num_topk=8; } td_;
    std::array<uint64_t,8*kOutputRecordWords> records{};
    // FUNCTIONS
};
constexpr bool check() {
    for(unsigned lane:{0U,1U}) for(unsigned field:{kOutputSrcWord,kOutputWeightWord,kOutputTopkWord}) {
        Sender sender;
        sender.lane=lane;
        auto* record=sender.records.data()+sender.QueuePeer(2)*kOutputRecordWords;
        record[2]=100; record[field]=0x1000000+lane*0x10000;
        sender.PutChannelOutput(2,field,7,0x800000,65);
        if(sender.trapped || sender.calls!=1 || sender.peer!=2) return false;
        const unsigned width=field==kOutputSrcWord?4:(field==kOutputTopkWord?8*sizeof(topk_idx_t):32);
        if(sender.destination!=record[field]+7*width || sender.source!=0x800000 || sender.bytes!=65*width) return false;
        sender.PutChannelOutput(2,field,99,0x800000,2);
        if(!sender.trapped || sender.calls!=1) return false;
        sender.trapped=false;
        sender.PutChannelOutput(2,99,0,0,1);
        if(!sender.trapped || sender.calls!=1) return false;
        sender.trapped=false;
        record[field]=0;
        sender.PutChannelOutput(2,field,0,0x800000,1);
        if(!sender.trapped || sender.calls!=1) return false;
        sender.trapped=false;
        record[2]=0;
        sender.PutChannelOutput(2,field,0,0,0);
        if(sender.trapped || sender.bytes!=0) return false;
        record[2]=UINT32_MAX; record[field]=4096; sender.trapped=false;
        const unsigned before=sender.calls;
        sender.PutChannelOutput(2,field,0,0x800000,UINT32_MAX);
        if(!sender.trapped || sender.calls!=before) return false;
    }
    return true;
}
static_assert(check());
"""
    path = tmp_path / "channel_address.cpp"
    path.write_text(source.replace("// FUNCTIONS", functions).replace("INDEX_TYPE", f"int{bits}_t"))
    result = subprocess.run(
        [compiler, "-std=c++17", "-I", str(ROOT / "csrc/direct"), "-c", str(path), "-o", str(tmp_path / "address.o")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if corrupt:
        assert result.returncode != 0 and "static assertion" in result.stderr
    else:
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("tile", [32, 64, 128, 256])
def test_real_metadata_and_wqe_bodies_constexpr(tmp_path, bits, corrupt, tile):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    kernels = ROOT / "csrc/direct/kernels"
    functions = vf_as_constexpr((kernels / "dispatch/channel_metadata.h").read_text(), "PackPlanMetadataVf")
    functions += vf_as_constexpr((kernels / "dispatch/channel_metadata.h").read_text(), "PackRemoteMetadataVf")
    functions += vf_as_constexpr((kernels / "common/udma_shared_sender.h").read_text(), "BuildWqes")
    if corrupt:
        functions = functions.replace(
            "expert - topk_idx_t(peer * experts_per_rank)", "expert - topk_idx_t(peer * experts_per_rank) + 1"
        )
    source = r"""
#include <cstdint>
#include <array>
#define __simt_vf__ constexpr
#define __launch_bounds__(n)
#define __ubuf__
#define __gm__
using topk_idx_t = INDEX_TYPE;
constexpr void asc_threadfence() {}
// FUNCTIONS
template<unsigned World, unsigned Tokens, bool Weighted>
constexpr bool pack_check() {
    constexpr unsigned C=3, Ch=1, K=8, S=((Tokens+7)/8)*8+8;
    std::array<int32_t,(Tokens+1)*World> heads{};
    std::array<topk_idx_t,(Tokens+1)*K> idx{};
    std::array<float,(Tokens+1)*K> weights{};
    std::array<topk_idx_t,World*C*S*K> output{}, expected{};
    std::array<float,World*C*S*K> ow{}, ew{};
    for(unsigned i=0;i<output.size();++i){output[i]=expected[i]=-77;ow[i]=ew[i]=-77;}
    for(unsigned t=0;t<Tokens;++t) for(unsigned k=0;k<K;++k) {
        idx[t*K+k] = (t%7==0 || k==3) ? -1 : topk_idx_t((t*13+k*5)%(World*16));
        weights[t*K+k]=float((t*K+k)%61)/64;
    }
    for(unsigned p=0;p<World;++p) {
        unsigned n=0;
        for(unsigned t=0;t<Tokens;++t) {
            bool routed=false;
            for(unsigned k=0;k<K;++k) routed |= idx[t*K+k]>=topk_idx_t(p*16) && idx[t*K+k]<topk_idx_t((p+1)*16);
            heads[t*World+p]=routed?int(n):-1;
            if(!routed)continue;
            for(unsigned k=0;k<K;++k) {
                const auto expert=idx[t*K+k];
                const bool valid=expert>=topk_idx_t(p*16) && expert<topk_idx_t((p+1)*16);
                const unsigned o=((p*C+Ch)*S+n)*K+k;
                expected[o]=valid?expert-p*16:topk_idx_t(-1);
                if(Weighted)ew[o]=valid?weights[t*K+k]:0;
            }
            ++n;
        }
    }
    for(unsigned start=0;start<Tokens;start+=TILE_TOKENS) {
        unsigned n=Tokens-start<TILE_TOKENS?Tokens-start:TILE_TOKENS;
        for(unsigned peer=0;peer<World;++peer) {
            std::array<int32_t,Tokens+1> plan{};
            unsigned rows=0, first=0;
            for(unsigned t=start;t<start+n;++t) if(heads[t*World+peer]>=0) {
                if(!rows) first=heads[t*World+peer];
                plan[rows++]=t;
            }
            const unsigned dest=((peer*C+Ch)*S+first)*K;
            for(unsigned tid=0;tid<128;++tid)
                PackPlanMetadataVf(plan.data(),idx.data()+start*K,weights.data()+start*K,
                    output.data()+dest,ow.data()+dest,nullptr,rows,start,K,peer,16,Weighted,tid);
        }
    }
    for(unsigned i=0;i<output.size();++i) if(output[i]!=expected[i] || ow[i]!=ew[i])return false;
    for(unsigned i=0;i<output.size();++i) { output[i]=-77; ow[i]=-77; }
    for(unsigned start=0;start<Tokens;start+=TILE_TOKENS) {
        const unsigned n=Tokens-start<TILE_TOKENS?Tokens-start:TILE_TOKENS;
        std::array<int32_t,World*(TILE_TOKENS+8)> plans{};
        std::array<uint32_t,World*4> desc{};
        for(unsigned p=0;p<World;++p) {
            desc[p*4]=p;
            desc[p*4+3]=1;
            for(unsigned t=start;t<start+n;++t) if(heads[t*World+p]>=0) {
                if(!desc[p*4+1])desc[p*4+2]=heads[t*World+p];
                plans[p*(TILE_TOKENS+8)+1+desc[p*4+1]++]=t;
            }
        }
        for(unsigned tid=0;tid<128;++tid)
            PackRemoteMetadataVf(plans.data(),desc.data(),idx.data()+start*K,weights.data()+start*K,
                output.data(),ow.data(),World,TILE_TOKENS+8,start,K,16,C,Ch,S,Weighted,tid);
    }
    for(unsigned i=0;i<output.size();++i) if(output[i]!=expected[i] || ow[i]!=ew[i])return false;
    return true;
}
constexpr bool wqe_check() {
    uint64_t wqes[32*8]{}, requests[32*4]{}, headers[4*2*8]{};
    for(unsigned i=0;i<64;++i)headers[i]=1000+i;
    for(unsigned i=0;i<24;++i) {
        // Eight channels, three fields, distinct source and destination strides.
        requests[i*4]=20000+(i/3)*700+i%3*128;
        requests[i*4+1]=40000+(i/3)*900+i%3*256;
        requests[i*4+2]=i%3==0?68:(i%3==1?1088:544);
        requests[i*4+3]=i%4;
    }
    for(unsigned tid=0;tid<32;++tid) BuildWqes(wqes,requests,headers,24,tid);
    for(unsigned i=0;i<24;++i) {
        for(unsigned j=0;j<5;++j) if(wqes[i*8+j]!=headers[requests[i*4+3]*2*8+j])return false;
        if(wqes[i*8+5]!=requests[i*4] || wqes[i*8+6]!=requests[i*4+2] || wqes[i*8+7]!=requests[i*4+1])return false;
    }
    for(unsigned i=24*8;i<32*8;++i)if(wqes[i])return false;
    return true;
}
static_assert(pack_check<8,0,false>());
static_assert(pack_check<8,33,false>());
static_assert(pack_check<16,65,true>());
static_assert(pack_check<8,TILE_TOKENS+1,true>());
static_assert(pack_check<256,2,true>());
static_assert(wqe_check());
"""
    path = tmp_path / "channel_metadata.cpp"
    path.write_text(
        source.replace("INDEX_TYPE", f"int{bits}_t")
        .replace("TILE_TOKENS", str(tile))
        .replace("// FUNCTIONS", functions)
    )
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True, timeout=30).stdout.lower()
    limit = "-fconstexpr-steps=10000000" if "clang" in version else "-fconstexpr-ops-limit=100000000"
    result = subprocess.run(
        [compiler, "-std=c++17", limit, "-c", str(path), "-o", str(tmp_path / "channel_metadata.o")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if corrupt:
        assert result.returncode != 0
        assert "static assertion" in result.stderr
    else:
        assert result.returncode == 0, result.stderr


def test_pack_and_local_copy_pipeline(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    kernels = ROOT / "csrc/direct/kernels"
    functions = cpp_functions((kernels / "dispatch/udma_dispatch_metadata.h").read_text(), "CopySelfChannelMetadata")
    harness = (
        (Path(__file__).with_name("eight_aiv_copy_oracle.cpp")).read_text().split("static unsigned sender_phase;", 1)[0]
    )
    harness = harness.replace("enum class HardEvent {", "enum class HardEvent { V_MTE2,")
    harness = harness.replace(
        "static unsigned list_loads;",
        "static unsigned list_loads;\nstatic uintptr_t plan_begin, plan_end;\nstatic uint64_t plan_bytes;",
    )
    dma_entry = (
        "void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyExtParams copy, DataCopyPadExtParams<T>)\n{"
    )
    assert harness.count(dma_entry) == 1, "GM-to-UB DMA mock instrumentation anchor must be unique"
    harness = harness.replace(
        dma_entry,
        dma_entry
        + "\n    if (uintptr_t(src.data)>=plan_begin && uintptr_t(src.data)<plan_end) plan_bytes+=copy.bytes;",
    )
    harness = harness.replace(
        "template <HardEvent>\nvoid SetFlag(int)\n{\n}",
        "static int credits[16][8]{};\nstatic unsigned event_sets;\n"
        "template <HardEvent E> void SetFlag(int e) { assert(!credits[int(E)][e]); credits[int(E)][e]=1; ++event_sets; }",
    )
    harness = harness.replace(
        "template <HardEvent>\nvoid WaitFlag(int)\n{\n}",
        "template <HardEvent E> void WaitFlag(int e) { assert(credits[int(E)][e]==1); credits[int(E)][e]=0; }",
    )
    harness += r"""
#include "common/dispatch_channel_metadata.h"
using namespace ascend_deepep;
struct { unsigned x; } threadIdx;
void asc_threadfence() {}
struct dim3 { unsigned x; explicit dim3(unsigned value):x(value){} };
template<auto Fn, class... Args> void asc_vf_call(dim3 n, Args... args) {
    for(threadIdx.x=0;threadIdx.x<n.x;++threadIdx.x)Fn(args...);
}
#define __CCE_KT_TEST__
#include "kernels/dispatch/channel_metadata.h"
namespace ascend_deepep {
// FUNCTIONS
}
"""
    path = tmp_path / "channel_calls.cpp"
    path.write_text(
        harness.replace("// FUNCTIONS", functions)
        + Path(__file__).with_name("channel_pack_pipeline_oracle.cpp").read_text()
    )
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(ROOT / "csrc/direct"),
            str(path),
            "-o",
            str(tmp_path / ("calls.exe" if os.name == "nt" else "calls")),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr

    try:
        subprocess.run([str(tmp_path / ("calls.exe" if os.name == "nt" else "calls"))], check=True, timeout=60)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            pytest.skip("compiled, but Windows application control blocked execution (4551)")
        raise


def test_dynamic_metadata_tile_layout_constexpr(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    source = r"""
#include "common/dispatch_channel_metadata.h"
#include "common/udma_schedule.h"
using namespace ascend_deepep;
constexpr bool check() {
    for(unsigned capacity:{8U,64U}) for(unsigned qps:{1U,2U})
    for(unsigned slots:{1U,2U}) for(unsigned k:{1U,8U,128U})
    for(unsigned index_bytes:{4U,8U}) for(bool weighted:{false,true}) {
        const unsigned live=udma::DispatchLiveSendUbBytes(capacity,qps);
        const unsigned descriptors=(capacity*16+31)/32*32;
        const unsigned budget=(221120-live)/slots-descriptors;
        const auto tile=SelectChannelMetadataTile(0,capacity,k,index_bytes,weighted,budget);
        if(!tile.tokens || live+slots*(descriptors+tile.bytes)>221120)return false;
        for(unsigned peers:{1U,capacity}) for(unsigned stride:{0U,8U,128U,512U,8192U}) {
            const auto cache=ExtendMetadataPlanCache(tile,peers,stride,budget);
            if(cache.plan_tokens<tile.tokens || cache.plan_tokens%8 ||
               cache.input.bytes>budget || cache.input.index_offset%32 ||
               cache.input.weight_offset%32 || cache.input.index_offset!=peers*cache.plan_tokens*4 ||
               cache.input.bytes-cache.input.index_offset!=tile.bytes-tile.index_offset)return false;
        }
    }
    for (unsigned world : {1U, 8U, 16U, 256U})
    for (unsigned k : {1U, 8U, 128U})
    for (unsigned bytes : {4U, 8U})
    for (bool weights : {false, true})
    for (unsigned budget : {196608U, 221120U, 98304U, 110560U}) {
        const auto automatic = SelectChannelMetadataTile(0, world, k, bytes, weights, budget);
        if (!automatic.tokens || automatic.bytes > budget) return false;
        for (unsigned tokens : {32U, 64U, 128U, 256U}) {
            const auto tile = MakeChannelMetadataTile(tokens, world, k, bytes, weights);
            if (tile.index_offset % 32 || tile.weight_offset % 32 || tile.bytes % 32) return false;
            if (tile.index_offset < tokens*world*4 || tile.weight_offset < tile.index_offset+tokens*k*bytes ||
                tile.bytes < tile.weight_offset+(weights?tokens*k*4:0)) return false;
            const auto selected = SelectChannelMetadataTile(tokens, world, k, bytes, weights, budget);
            if (bool(selected.tokens) != (tile.bytes <= budget)) return false;
            if (selected.tokens && selected.tokens != tokens) return false;
        }
    }
    return true;
}
static_assert(check());
static_assert(SelectChannelMetadataTile(0,256,128,8,true,221120/2).tokens == 32);
static_assert(SelectChannelMetadataTile(64,256,128,8,true,221120/2).tokens == 0);
static_assert(MakeChannelMetadataTile(128,8,8,8,true).bytes == 16384);
static_assert(MakeChannelMetadataTile(128,16,8,8,true).bytes == 20480);
static_assert(SelectChannelMetadataTile(0,256,128,8,true,221120).tokens == 64);
static_assert(SelectChannelMetadataTile(128,256,128,8,true,221120).tokens == 0);
static_assert(SelectChannelMetadataTile(64,256,128,8,true,221120).bytes == 163840);
static_assert(SelectChannelMetadataTile(17,8,8,8,true,221120).tokens == 0);
static_assert(SelectChannelMetadataTile(0,257,8,8,true,221120).tokens == 0);
static_assert(SelectChannelMetadataTile(0,8,129,8,true,221120).tokens == 0);
"""
    path = tmp_path / "tile_layout.cpp"
    path.write_text("#include <initializer_list>\n" + source)
    result = subprocess.run(
        [compiler, "-std=c++17", "-c", "-I", str(ROOT / "csrc/direct"), str(path), "-o", str(tmp_path / "tile.o")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr


def test_self_weight_output_is_not_overwritten_by_staging_constexpr(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    kernel = (ROOT / "csrc/direct/kernels/dispatch/dispatch_udma.cpp").read_text()
    begin = kernel.index("            const int32_t self_begin = tiling.channel_metadata")
    end = kernel.index("\n        }\n    }\n\n    trace.FinishOutput();", begin)
    body = kernel[begin:end]
    source = r"""
#include <array>
#include <cstdint>
constexpr int DispatchMin(int a,int b) { return a<b?a:b; }
struct Trace { enum {WeightOutputBytes}; constexpr void Accumulate(int,uint64_t) {} };
struct Prefix { int begin,end; constexpr int GetValue(int i) const {return i==1?begin:end;} };
namespace ascend_deepep {
constexpr void CopyDispatchRecvToGm(std::array<unsigned,16>& out,int,int,uint64_t start,uint64_t bytes,int,int) {
    for(unsigned i=start/4;i<(start+bytes)/4;++i) ++out[i];
}
}
constexpr bool check(int recv_token_begin,int worker_tokens,int sb,int se,bool channel) {
    std::array<unsigned,16> recv_topk_weights{};
    struct {bool channel_metadata;int topk_weights_buffer_offset;} tiling{channel,0};
    const Prefix rank_prefix_gm{sb,se};
    constexpr int rank=1,world_size=4,topk_weights_row_bytes=4;
    constexpr int symmetric_buffer=0,staging_pingpong_ub=0,staging_capacity=0;
    Trace trace;
// BODY
    for(int row=0;row<16;++row) {
        const bool expected=row>=recv_token_begin && row<recv_token_begin+worker_tokens &&
            !(channel && row>=sb && row<se);
        if(recv_topk_weights[row]!=unsigned(expected))return false;
    }
    return true;
}
constexpr bool all() {
    for(int sb=0;sb<=16;++sb) for(int se=sb;se<=16;++se)
    for(int begin=0;begin<=16;++begin) for(int count=0;count<=16-begin;++count)
        if(!check(begin,count,sb,se,true)||!check(begin,count,sb,se,false))return false;
    return true;
}
static_assert(all());
"""
    path = tmp_path / "self_output.cpp"
    path.write_text(source.replace("// BODY", body))
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True, timeout=30).stdout.lower()
    limit = "-fconstexpr-steps=100000000" if "clang" in version else "-fconstexpr-ops-limit=100000000"
    result = subprocess.run(
        [compiler, "-std=c++17", limit, "-c", str(path), "-o", str(tmp_path / "self.o")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("corrupt", [False, True])
def test_direct_topk_handle_reads_final_rows_not_stale_shmem(tmp_path, corrupt):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    kernel = (ROOT / "csrc/direct/kernels/dispatch/dispatch_udma.cpp").read_text()
    marker = (
        "tiling.metadata_direct_output && !tiling.omit_topk_output\n                    ? tiling.metadata_topk_output"
    )
    begin = kernel.index(marker)
    end = kernel.index("),", begin)
    selection = kernel[begin:end].replace("reinterpret_cast<uint64_t>(symmetric_buffer)", "symmetric_buffer")
    if corrupt:
        selection = "symmetric_buffer + tiling.topk_idx_buffer_offset"
    vf = vf_as_constexpr((ROOT / "csrc/direct/kernels/common/v2_handle.h").read_text(), "V2HandleSourceVf")
    source = r"""
#include <array>
#include <cstdint>
#define __simt_vf__ constexpr
#define __launch_bounds__(n)
#define __gm__
using topk_idx_t = int64_t;
constexpr void asc_threadfence() {}
// VF
constexpr bool check(bool direct, bool omit) {
    struct { bool metadata_direct_output, omit_topk_output;
             uint64_t metadata_topk_output, topk_idx_buffer_offset; } tiling{direct,omit,4096,128};
    constexpr uint64_t symmetric_buffer=8192;
    const uint64_t selected = SELECTION;
    // The direct tensor has a different first valid route than poisoned staging.
    std::array<topk_idx_t,8> final_topk{-1,-1,3,-1,-1,-1,-1,4};
    std::array<topk_idx_t,8> staging{1,-1,-1,-1,2,-1,-1,-1};
    const auto* indices=selected==4096?final_topk.data():staging.data();
    std::array<int32_t,4> ranks{1,0,2,0};
    std::array<int32_t,2> src{7,9};
    std::array<int32_t,18> output{};
    for(unsigned tid=0;tid<128;++tid)
        V2HandleSourceVf(ranks.data(),src.data(),indices,output.data(),0,2,64,4,3,0,1,tid);
    if(output[0]!=7 || output[6]!=73) return false;
    if(output[1]!=(direct&&!omit?2:0) || output[7]!=(direct&&!omit?7:4)) return false;
    for(unsigned i=12;i<18;++i) if(output[i]!=-1) return false;
    return true;
}
static_assert(check(false,false) && check(false,true) && check(true,false) && check(true,true));
"""
    path = tmp_path / "topk_consumer.cpp"
    path.write_text(source.replace("// VF", vf).replace("SELECTION", selection))
    result = subprocess.run(
        [compiler, "-std=c++17", "-c", str(path), "-o", str(tmp_path / "consumer.o")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if corrupt:
        assert result.returncode != 0 and "static assertion" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
