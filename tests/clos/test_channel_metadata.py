"""Compile/evaluate the real packing and WQE bodies without a device SDK."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.utils.cpp import cpp_functions

ROOT = Path(__file__).resolve().parents[2]


def test_output_publication_preserves_epoch_and_rejects_invalid_capacity(tmp_path):
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
            direct&&!cached?2048:0,direct?4096:0,direct&&!cached?4:0,direct?32:0);
        assert(!trapped && writes.back()==64+kOutputEpochWord);
        assert(writes.size()==(direct?8:6));
        assert(control[64+kOutputSrcWord]==(direct?(cached?0:2048):77));
        assert(control[64+kOutputWeightWord]==(direct?4096:77));
    }
    writes.clear(); trapped=false;
    PublishDispatchOutput(reinterpret_cast<uint8_t*>(control.data()),0,1,10,1024,0,3,3,16,0,0,0,4096,4,32);
    assert(trapped && writes.empty());
}
"""
    path = tmp_path / "publication.cpp"
    path.write_text(source.replace("// FUNCTIONS", functions))
    binary = tmp_path / ("publication.exe" if os.name == "nt" else "publication")
    result = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-I", str(ROOT / "csrc/direct"), str(path), "-o", str(binary)],
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


def test_channel_output_uses_remote_tensor_va_and_queue_lane(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    functions = cpp_functions(
        (ROOT / "csrc/direct/kernels/dispatch/udma_dispatch_sender.h").read_text(), "PutChannelOutput"
    )
    source = r"""
#include <cassert>
#include <array>
#include "common/dispatch_output.h"
#define __aicore__
#define __ubuf__
#define __gm__
using namespace ascend_deepep::udma;
constexpr unsigned kDispatchBasesOffset=32;
bool trapped=false;
void trap() { trapped=true; }
struct Base {
    static unsigned lane, peer, bytes, calls;
    static uint64_t destination, source;
    static unsigned QueuePeer(unsigned p) { return lane*4+p; }
    static void PutAbsolute(unsigned p,uint64_t dst,uint8_t* src,unsigned n) {
        ++calls; peer=p; destination=dst; source=reinterpret_cast<uint64_t>(src); bytes=n;
    }
};
unsigned Base::lane,Base::peer,Base::bytes,Base::calls;
uint64_t Base::destination,Base::source;
struct Sender {
    struct { unsigned num_topk=8; } td_;
    alignas(32) uint8_t scratch_[32+8*64]{};
    // FUNCTIONS
};
int main() {
    Sender sender;
    for(unsigned lane:{0U,1U}) for(unsigned field:{kOutputSrcWord,kOutputWeightWord}) {
        Base::lane=lane; Base::calls=0; trapped=false;
        auto* record=reinterpret_cast<uint64_t*>(sender.scratch_+32)+Base::QueuePeer(2)*8;
        record[2]=100; record[field]=0x1000000+lane*0x10000;
        sender.PutChannelOutput(2,field,7,0x800000,65);
        assert(!trapped && Base::calls==1 && Base::peer==2);
        const unsigned width=field==kOutputSrcWord?4:32;
        assert(Base::destination==record[field]+7*width && Base::source==0x800000 && Base::bytes==65*width);
        sender.PutChannelOutput(2,field,99,0x800000,2);
        assert(trapped && Base::calls==1);
        trapped=false;
        sender.PutChannelOutput(2,99,0,0,1);
        assert(trapped && Base::calls==1);
        trapped=false;
        record[field]=0;
        sender.PutChannelOutput(2,field,0,0x800000,1);
        assert(trapped && Base::calls==1);
    }
}
"""
    path = tmp_path / "channel_address.cpp"
    path.write_text(source.replace("// FUNCTIONS", functions))
    binary = tmp_path / ("channel_address.exe" if os.name == "nt" else "channel_address")
    result = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-I", str(ROOT / "csrc/direct"), str(path), "-o", str(binary)],
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


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("tile", [32, 64, 128, 256])
def test_real_metadata_and_wqe_bodies_constexpr(tmp_path, bits, corrupt, tile):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    kernels = ROOT / "csrc/direct/kernels"
    functions = vf_as_constexpr((kernels / "notify/notify_channel_metadata.h").read_text(), "PackChannelMetadataVf")
    functions += vf_as_constexpr((kernels / "common/udma_shared_sender.h").read_text(), "BuildWqes")
    functions += vf_as_constexpr((kernels / "dispatch/udma_dispatch_metadata.h").read_text(), "CopySelfChannelWordsVf")
    if corrupt:
        functions = functions.replace("uint32_t(ordinal)", "uint32_t(ordinal) + 1U")
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
        for(unsigned tid=0;tid<128;++tid)
            PackChannelMetadataVf(heads.data()+start*World,idx.data()+start*K,weights.data()+start*K,
                output.data(),ow.data(),n,K,World,16,C,Ch,S,Weighted,tid);
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
        for(unsigned j=0;j<5;++j) if(wqes[i*8+j]!=headers[(requests[i*4+3]*2+(i==23))*8+j])return false;
        if(wqes[i*8+5]!=requests[i*4] || wqes[i*8+6]!=requests[i*4+2] || wqes[i*8+7]!=requests[i*4+1])return false;
    }
    for(unsigned i=24*8;i<32*8;++i)if(wqes[i])return false;
    uint32_t input[129]{}, output[131]{};
    for(unsigned i=0;i<129;++i)input[i]=i+3;
    for(unsigned tid=0;tid<128;++tid)CopySelfChannelWordsVf(input,output+1,129,tid);
    if(output[0] || output[130])return false;
    for(unsigned i=0;i<129;++i)if(output[i+1]!=input[i])return false;
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
    functions = cpp_functions(
        (kernels / "notify/notify_channel_metadata.h").read_text(),
        "PackChannelMetadataVf",
        "LoadChannelMetadataInput",
        "PackChannelMetadataDoubleBuffered",
        "PackChannelMetadata",
    )
    functions += cpp_functions(
        (kernels / "dispatch/udma_dispatch_metadata.h").read_text(), "CopySelfChannelWordsVf", "CopySelfChannelMetadata"
    )
    harness = (Path(__file__).with_name("eight_aiv_copy_oracle.cpp")).read_text().split("using Row =", 1)[0]
    harness = harness.replace("enum class HardEvent {", "enum class HardEvent { MTE2_V, V_MTE2,").replace(
        "    T GetValue(uint64_t i) const",
        "    T* GetPhyAddr() const { return data; }\n    T GetValue(uint64_t i) const",
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
#define __simt_vf__
#define __launch_bounds__(n)
#define __ubuf__
#include "common/dispatch_channel_metadata.h"
struct { unsigned x; } threadIdx;
void asc_threadfence() {}
struct dim3 { unsigned x; explicit dim3(unsigned value):x(value){} };
template<auto Fn, class... Args> void asc_vf_call(dim3 n, Args... args) {
    for(threadIdx.x=0;threadIdx.x<n.x;++threadIdx.x)Fn(args...);
}
// FUNCTIONS
void check(Resource& resource, GM_ADDR head, GM_ADDR plan, GM_ADDR workspace, const DispatchTilingData& td) {
    PackChannelMetadata(resource,head,td,0);
    CopySelfChannelMetadata(resource,plan,workspace,td,0,3,17,32);
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
using namespace ascend_deepep;
constexpr bool check() {
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
