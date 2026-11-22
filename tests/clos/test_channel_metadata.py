"""Compile/evaluate the real packing and WQE bodies without a device SDK."""

from pathlib import Path
import shutil
import subprocess

import pytest

from tests.utils.cpp import cpp_functions

ROOT = Path(__file__).resolve().parents[2]


def vf_as_constexpr(source, name):
    body = cpp_functions(source, name)
    # SIMT lanes run independently; inject the lane id as a constexpr argument.
    end = body.rfind(")", 0, body.index("{"))
    return (body[:end] + ", uint32_t tid" + body[end:]).replace("threadIdx.x", "tid")


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("corrupt", [False, True])
def test_real_metadata_and_wqe_bodies_constexpr(tmp_path, bits, corrupt):
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
    for(unsigned start=0;start<Tokens;start+=32) {
        unsigned n=Tokens-start<32?Tokens-start:32;
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
static_assert(pack_check<256,2,true>());
static_assert(wqe_check());
"""
    path = tmp_path / "channel_metadata.cpp"
    path.write_text(source.replace("INDEX_TYPE", f"int{bits}_t").replace("// FUNCTIONS", functions))
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


def test_pack_and_local_copy_template_calls_compile(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    kernels = ROOT / "csrc/direct/kernels"
    functions = cpp_functions(
        (kernels / "notify/notify_channel_metadata.h").read_text(), "PackChannelMetadataVf", "PackChannelMetadata"
    )
    functions += cpp_functions(
        (kernels / "dispatch/udma_dispatch_metadata.h").read_text(), "CopySelfChannelWordsVf", "CopySelfChannelMetadata"
    )
    harness = (Path(__file__).with_name("eight_aiv_copy_oracle.cpp")).read_text().split("using Row =", 1)[0]
    harness = harness.replace("enum class HardEvent {", "enum class HardEvent { MTE2_V, V_MTE2, V_S,").replace(
        "    T GetValue(uint64_t i) const",
        "    T* GetPhyAddr() const { return data; }\n    T GetValue(uint64_t i) const",
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
    path.write_text(harness.replace("// FUNCTIONS", functions))
    result = subprocess.run(
        [compiler, "-std=c++17", "-c", "-I", str(ROOT / "csrc/direct"), str(path), "-o", str(tmp_path / "calls.o")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
