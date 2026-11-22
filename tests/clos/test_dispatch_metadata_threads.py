"""Execute actual metadata VF bodies against an independent sparse-route oracle."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.utils.cpp import cpp_functions

ROOT = Path(__file__).resolve().parents[2]


def test_metadata_vf_thread_variants(tmp_path):
    source = (ROOT / "csrc/direct/kernels/dispatch/udma_dispatch_metadata.h").read_text()
    functions = cpp_functions(source, "DispatchSourceIndicesVf", "DispatchTopkScatterSimtVf")
    host = (ROOT / "csrc/direct/ops/dispatch.cpp").read_text()
    begin = host.index("    const auto enabled_option =")
    options = host[begin : host.index("    const bool host_prep", begin)]
    begin = host.index("    tiling.metadata_src_threads =")
    fields = host[begin : host.index("    const bool cached_channel_ep16", begin)]
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++17 compiler required for production VF oracle")
    program = (
        r"""
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>
#define __simt_vf__
#define __launch_bounds__(n)
#define __gm__
namespace ascend_deepep { using topk_idx_t = int64_t; }
struct { unsigned x; } threadIdx;
void asc_threadfence() {}
uint8_t* base;
using Key = std::pair<unsigned, uint64_t>;
std::map<Key,double> actual, expected;
template<class T> void put(T* address,T value,unsigned peer) {
    Key key{peer,reinterpret_cast<uint8_t*>(address)-base};
    assert(actual.emplace(key,double(value)).second); // No duplicate writers.
}
namespace simt {
void aclshmemx_instance_ctx_set(unsigned) {}
void aclshmem_int32_p(int32_t* p,int32_t v,unsigned r) {put(p,v,r);}
void aclshmem_int64_p(int64_t* p,int64_t v,unsigned r) {put(p,v,r);}
void aclshmem_float_p(float* p,float v,unsigned r) {put(p,v,r);}
}
"""
        + functions
        + r"""
template<unsigned Threads,unsigned Lanes> void check(unsigned world,unsigned tokens,unsigned rank,unsigned ch,bool weighted) {
    constexpr unsigned channels=2,k=8,stride=256,idx_offset=65536,weight_offset=262144;
    unsigned begin=ch*3,nt=tokens+begin;
    std::vector<uint8_t> storage(1024*1024);base=storage.data();
    std::vector<int32_t> rp(world*world),cp(world*channels),plan(world*channels*stride,-1),head(nt*world,-1);
    std::vector<int64_t> idx(nt*k,-1);std::vector<float> weights(nt*k);
    std::vector<unsigned> count(world,0);
    for(unsigned peer=0;peer<world;++peer) {
        if(rank) rp[(rank-1)*world+peer]=11+peer;
        if(ch) cp[peer*channels+ch-1]=3+peer;
    }
    for(unsigned t=begin;t<nt;++t) {
        std::vector<bool> seen(world,false);
        for(unsigned j=0;j<k;++j) {
            auto i=t*k+j;weights[i]=float(i%61)/64;
            if(t%7==0 || j==3) continue;
            idx[i]=(t*7+j*3)%(world*32);unsigned peer=idx[i]/32;
            if(!seen[peer]) {
                seen[peer]=true;head[t*world+peer]=count[peer];
                plan[(peer*channels+ch)*stride+count[peer]++]=t;
            }
        }
    }
    expected.clear();actual.clear();
    for(unsigned peer=0;peer<world;++peer) {
        unsigned start=(rank?11+peer:0)+(ch?3+peer:0);
        cp[peer*channels+ch]=(ch?3+peer:0)+count[peer];
        for(unsigned t=begin;t<nt;++t) if(head[t*world+peer]>=0)
            expected[{peer,(start+head[t*world+peer])*4ULL}]=t;
    }
    for(threadIdx.x=0;threadIdx.x<Threads;++threadIdx.x)
        DispatchSourceIndicesVf<Threads,Lanes>(plan.data(),rp.data(),cp.data(),base,0,rank,world,channels,ch,stride,0);
    assert(actual==expected);
    actual.clear();expected.clear();
    for(unsigned t=begin;t<nt;++t) for(unsigned j=0;j<k;++j) {
        auto i=t*k+j;if(idx[i]<0)continue;
        unsigned peer=idx[i]/32,row=(rank?11+peer:0)+(ch?3+peer:0)+head[t*world+peer];
        expected[{peer,idx_offset+(row*k+j)*8ULL}]=idx[i]%32;
        if(weighted)expected[{peer,weight_offset+(row*k+j)*4ULL}]=weights[i];
    }
    for(threadIdx.x=0;threadIdx.x<Threads;++threadIdx.x)
        DispatchTopkScatterSimtVf<Threads>(idx.data(),weights.data(),base,head.data(),rp.data(),cp.data(),rank,channels,ch,
            begin,tokens,k,world,32,idx_offset,weight_offset,k*8,k*4,weighted,0);
    assert(actual==expected);
    actual[{0,999999}]=1;assert(actual!=expected); // Oracle detects corruption.
}
void env(const char* name,const char* value) {
#ifdef _WIN32
    _putenv_s(name,value?value:"");
#else
    if(value)setenv(name,value,1);else unsetenv(name);
#endif
}
void host_check(bool use_udma,unsigned world_size,unsigned a,unsigned b) {
    // HOST_OPTIONS
    struct { unsigned metadata_src_threads,metadata_src_lanes,metadata_topk_threads; } tiling{};
    // HOST_FIELDS
    assert(tiling.metadata_src_threads==(a?256U:128U));
    assert(tiling.metadata_src_lanes==(a && world_size<=8?32U:16U));
    assert(tiling.metadata_topk_threads==(b?256U:128U));
}
int main() {
    for(const char* a:{static_cast<const char*>(nullptr),"0","1","bad"}) for(const char* b:{static_cast<const char*>(nullptr),"0","1","bad"})
    for(unsigned ep:{8U,16U}) for(bool udma:{false,true}) {
        env("DEEPEP_CLOS_DISPATCH_SRC_IDX_256",a);env("DEEPEP_CLOS_DISPATCH_TOPK_256",b);
        bool on_a=a && std::strcmp(a,"1")==0,on_b=b && std::strcmp(b,"1")==0;
        bool invalid=(a && std::strcmp(a,"bad")==0)||(b && std::strcmp(b,"bad")==0)||(!udma&&(on_a||on_b));
        bool failed=false;
        try {host_check(udma,ep,on_a,on_b);}catch(const std::runtime_error&){failed=true;}
        assert(failed==invalid);
    }
    for(unsigned ep:{4U,8U,16U,32U}) for(unsigned n:{0U,1U,17U,129U})
    for(unsigned rank:{0U,3U}) for(unsigned ch:{0U,1U}) for(bool w:{false,true}) {
        check<128,16>(ep,n,rank,ch,w);
        check<256,16>(ep,n,rank,ch,w);
        check<256,32>(ep,n,rank,ch,w);
    }
}
"""
    )
    program = program.replace("// HOST_OPTIONS", options).replace("// HOST_FIELDS", fields)
    program = (
        '#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error("invalid option"); } while (0)\n'
        + program
    )
    path = tmp_path / "metadata.cpp"
    path.write_text(program)
    executable = tmp_path / ("metadata.exe" if os.name == "nt" else "metadata")
    subprocess.run([compiler, "-std=c++17", "-O2", str(path), "-o", str(executable)], check=True, timeout=180)
    try:
        subprocess.run([str(executable)], check=True, timeout=60)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            pytest.skip("compiled; Windows application control blocked execution (4551)")
        raise
