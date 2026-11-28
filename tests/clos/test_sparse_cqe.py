"""Check production payload WQEs, receive mappings and index lifetimes on CPU."""

from pathlib import Path
import shutil
import subprocess

import pytest

from tests.utils.cpp import cpp_functions


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("limit,corrupt", [(n, False) for n in (1, 2, 3, 4, 5, 8, 16, 31, 32)] + [(4, True)])
def test_coalesced_combine_payload_mapping(tmp_path, limit, corrupt):
    """Expand actual production WQEs back into rows and compare every address.

    Compile-time checks include tails, span boundaries, multiple headers/QPs,
    dense peer ordinals, capacity tails and the conservative byte cap.
    """
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++17 compiler required")
    text = (ROOT / "csrc/direct/kernels/combine/udma_combine_sender.h").read_text()
    body = cpp_functions(text, "BuildCombineWqesCoalesced")
    end = body.rfind(")", 0, body.index("{"))
    body = (body[:end] + ", Warp& warp" + body[end:]).replace("threadIdx.x", "warp.lane")
    body = body.replace("asc_ballot(", "warp.ballot(").replace("asc_shfl_up(", "warp.up(")
    program = r"""
#include <cstdint>
#define __simt_vf__ constexpr
#define __launch_bounds__(n)
#define __ubuf__
namespace __asc_simt_vf { void __trap(); }
constexpr int32_t index_at(uint64_t seed, unsigned pos) {
    return int32_t(seed + pos);
}
// Emulate exactly the three ballots and two shuffles. The scalar golden
// computes masks independently, then every production lane must supply the
// expected predicate/value; wrong participation/order fails the test too.
struct Warp {
    unsigned lane=0, ballots=0, shuffles=0, run_heads=0, heads=0;
    unsigned tokens[32]{}, owners[32]{};
    bool valid=true;
    constexpr unsigned ballot(bool predicate) {
        unsigned mask=ballots==0 ? 0 : ballots==1 ? run_heads : heads;
        valid &= ballots<3 && predicate==bool((mask>>lane)&1U);
        ++ballots;
        return mask;
    }
    constexpr unsigned up(unsigned value,unsigned delta,unsigned width) {
        const auto* values=shuffles==0 ? tokens : owners;
        valid &= shuffles<2 && delta==1 && width==32 && value==values[lane];
        ++shuffles;
        return values[lane ? lane-1 : 0];
    }
};
// PRODUCTION
constexpr bool check(unsigned count,unsigned width,unsigned bytes,unsigned pattern, bool adjacent=false) {
    uint64_t wqes[256]{}, spans[128]{}, headers[16*32]{}, bases[32]{};
    for(unsigned p=0;p<32;++p) {
        bases[p]=0x100000000ULL+uint64_t(p)*10000000;
        for(unsigned h=0;h<16;++h)headers[p*16+h]=p*100+h;
    }
    unsigned n=0, output=999;
    for(unsigned start=0;start<count;start+=width,++n) {
        unsigned len=count-start<width ? count-start : width;
        spans[n*4]=adjacent ? 8192-count+start : n*128+pattern;
        spans[n*4+1]=(uint64_t(len)<<32)|(17+start);
        spans[n*4+2]=n;
        spans[n*4+3]=start;
    }
    Warp warp;
    unsigned cap=bytes<=65536 ? 65536/bytes : 1;
    if(cap>LIMIT)cap=LIMIT;
    for(unsigned s=0;s<n;++s) {
        unsigned begin=unsigned(spans[s*4+3]), len=unsigned(spans[s*4+1]>>32);
        for(unsigned pos=0;pos<len;++pos) {
            unsigned lane=begin+pos;
            warp.tokens[lane]=unsigned(index_at(spans[s*4],pos));
            warp.owners[lane]=s;
            if(pos==0 || index_at(spans[s*4],pos)!=index_at(spans[s*4],pos-1)+1)
                warp.run_heads |= 1U<<lane;
        }
        for(unsigned pos=0;pos<len;) {
            warp.heads |= 1U<<(begin+pos);
            unsigned rows=1;
            while(rows<cap && pos+rows<len &&
                  index_at(spans[s*4],pos+rows)==index_at(spans[s*4],pos)+int(rows))++rows;
            pos+=rows;
        }
    }
    // Reverse order also proves no output relies on another lane's UB writes.
    for(unsigned t=32;t>0;--t) {
        warp.lane=t-1; warp.ballots=0; warp.shuffles=0;
        BuildCombineWqesCoalesced(wqes,spans,headers,bases,&output,0x200000000ULL,
                                 512,bytes,8192,n,count,LIMIT,warp);
        if(!warp.valid || warp.ballots!=(count?3U:0U) || warp.shuffles!=(count?2U:0U))return false;
    }
    if(output>count || (count && !output))return false;
    unsigned cursor=0, expected_count=0;
    for(unsigned s=0;s<n;++s) {
        unsigned len=unsigned(spans[s*4+1]>>32), pos=0;
        while(pos<len) {
            unsigned cap=bytes<=65536 ? 65536/bytes : 1;
            if(cap>LIMIT)cap=LIMIT;
            unsigned expected_rows=1;
            while(expected_rows<cap && pos+expected_rows<len &&
                  index_at(spans[s*4],pos+expected_rows)==index_at(spans[s*4],pos)+int(expected_rows))
                ++expected_rows;
            auto* w=wqes+cursor*8;
            if(w[6]!=uint64_t(expected_rows)*bytes)return false;
            for(unsigned h=0;h<5;++h)if(w[h]!=headers[s*16+h])return false;
            for(unsigned r=0;r<expected_rows;++r) {
                if(w[5]+uint64_t(r)*bytes!=bases[s]+512+uint64_t(index_at(spans[s*4],pos+r))*bytes)return false;
                if(w[7]+uint64_t(r)*bytes!=0x200000000ULL+
                   (uint64_t(uint32_t(spans[s*4+1]))+pos+r)*bytes)return false;
            }
            pos+=expected_rows; ++cursor; ++expected_count;
        }
    }
    if(output!=expected_count)return false;
    for(unsigned i=output*8;i<256;++i)if(wqes[i])return false;
    return true;
}
constexpr bool matrix(unsigned bytes) {
    for(unsigned count : {0U,1U,2U,15U,16U,17U,31U,32U})
        for(unsigned width : {1U,3U,5U,16U,32U})
            for(unsigned pattern=0;pattern<5;++pattern)
                if(!check(count,width,bytes,pattern))return false;
    // Consecutive destinations across spans must still produce distinct WQEs;
    // the final valid token also touches the capacity boundary.
    for(unsigned width : {1U,3U,5U,16U,32U})
        if(!check(32,width,bytes,0,true))return false;
    return true;
}
static_assert(matrix(5632));
static_assert(matrix(8192));
static_assert(matrix(12288));
static_assert(matrix(65568));
"""
    program = "#include <initializer_list>\n" + program.replace("// PRODUCTION", body).replace("LIMIT", str(limit))
    if corrupt:
        program = program.replace("wqe[6] = uint64_t(rows) * x_bytes;", "wqe[6] = x_bytes;")
    path = tmp_path / "coalesce.cpp"
    path.write_text(program)
    budget = "-fconstexpr-steps=100000000" if "clang" in Path(compiler).name else "-fconstexpr-ops-limit=100000000"
    result = subprocess.run(
        [compiler, "-std=c++17", budget, "-c", str(path), "-o", str(tmp_path / "coalesce.o")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if corrupt:
        assert result.returncode != 0 and "static assertion" in result.stderr, result.stderr
    else:
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("corrupt", [False, True])
def test_dense_payload_reduce_mapping(tmp_path, corrupt):
    """Compile-time sparse-route scatter golden, independent of reduce ownership."""
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++17 compiler required")
    kernel = (ROOT / "csrc/direct/kernels/combine/combine_udma.cpp").read_text()
    mapping = cpp_functions(kernel, "CompactWeightSlot").replace("__aicore__ inline", "constexpr")
    if corrupt:
        mapping = mapping.replace("channel - 1U", "channel")
    program = r"""
#include <cstdint>
#include <initializer_list>
#define ASCEND_DEEPEP_ARCH 3510
#include "common/types.h"
#include "common/platform.h"
namespace AscendC {
template<class T> struct LocalTensor {
    const T* data;
    constexpr T GetValue(unsigned i) const { return data[i]; }
};
}
// MAPPING
constexpr bool check(unsigned world, unsigned channels, unsigned cores, unsigned generation) {
    constexpr unsigned tokens=67, capacity=73;
    int32_t prefix[128*64]{}, heads[tokens*128]{}, expected[tokens*128]{};
    ascend_deepep::CombineTilingData td{};
    td.world_size=world; td.num_channels=channels; td.num_tokens=tokens; td.token_capacity=capacity;
    const unsigned channel_tokens=(tokens+channels-1)/channels;
    for(unsigned p=0;p<world;++p) {
        unsigned dense_row=0;
        for(unsigned c=0;c<channels;++c) {
            unsigned ordinal=0;
            for(unsigned t=c*channel_tokens;t<tokens && t<(c+1)*channel_tokens;++t) {
                // Empty peers, repeated channel prefixes, sparse tokens and tails.
                bool route=p%7!=0 && (t*3+p+generation)%5<2;
                heads[t*world+p]=route ? int32_t(ordinal++) : -1;
                expected[t*world+p]=route ? int32_t(dense_row++) : -1;
            }
            prefix[p*channels+c]=int32_t(dense_row);
        }
    }
    AscendC::LocalTensor<int32_t> ub{prefix};
    unsigned per_core=(tokens+cores-1)/cores;
    for(unsigned core=0;core<cores;++core)
        for(unsigned t=core*per_core;t<tokens && t<(core+1)*per_core;++t)
            for(unsigned p=0;p<world;++p) {
                int32_t ordinal=heads[t*world+p];
                if(ordinal<0)continue;
                uint64_t slot=CompactWeightSlot(td,ub,p,t,unsigned(ordinal));
                if(slot!=uint64_t(p)*capacity+unsigned(expected[t*world+p]))return false;
                if(slot<uint64_t(p)*capacity || slot>=uint64_t(p+1)*capacity)return false;
            }
    // Same production UB ledger: aligned metadata + five buffers at 14 B/element.
    unsigned bytes=(world*channels*4+31)/32*32;
    unsigned chunk=(ascend_deepep::platform::kKernelUbBytes-bytes)/(14*64)*64;
    return chunk>=64 && bytes+14*chunk<=ascend_deepep::platform::kKernelUbBytes;
}
constexpr bool matrix() {
    for(unsigned w : {4U,8U,16U,32U,64U,128U})
        for(unsigned ch : {8U,16U,32U,64U})
            for(unsigned cores : {8U,16U,32U,48U,64U})
                for(unsigned gen : {0U,1U})
                    if(!check(w,ch,cores,gen))return false;
    return true;
}
static_assert(matrix());
"""
    path = tmp_path / "dense_mapping.cpp"
    path.write_text(program.replace("// MAPPING", mapping))
    budget = "-fconstexpr-steps=100000000" if "clang" in Path(compiler).name else "-fconstexpr-ops-limit=100000000"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            budget,
            "-I",
            str(ROOT / "csrc/direct"),
            "-c",
            str(path),
            "-o",
            str(tmp_path / "dense.o"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert (result.returncode != 0) == corrupt, result.stderr
    if corrupt:
        assert "static assertion" in result.stderr, result.stderr


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
