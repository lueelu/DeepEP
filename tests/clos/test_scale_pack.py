"""Execute the production SIMT scale gather on CPU with independent row checks."""

from pathlib import Path
import shutil
import subprocess

import pytest

from tests.utils.cpp import cpp_functions


def test_serial_aggregate_self_copy_native_gate(tmp_path):
    """Execute native tiling selection with serial and overlapping producers."""
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required")
    root = Path(__file__).resolve().parents[2]
    source = (root / "csrc/direct/ops/combine.cpp").read_text()
    begin = source.index("    if (use_udma && epilogue && !tiling.udma_token_schedule")
    end = source.index("    tiling.udma_weight_mode", begin)
    gate = source[begin:end]
    harness = (
        r"""
#include <cassert>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("gate"); } while (0)
namespace ascend_deepep::udma {
int MetadataFirst(int, int, int, int, int, int) { return 6; }
}
struct Tiling {
    bool udma_token_schedule = false, udma_self_copy_after_submit = false;
    int udma_self_copy_first_aiv = 0;
    int world_size = 4, udma_peers_per_worker = 1, block_dim = 48;
    int udma_peer_order = 0, udma_ranks_per_group = 4, udma_workers_per_peer = 2;
};
int main() {
    for (bool expanded_x : {false,true})
    for (bool aggregate : {false,true})
    for (int pipeline_chunk_tokens : {0,32})
    for (bool topk_weights : {false,true})
    for (bool use_udma : {false,true})
    for (bool epilogue : {false,true})
    for (bool token : {false,true})
    for (int mode : {0,1,2}) {
        Tiling tiling;
        tiling.udma_token_schedule = token;
        struct { int self_copy; } udma_launch{mode};
"""
        + gate
        + r"""
        bool serial_peer = use_udma && epilogue && !token && !aggregate && !pipeline_chunk_tokens;
        assert(tiling.udma_self_copy_after_submit == (serial_peer && mode == 2));
        assert(tiling.udma_self_copy_first_aiv ==
               (serial_peer && mode == 1 && !topk_weights ? 6 : 0));
    }
}
"""
    )
    path = tmp_path / "combine_gate.cpp"
    executable = tmp_path / "combine_gate"
    path.write_text(harness)
    subprocess.run([compiler, "-std=c++17", "-O2", str(path), "-o", str(executable)], check=True, timeout=60)
    try:
        subprocess.run([str(executable)], check=True, timeout=30)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            pytest.skip("compiled, but Windows application control blocked execution (4551)")
        raise


def test_batched_notify_count_tiles_and_source_ownership(tmp_path):
    """Run production integer reductions against scalar sums, including tails."""
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required")
    root = Path(__file__).resolve().parents[2]
    functions = cpp_functions(
        (root / "csrc/direct/kernels/notify/notify_dispatch.cpp").read_text(),
        "NotifyCountTileWidth",
        "NotifyCountRowsPerOwner",
        "PrefixNotifyCountsVf",
        "ReduceNotifyExpertsVf",
    )
    harness = (
        r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#define __aicore__
#define __simt_vf__
#define __launch_bounds__(...)
#define __gm__
#define __ubuf__
constexpr uint32_t kMaxVectorTileElems = 16384, kNotifyInt32AlignElems = 8;
struct { uint32_t x; } threadIdx;
void asc_threadfence() {}
"""
        + functions
        + r"""
int main() {
    assert(NotifyCountTileWidth(0, 16384) == 0);
    assert(NotifyCountTileWidth(128, 1023) == 0);
    assert(NotifyCountTileWidth(2049, UINT64_MAX) == 0);
    assert(NotifyCountTileWidth(256, UINT64_MAX) == 64);
    for (uint32_t world : {2U,3U,4U,7U,8U,16U,32U,64U,128U,192U,256U,512U})
    for (uint32_t cores : {1U,2U,6U,8U,16U,20U,32U,48U,64U}) {
        uint32_t rs=(world+7)/8*8, group=NotifyCountRowsPerOwner(world);
        std::vector<int> owners(world,-1), lines((world*world+7)/8,-1);
        for(uint32_t c=0;c<cores;++c)
        for(uint32_t first=c*group;first<world;first+=cores*group)
        for(uint32_t s=first;s<std::min(world,first+group);++s) {
            assert(owners[s]==-1); owners[s]=c;
            for(uint32_t l=s*world/8;l<=(s*world+world-1)/8;++l) {
                assert(lines[l]==-1 || lines[l]==int(c)); lines[l]=c;
            }
        }
        for(auto c:owners)assert(c>=0);
        uint32_t tile=NotifyCountTileWidth(world,16384);
        uint32_t ec=NotifyCountTileWidth(world,uint64_t(cores)*rs);
        if(!ec)continue; // Capacity fallback remains the original kernel.
        assert(uint64_t(world)*tile<=16384 && uint64_t(world)*ec<=uint64_t(cores)*rs);
        for(uint32_t iteration=0;iteration<2;++iteration) {
            // Preserve an independent input: output overwrites the raw GM stage.
            std::vector<int32_t> counts(world*world), stage(world*world+8,-123), pub(world*world+8,-123);
            for(uint32_t s=0;s<world;++s)for(uint32_t d=0;d<world;++d)
                stage[s*world+d]=counts[s*world+d]=(!iteration || d==0 ? 0 : (s*13+d*7)%97);
            for(uint32_t b=0;b<world;b+=tile) {
                uint32_t n=std::min(tile,world-b), stride=(n+7)/8*8;
                std::vector<int32_t> ub(world*stride,-999);
                for(uint32_t s=0;s<world;++s)for(uint32_t j=0;j<n;++j)ub[s*stride+j]=stage[s*world+b+j];
                for(threadIdx.x=0;threadIdx.x<128;++threadIdx.x)
                    PrefixNotifyCountsVf(ub.data(),stage.data(),pub.data(),world,b,n,stride);
            }
            for(uint32_t d=0;d<world;++d) {
                int32_t sum=0;
                for(uint32_t s=0;s<world;++s) {
                    sum+=counts[s*world+d];
                    assert(stage[s*world+d]==sum && pub[s*world+d]==sum);
                }
            }
            for(uint32_t i=0;i<8;++i)assert(stage[world*world+i]==-123 && pub[world*world+i]==-123);
            for(uint32_t local : {1U,3U,4U,7U,8U,9U,31U,64U,129U,257U}) {
                uint32_t width=std::min(ec,(local+7)/8*8);
                std::vector<int32_t> out(local+8,-123), expected(local,0);
                for(uint32_t s=0;s<world;++s)for(uint32_t e=0;e<local;++e)
                    expected[e]+=(!iteration || e==0 ? 0 : (s*11+e*17)%53);
                for(uint32_t b=0;b<local;b+=width) {
                    uint32_t n=std::min(width,local-b);
                    std::vector<int32_t> scratch(world*width,-999);
                    for(uint32_t s=0;s<world;++s)for(uint32_t e=0;e<n;++e)
                        scratch[s*width+e]=(!iteration || b+e==0 ? 0 : (s*11+(b+e)*17)%53);
                    for(threadIdx.x=0;threadIdx.x<128;++threadIdx.x)
                        ReduceNotifyExpertsVf(scratch.data(),out.data(),world,b,n,width);
                }
                for(uint32_t e=0;e<local;++e)assert(out[e]==expected[e]);
                for(uint32_t i=0;i<8;++i)assert(out[local+i]==-123);
            }
        }
    }
}
"""
    )
    source, binary = tmp_path / "notify_tiles.cpp", tmp_path / "notify_tiles"
    source.write_text(harness)
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True, capture_output=True)
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=60)


def test_notify_prefix_publication_and_host_count_packing(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required")
    root = Path(__file__).resolve().parents[2]
    prefix = cpp_functions(
        (root / "csrc/direct/kernels/notify/notify_dispatch_plan.h").read_text(),
        "PrefixAndPublishChannelsVf",
    )
    publish = cpp_functions(
        (root / "csrc/direct/kernels/notify/notify_dispatch.cpp").read_text(), "PackNotifyHostCountsVf"
    )
    harness = (
        r"""
#include <cassert>
#include <cstdint>
#include <vector>
#define __simt_vf__
#define __launch_bounds__(...)
#define __gm__
#define __ubuf__
struct { uint32_t x; } threadIdx;
void asc_threadfence() {}
"""
        + prefix
        + publish
        + r"""
int main() {
    for (uint32_t world : {2U,4U,8U,64U,128U,256U})
    for (uint32_t channels : {2U,6U,8U,20U,48U,64U}) {
        uint32_t rs=(world+7)/8*8, cs=(channels+7)/8*8;
        std::vector<int32_t> input(rs*channels, -999), out(world*channels+8,-123);
        std::vector<int32_t> padded(world*cs+8,-123);
        for(uint32_t c=0;c<channels;++c)for(uint32_t r=0;r<world;++r)
            input[c*rs+r]=(r==0 ? 0 : (c*13+r*7)%9);
        for(threadIdx.x=0;threadIdx.x<128;++threadIdx.x)
            PrefixAndPublishChannelsVf(input.data(),out.data(),padded.data(),world,rs,channels,cs);
        for(uint32_t r=0;r<world;++r) {
            int32_t expected=0;
            for(uint32_t c=0;c<cs;++c) {
                if(c<channels) {
                    expected+=input[c*rs+r];
                    assert(out[r*channels+c]==expected);
                    assert(padded[r*cs+c]==expected);
                } else assert(padded[r*cs+c]==0);
            }
        }
        for(uint32_t i=0;i<8;++i) {
            assert(out[world*channels+i]==-123);
            assert(padded[world*cs+i]==-123);
        }
    }
    for(uint32_t experts : {1U,3U,64U,129U,256U}) {
        std::vector<int32_t> ranks(16,0), counts(experts,0);
        std::vector<uint32_t> host(experts+2,0);
        for(uint32_t round=0;round<20;++round) {
            for(uint32_t i=0;i<=experts;++i)host[i]=0;
            host[experts+1]=0xdeadbeef;
            ranks[13]=round;
            for(uint32_t i=0;i<experts;++i) counts[i]=i%2 ? round*7 : 0;
            // Exercise the actual UB packer, including a nonzero chunk offset.
            // The kernel publishes this packed snapshot to mapped host via MTE3.
            for(uint32_t offset=0;offset<=experts;offset+=7) {
                uint32_t remaining=experts+1-offset, length=remaining<7?remaining:7;
                for(threadIdx.x=0;threadIdx.x<128;++threadIdx.x)
                    PackNotifyHostCountsVf(ranks.data()+13,counts.data(),host.data()+offset,offset,length);
            }
            assert(host[0]==round+1);
            for(uint32_t i=0;i<experts;++i)assert(host[i+1]==uint32_t(counts[i])+1);
            assert(host[experts+1]==0xdeadbeef);
        }
    }
}
"""
    )
    source, binary = tmp_path / "notify.cpp", tmp_path / "notify"
    source.write_text(harness)
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True, capture_output=True)
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=30)


def test_peer_scale_pack_preserves_bits_strides_and_empty_channels(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required")
    root = Path(__file__).resolve().parents[2]
    source = (root / "csrc/direct/kernels/dispatch/dispatch_scale_pack.h").read_text()
    kernel = cpp_functions(source, "PackDispatchPeerScalesVf")
    harness = (
        r"""
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define __simt_vf__
#define __launch_bounds__(...)
#define __gm__
struct { uint32_t x; } threadIdx;
constexpr uint32_t kDispatchScalePackThreads = 128;
void asc_threadfence() {}
namespace __asc_simt_vf { void __trap() { throw std::runtime_error("invalid plan"); } }
"""
        + kernel
        + r"""
int main() {
    for (uint32_t tokens : {0U, 1U, 37U, 257U})
    for (uint32_t channels : {2U, 8U, 32U})
    for (uint32_t workers : {1U, 3U, 48U})
    for (bool column_major : {false, true}) {
        const uint32_t world = 8, rank = 3, scales = 56;
        const uint32_t chunk = (tokens + channels - 1) / channels;
        const uint32_t stride = (chunk + 7) / 8 * 8;
        const uint64_t ts = column_major ? 1 : scales + 3;
        const uint64_t ss = column_major ? tokens + 5 : 1;
        std::vector<uint32_t> input((uint64_t(tokens) + 1) * ts + scales * ss, 0);
        for (uint32_t t = 0; t < tokens; ++t)
            for (uint32_t c = 0; c < scales; ++c)
                input[t * ts + c * ss] = 0x7f800001U ^ (t * 131U + c * 65537U);
        const uint32_t untouched = 0xdeadbeefU;
        std::vector<uint32_t> output(uint64_t(world) * tokens * scales, untouched);
        std::vector<int32_t> plan(world * channels * stride, -1), prefix(world * channels, 0);
        for (uint32_t p = 0; p < world; ++p) {
            uint32_t total = 0;
            for (uint32_t ch = 0; ch < channels; ++ch) {
                uint32_t count = 0;
                for (uint32_t t = ch * chunk; t < tokens && t < (ch + 1) * chunk; ++t)
                    if (p != 0 && (t + p) % 3 != 0) {
                        plan[(p * channels + ch) * stride + count++] = t;
                        ++total;
                    }
                prefix[p * channels + ch] = total;
            }
        }
        for (uint32_t worker = 0; worker < workers; ++worker)
            for (threadIdx.x = 0; threadIdx.x < 128; ++threadIdx.x)
                PackDispatchPeerScalesVf(input.data(), output.data(), plan.data(), prefix.data(),
                    rank, world, channels, stride, tokens, scales, ts, ss, worker, workers);
        for (uint32_t p = 0; p < world; ++p) {
            uint32_t row = 0;
            if (p != rank && p != 0)
                for (uint32_t t = 0; t < tokens; ++t)
                    if ((t + p) % 3 != 0) {
                        for (uint32_t c = 0; c < scales; ++c)
                            assert(output[(uint64_t(p) * tokens + row) * scales + c] ==
                                   (0x7f800001U ^ (t * 131U + c * 65537U)));
                        ++row;
                    }
            for (; row < tokens; ++row)
                for (uint32_t c = 0; c < scales; ++c)
                    assert(output[(uint64_t(p) * tokens + row) * scales + c] == untouched);
        }
        // A producer must reject corrupt routing before reading beyond input.
        if (tokens) {
            prefix[channels] = 1;
            plan[channels * stride] = tokens;
            threadIdx.x = 0;
            bool failed = false;
            try {
                PackDispatchPeerScalesVf(input.data(), output.data(), plan.data(), prefix.data(),
                    rank, world, channels, stride, tokens, scales, ts, ss, 0, 1);
            } catch (const std::runtime_error&) { failed = true; }
            assert(failed);
        }
    }
}
"""
    )
    path = tmp_path / "scale_pack.cpp"
    executable = tmp_path / "scale_pack"
    path.write_text(harness)
    subprocess.run([compiler, "-std=c++17", "-O2", str(path), "-o", str(executable)], check=True, timeout=60)
    subprocess.run([str(executable)], check=True, timeout=30)
