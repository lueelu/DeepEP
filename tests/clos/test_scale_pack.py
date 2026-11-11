"""Execute the production SIMT scale gather on CPU with independent row checks."""

from pathlib import Path
import shutil
import subprocess

import pytest

from tests.utils.cpp import cpp_functions


def test_notify_prefix_publication_and_host_zero_counts(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required")
    root = Path(__file__).resolve().parents[2]
    prefix = cpp_functions(
        (root / "csrc/direct/kernels/notify/notify_dispatch_plan.h").read_text(),
        "PrefixAndPublishChannelsVf",
    )
    publish = cpp_functions((root / "csrc/direct/kernels/common/v2_handle.h").read_text(), "NotifyHostCountsVf")
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
            for(threadIdx.x=0;threadIdx.x<128;++threadIdx.x)
                NotifyHostCountsVf(ranks.data(),counts.data(),host.data(),1,4,experts);
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
