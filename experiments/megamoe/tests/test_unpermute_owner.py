# Copyright (c) 2026, Lu Lu

"""Check host buffers against every device output partition after owner removal."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_unpermute_owner_exclusion_preserves_exact_output_coverage_and_buffers(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Unpermute compilation requires g++ or clang++")
    directory = Path(__file__).parent
    kernels = directory.parent / "csrc_ep32/kernels/mega_moe"
    host = (directory.parent / "csrc/forward.cpp").read_text(encoding="utf-8")
    config_functions = host[
        host.index("MegaMoeUnpermuteBufferConfig UnpermuteConfig(") : host.index("PlatformInfo GetPlatformInfo()")
    ]
    source = tmp_path / "unpermute.cpp"
    source.write_text(
        """
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#include "mega_moe_peermem.hpp"
using namespace MegaMoeImpl;
template <typename T> T CeilDiv(T v, T d) { return (v + d - 1) / d; }
template <typename T> T AlignUp(T v, T a) { return CeilDiv(v, a) * a; }
"""
        + config_functions
        + """
int main() {
    for (auto topology : {TOPO_TYPE_MTE, TOPO_TYPE_MTE_ROCE})
    for (uint32_t physical : {2U, 4U, 16U, 48U, 56U, 64U})
    for (uint32_t bs = 1U; bs <= 65U; ++bs)
    for (uint32_t topk : {1U, 8U, 32U}) {
        if (topology == TOPO_TYPE_MTE_ROCE && physical < 28U) continue;
        MegaMoeTilingData t{};
        t.topoType = topology;
        t.blockAivNum = physical;
        t.bs = bs;
        t.topK = topk;
        t.h = 3584U;
        SetUnpermuteConfigs(t, 192U * 1024U - 256U);
        const uint32_t jobs = UnpermuteJobCount(&t);
        assert(jobs == physical - (topology == TOPO_TYPE_MTE_ROCE ? 16U : 0U));
        std::vector<unsigned> owners(bs, 0U);
        for (uint32_t p = 0; p < physical; ++p) {
            const uint32_t logical = UnpermuteJobIndex(&t, p);
            const bool expectedOwner = topology == TOPO_TYPE_MTE_ROCE &&
                p % 2U == 0U && p / 2U < 16U;
            assert(IsHierarchicalCombineOwner(&t, p) == expectedOwner);
            if (expectedOwner) assert(HierarchicalCombineLane(&t, p) == p / 2U);
            if (IsHierarchicalCombineOwner(&t, p)) {
                assert(logical >= jobs);
                continue;
            }
            assert(logical < jobs);
            const uint32_t chunk = CeilDiv(bs, jobs), start = logical * chunk;
            if (start >= bs) continue;
            const uint32_t count = std::min(chunk, bs - start);
            const auto& c = logical < t.unpermuteFullTokenChunkCoreCount
                ? t.unpermuteConfigForFullTokenChunk : t.unpermuteConfigForTailTokenChunk;
            assert(c.tokensPerBatch > 0 && static_cast<uint32_t>(c.tokensPerBatch) <= count);
            assert(c.inputBufferCount >= MIN_UNPERMUTE_INPUT_BUFFER_COUNT);
            for (uint32_t row = start; row < start + count; ++row) ++owners[row];
        }
        assert(std::all_of(owners.begin(), owners.end(), [](unsigned n) { return n == 1U; }));
    }
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / "unpermute"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{kernels / 'common'}",
            f"-I{directory / 'layout_stubs'}",
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
