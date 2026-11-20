# Copyright (c) 2026, Lu Lu

"""Compile the host tiling and device payload-capacity calculation together."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_relay_packet_capacity_and_bs32_fast_path_buffers(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Relay capacity compilation requires g++ or clang++")
    directory = Path(__file__).parent
    kernels = directory.parent / "csrc_ep32/kernels/mega_moe"
    host = (directory.parent / "csrc_ep32/forward.cpp").read_text(encoding="utf-8")
    host = host[host.index("uint32_t DispatchCopyBytes(") : host.index("uint64_t HostFlagElements(")]
    device = (kernels / "stage/mega_moe_send_mask.hpp").read_text(encoding="utf-8")
    start = device.index("__aicore__ inline int32_t RailItemsPerBatch(")
    device = device[start : device.index("\n}", start) + 2]
    source = tmp_path / "relay_capacity.cpp"
    source.write_text(
        """
#include <algorithm>
#include <cassert>
#include <cstdint>
#define __aicore__
#define TORCH_CHECK(ok, ...) assert(ok)
#include "mega_moe_peermem.hpp"
using namespace MegaMoeImpl;
template <typename T> T CeilDiv(T v, T d) { return (v + d - 1) / d; }
template <typename T> T AlignUp(T v, T a) { return CeilDiv(v, a) * a; }
"""
        + host
        + device
        + """
int main() {
    for (auto topology : {TOPO_TYPE_MTE, TOPO_TYPE_MTE_ROCE})
    for (uint32_t bs : {1U, 31U, 32U, 33U, 48U, 64U, 72U, 96U, 128U, 138U, 144U, 255U, 256U})
    for (uint32_t topk : {1U, 6U, 8U, 32U}) {
        MegaMoeTilingData t{};
        t.topoType = topology; t.bs = t.numMaxTokensPerRank = bs; t.topK = topk;
        t.groupedMatmulMode = GROUPED_MATMUL_MODE_A8W4;
        t.epWorldSize = 32; t.dispatchRanksPerNode = 16; t.blockAivNum = 56;
        constexpr uint32_t fixed = 32768, ub = 192U * 1024U - HIERARCHICAL_ROCE_UB_BYTES;
        t.h = bs >= 96 ? 5120U : 3584U; t.moeExpertPerRank = bs >= 96 ? 13U : 12U;
        for (uint32_t limit : {bs, 4096U}) {
            t.numMaxTokensPerRank = limit;
            auto d = DispatchConfig(t, ub);
            const uint32_t used = AlignUp(32U*t.moeExpertPerRank*4U,32U) + AlignUp(t.moeExpertPerRank*4U,32U)
                       + 4U*d.routeItemsPerBatch + d.bufferCount*(d.copyBufferBytes+32U);
            assert(used <= ub);
            if (topology == TOPO_TYPE_MTE_ROCE) {
                assert(static_cast<uint32_t>(d.routeItemsPerBatch) >= std::max(16U * AlignUp(bs,8U), AlignUp(limit,256U)));
                auto tight = DispatchConfig(t, AlignUp(32U*t.moeExpertPerRank*4U,32U) + AlignUp(t.moeExpertPerRank*4U,32U)
                       + MIN_DISPATCH_BUFFER_COUNT*(d.copyBufferBytes+32U) + 64U);
                assert(tight.routeItemsPerBatch == 16 && tight.bufferCount == MIN_DISPATCH_BUFFER_COUNT);
            }
        }
        t.numMaxTokensPerRank = bs;
        auto c = SendMaskConfig(t, fixed, 8U, ub);
        const uint32_t total = bs * topk;
        const uint32_t capacity = UseDecodeRailRouting(&t) ? RailItemsPerBatch(c) : c.routeItemsPerBatch;
        assert(static_cast<uint32_t>(c.routeBatchCount) == CeilDiv(total, capacity));
        assert(fixed + 2U * c.routeItemsPerBatch * 4U + c.bufferCount * c.bufferBytes <= ub);
        if (UseRouteAllGather(&t)) {
            assert(static_cast<uint32_t>(c.bufferCount * c.bufferBytes) >= t.moeExpertPerRank * AlignUp(bs, 32U) * 4U);
        }
        if (UseDecodeRailRouting(&t)) {
            const uint32_t packets = CeilDiv(std::min(total, capacity), RAIL_PACKET_DATA_WORDS);
            assert(packets * RAIL_PACKET_WORDS <= static_cast<uint32_t>(c.routeItemsPerBatch));
        } else {
            assert(static_cast<uint32_t>(c.routeItemsPerBatch) == AlignUp(total, 256U));
        }
        if (bs == 32U && topk == 8U) {
            assert(UseDecodeRailRouting(&t));
            assert(c.routeItemsPerBatch == 512 && RailItemsPerBatch(c) == 508 && c.routeBatchCount == 1);
            // Both gates in TryRelayCompactRoutes must pass for 7/8 owned experts.
            for (uint32_t experts : {7U, 8U}) {
                const uint32_t routeBytes = experts * AlignUp(bs, uint32_t(INT32_PER_256B)) * 4U;
                assert(total <= static_cast<uint32_t>(RailItemsPerBatch(c)));
                assert(routeBytes <= static_cast<uint32_t>(c.bufferCount * c.bufferBytes) ||
                       routeBytes <= static_cast<uint32_t>(c.routeItemsPerBatch) * 4U);
            }
        }
    }
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / "relay_capacity"
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
