# Copyright (c) 2026, Lu Lu
"""Compile the real Host helper and compare Dispatch/Combine wire ABIs."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_shared_metadata_workspace_and_tiling(tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler:
        command = [compiler]
    elif importlib.util.find_spec("ziglang"):
        command = [sys.executable, "-m", "ziglang", "c++"]
    else:
        pytest.skip("Wire ABI check requires a C++17 compiler")
    root = Path(__file__).resolve().parents[2]
    combine_source = r"""
#include <cassert>
#include <cstddef>
#include <vector>
#include <deep_ep/common/compiled.hpp>
using deep_ep::EpServerDedupReadIndexBlock;
using deep_ep::EpServerDedupTokenType;
#include "PREFIX/ep_memory_server_dedup_count_layout.h"
#include "PREFIX/ll_rdma_layout.h"
using namespace NS;
std::vector<uint64_t> NAME() {
    std::vector<uint64_t> values{
        sizeof(CommArgs), offsetof(CommArgs, rank), offsetof(CommArgs, rankSize),
        offsetof(CommArgs, localRankSize), offsetof(CommArgs, peerMems), offsetof(CommArgs, fftsVal),
        offsetof(CommArgs, reserved0), offsetof(CommArgs, reservedAddresses), offsetof(CommArgs, reservedConfig),
        kLlPhysicalRanksPerHost, kEpServerDedupMagicStride, kEpServerDedupControlPrefixBytes,
        kEpServerDedupPeerBytes, kEpServerDedupWindowBytes, kEpServerDedupWindowUsableBytes,
        sizeof(EpServerDedupReadIndexBlock), offsetof(EpServerDedupReadIndexBlock, weightBits),
        uint64_t(EpServerDedupTokenType::Direct), uint64_t(EpServerDedupTokenType::FirstHop),
        uint64_t(EpServerDedupTokenType::Terminal), RDMA::kCoreBytes, RDMA::kBackend,
    };
    for (unsigned cores : {16, 28, 56, 64, 72})
        for (unsigned experts : {128, 384, 1024})
            assert(DispatchServerDedupCountLayout(experts, cores).stateBytes < 1024U * 1024U);
    for (unsigned world : {16, 32, 64, 128, 256})
        values.push_back(LlTilingKeyForWorld(world));
    for (int magic : {0, 1, 0, 1}) values.push_back(DispatchServerDedupWindowOffset(magic));
    return values;
}
"""
    dispatch_source = r"""
#include <cassert>
#include <cstddef>
#include <vector>
#include <deep_ep/comm/comm_args.hpp>
#include <deep_ep/common/compiled.hpp>
#include <deep_ep/layout/ep/token.hpp>
#include <deep_ep/layout/ep/workspace.hpp>
using namespace deep_ep;
std::vector<uint64_t> dispatch() {
    std::vector<uint64_t> values{
        sizeof(CommArgs), offsetof(CommArgs, rank), offsetof(CommArgs, rankSize),
        offsetof(CommArgs, localRankSize), offsetof(CommArgs, peerMems), offsetof(CommArgs, fftsVal),
        offsetof(CommArgs, reserved0), offsetof(CommArgs, reservedAddresses), offsetof(CommArgs, reservedConfig),
        kNumRanksPerHost, kEpServerDedupMagicStride, kEpServerDedupControlPrefixBytes,
        kEpServerDedupPeerBytes, kEpServerDedupWindowBytes, kEpServerDedupWindowUsableBytes,
        sizeof(EpServerDedupReadIndexBlock), offsetof(EpServerDedupReadIndexBlock, weightBits),
        uint64_t(EpServerDedupTokenType::Direct), uint64_t(EpServerDedupTokenType::FirstHop),
        uint64_t(EpServerDedupTokenType::Terminal), rdma::kCoreBytes, rdma::kBackend,
    };
    for (unsigned cores : {16, 28, 56, 64, 72})
        for (unsigned experts : {128, 384, 1024})
            assert(DispatchServerDedupCountLayout(experts, cores).stateBytes < 1024U * 1024U);
    for (unsigned world : {16, 32, 64, 128, 256})
        values.push_back(LlTilingKeyForWorld(world));
    for (int magic : {0, 1, 0, 1}) values.push_back(DispatchServerDedupWindowOffset(magic));
    return values;
}
"""
    (tmp_path / "dispatch.cpp").write_text(dispatch_source)
    path = tmp_path / "combine.cpp"
    path.write_text(
        combine_source.replace("PREFIX", "ubx/kernels").replace("NS", "UbxDispatch").replace("RDMA", "UbxRdma")
        .replace("NAME", "combine")
    )
    paths = [str(tmp_path / "dispatch.cpp"), str(path)]
    main = tmp_path / "main.cpp"
    main.write_text(r"""
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "ubx/v2.hpp"
#include "ubx/kernels/combine_layout.h"
#include "ubx/kernels/ep_memory_server_dedup_combine_tiling.h"
std::vector<uint64_t> dispatch();
std::vector<uint64_t> combine();
int main() {
    assert(dispatch() == combine());
    const unsigned maxUb = UbxCombine::kEpServerDedupCombineMaxUbBytes;
    assert(deepep::ubx::combine_ub_bytes(256 * 1024) == maxUb);
    assert(deepep::ubx::combine_ub_bytes(216 * 1024 + 31) == 216 * 1024);

    for (int world : {16, 32, 64, 128, 256})
        for (int tokens : {1, 4, 24, 32, 48, 64, 72, 256})
            for (int cores : {16, 28, 56})
                for (uint8_t dtype : {3, 11}) {
                    // E=512 permits every tested world; TopK can exceed experts/rank.
                    const auto c = deepep::ubx::v2_capacity(world, tokens, 3584, 8, 512, cores, 16, maxUb);
                    assert(c == uint64_t(world * tokens * std::min(8, 512 / world)));
                    const auto bytes = deepep::ubx::v2_tiling(world, tokens, 3584, 8, 512, dtype, 256, cores, 16, maxUb);
                    UbxCombine::EpServerDedupCombineTilingData td{};
                    std::memcpy(&td, bytes.data(), sizeof(td));
                    assert(td.bs == uint32_t(tokens) && td.h == 3584 && td.moeExpertNum == 512 && td.topK == 8);
                    assert(td.dtype == dtype && td.numMaxTokensPerRank == 256 && td.quantMode == 0);
                    assert(td.tilingKey == (world <= 16 ? 4 : 16));
                }
    for (unsigned budgetKiB : {200U, 216U, 240U}) {
        const unsigned ub = deepep::ubx::combine_ub_bytes(budgetKiB * 1024);
        assert(ub == budgetKiB * 1024);
        // The actual host admission and tiling paths on a 56-vector-core device.
        for (int bs : {1, 2, 4, 32, 48, 64, 256}) {
            const auto bytes = deepep::ubx::v2_tiling(16, bs, 3584, 8, 384, 11, 256, 56, 16, ub);
            UbxCombine::EpServerDedupCombineTilingData td{};
            std::memcpy(&td, bytes.data(), sizeof(td));
            assert(td.bs == uint32_t(bs) && td.h == 3584);
            const auto normal = UbxCombine::PlanPutRows(bs * 8U, bs, 3584, 8, 28, ub);
            assert(normal.rows == bs * 8U && normal.lanes == 8);
        }
        for (unsigned h : {3584U, 3840U, 4096U}) {
            auto plan = [h, ub](unsigned rows) { return UbxCombine::PlanPutRows(rows, 256, h, 8, 28, ub); };
            for (unsigned rows : {0U, 256U, 10001U}) {
                const auto whole = plan(rows);
                assert(whole.rows == rows && whole.lanes && whole.receive.inputLanes);
            }
            unsigned fit = 10001U;
            while (plan(fit + 1).rows == fit + 1) ++fit;
            const auto spill = plan(fit + 1);
            assert(spill.rows == 2048U && spill.lanes && spill.receive.inputLanes);
            const unsigned maximum = deepep::ubx::v2_capacity(32, 256, h, 8, 384, 56, 16, ub);
            assert(maximum == 65536U);
            for (unsigned rows = 0; rows <= maximum; ++rows) {
                const auto actual = plan(rows);
                const auto& recv = actual.receive;
                assert(actual.rows == (rows <= fit ? rows : spill.rows));
                assert(actual.lanes && recv.inputLanes && recv.outputLanes && recv.totalBytes <= ub);
                assert(recv.residentBytes + recv.scratchBytes == ub);
                assert(recv.residentBytes + recv.inputOffset + UbxCombine::EpCombineRouteSnapshotBytes(recv) == ub);
                const unsigned sent = UbxCombine::kEpServerDedupCombineFixedUbBytes + UbxCombine::EpCombineRdmaBytes(h)
                    + UbxCombine::EpCombineRowUbBytes(actual.rows) + actual.lanes * ((2 * h + 479) / 480) * (480 + 512);
                assert(sent <= ub);
            }
            // Final aggregation can use slack after the partial reduction's outputs.
            // Check every possible rotation of its retained slots against the physical end.
            for (unsigned rows : {0U, fit, fit + 1, maximum}) {
                const auto recv = plan(rows).receive;
                for (unsigned tasks = 0; tasks <= 2 * recv.outputLanes; ++tasks) {
                    const auto final = UbxCombine::EpCombineM5Layout(recv, tasks);
                    assert(final.inputLanes > 0);
                    for (unsigned lane = 0; lane < final.inputLanes; ++lane)
                        assert(recv.residentBytes + final.inputOffsets[lane] + recv.inputBytes <= ub);
                }
            }
        }
    }
    // With fewer cores and a larger hidden size, the receive arena cannot fit
    // even the first 2048-row fallback. The same host/device planner halves it.
    const auto capacity = deepep::ubx::v2_capacity(32, 32, 12800, 4, 512, 16, 16, maxUb);
    const auto reduced = UbxCombine::PlanPutRows(capacity, 32, 12800, 4, 8, maxUb);
    assert(capacity == 4096 && reduced.rows == 1024 && reduced.lanes && reduced.receive.inputLanes);
    int rejected = 0;
    try { deepep::ubx::v2_capacity(32, 32, 3584, 8, 384, 56, 4, maxUb); }
    catch (const std::invalid_argument&) { ++rejected; }
    try { deepep::ubx::v2_tiling(32, 32, 3584, 8, 384, 11, 8, 56, 16, maxUb); }
    catch (const std::invalid_argument&) { ++rejected; }
    try { deepep::ubx::combine_ub_bytes(-1); }
    catch (const std::invalid_argument&) { ++rejected; }
    try { deepep::ubx::v2_capacity(16, 32, 3584, 8, 384, 56, 16, 32U * 1024U); }
    catch (const std::invalid_argument&) { ++rejected; }
    assert(rejected == 4);
}
""")
    binary = tmp_path / "layout.exe"
    subprocess.run(
        [
            *command,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-UNDEBUG",
            "-I" + str(root / "csrc"),
            "-I" + str(root / "deep_ep" / "include"),
            *paths,
            str(main),
            str(root / "csrc/ubx/v2.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=30)
