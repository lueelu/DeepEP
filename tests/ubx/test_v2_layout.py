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
    source = r"""
#include <cassert>
#include <cstddef>
#include <vector>
#include "PREFIX/ep_memory_server_dedup_protocol.h"
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
        DISPATCH_DEDUP_DATA_TYPE_FP16, DISPATCH_DEDUP_DATA_TYPE_BFP16,
    };
    for (unsigned cores : {16, 28, 56, 64, 72})
        for (unsigned experts : {128, 384, 1024})
            assert(DispatchServerDedupCountLayout(experts, cores).stateBytes < 1024U * 1024U);
    for (unsigned world : {16, 32, 64, 128, 256}) {
        values.push_back(LlTilingKeyForWorld(world));
        for (unsigned capacity : {1, 32, 48, 256})
            values.push_back(EpServerDedupEnabled(capacity, 1, world));
    }
    for (int magic : {0, 1, 0, 1}) values.push_back(DispatchServerDedupWindowOffset(magic));
    return values;
}
"""
    paths = []
    for name, prefix, namespace, rdma in (
        ("dispatch", "kernels/dispatch/low_latency/common", "DispatchDedup", "LlRdma"),
        ("combine", "ubx/kernels", "UbxDispatch", "UbxRdma"),
    ):
        path = tmp_path / f"{name}.cpp"
        path.write_text(
            source.replace("PREFIX", prefix).replace("NS", namespace).replace("RDMA", rdma).replace("NAME", name)
        )
        paths.append(str(path))
    main = tmp_path / "main.cpp"
    main.write_text(r"""
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "ubx/v2.hpp"
#include "ubx/kernels/ep_memory_server_dedup_combine_tiling.h"
std::vector<uint64_t> dispatch();
std::vector<uint64_t> combine();
int main() {
    assert(dispatch() == combine());
    for (int world : {16, 32, 64, 128, 256})
        for (int tokens : {1, 4, 24, 32, 48, 64, 72, 256})
            for (int cores : {16, 28, 56})
                for (uint8_t dtype : {3, 11}) {
                    // E=512 permits every tested world; TopK can exceed experts/rank.
                    const auto c = deepep::ubx::v2_capacity(world, tokens, 3584, 8, 512, cores, 16);
                    assert(c == uint64_t(world * tokens * std::min(8, 512 / world)));
                    const auto bytes = deepep::ubx::v2_tiling(world, tokens, 3584, 8, 512, dtype, 256, cores, 16);
                    UbxCombine::EpServerDedupCombineTilingData td{};
                    std::memcpy(&td, bytes.data(), sizeof(td));
                    assert(td.bs == uint32_t(tokens) && td.h == 3584 && td.moeExpertNum == 512 && td.topK == 8);
                    assert(td.dtype == dtype && td.numMaxTokensPerRank == 256 && td.quantMode == 0);
                    assert(td.tilingKey == (world <= 16 ? 4 : 16));
                }
    int rejected = 0;
    try { deepep::ubx::v2_capacity(32, 32, 3584, 8, 384, 56, 4); }
    catch (const std::invalid_argument&) { ++rejected; }
    try { deepep::ubx::v2_tiling(32, 32, 3584, 8, 384, 11, 8, 56, 16); }
    catch (const std::invalid_argument&) { ++rejected; }
    assert(rejected == 2);
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
