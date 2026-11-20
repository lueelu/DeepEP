# Copyright (c) 2026, Lu Lu

"""Compare the actual host reset budget with the production workspace layout."""

from pathlib import Path
import shutil
import subprocess

import pytest


def host_flag_budget_source():
    directory = Path(__file__).parent
    source = (directory.parent / "csrc_ep32/forward.cpp").read_text(encoding="utf-8")
    start = source.index("uint64_t HostFlagElements(")
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    host_function = source[start:end]
    # Compile the production function without the unrelated Torch/CANN host
    # launch dependencies. Its independently allocated workspace is the oracle.
    return (
        """
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#include "mega_moe_workspace.hpp"
template <typename T> T CeilDiv(T value, T divisor) {
    return (value + divisor - 1) / divisor;
}
"""
        + host_function
        + """
int main() {
    using namespace MegaMoeImpl;
    std::vector<uint8_t> allocation(16 * 1024 * 1024 + 511);
    auto* base = reinterpret_cast<uint8_t*>(
        (reinterpret_cast<uintptr_t>(allocation.data()) + 511) & ~uintptr_t{511});
    for (auto topology : {TOPO_TYPE_MTE, TOPO_TYPE_MTE_ROCE})
    for (auto mode : {GROUPED_MATMUL_MODE_GENERAL, GROUPED_MATMUL_MODE_A8W4})
    for (uint32_t quant : {0U, 1U})
    for (uint32_t shared : {0U, 1U, 2U})
    for (uint32_t cores : {1U, 8U, 16U, 28U, 32U})
    for (uint32_t experts : {1U, 12U, 128U})
    for (uint32_t output : {1U, 256U, 257U, 513U}) {
        MegaMoeTilingData t{};
        t.topoType = topology;
        t.groupedMatmulMode = mode;
        t.combineQuantMode = quant;
        t.sharedExpertNum = shared;
        t.aicNum = cores;
        t.blockAivNum = cores * 2;
        t.moeExpertPerRank = experts;
        t.maxOutputSize = output;
        t.numMaxTokensPerRank = t.bs = 32;
        t.epWorldSize = 32;
        t.dispatchRanksPerNode = 16;
        t.topK = 8;
        t.h = 1024;
        t.hiddenDim = 256;
        WorkspaceInfo workspace(base, &t);
        assert(workspace.workspaceSize < 16 * 1024 * 1024);
        assert(HostFlagElements(t) == static_cast<uint64_t>(workspace.flagResetElementCount));

    }
}
"""
    )


def test_host_flag_budget_matches_device_layout(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Host flag-budget compilation requires g++ or clang++")
    directory = Path(__file__).parent
    source = tmp_path / "host_flag_budget.cpp"
    source.write_text(host_flag_budget_source(), encoding="utf-8")
    binary = tmp_path / "host_flag_budget"
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{directory / 'layout_stubs'}",
            f"-I{directory.parent / 'csrc_ep32/kernels/mega_moe/common'}",
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
