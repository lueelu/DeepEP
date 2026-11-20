# Copyright (c) 2026, Lu Lu
"""Numerical/readiness checks using production reduction and unpermute code."""

from pathlib import Path
import shutil
import subprocess

import pytest


def function_source(text, name, result="void"):
    start = text.index(f"__aicore__ inline {result} {name}(")
    opening = text.index("{", start)
    end, depth = opening + 1, 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


@pytest.mark.parametrize("dispatch_timer", [False, True], ids=["timer-off", "timer-on"])
def test_production_weighted_gateway_reduction_and_unpermute(tmp_path, dispatch_timer):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Production reduction checks require a C++ compiler")
    directory = Path(__file__).parent
    kernel = directory.parent / "csrc_ep32/kernels/mega_moe"
    dispatch = (kernel / "stage/mega_moe_hierarchical_dispatch.hpp").read_text(encoding="utf-8")
    fixture = (directory / "hierarchical_reduce_test.cpp").read_text(encoding="utf-8")
    token_dispatch = (kernel / "stage/mega_moe_token_dispatch.hpp").read_text(encoding="utf-8")
    row_types = token_dispatch[
        token_dispatch.index("struct ExpertDispatchCoreRange {") : token_dispatch.index("// 由 tiling/peermem")
    ]
    template = "template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>\n"
    queue = "\n".join(
        (
            "template <typename ActivationType>\n" + function_source(token_dispatch, "ObserveDispatchGateway"),
            "template <typename ActivationType>\n" + function_source(token_dispatch, "ReadyDispatchRows", "uint32_t"),
            template + function_source(token_dispatch, "FlushDispatchRouteBatch"),
            "template <typename ActivationType>\n" + function_source(token_dispatch, "LoadDispatchIndices"),
            template + function_source(token_dispatch, "SubmitDispatchProbe"),
            template + function_source(token_dispatch, "DispatchDeferredSources"),
            template + function_source(token_dispatch, "DispatchRankTokens", "int32_t"),
            template + function_source(token_dispatch, "FinishDispatchSources"),
            "template <typename ActivationType>\n"
            + function_source(token_dispatch, "FindDispatchSourceRank", "uint32_t"),
            "template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch>\n"
            + function_source(token_dispatch, "DispatchOwnedExpertRows"),
        )
    )
    timer = (kernel / "common/ascend_timer_v2_device.hpp").read_text(encoding="utf-8")
    relay_scope = timer[timer.index("class AscendTimerRelayScope {") : timer.index("class AscendTimerFixedScope {")]
    fixture = fixture.replace("/* DISPATCH_RELAY_SCOPE */", relay_scope)
    fixture = fixture.replace("/* DISPATCH_ROW_TYPES */", row_types).replace("/* PRODUCTION_DISPATCH_QUEUE */", queue)
    send_mask = (kernel / "stage/mega_moe_send_mask.hpp").read_text(encoding="utf-8")
    fixture = fixture.replace("/* BF16_PACKER */", function_source(send_mask, "PackBf16Bs96RailMetadata"))
    source = tmp_path / "hierarchical_reduce.cpp"
    source.write_text(
        fixture.replace("/* PRODUCTION_WEIGHT_PACKER */", function_source(dispatch, "PackHierarchicalCombineWeights")),
        encoding="utf-8",
    )
    binary = tmp_path / "hierarchical_reduce"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            *(["-DTEST_DISPATCH_TIMER"] if dispatch_timer else []),
            "-Wall",
            "-Wextra",
            "-Werror",
            "-ffp-contract=off",
            "-fno-strict-aliasing",  # Device register views alias signed/unsigned vector registers.
            f"-I{directory}",
            f"-I{directory / 'layout_stubs'}",
            f"-I{directory / 'combine_vf_stubs'}",
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
