# Copyright (c) 2026, Lu Lu
"""Static contract of the optional DeepJIT/SuperKernel entries.

These checks run without an NPU or torch: they only verify that the Host
specialization keys, the device macro names and the entry wiring agree, and
that nested components no longer read the fused-grid block accessors.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "csrc"


def read(relative):
    return (CSRC / relative).read_text(encoding="utf-8")


def test_entry_headers_exist_and_gate_super_kernel():
    for relative, guard in (
        ("kernels/dispatch/low_latency/dispatch_entry.h", "#ifndef DEEPEP_LL_JIT"),
        ("kernels/combine/low_latency/combine_entry.h", "#ifndef DEEPEP_LL_JIT"),
        ("ubx/kernels/combine_entry.h", "#ifndef DEEPEP_UBX_JIT"),
    ):
        text = read(relative)
        assert "__global__ __aicore__" in text
        assert "__sk__ __vector__" in text
        assert "SK_BIND(" in text
        assert guard in text
        assert "DestroyWithoutPipeAll" in text or "kernel.Destroy()" in text


def test_jit_bridge_keys_match_entry_macros():
    bridge = read("runtime/ll_jit.cpp")
    dispatch = read("kernels/dispatch/low_latency/dispatch_entry.h")
    for macro in (
        "DEEPEP_LL_JIT_SPEC_BS",
        "DEEPEP_LL_JIT_SPEC_H",
        "DEEPEP_LL_JIT_SPEC_TOPK",
        "DEEPEP_LL_JIT_SPEC_EXPERTS",
        "DEEPEP_LL_JIT_SPEC_CORES",
        "DEEPEP_LL_JIT_SPEC_WORLD",
        "DEEPEP_LL_JIT_SPEC_INPUT_CLASS",
        "DEEPEP_LL_JIT_SPEC_GROUP",
        "DEEPEP_LL_JIT_SPEC_INPUT_MODE",
        "DEEPEP_LL_JIT_SPEC_E4M3",
        "DEEPEP_LL_JIT_SPEC_DEDUP",
    ):
        assert macro in bridge
        assert macro in dispatch
    ubx = read("ubx/kernels/combine_entry.h")
    assert '"DEEPEP_UBX_JIT"' in bridge
    assert "_SPECIALIZED 1" in bridge
    assert "DEEPEP_UBX_JIT_SPEC_GROUP" in bridge
    for macro in ("DEEPEP_UBX_JIT_SPEC_BS", "DEEPEP_UBX_JIT_SPEC_GROUP", "DEEPEP_UBX_JIT_SPEC_DEDUP"):
        assert macro in ubx


def test_nested_components_use_logical_core_context():
    for relative in (
        "kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h",
        "kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h",
        "kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h",
        "kernels/dispatch/low_latency/device/ll_transport.h",
        "kernels/dispatch/low_latency/device/ll_count_frame_transport.h",
        "ubx/kernels/combine_server_dedup_kernel.h",
        "ubx/kernels/combine_server_dedup_partial.h",
        "ubx/kernels/ll_transport.h",
        "ubx/kernels/combine_transfer_batch.h",
        "ubx/kernels/combine_wqe_prepare.h",
    ):
        # Strip comments: the explanatory text may name the accessor it replaces.
        code = "\n".join(line.split("//", 1)[0] for line in read(relative).splitlines())
        assert "GetBlockNum()" not in code and "GetBlockIdx()" not in code, relative


def test_launch_functions_expose_explicit_jit_switch():
    host = read("ops/low_latency.cpp")
    assert "low_latency_jit_prepare_dispatch" in host
    assert "low_latency_jit_prepare_combine" in host
    assert host.count('py::arg("jit") = false') == 2
    assert "ll_jit_launch_dispatch" in host
    assert "ll_jit_launch_combine" in host
    assert "ll_jit_launch_combine_ubx" in host


def test_python_runtime_and_graph_wrapper():
    runtime = (ROOT / "deep_ep" / "_runtime.py").read_text(encoding="utf-8")
    assert 'os.environ.get("DEEPEP_LL_JIT"' in runtime
    assert "low_latency_jit_available" in runtime
    graph = (ROOT / "deep_ep" / "backends" / "ubx" / "graph.py").read_text(encoding="utf-8")
    assert "super_kernel_scope_begin" in graph
    assert "super_kernel_optimize" in graph
    assert "class LLGraph" in graph
