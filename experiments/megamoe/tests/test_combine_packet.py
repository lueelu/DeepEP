# Copyright (c) 2026, Lu Lu
"""Run the production gateway sender with deferred DMA and guarded row slots."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_packet_sender_payload_markers_bounds_and_ring_lifetime(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Packet Combine checks require a C++ compiler")
    directory = Path(__file__).parent
    production = (directory.parent / "csrc_ep32/kernels/mega_moe/stage/mega_moe_gmm2_combine.hpp").read_text(
        encoding="utf-8"
    )
    routes = production[production.index("struct CombineTokenRoute") : production.index("// 为 tile 内")]
    sender = production[production.index("template <typename Element>\n") : production.index("// 通过本地 MTE")]
    config = production[
        production.index("struct WaveCombineBufferConfig") : production.index(
            "// 非 layered 路径统一使用的 Combine 行环。"
        )
    ]
    init = production[
        production.index("template <uint8_t CombineMode, bool IncludeAiv0") : production.index("// 默认由 AIV1")
    ]
    drain = production[
        production.index("__aicore__ inline void DrainCombineRowBuffers") : production.index("// 加载一段")
    ]
    send = production[
        production.index("template <uint8_t CombineMode, bool IsBufferReuse>") : production.index("// 使用行级流水")
    ]
    preload = production[
        production.index("__aicore__ inline void PreloadWaveCombineMetaInfo") : production.index("// 搬运并按需量化")
    ]
    pipeline = (directory.parent / "csrc_ep32/kernels/mega_moe/stage/mega_moe_combine_pipeline.hpp").read_text(
        encoding="utf-8"
    )
    pipeline = pipeline[
        pipeline.index("class CrossExpertCombinePipeline") : pipeline.index("}  // namespace MegaMoeImpl")
    ]
    fixture = (directory / "combine_packet_test.cpp").read_text(encoding="utf-8")
    for key, code in {
        "ROUTES": routes,
        "SENDER": sender,
        "CONFIG": config,
        "INIT": init,
        "DRAIN": drain,
        "SEND": send,
        "PRELOAD": preload,
        "PIPELINE": pipeline,
    }.items():
        fixture = fixture.replace(f"/* PRODUCTION_{key} */", code)
    source = tmp_path / "combine_packet.cpp"
    source.write_text(fixture, encoding="utf-8")
    binary = tmp_path / "combine_packet"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-UNDEBUG",
            f"-I{directory}",
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
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
