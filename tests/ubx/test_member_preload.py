# Copyright (c) 2026, Lu Lu
"""Exercise the partial reducer's real DMA descriptors and event ownership on the host."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_member_preload_preserves_order_and_lane_bounds(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("Requires a host C++ compiler")
    kernels = Path(__file__).resolve().parents[2] / "csrc/ubx/kernels"
    partial = (kernels / "combine_server_dedup_partial.h").read_text()
    methods = partial[partial.index("    __aicore__ inline void Load(uint32_t row,"):
                      partial.index("    template <class T>\n    __aicore__ inline void Aggregate(")]
    wire = (kernels / "combine_stage_wire.h").read_text()
    wire = wire[wire.index("__aicore__ inline void LoadStageWire("):wire.index("}  // namespace UbxCombine")]
    source = r"""
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#define __aicore__
template<class T> struct View {
    T* data;
    View operator[](unsigned index) const { return {data + index}; }
    template<class U> View<U> ReinterpretCast() const { return {reinterpret_cast<U*>(data)}; }
};
namespace AscendC { template<class T> using LocalTensor = View<T>; }
using GM_ADDR = uint8_t*;
namespace UbxTransport { View<uint8_t> Tensor(GM_ADDR p, bool uncached) { assert(uncached); return {p}; } }
struct Copy { unsigned count, bytes, srcGap, dstGap, reserved; };
template<class T> struct Pad { bool enabled; unsigned left, right; T value; };
unsigned copies;
template<class T> void DataCopyPad(View<T> dst, View<T> src, Copy c, Pad<T>) {
    assert(!c.dstGap);
    ++copies;
    for (unsigned i = 0; i < c.count; ++i)
        std::memcpy(reinterpret_cast<uint8_t*>(dst.data) + i * c.bytes,
                    reinterpret_cast<uint8_t*>(src.data) + i * (c.bytes + c.srcGap), c.bytes);
}
WIRE
enum class HardEvent { V_MTE2, MTE2_V };
struct Reducer {
    unsigned inputSequence_, inputLanes_, inputBytes_, h_;
    View<uint16_t> expert_, inputs_;
    GM_ADDR inbox;
    std::vector<int> events;  // free -> acquired by MTE2 -> readable by Vector
    template<HardEvent E> void WaitFlag(unsigned lane) {
        static_assert(E == HardEvent::V_MTE2);
        assert(events.at(lane) == 0);
        events[lane] = 1;
    }
    template<HardEvent E> void SetFlag(unsigned lane) {
        static_assert(E == HardEvent::MTE2_V);
        assert(events.at(lane) == 1);
        events[lane] = 2;
    }
    GM_ADDR Inbox(unsigned source, unsigned token, unsigned member) {
        assert(source == 2 && token == 3 && member);
        return inbox + member * (inputBytes_ / 480U * 512U);
    }
METHODS
};
void checkPreload(unsigned h, unsigned lanes, unsigned sequence, unsigned members) {
    const unsigned pitch = (2 * h + 479) / 480 * 480;
    const unsigned packet = pitch / 480 * 512;
    std::vector<uint16_t> input(lanes * pitch / 2 + 32, 0xffff), expert(2 * h);
    std::vector<uint16_t> slots(members * packet / 2, 0xfffe);
    for (unsigned i = 0; i < h; ++i) {
        expert[h + i] = uint16_t(i);
        for (unsigned m = 1; m < members; ++m)
            slots[m * packet / 2 + i / 240 * 256 + i % 240] = uint16_t(m * 1031 + i);
    }
    Reducer reducer{sequence, lanes, pitch, h, {expert.data()}, {input.data() + 16},
                    reinterpret_cast<uint8_t*>(slots.data()), std::vector<int>(lanes)};
    copies = 0;
    reducer.PreloadInputs(1, 2, 3, members);
    // One local read and one inbox descriptor for two members, including H tails.
    if (sequence == 0 && lanes >= 3 && members == 3) assert(copies == 2);
    for (unsigned m = 0; m < members; ++m) {
        const unsigned lane = (sequence + m) % lanes;
        assert(reducer.events[lane] == 2);
        for (unsigned i = 0; i < h; ++i)
            assert(input[16 + lane * pitch / 2 + i] == uint16_t(m * 1031 + i));
        reducer.events[lane] = 0;
        if (m + lanes < members) reducer.Load(1, 2, 3, m + lanes);
    }
    for (int event : reducer.events) assert(event == 0);
    for (unsigned i = 0; i < 16; ++i)
        assert(input[i] == 0xffff && input[input.size() - 1 - i] == 0xffff);
}
int main() {
    for (unsigned h : {16U, 240U, 256U, 3584U, 3840U, 4096U})
        for (unsigned lanes = 1; lanes <= 6; ++lanes)
            for (unsigned sequence = 0; sequence < lanes; ++sequence)
                for (unsigned members : {2U, 3U, 4U, 7U, 16U, 32U})
                    checkPreload(h, lanes, sequence, members);
}
""".replace("WIRE", wire).replace("METHODS", methods)
    unit = tmp_path / "preload.cpp"
    unit.write_text(source)
    binary = tmp_path / "preload"
    subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(unit), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
