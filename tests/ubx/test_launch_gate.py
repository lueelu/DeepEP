# Copyright (c) 2026, Lu Lu
"""Exercise the production launch gate with delayed host submissions."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def gate_binary(tmp_path_factory):
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler:
        command = [compiler]
    elif importlib.util.find_spec("ziglang"):
        command = [sys.executable, "-m", "ziglang", "c++"]
    else:
        pytest.skip("Launch gate check requires a C++17 compiler")
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path_factory.mktemp("ubx-launch-gate")
    (directory / "acl").mkdir()
    (directory / "acl/acl.h").write_text(
        """#pragma once
using aclrtStream = void*;
using aclrtNotify = void*;
int aclrtCreateNotify(aclrtNotify*, unsigned);
int aclrtCreateStream(aclrtStream*);
int aclrtDestroyNotify(aclrtNotify);
int aclrtDestroyStream(aclrtStream);
int aclrtWaitAndResetNotify(aclrtNotify, aclrtStream, unsigned);
int aclrtRecordNotify(aclrtNotify, aclrtStream);
int aclrtSynchronizeStreamWithTimeout(aclrtStream, int);
"""
    )
    source = directory / "gate.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <functional>
#include <string>
#include <vector>
#include "ubx/launch_gate.hpp"
static int target_id, control_id, notify_id;
static void* target = &target_id;
static bool held = false, released = false, completed = false;
static bool fail_completion = false;
static int clock_ticks = 0, begin = -1, end = -1;
static std::vector<std::function<void()>> queue;
static std::vector<int> executed;
static void progress() {
    if (held) return;
    for (auto& job : queue) job();
    queue.clear();
}
static void enqueue(int kind) {
    queue.push_back([kind] {
        executed.push_back(kind);
        if (kind == 0) clock_ticks += 200; // HCCL dependency, outside Events
        if (kind == 1) begin = clock_ticks;
        if (kind == 2) clock_ticks += 5;   // Combine
        if (kind == 3) end = clock_ticks;
    });
    progress();
    clock_ticks += 100; // arbitrarily slow host between submissions
    assert(executed.empty());
}
int aclrtCreateNotify(aclrtNotify* p, unsigned) { *p = &notify_id; return 0; }
int aclrtCreateStream(aclrtStream* p) { *p = &control_id; return 0; }
int aclrtWaitAndResetNotify(aclrtNotify n, aclrtStream s, unsigned) {
    assert(n == &notify_id && s == target); held = true; return 0;
}
int aclrtRecordNotify(aclrtNotify n, aclrtStream s) {
    assert(n == &notify_id && s == &control_id && held && queue.size() == 4);
    held = false; released = true; progress(); return 0;
}
int aclrtSynchronizeStreamWithTimeout(aclrtStream s, int) {
    if (s == target) {
        if (fail_completion) return 1;
        assert(released && queue.empty()); completed = true;
    } else assert(s == &control_id && completed);
    return 0;
}
int aclrtDestroyStream(aclrtStream s) { assert(s == &control_id && completed); return 0; }
int aclrtDestroyNotify(aclrtNotify n) { assert(n == &notify_id && completed); return 0; }
int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string mode = argv[1];
    fail_completion = mode == "completion-failure";
    deepep::ubx::LaunchGate gate(target);
    gate.enqueue([&] {
        enqueue(0);
        if (mode == "callback-failure") throw std::runtime_error("injected callback failure");
    }, [&] {
        enqueue(1); enqueue(2);
        if (mode == "launch-failure") throw std::runtime_error("injected launch failure");
        enqueue(3);
    });
    gate.complete();
    assert((executed == std::vector<int>{0, 1, 2, 3}));
    assert(end - begin == 5);
}
"""
    )
    binary = directory / ("gate.exe" if sys.platform == "win32" else "gate")
    subprocess.run(
        [*command, "-std=c++17", "-I" + str(directory), "-I" + str(root / "csrc"), str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return binary


@pytest.mark.parametrize("case", ["normal", "callback-failure", "launch-failure", "completion-failure"])
def test_submission_and_completion(gate_binary, case):
    result = subprocess.run([gate_binary, case], capture_output=True, text=True, timeout=10)
    assert result.returncode == (0 if case == "normal" else 1), result.stderr
    if case != "normal":
        assert "UBX launch gate failed:" in result.stderr
