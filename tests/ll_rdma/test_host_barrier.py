"""DeepEP-owned Host barrier checks; SHMEM remains untouched."""
from pathlib import Path
import os
import random
import shutil
import subprocess
import pytest
ROOT = Path(__file__).resolve().parents[2]


def test_real_helper_against_cpu_transport_doubles(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if compiler is None and not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG or install a C++17 compiler for the Host barrier helper")
    command = [compiler] if compiler else [zig, "c++"]
    binary = tmp_path / "host-barrier.exe"
    subprocess.run([*command, "-std=c++17", "-UNDEBUG", "-pthread", "-iquote", str(ROOT),
                    str(Path(__file__).with_name("host_barrier_cpu.cpp")), "-o", str(binary)],
                   check=True, timeout=180)
    subprocess.run([str(binary)], check=True, timeout=60)


def test_project_build_owns_barrier():
    cmake = (ROOT / "cmake/device/CMakeLists.txt").read_text()
    ll_sources = cmake.split("set(_ll_sources", 1)[1].split("add_library(deepep_kernel_ll_plain", 1)[0]
    assert "csrc/kernels/dispatch/low_latency.cpp" in ll_sources
    assert "csrc/kernels/combine/low_latency.cpp" in ll_sources
    assert "variants" not in ll_sources
    for variant in ("plain", "reference"):
        assert f"add_library(deepep_kernel_ll_{variant} OBJECT ${{_ll_sources}})" in cmake
    assert "add_library(deepep_kernels SHARED $<TARGET_OBJECTS:${_ll_variant}>)" in cmake
    # The legacy helper remains testable but is absent from the LL build and
    # from Session's runtime launch path; NDR uses its existing Host collective.
    assert "csrc/kernels/runtime/host_barrier.cpp" not in cmake
    launcher = (ROOT / "csrc/kernels/runtime/host_barrier.cpp").read_text()
    assert "deepep_ndr_host_barrier_kernel" in launcher
    session = (ROOT / "csrc/runtime/session.cpp").read_text()
    assert "deepep_ndr_host_barrier_launch_v2" not in session
    assert "NDR barrier requires HCCL Host coordination" in session
    assert "SHMEM barrier completion" in session
    assert not (ROOT / "scripts/shmem_barrier_patch.py").exists()
    assert "source_patches" in (ROOT / "scripts/shmem_sdk_identity.py").read_text()


def test_initialization_diagnostics_and_bounded_local_poll():
    helper = (ROOT / "csrc/kernels/runtime/host_barrier.h").read_text()
    wait = helper.split("void deepep_barrier_wait", 1)[1].split("void deepep_host_barrier", 1)[0]
    assert "GetSystemCycle()" in wait and ">= wait_cycles" in wait
    assert "AscendC::Trap()" in wait and "predecessor=%d slot=%d" in wait
    session = (ROOT / "csrc/runtime/session.cpp").read_text()
    for phase in ("SHMEM init enter", "SHMEM init return", "workspace allocation enter",
                  "workspace memset return", "barrier launched; synchronize enter"):
        assert phase in session
    launcher = (ROOT / "tests/elastic/test_ep_low_latency.py").read_text()
    assert "if args.trace_ops:" in launcher and "os.environ['DEEPEP_LL_INIT_TRACE'] = '1'" in launcher


@pytest.mark.parametrize("size", [2, 3, 16, 17, 32, 64, 128, 256])
@pytest.mark.parametrize("seed", range(4))
def test_dissemination_with_delays_and_repeated_generations(size, seed):
    """No rank may finish a generation before every rank entered it.

    Random delivery models delayed RDMA; send completion precedes counter reuse.
    Explicit shuffled team membership exercises mapping without rank arithmetic.
    """
    rng = random.Random(seed)
    members = rng.sample(range(1024), size)
    rounds = (size - 1).bit_length()
    slots = [[0] * rounds for _ in members]
    generation = [0] * size
    phase = [0] * size
    state = ["enter"] * size
    entered = [set() for _ in range(9)]
    pending = []
    steps = 0
    while any(g < 8 or s != "done" for g, s in zip(generation, state)):
        steps += 1
        assert steps < size * 8 * rounds * 300
        if pending and rng.randrange(3) == 0:
            sender, dest, slot, value = pending.pop(rng.randrange(len(pending)))
            slots[dest][slot] = value
            assert generation[sender] == value  # quiet protects source counter
            state[sender] = "wait"
            continue
        rank = rng.randrange(size)
        if state[rank] == "enter":
            generation[rank] += 1
            entered[generation[rank]].add(rank)
            phase[rank] = 0
            state[rank] = "send"
        elif state[rank] == "send":
            dest = (rank + (1 << phase[rank])) % size
            peer = members[dest]
            assert members.index(peer) == dest
            pending.append((rank, dest, phase[rank], generation[rank]))
            state[rank] = "quiet"
        elif state[rank] == "wait":
            if slots[rank][phase[rank]] not in (generation[rank], generation[rank] + 1):
                continue
            phase[rank] += 1
            if phase[rank] == rounds:
                assert len(entered[generation[rank]]) == size
                state[rank] = "done" if generation[rank] == 8 else "enter"
            else:
                state[rank] = "send"
