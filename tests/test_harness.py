# Copyright (c) 2026, Lu Lu
# Modified by lishaoxun 2026

"""Check collection gates, lifecycle failures and output matching on CPU."""

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.utils.cases import make_case
from tests.utils.checks import assert_close
from tests.utils.device import _semantic_counts, _valid_positions, observed_rows
from tests.utils.envs import Launch, owned_process_group, read_launch
from tests.utils.reference import dispatch, expert_payload


@pytest.mark.parametrize(
    "change",
    [
        {"WORLD_SIZE": "0"},
        {"RANK": "2"},
        {"LOCAL_RANK": "-1"},
        {"LOCAL_WORLD_SIZE": "0"},
        {"MASTER_PORT": "bad"},
        {"MASTER_ADDR": ""},
    ],
)
def test_invalid_launch_is_rejected(change):
    env = dict(
        RANK="0",
        LOCAL_RANK="0",
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
        MASTER_ADDR="localhost",
        MASTER_PORT="29500",
    )
    with pytest.raises(RuntimeError):
        read_launch(env | change)


def test_missing_launch_fails_before_framework_import():
    with pytest.raises(RuntimeError, match="torchrun"):
        read_launch({})


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_launch_accepts_qualification_ep_sizes(world_size):
    env = dict(
        RANK=str(world_size - 1),
        LOCAL_RANK=str(world_size - 1),
        WORLD_SIZE=str(world_size),
        LOCAL_WORLD_SIZE=str(world_size),
        MASTER_ADDR="localhost",
        MASTER_PORT="29500",
    )
    assert read_launch(env).world_size == world_size


def mock_framework(initialized=False):
    dist = Mock()
    state = {"initialized": initialized}
    dist.is_initialized.side_effect = lambda: state["initialized"]
    dist.init_process_group.side_effect = lambda **kwargs: state.update(initialized=True)
    dist.destroy_process_group.side_effect = lambda: state.update(initialized=False)
    npu = Mock()
    npu.is_available.return_value = True
    npu.device_count.return_value = 2
    return SimpleNamespace(distributed=dist, npu=npu), state


@pytest.mark.parametrize("failure", [None, "body", "init"])
def test_owned_group_cleanup_including_failure(failure):
    torch, state = mock_framework()
    if failure == "init":

        def fail_init(**kwargs):
            state["initialized"] = True
            raise RuntimeError("initialization failed")

        torch.distributed.init_process_group.side_effect = fail_init

    def run():
        with owned_process_group(torch, Launch(0, 0, 2)) as group:
            assert group is torch.distributed.group.WORLD
            if failure == "body":
                raise RuntimeError("communication failed")

    if failure:
        with pytest.raises(RuntimeError, match="failed"):
            run()
    else:
        run()
    torch.distributed.destroy_process_group.assert_called_once()
    assert not state["initialized"]
    assert torch.distributed.init_process_group.call_args.kwargs["timeout"].total_seconds() == 90


def test_existing_group_is_never_reused_or_destroyed():
    torch, state = mock_framework(initialized=True)
    with pytest.raises(RuntimeError, match="already exists"):
        with owned_process_group(torch, Launch(0, 0, 2)):
            pytest.fail("must reject before yielding")
    torch.npu.set_device.assert_not_called()
    torch.distributed.destroy_process_group.assert_not_called()
    assert state["initialized"]


def test_missing_device_fails_without_creating_group():
    torch, _ = mock_framework()
    torch.npu.is_available.return_value = False
    with pytest.raises(RuntimeError, match="unavailable"):
        with owned_process_group(torch, Launch(0, 0, 2)):
            pytest.fail("must reject before yielding")
    torch.distributed.init_process_group.assert_not_called()
    torch.distributed.destroy_process_group.assert_not_called()


@pytest.mark.parametrize("expanded", [False, True])
def test_payload_matching_survives_scheduling_permutation(expanded):
    case = make_case("repeated_slot")
    layout = dispatch(case, expanded=expanded)
    rows = list(layout.ranks[1])
    # Reverse duplicate rows within expert 3, including distinct slot weights.
    rows[-2:] = rows[-2:][::-1]
    raw = (
        [row.x for row in rows],
        None if expanded else [row.indices for row in rows],
        [row.weights[0] if expanded else row.weights for row in rows],
        layout.counts[1],
    )
    observed = observed_rows(case, 1, expanded, raw)
    assert observed.ranks[1] == tuple(rows)
    # Roundtrip must consume received values, not re-create clean source values.
    altered = replace(
        observed,
        ranks=((), (replace(rows[0], x=(0,) * case.hidden),) + tuple(rows[1:])),
    )
    clean = expert_payload(case, altered)[1]
    roundtrip = expert_payload(case, altered, use_received_x=True)[1]
    with pytest.raises(AssertionError):
        assert_close(roundtrip[0].x, clean[0].x)


@pytest.mark.parametrize("mode", ["disabled", "missing_capabilities", "requested", "empty_selection"])
def test_pytest_device_gate(mode):
    root = Path(__file__).resolve().parents[1]
    target = "tests/legacy/test_workflows.py::test_cached_dispatch"
    args = [sys.executable, "-m", "pytest", "-q", "-x", target]
    if mode != "disabled":
        args.append("--run-device")
    if mode in {"requested", "empty_selection"}:
        args.append("--device-capabilities=all")
    if mode == "empty_selection":
        args += ["-m", "not device"]
    env = dict(os.environ, PYTHONPATH=str(root), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        env.pop(name, None)
    result = subprocess.run(args, cwd=root, env=env, text=True, capture_output=True, timeout=30)
    output = result.stdout + result.stderr
    if mode == "disabled":
        assert result.returncode == 0 and "1 skipped" in output, output
    else:
        assert result.returncode != 0, output
        assert "1 skipped" not in output, output
        expected = {
            "missing_capabilities": "requires --device-capabilities",
            "requested": "torchrun",
            "empty_selection": "at least one selected, supported",
        }[mode]
        assert expected in output


def test_device_gate_rejects_multiple_cases_without_reinitialize():
    root = Path(__file__).resolve().parents[1]
    args = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--run-device",
        "--device-capabilities=legacy",
        "-k",
        "dispatch and normal",
        "tests/legacy/test_workflows.py",
    ]
    env = dict(os.environ, PYTHONPATH=str(root), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    result = subprocess.run(args, cwd=root, env=env, text=True, capture_output=True, timeout=30)
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "select exactly one supported device case" in output


def test_helpers_import_without_optional_dependencies():
    root = Path(__file__).resolve().parents[1]
    code = """
import sys
class RejectOptional:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'numpy', 'mpi4py', 'deep_ep'):
            raise RuntimeError('Unexpected backend dependency: ' + fullname)
sys.meta_path.insert(0, RejectOptional())
from tests.utils import cases, checks, reference, envs, device
case = cases.make_case('normal')
assert reference.dispatch(case).counts
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_aligned_expanded_positions_exclude_padding():
    case = make_case("normal", expert_alignment=128)
    handle = SimpleNamespace(psum_num_recv_tokens_per_expert=[3, 130])
    assert _valid_positions(case, True, handle, 5) == (0, 1, 2, 128, 129)


def test_aligned_dispatch_uses_unaligned_semantic_counts():
    handle = SimpleNamespace(num_unaligned_recv_tokens_per_expert=[3, 2])
    assert _semantic_counts(handle, [128, 128], 128) == [3, 2]
