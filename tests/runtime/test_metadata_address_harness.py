"""CPU checks for the address-collision harness, without loading Torch or NPU."""

import ctypes
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def harness():
    path = Path(__file__).parents[1] / "clos/source/test_metadata_address.py"
    spec = importlib.util.spec_from_file_location("metadata_address_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_libc(monkeypatch, harness, address):
    libc = SimpleNamespace(mmap=Mock(return_value=address), munmap=Mock(return_value=0))
    monkeypatch.setattr(harness, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(harness.ctypes, "CDLL", Mock(return_value=libc))
    return libc


def test_reserve_exact_address_and_release_once(monkeypatch, harness):
    libc = fake_libc(monkeypatch, harness, harness.OLD_META_ADDRESS)
    reservation = harness.HostReservation()
    libc.mmap.assert_called_once_with(harness.OLD_META_ADDRESS, 32 << 20, 0, 0x100022, -1, 0)
    libc.munmap.assert_not_called()
    reservation.close()
    reservation.close()
    libc.munmap.assert_called_once_with(harness.OLD_META_ADDRESS, 32 << 20)


def test_failed_reservation_is_not_a_reproduction(monkeypatch, harness):
    libc = fake_libc(monkeypatch, harness, ctypes.c_void_p(-1).value)
    with pytest.raises(OSError, match="Host mmap injection failed"):
        harness.HostReservation()
    libc.munmap.assert_not_called()


def test_ignored_noreplace_is_rejected_and_unmapped(monkeypatch, harness):
    libc = fake_libc(monkeypatch, harness, 0x12340000)
    with pytest.raises(RuntimeError, match="did not honor MAP_FIXED_NOREPLACE"):
        harness.HostReservation()
    libc.munmap.assert_called_once_with(0x12340000, 32 << 20)


@pytest.mark.parametrize("api", ["aclshmemx_init_attr", "aclshmemx_init_attr_with_buffers"])
def test_only_native_shmem_init_failure_matches(harness, api):
    error = RuntimeError(f"CANN SHMEM call failed: {api}(args), status=-1, location=file:123")
    assert harness.expected_init_error(error, SimpleNamespace(expect_init="failure"))
    assert not harness.expected_init_error(error, SimpleNamespace(expect_init="success"))


@pytest.mark.parametrize(
    "message",
    [
        "HCCL timeout",
        "Host mmap injection failed",
        "cannot import torch_npu",
        "aclshmemx_init_attr symbol not found",
        "device barrier failed",
    ],
)
def test_unrelated_errors_do_not_match(harness, message):
    assert not harness.expected_init_error(RuntimeError(message), SimpleNamespace(expect_init="failure"))
