"""Real CPU process groups exercise selection without an NPU or native extension."""

from datetime import timedelta
import json
from pathlib import Path
import time

import pytest


def _selection_worker(rank, store, result_dir):
    import torch.distributed as dist
    from deep_ep import _machine

    dist.init_process_group("gloo", init_method=store, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    results = []
    try:
        # A subgroup must be passed through, not silently replaced by WORLD.
        group = dist.new_group([0, 1], backend="gloo", timeout=timedelta(seconds=30))
        for kind in ("pod", "clos_only", "card_4pmesh"):
            assert _machine.select(kind, group)["machine_type"] == kind
            results.append(kind)
        _machine.detect = lambda _: {"machine_type": "clos_only", "source": "test"}
        assert _machine.select("auto", group)["machine_type"] == "clos_only"
        results.append("auto")
        with pytest.raises(ValueError, match="different machine backends"):
            _machine.select("pod" if rank == 0 else "clos_only", group)
        results.append("mixed_rejected")
        with pytest.raises(RuntimeError, match="failed collectively"):
            _machine.select("pod" if rank == 0 else "invalid", group)
        results.append("peer_error_rejected")
        assert _machine.select("pod", group)["machine_type"] == "pod"
        results.append("collective_order_preserved")
        import deep_ep
        from deep_ep import buffer as legacy
        from deep_ep.buffers import elastic

        # An old argument on just one rank must stop both before any backend
        # initializes. Matching/None values are removed arguments too.
        elastic.require_native = legacy.require_native = lambda _: None
        for api in (deep_ep.Buffer, deep_ep.ElasticBuffer):
            for kind in ("auto", "pod", "clos_only", "card_4pmesh"):
                for old_value in (None, "auto", "mte", "udma"):
                    options = {"transport": old_value} if rank == 0 else {}
                    with pytest.raises(RuntimeError, match="transport is selected by machine_type"):
                        api(group, machine_type=kind, **options)
        assert _machine.select("pod", group)["machine_type"] == "pod"
        results.append("removed_transport_rejected_collectively")
        from deep_ep import _card_adapter
        from types import SimpleNamespace

        _card_adapter.host_identity = lambda: "same-host"
        _card_adapter.prepare(group, {})
        with pytest.raises(RuntimeError, match="preflight failed collectively"):
            _card_adapter.prepare(group, {"udma_schedule": "token" if rank == 0 else "peer"})
        _card_adapter.host_identity = lambda: f"host-{rank}"
        with pytest.raises(ValueError, match="same host"):
            _card_adapter.prepare(group, {})
        aborted = []
        instance = _card_adapter._MappedMte()
        instance._native = SimpleNamespace(
            mte_mapping_errors=lambda: ["unmapped peer"] if rank == 0 else [],
            abort_initialization=lambda: aborted.append(True),
        )
        with pytest.raises(RuntimeError, match="mapping check failed collectively"):
            instance._validate_native_mappings(group)
        assert aborted == [True] and instance._native is None
        results.append("card_collective_guards")
        Path(result_dir, f"rank{rank}.json").write_text(json.dumps(results), encoding="utf-8")
        dist.destroy_process_group(group)
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_selection(tmp_path):
    dist = pytest.importorskip("torch.distributed")
    if not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable in this Torch build")
    import torch.multiprocessing as mp

    context = mp.spawn(
        _selection_worker,
        args=((tmp_path / "store").resolve().as_uri(), str(tmp_path)),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("CPU process-group selection timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    for rank in range(2):
        assert json.loads((tmp_path / f"rank{rank}.json").read_text()) == [
            "pod",
            "clos_only",
            "card_4pmesh",
            "auto",
            "mixed_rejected",
            "peer_error_rejected",
            "collective_order_preserved",
            "removed_transport_rejected_collectively",
            "card_collective_guards",
        ]


def test_real_mpi_communicator_selection():
    pytest.importorskip("torch.distributed")
    mpi = pytest.importorskip("mpi4py.MPI")
    from deep_ep import _machine

    for kind in ("pod", "clos_only", "card_4pmesh"):
        assert _machine.select(kind, mpi.COMM_SELF)["machine_type"] == kind
