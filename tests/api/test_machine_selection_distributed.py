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
        for kind, alias in (("mesh_clos", "pod"), ("global_clos", "clos_only"), ("mesh_pcie", "card_4pmesh")):
            assert _machine.select(kind, group)["machine_type"] == kind
            assert _machine.select(alias if rank == 0 else kind, group)["machine_type"] == kind
            results.append(kind)
        _machine.detect = lambda _: {"machine_type": "global_clos", "source": "test"}
        assert _machine.select("auto", group)["machine_type"] == "global_clos"
        results.append("auto")
        with pytest.raises(ValueError, match="different machine backends"):
            _machine.select("mesh_clos" if rank == 0 else "global_clos", group)
        results.append("mixed_rejected")
        with pytest.raises(RuntimeError, match="failed collectively"):
            _machine.select("mesh_clos" if rank == 0 else "invalid", group)
        results.append("peer_error_rejected")
        assert _machine.select("mesh_clos", group)["machine_type"] == "mesh_clos"
        results.append("collective_order_preserved")
        import deep_ep
        from deep_ep import buffer as legacy
        from deep_ep.buffers import elastic
        from deep_ep.backends.direct import api as _clos_adapter
        from deep_ep.backends import mesh_pcie as _card_adapter

        # Rank-local legacy arguments must not change the chosen engine or the
        # order of topology collectives. Constructors are CPU-only doubles.
        elastic.require_native = legacy.require_native = lambda _: None

        class Engine:
            def __init__(self, *args, **kwargs):
                self.transport = kwargs["transport"]

        def pod_initialize(self, *args, **kwargs):
            assert "transport" not in kwargs

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_clos_adapter, "buffer_class", lambda _: Engine)
            patch.setattr(_card_adapter, "buffer_class", lambda _: Engine)
            for api in (deep_ep.Buffer, deep_ep.ElasticBuffer):
                patch.setattr(api.__bases__[0], "__init__", pod_initialize)
                for kind in ("auto", "mesh_clos", "global_clos", "mesh_pcie"):
                    for old_value in (None, "auto", "mte", "udma", "invalid"):
                        for options in (
                            {"transport": old_value} if rank == 0 else {},
                            {"transport": "mte" if rank == 0 else "udma"},
                        ):
                            instance = api(group, machine_type=kind, **options)
                            expected = "global_clos" if kind == "auto" else kind
                            assert instance.machine_type == expected
                            if expected != "mesh_clos":
                                assert instance.transport == ("mte" if expected == "mesh_pcie" else "udma")
        assert _machine.select("mesh_clos", group)["machine_type"] == "mesh_clos"
        results.append("legacy_transport_ignored_collectively")
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
            "mesh_clos",
            "global_clos",
            "mesh_pcie",
            "auto",
            "mixed_rejected",
            "peer_error_rejected",
            "collective_order_preserved",
            "legacy_transport_ignored_collectively",
            "card_collective_guards",
        ]


def test_real_mpi_communicator_selection():
    pytest.importorskip("torch.distributed")
    mpi = pytest.importorskip("mpi4py.MPI")
    from deep_ep import _machine

    for kind in ("mesh_clos", "global_clos", "mesh_pcie"):
        assert _machine.select(kind, mpi.COMM_SELF)["machine_type"] == kind
