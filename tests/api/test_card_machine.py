"""MESH_PCIE classification and public V1/V2 constructor policy (CPU only)."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deep_ep import _machine


def card_topology():
    # Deliberately nonconsecutive IDs: never derive the Mesh domain via rank//4.
    nodes = ["2", "5", "9", "14"]
    root = {
        "rank_list": [
            {
                "device_id": 3,
                "local_id": "5",
                "level_list": [
                    {
                        "net_type": "TOPO_FILE_DESC",
                        "rank_addr_list": [{"addr_type": "EID", "addr": "10" * 16, "ports": ["0/4"]}],
                    },
                    {
                        "net_type": "CLOS",
                        "rank_addr_list": [{"addr_type": "IPV4", "addr": "192.0.2.1", "ports": ["d2h"]}],
                    },
                ],
            }
        ],
    }
    edges = [
        {"local_a": a, "local_b": b, "local_a_ports": ["0/4"], "local_b_ports": ["0/5"]}
        for index, a in enumerate(nodes)
        for b in nodes[index + 1 :]
    ]
    edges.append({"local_a": "5", "local_a_ports": ["d2h"], "topo_type": "CLOS"})
    return root, {"edge_list": edges}


def test_card_rootinfo_and_board_detection(monkeypatch, tmp_path):
    root, topo = card_topology()
    assert _machine.classify_root(root, topo, 3) == "mesh4"
    topo_path, root_path = tmp_path / "topo.json", tmp_path / "root.json"
    topo_path.write_text(json.dumps(topo))
    root["topo_file_path"] = str(topo_path)
    root_path.write_text(json.dumps(root))
    monkeypatch.setattr(_machine, "physical_device", lambda: 3)
    monkeypatch.setattr(_machine, "board_id", lambda _: 0x6C)
    result = _machine.detect(root_path)
    assert result["machine_type"] == "mesh_pcie"
    assert result["source"] == "rootinfo" and result["physical_device"] == 3
    monkeypatch.setattr(_machine, "board_id", lambda _: 3)
    with pytest.raises(ValueError, match="requires MESH_PCIE"):
        _machine.detect(root_path)


@pytest.mark.parametrize("fault", ["missing_edge", "extra_peer", "host_only", "invalid_eid"])
def test_card_incomplete_topology_is_rejected(fault):
    root, topo = card_topology()
    if fault == "missing_edge":
        topo["edge_list"].pop(0)
    elif fault == "extra_peer":
        edge = copy.deepcopy(topo["edge_list"][0])
        edge["local_b"] = "22"
        topo["edge_list"].append(edge)
    elif fault == "host_only":
        root["rank_list"][0]["level_list"].pop(0)
    else:
        root["rank_list"][0]["level_list"][0]["rank_addr_list"][0]["addr"] = "bad"
    with pytest.raises(ValueError):
        _machine.classify_root(root, topo, 3)


def test_card_without_rootinfo_uses_board_rule(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda _: False)
    monkeypatch.setattr(_machine, "physical_device", lambda: 3)
    monkeypatch.setattr(_machine, "board_id", lambda _: 0x6C)
    result = _machine.select()
    assert result["machine_type"] == "mesh_pcie" and result["board"] == 0x6C
    with pytest.raises(RuntimeError, match="explicit rootinfo is missing"):
        _machine.select(rootinfo_path="missing.json")


@pytest.fixture
def card(monkeypatch):
    pytest.importorskip("torch")
    from deep_ep.backends import mesh_pcie as _card_adapter

    monkeypatch.setattr(_card_adapter, "host_identity", lambda: "one-host")
    monkeypatch.setattr(_card_adapter, "gather_records", lambda _, record: [record, copy.deepcopy(record)])
    return _card_adapter


@pytest.mark.parametrize(
    "value", ["9bb60993-7841-4082-a37b-f9ce989ddab9\n", "", "bad", "00000000-0000-0000-0000-000000000000"]
)
def test_host_identity_requires_valid_boot_id(monkeypatch, value):
    pytest.importorskip("torch")
    from deep_ep.backends import mesh_pcie as _card_adapter

    monkeypatch.setattr(Path, "read_text", lambda _: value)
    if value.startswith("9bb6"):
        assert _card_adapter.host_identity() == value.strip()
    else:
        with pytest.raises(ValueError):
            _card_adapter.host_identity()


def test_card_preflight_does_not_change_constructor_options(card):
    options = {"udma_schedule": "peer", "num_bytes": 123456}
    card.prepare(object(), options)
    assert options == {"udma_schedule": "peer", "num_bytes": 123456}


def test_incompatible_schedule_rejected_collectively(card):
    with pytest.raises(RuntimeError, match="preflight failed collectively"):
        card.prepare(object(), {"udma_schedule": "token"})


def test_card_requires_same_host_and_group(card, monkeypatch):
    with pytest.raises(ValueError, match="requires an EP"):
        card.prepare(None, {})
    monkeypatch.setattr(card, "gather_records", lambda _, record: [record, {"host": "another-host", "error": None}])
    with pytest.raises(ValueError, match="same host"):
        card.prepare(object(), {})
    monkeypatch.setattr(card, "gather_records", lambda _, record: [record, {"host": None, "error": "boot ID missing"}])
    with pytest.raises(RuntimeError, match="boot ID missing"):
        card.prepare(object(), {})


@pytest.mark.parametrize("kind", ["Buffer", "ElasticBuffer"])
@pytest.mark.parametrize("fault", [None, "local", "remote", "query", "old_library"])
def test_public_card_construction_checks_before_device_barrier(kind, fault, card, monkeypatch):
    pytest.importorskip("torch")
    import deep_ep
    from deep_ep import buffer as legacy
    from deep_ep.buffers import elastic as public_elastic
    from deep_ep.backends.direct import buffer as source_v1, elastic as source_v2

    calls = []
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 2)

    class Native:
        def __init__(self, *args):
            assert args[3] == "mte"
            calls.append("native_init")

        def mte_mapping_errors(self):
            calls.append("mapping")
            if fault == "old_library":
                raise AttributeError("mte_mapping_errors unavailable")
            if fault == "query":
                raise RuntimeError("query unavailable")
            return ["peer 1 workspace unmapped"] if fault == "local" else []

        def abort_initialization(self):
            calls.append("abort")
            if fault == "old_library":
                raise AttributeError("abort_initialization unavailable")

        def num_device_aiv(self):
            return 64

    monkeypatch.setattr(_machine, "select", lambda *args, **kw: {"machine_type": "mesh_pcie", "source": "test"})
    monkeypatch.setattr(legacy, "require_native", lambda _: None)
    monkeypatch.setattr(public_elastic, "require_native", lambda _: None)
    monkeypatch.setattr(source_v1, "_is_capturing", lambda: False)
    for module in (source_v1, source_v2):
        monkeypatch.setattr(module, "get_native_class", lambda _: Native)
        monkeypatch.setattr(module, "_prepare_shmem_bootstrap", lambda *args: ([], "test"))
        monkeypatch.setattr(module, "_create_communication_stream", lambda: SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(source_v2, "_validate_group_consensus", lambda *args: None)
    monkeypatch.setattr(source_v2.ElasticBuffer, "barrier", lambda *args, **kw: calls.append("device_barrier"))
    if fault == "remote":
        monkeypatch.setattr(
            card,
            "gather_records",
            lambda _, record: [record, ["peer 0 workspace unmapped"]] if isinstance(record, list) else [record, record],
        )
    public = getattr(deep_ep, kind)
    kwargs = {"machine_type": "mesh_pcie", "num_bytes": 128 * 1024 * 1024}
    if fault:
        with pytest.raises(RuntimeError, match="mapping check failed collectively"):
            public(group, **kwargs)
        assert calls == ["native_init", "mapping", "abort"]
    else:
        result = public(group, **kwargs)
        assert isinstance(result, public) and result.machine_type == "mesh_pcie"
        assert result.transport == "mte"
        assert calls == ["native_init", "mapping"] + (["device_barrier"] if kind == "ElasticBuffer" else [])
        # The CPU double has no device resources to release.
        result._destroyed = True


def test_v2_card_hint_uses_migrated_layout(monkeypatch):
    pytest.importorskip("torch")
    from deep_ep.buffers import elastic
    from deep_ep.backends.direct import api as _clos_adapter

    monkeypatch.setattr(elastic, "require_native", lambda _: None)
    monkeypatch.setattr(_clos_adapter.ElasticBuffer, "get_buffer_size_hint", lambda *args: 123456)
    assert elastic.ElasticBuffer.get_buffer_size_hint(None, 129, 6144, 8, machine_type="mesh_pcie") == 123456
