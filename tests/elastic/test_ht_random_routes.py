"""Host checks for the HT random routing port; these do not certify NPU execution."""

import importlib.util
from pathlib import Path
import sys

import pytest

from tests.elastic import test_combine as combine
from tests.elastic import test_dispatch as dispatch


@pytest.fixture
def reference_planner(monkeypatch):
    path = Path(__file__).with_name("combine_reference.py")
    spec = importlib.util.spec_from_file_location("_random_route_reference", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("world", (2, 4, 8, 16, 32, 64, 128, 256))
@pytest.mark.parametrize("seed", (0, 1, 42, -1, 2**32 + 7))
def test_random_matches_reference_generator(reference_planner, world, seed):
    # Include non-power-of-two expert counts, repeated slots and high source ranks.
    case = reference_planner.CombineCase(world, 3, 16, world * 3, "random", seed)
    routes = dispatch.make_routes(world, case.tokens, case.topk, case.experts, 0, "random", seed=seed)
    assert routes == [
        [reference_planner.route_experts(source, token, case) for token in range(case.tokens)]
        for source in range(world)
    ]


def test_random_reproducibility_and_duplicate_slots():
    routes = dispatch.make_routes(16, 17, 16, 32, 0, "random", seed=42)
    assert routes == dispatch.make_routes(16, 17, 16, 32, 0, "random", seed=42)
    assert routes == dispatch.make_routes(16, 17, 16, 32, 9, "random", seed=42)
    assert routes != dispatch.make_routes(16, 17, 16, 32, 0, "random", seed=43)
    assert routes != dispatch.make_routes(16, 17, 16, 32, 0, "normal")
    assert any(len(set(row)) < len(row) for rows in routes for row in rows)
    assert all(len(row) == 16 and all(0 <= expert < 32 for expert in row) for rows in routes for row in rows)
    golden = dispatch.build_notify_golden(routes, 32, 256)
    assert sum(golden["rank_rows"]) == 16 * 17 * 16
    assert sum(golden["gather_rows"]) > 0
    assert len(set(golden["rank_rows"])) > 1
    for rank in range(16):
        plan = combine.build_reference(routes, 32, rank)
        assert plan["rank_rows"] == golden["rank_rows"]
        assert plan["gather_rows"] == golden["gather_rows"]
        assert [(i // (17 * 16), i // 16 % 17, i % 16) for i in plan["weight_return_meta"]] == (
            dispatch.expert_rows(routes, 32, rank)
        )


@pytest.mark.parametrize("mode", sorted(set(dispatch._ROUTES + combine._ROUTES) - {"random"}))
def test_seed_does_not_change_existing_routes(mode):
    assert dispatch.make_routes(16, 3, 6, 32, 0, mode, seed=1) == dispatch.make_routes(16, 3, 6, 32, 0, mode, seed=42)


@pytest.mark.parametrize("entry", (dispatch, combine))
@pytest.mark.parametrize(
    "route, seed_args, seed", (("random", [], 1), ("random", ["--seed", "42"], 42), ("all", [], 1))
)
def test_random_cli_and_all(entry, route, seed_args, seed, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [entry.__file__, "--ep-size", "256", "--route", route, "--output", str(tmp_path / "output"), *seed_args],
    )
    if entry is combine:
        monkeypatch.setitem(combine._dispatch, "_launch", lambda args, worker: calls.append(args))
    else:
        monkeypatch.setattr(dispatch, "_launch", lambda args, worker: calls.append(args))
    entry.main()
    (args,) = calls
    assert (args.route, args.seed) == (route, seed)
    assert "random" in entry._ROUTES
