import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "ep32_replay", Path(__file__).resolve().parents[1] / "scripts/check_ep32_replay.py"
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def test_rounds_vary_every_input_and_rotate_physical_slots():
    cfg = dict(seed=42, model=dict(hidden=128, num_experts=32, num_topk=3))
    base = replay.make_round(cfg, 1, 0, world=2, bs=8)
    last = base
    hashes = []
    for step in range(replay.STEPS):
        current = replay.make_round(cfg, 1, step, world=2, bs=8)
        hashes.append(replay.digest(current))
        np.testing.assert_array_equal(current[2], (base[2] + step) % 32)
        assert replay.route_histogram(current[2], 32).sum() == 24
        if step:
            assert all(not np.array_equal(previous, this) for previous, this in zip(last, current))
        last = current
    assert len(set(hashes)) == 160
    assert replay.digest(replay.make_round(cfg, 1, 159, world=2, bs=8)) == hashes[-1]
    with pytest.raises(ValueError, match="distinct physical"):
        replay.route_histogram(np.array([[3, 3]], dtype=np.int32), 32)


def test_compare_all_calls_counts_and_independent_oracles(tmp_path, monkeypatch):
    for key, value in dict(STEPS=3, TAIL=(2, 0), BS=2, LOCAL=2, MODEL=dict(hidden=4)).items():
        monkeypatch.setattr(replay, key, value)
    args = SimpleNamespace(
        fixed_perf=False, **{key: tmp_path / key for key in ("bank", "reference", "replay", "candidate_eager")}
    )
    rank = 1
    folders = {key: folder / f"rank_{rank}" for key, folder in vars(args).items() if isinstance(folder, Path)}
    for f in folders.values():
        f.mkdir(parents=True)
    hashes = ["a", "b", "c"]
    physical = np.arange(6).reshape(3, 2)
    out = [np.full((2, 4), step, np.float32) for step in range(3)]
    np.savez(folders["bank"] / "golden.npz", output=out[0], counts=physical[0])
    replay.put_json(folders["bank"] / "bank.json", dict(hashes=hashes))
    for key in ("reference", "replay", "candidate_eager"):
        folder = folders[key]
        replay.put_json(
            folder / "run.json",
            dict(
                mode=key.replace("_", "-"),
                rank=rank,
                hashes=hashes,
                routed=["r"],
                shared=["s"],
                wave=5,
                bs=2,
                model=replay.MODEL,
                clamp=10,
                input="fp8",
                device="fake",
                cores={},
                device_properties="fake",
            ),
        )
        np.save(folder / "physical_counts.npy", physical)
        for i, step in enumerate([0, 1, 2] + (list(replay.TAIL) if key == "replay" else [])):
            np.savez(folder / f"result_{i}.npz", output=out[step], counts=physical[step])
    result = replay.compare_rank(args, rank)
    assert result["passed"] and result["stale_detected"] == 2
    np.savez(folders["replay"] / "result_3.npz", output=out[1], counts=physical[2] + 1)
    errors = replay.compare_rank(args, rank)["failures"]
    assert errors[0]["physical_slots"] == [2, 3]
    assert errors[1]["mismatched"] == 8 and errors[1]["max_abs"] == 1 and errors[1]["iteration"] == 3
    assert errors[2]["comparison"] == ["replay", "candidate-eager"]
    np.savez(folders["reference"] / "result_2.npz", output=out[1], counts=physical[2])
    assert replay.compare_rank(args, rank)["stale_detected"] == 1
    # Fixed runs use only round zero and require matching archived data and timer-off builds.
    args.fixed_perf, args.replay = True, None
    weights = dict(routed=["r"], shared=["s"])
    replay.put_json(
        folders["bank"] / "bank.json",
        dict(hashes=hashes, input="bf16", weights=weights, config=dict(shared_fixture="native-fp8")),
    )
    for key in ("reference", "candidate_eager"):
        path = folders[key] / "run.json"
        record = dict(json.loads(path.read_text()), fixed_perf=True, input="bf16", timer_enabled=False)
        replay.put_json(path, record)
        np.save(folders[key] / "physical_counts.npy", physical[:1])
    fixed = replay.compare_rank(args, rank)
    assert fixed["passed"] and fixed["rounds"] == 1 and fixed["alternate"] == 0
    args.baseline_eager, args.reference = args.reference, None
    for key, protocol, lanes in (("reference", 4, 16), ("candidate_eager", 5, 24)):
        path = folders[key] / "run.json"
        record = dict(json.loads(path.read_text()), mode="candidate-eager", bf16_bs96_active=protocol == 5,
                      compiled=dict(core_combine_reduce_version=protocol, combine_reduce_version=protocol,
                                    combine_qps=lanes, dispatch_qps=8, dispatch_workers=14))
        replay.put_json(path, record)
    assert replay.compare_rank(args, rank)["passed"]
    record["compiled"]["combine_qps"] = 16
    replay.put_json(path, record)
    assert not replay.compare_rank(args, rank)["passed"]
    np.savez(folders["candidate_eager"] / "result_0.npz", output=-out[0], counts=physical[0])
    assert replay.compare_rank(args, rank)["failures"][-1]["mismatched"] == 8
    assert replay.differences(np.array([np.nan]), np.array([np.nan]), (1,))["nonfinite"] == 1


@pytest.mark.parametrize("input_kind,fixture", [("bf16", "native-fp8"), ("fp8", "native-fp8"), ("fp8", "fp4-subset")])
def test_bank_roundtrip_and_global_counts(tmp_path, monkeypatch, input_kind, fixture):
    from deep_ep_experimental.megamoe import benchmark

    config = benchmark.load_config("ep32_bs96")
    config["model"] = dict(hidden=128, intermediate_hidden=128, num_experts=4, num_topk=2, num_shared_experts=1)
    for key, value in dict(MODEL=config["model"], BS=4, LOCAL=2, WORLD=2, STEPS=3).items():
        monkeypatch.setattr(replay, key, value)
    monkeypatch.setattr(benchmark, "load_config", lambda name: config)
    args = SimpleNamespace(
        bank=tmp_path / "bank",
        golden=tmp_path / "golden",
        first_rank=0,
        end_rank=2,
        workers=1,
        input=input_kind,
        shared_fixture=fixture,
    )
    replay.bank(args)
    for rank in range(2):
        folder = args.bank / f"rank_{rank}"

        def gather(counts, tag):
            step = int(tag.split()[-1])
            with np.load(args.bank / f"rank_{1 - rank}" / f"input_{step}.npz") as peer:
                peer_counts = np.bincount(peer["ids"].ravel(), minlength=4)
            return [counts, peer_counts]

        bank, meta, counts = replay.load_inputs(folder, rank, SimpleNamespace(all_gather=gather))
        assert len(bank) == len(set(meta["hashes"])) == 3
        assert meta["input"] == input_kind
        if input_kind == "bf16":
            x = benchmark.inputs(config, rank, 4, world=2)[0]
            np.testing.assert_array_equal(bank[0][0], (x.view(np.uint32) >> 16).astype(np.uint16))
            assert len(bank[0]) == 3
        with np.load(folder / "golden.npz") as gold:
            np.testing.assert_array_equal(counts[0], gold["counts"])
        for role in ("routed", "shared"):
            for i in range(2):
                with np.load(folder / f"{role}{i}.npz") as data:
                    assert replay.digest((data["w"], data["sf"])) == meta["weights"][role][i]
                    if role == "shared" and fixture == "native-fp8":
                        magnitude = data["w"] & 127
                        assert set(np.unique(magnitude)) == set(range(48, 80))


def test_prepared_rows_and_reference_wave_api():
    rows = np.arange(512 * 2).reshape(512, 2)
    np.testing.assert_array_equal(
        replay.interleave_rows(rows), np.concatenate([rows[:128], rows[256:384], rows[128:256], rows[384:]])
    )
    calls = []
    old = SimpleNamespace(set_wave_groups_for_benchmark=calls.append, _plan=SimpleNamespace(wave_groups=5))
    new = SimpleNamespace(set_wave=calls.append, _plan=SimpleNamespace(effective_wave=5))
    for buf, reference in ((old, True), (new, False)):
        replay.set_wave(buf, reference)
        assert replay.effective_wave(buf, reference) == 5
    assert calls == [5, 5]


def test_runtime_protocol_matches_each_wheel():
    for core, extension, reference, valid in (
        (2, 1, True, True),
        (1, 2, True, False),
        (4, 4, False, True),
        (5, 5, False, True),
        (5, 4, False, False),
        (4, 1, False, False),
        (2, 4, False, False),
    ):
        args = (
            SimpleNamespace(megamoe_combine_reduce_version=core),
            SimpleNamespace(combine_reduce_version=extension),
            reference,
        )
        if valid:
            result = replay.compiled_configuration(*args)
            assert (result["core_combine_reduce_version"], result["combine_reduce_version"]) == (core, extension)
        else:
            with pytest.raises(ValueError, match="compiled EP32"):
                replay.compiled_configuration(*args)


@pytest.mark.parametrize("role", ["routed", "shared"])
def test_weight_adapter_uses_parallel_sequences_and_preserves_bits(tmp_path, monkeypatch, role):
    import torch
    from deep_ep_experimental.megamoe.reference import make_weights

    shared = role == "shared"
    pairs = make_weights(1 if shared else 2, 128, 256, 37, native_fp8=shared)
    for j, (w, sf) in enumerate(pairs):
        np.savez(tmp_path / f"{role}{j}.npz", w=w, sf=sf)
    monkeypatch.setattr(torch.Tensor, "npu", lambda self: self.clone(), raising=False)
    monkeypatch.setattr(torch, "float8_e8m0fnu", torch.uint8, raising=False)
    casts = []

    def cast(w, fmt, *, customize_dtype, input_dtype):
        assert fmt == 29 and customize_dtype == torch.float8_e4m3fn and input_dtype == "packed-fp4"
        casts.append(w)
        return w

    def prepare(first, second, *, input_layout, planar_l1, shared):
        assert input_layout == "prepared" and shared == (role == "shared")
        for j, group in enumerate((first, second)):
            assert isinstance(group, tuple) and all(isinstance(items, list) for items in group)
            for i, (w, sf) in enumerate(zip(*group, strict=True)):
                for value, source in zip((w, sf), pairs[j], strict=True):
                    expected = replay.interleave_rows(source[i]) if j == 0 else source[i]
                    np.testing.assert_array_equal(value.view(torch.uint8).numpy(), expected)
        if shared:
            assert planar_l1 is None
        else:
            for i, (w, sf) in enumerate(zip(*planar_l1, strict=True)):
                np.testing.assert_array_equal(w.numpy(), pairs[0][0][i])
                np.testing.assert_array_equal(sf.numpy(), pairs[0][1][i])
        return SimpleNamespace(l1=first, l2=second)

    _, hashes = replay.load_weights(
        tmp_path, role, torch, prepare, SimpleNamespace(npu_format_cast=cast, float4_e2m1fn_x2="packed-fp4")
    )
    assert len(casts) == (0 if shared else 6)
    assert hashes == [replay.digest(p) for p in pairs]
