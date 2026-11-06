# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026

"""Opt-in MXFP8 diagnosis; reference replays do not change the precision verdict."""

import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch

from atk_compare import DEFAULT_RATIOS, compare_with_atk
from npu_baseline import (
    _grouped_matmul,
    _grouped_wgrad,
    _mxfp8_saved_roundtrip,
    _swiglu_backward_fp32,
    _swiglu_fp32,
)
from precision_metrics import OUTPUTS, write_report


def difference(actual, reference):
    """Auxiliary error facts, with no tolerance or replacement ATK decision."""
    a, b = (value.detach().cpu() for value in (actual, reference))
    if a.shape != b.shape:
        raise ValueError(f"Diagnostic shapes differ: {a.shape} != {b.shape}")
    changed = (a != b).reshape(-1)
    a64, b64 = a.double().reshape(-1), b.double().reshape(-1)
    finite = torch.isfinite(a64) & torch.isfinite(b64)
    delta = (a64 - b64)[finite]
    maximum = float(delta.abs().max()) if delta.numel() else 0.0
    different = int(changed.sum())
    return {
        "shape": list(a.shape),
        "numel": a.numel(),
        "equal": torch.equal(a, b),
        "different": different,
        "nonfinite_pairs": int((~finite).sum()),
        "max_abs_finite": maximum,
        "rmse_finite": float(delta.square().mean().sqrt()) if delta.numel() else None,
        "relative_l2_finite": float(delta.norm() / b64[finite].norm().clamp_min(torch.finfo(torch.float64).tiny)),
        "first_different_flat_index": int(changed.to(torch.int8).argmax()) if different else None,
        "zeroed": int(((a == 0) & (b != 0)).sum()),
    }


@torch.no_grad()
def replay_saved_state(inputs, trace, device, planes, limit):
    """Reuse one reference raw_z/dHp; only the selected saved planes are rounded.

    This deliberately shares the contract codec and activation helpers. It
    isolates error propagation but cannot independently qualify the native
    encoder/decoder, nor does it capture fused-kernel intermediate tensors.
    """
    if not set(planes) <= {"gate", "up"}:
        raise ValueError(f"Unknown saved planes: {planes}")
    rows, slots, counts = (trace[key] for key in ("rows", "slots", "counts"))
    result = {
        "ids": inputs["ids"],
        "routes": [],
        "counts": counts,
        "reference": "megamoe",
        "topk_weight_dtype": inputs["p"].dtype,
    }
    if not rows.numel():
        result.update(
            dw1=torch.zeros_like(inputs["w1"], device="cpu"),
            dw2=torch.zeros_like(inputs["w2"], device="cpu"),
        )
        return result, {}

    def tensor(value):
        return value.to(device=device, dtype=torch.bfloat16).contiguous()

    x, dy = (tensor(value) for value in (inputs["x"][rows], inputs["dy"][rows]))
    p = inputs["p"][rows, slots, None].to(device=device).contiguous()
    raw_z, dhp = tensor(trace["raw_z"]), tensor(trace["dhp"])
    gate, up = raw_z.chunk(2, dim=-1)
    z = torch.cat(
        tuple(
            _mxfp8_saved_roundtrip(value) if name in planes else value for name, value in (("gate", gate), ("up", up))
        ),
        dim=-1,
    )
    h = _swiglu_fp32(z, limit, raw_z)
    hp = (h * p.float()).to(torch.bfloat16)
    dp = (dhp.float() * h.float()).sum(-1).to(p.dtype)
    dz = _swiglu_backward_fp32(dhp.float() * p.float(), z, limit, raw_z)
    weight1 = tensor(inputs["w1"].transpose(1, 2))
    dx = _grouped_matmul(dz, weight1.transpose(1, 2).contiguous(), counts)
    dw1 = _grouped_wgrad(x, dz, counts).reshape_as(weight1).transpose(1, 2).contiguous()
    dw2 = _grouped_wgrad(hp, dy, counts).reshape_as(inputs["w2"].transpose(1, 2)).transpose(1, 2).contiguous()
    result.update(dw1=dw1.cpu(), dw2=dw2.cpu())
    # Forward never consumes the compressed planes. Reuse the same captured
    # reference y in every replay, along with the same raw_z and dHp.
    dx, dp = dx.cpu(), dp.cpu()
    offset = 0
    for expert, size in enumerate(counts.tolist()):
        stop = offset + size
        if size:
            result["routes"].append(
                {
                    "rows": rows[offset:stop],
                    "slots": slots[offset:stop],
                    "y": trace["forward_y"][offset:stop],
                    "dx": dx[offset:stop],
                    "dp": dp[offset:stop],
                }
            )
        else:
            result["dw1"][expert].zero_()
            result["dw2"][expert].zero_()
        offset = stop
    stages = {name: value.detach().cpu().clone() for name, value in dict(z=z, h=h, hp=hp, dp=dp, dz=dz).items()}
    return result, stages


def runtime_identity():
    from deep_ep._native import require_native

    require_native("MegaMoE MXFP8 diagnostics")

    def fingerprint(path):
        path = Path(path).resolve()
        with path.open("rb") as stream:
            return {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}

    return {
        "python": sys.executable,
        "torch": torch.__version__,
        "native_libraries": {
            name: fingerprint(importlib.import_module(name).__file__)
            for name in ("deep_ep._C", "deep_ep_experimental.megamoe_train._C")
        },
        "kernel_library": fingerprint(
            Path(importlib.import_module("deep_ep_experimental.megamoe_train").__file__).parent
            / "libdeepep_megamoe_train_bf16.so"
        ),
        "build_info": json.loads(
            (Path(importlib.import_module("deep_ep").__file__).parent / "_build_info.json").read_text(encoding="utf-8")
        ),
        "test_sources": {
            name: fingerprint(Path(__file__).with_name(name))
            for name in ("test_mega_moe.py", "npu_baseline.py", "mxfp8_diagnostics.py")
        },
    }


def capture_fused_saved_tensors(fused, mode, path):
    """Observe the existing autograd saved state without changing its contents.

    The BF16 bridge saves training_saved_state and dispatch_output as its two
    UInt8 tensors, in that order. Detach the packed handle to avoid a graph
    reference cycle; backward still receives the original device/storage.
    """
    captured = []

    def pack(value):
        if value.dtype == torch.uint8:
            captured.append(value.detach().cpu().clone())
        return value.detach()

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        result = fused(mode)
    if len(captured) != 2:
        raise RuntimeError(f"Expected saved-state/dispatch UInt8 pair, captured {len(captured)} tensors")
    torch.save(dict(saved_state=captured[0], dispatch_output=captured[1]), path)
    return result


def diagnose_mxfp8(inputs, trace, actual, contract, baseline, truth, args, rank, device, directory, *, combine, fused):
    directory.mkdir(parents=True, exist_ok=False)
    torch.save(inputs, directory / "inputs.pt")
    torch.save(trace, directory / "reference_trace.pt")
    torch.save(dict(actual=actual, contract=contract, benchmark=baseline, golden=truth), directory / "outputs.pt")
    report = {
        "rank": rank,
        "identity": runtime_identity(),
        "scope": "Fused saved-state snapshots, save-mode controls and reference-side NPU replay",
        "affects_case_verdict": False,
        "codec_note": "Replay shares the contract codec; native encode/decode correctness is still unverified",
        "ratios": list(args.atk_ratios or DEFAULT_RATIOS),
        "reference_rows": int(trace["rows"].numel()),
        "fused_repeats": [],
        "fused_controls": {},
        "replays": {},
    }

    def checkpoint():
        write_report(directory / "diagnosis.json", report)

    checkpoint()
    # Three total fused executions, including the official result already
    # captured by the caller. Do not infer stability from reference replays.
    for iteration in (1, 2):
        repeated = capture_fused_saved_tensors(fused, args.save_gmm1_output, directory / f"saved_repeat_{iteration}.pt")
        changes = {name: difference(repeated[name], actual[name]) for name in (*OUTPUTS, "counts")}
        report["fused_repeats"].append({"iteration": iteration, "changes": changes})
        torch.save(repeated, directory / f"fused_repeat_{iteration}.pt")
        print(
            f"[MXFP8 REPEAT] rank={rank} iteration={iteration} equal={all(v['equal'] for v in changes.values())}",
            flush=True,
        )
        checkpoint()

    for mode in ("bf16", "none"):
        control = capture_fused_saved_tensors(fused, mode, directory / f"saved_{mode}.pt")
        torch.save(control, directory / f"fused_{mode}.pt")
        atk = compare_with_atk(control, baseline, truth, directory / f"atk_fused_{mode}", report["ratios"])
        record = {
            "atk": atk,
            "counts_equal": torch.equal(control["counts"], actual["counts"]),
            "vs_fused_mxfp8": {name: difference(control[name], actual[name]) for name in OUTPUTS},
        }
        report["fused_controls"][mode] = record
        print(f"[MXFP8 CONTROL] rank={rank} fused save={mode} ATK={'PASS' if atk['passed'] else 'FAIL'}", flush=True)
        checkpoint()

    uncompressed_stages = None
    for name, planes in (
        ("uncompressed", ()),
        ("gate_only", ("gate",)),
        ("up_only", ("up",)),
        ("both", ("gate", "up")),
    ):
        partial, stages = replay_saved_state(inputs, trace, device, planes, args.swiglu_limit)
        result = combine(partial)
        torch.save(dict(stages=stages, outputs=result), directory / f"replay_{name}.pt")
        if uncompressed_stages is None:
            uncompressed_stages = stages
        record = {
            "kind": "reference_replay_not_fused_kernel",
            "quantized_planes": list(planes),
            "vs_uncompressed_stages": {
                key: difference(value, uncompressed_stages[key]) for key, value in stages.items()
            },
            "vs_fused_mxfp8": {key: difference(result[key], actual[key]) for key in OUTPUTS},
            "atk": compare_with_atk(result, baseline, truth, directory / f"atk_replay_{name}", report["ratios"]),
        }
        if stages:
            decoded = stages["z"].chunk(2, dim=-1)
            original = trace["raw_z"].chunk(2, dim=-1)
            record["saved_plane_errors"] = {
                key: difference(value, raw) for key, value, raw in zip(("gate", "up"), decoded, original, strict=True)
            }
        if name == "both":
            record["vs_captured_reference_stages"] = {
                key: difference(value, trace[key]) for key, value in stages.items()
            }
            record["vs_contract_outputs"] = {key: difference(result[key], contract[key]) for key in OUTPUTS}
        report["replays"][name] = record
        print(
            f"[MXFP8 REPLAY] rank={rank} {name} ATK={'PASS' if record['atk']['passed'] else 'FAIL'} (reference only)",
            flush=True,
        )
        checkpoint()
    return {"report": str(directory / "diagnosis.json"), "affects_case_verdict": False}
