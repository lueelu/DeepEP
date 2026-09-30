# Copyright (c) 2026, Lu Lu
# Modified by huangxiaolan 2026

"""Validate and summarize cached dispatch device timing and per-AIV diagnostics."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median


def decode_profile(records, rank, fp8=True, weights=False, world=64, qp1_workers=None):
    if len(records) != 64:
        raise ValueError("Expected 64 AIV records")
    decoded = []
    for core, words in enumerate(records):
        if len(words) != 32 or words[26:29] != [0x44504550, rank, core] or words[30] not in (1, 2, 3, 4, 5, 6, 7):
            raise ValueError(f"Missing/stale profile rank={rank} core={core}")

        def interval(start, end):
            if not words[start] or words[end] < words[start]:
                raise ValueError(f"Invalid profile interval rank={rank} core={core} slots={start},{end}")
            return (words[end] - words[start]) / 1e6  # Ascend 950, 1000 ticks/us.

        rear_role = "auxiliary" if words[30] >= 3 and weights else "scale" if fp8 else "idle"
        if qp1_workers is not None and qp1_workers not in (4, 8):
            raise ValueError("QP1 workers must be 4 or 8")
        clos_end = 16 + qp1_workers if qp1_workers is not None else (24 if words[30] >= 4 and world == 128 else 20)
        role = (
            rear_role
            if core >= 32
            else "relay_helper"
            if core == 7
            else "clos_qp0"
            if 8 <= core < 16
            else "clos_qp1"
            if 16 <= core < clos_end
            else "local_forward"
        )
        if world <= 8 and 7 <= core < clos_end:
            role = "idle"
        phases = {
            "control_init_ms": interval(0, 1),
            "entry_barrier_ms": interval(1, 2),
            "address_publish_ms": interval(2, 3),
            "primary_work_ms": interval(3, 4),
            "forward_ms": interval(4, 5),
            "payload_tail_ms": interval(5, 6),
            "finish_wait_ms": interval(6, 7),
            "total_ms": interval(0, 7),
        }
        if words[30] >= 7:
            phases["entry_setup_ms"] = phases.pop("entry_barrier_ms")
        if role.startswith("clos_"):
            phases.update(
                clos_init_ms=interval(3, 8),
                relay_submit_ms=interval(8, 9),
                direct_submit_ms=interval(9, 10),
                final_cq_wait_ms=interval(10, 11),
                helper_ack_wait_ms=interval(11, 12),
                quiet_ms=interval(12, 13),
            )
        if words[30] >= 5 and core < min(world, 8) - 1:
            if words[10] > words[5] or words[13] != rank // 8 * 8 + (rank % 8 + core + 1) % min(world, 8):
                raise ValueError(f"Invalid FM completion/peer rank={rank} core={core}")
            if not 0 <= words[12] <= words[11] <= 64 * words[12]:
                raise ValueError(f"Invalid FM batch counts rank={rank} core={core}")
            phases.update(
                fm_ready_and_submit_ms=interval(4, 8), fm_terminal_so_ms=interval(8, 9), fm_quiet_ms=interval(9, 10)
            )
        if role == "scale" or (words[30] >= 6 and core >= 32 and fp8):
            phases.update(
                scale_send_ms=interval(31 if words[30] >= 2 else 3, 14),
                scale_receive_ms=interval(14, 15),
                scale_flag_wait_ms=words[24] / 1e6,
                scale_unpack_ms=words[25] / 1e6,
            )
            if words[30] == 6:
                phases["scale_intra_pod_send_ms"] = interval(16, 17)
                if world > 64:
                    phases["scale_cross_pod_send_ms"] = interval(18, 19)
            elif words[30] >= 7:
                phases["scale_single_pass_send_ms"] = interval(16, 17)
        if words[30] >= 6 and core >= 32 and weights:
            phases["weight_total_ms"] = interval(3, 31)
        if role == "relay_helper":
            for owner in range(8):
                phases[f"owner_{owner}_ack_from_work_start_ms"] = interval(3, 16 + owner)
        decoded.append(
            {"rank": rank, "core": core, "role": role, "generation": words[29], "phases_ms": phases, "raw_ticks": words}
        )
        if words[30] >= 5 and core < min(world, 8) - 1:
            decoded[-1]["fm"] = {"transport": "urma", "wqes": words[11], "dbs": words[12], "peer": words[13]}
    return decoded


def decode_weight_profile(records, rank):
    if len(records) != 32:
        raise ValueError("Expected 32 weight worker records")
    decoded = []
    for owner, words in enumerate(records):
        if len(words) != 16 or words[15] != 1 or words[12] or not 0 < words[0] <= words[1] <= words[2]:
            raise ValueError(f"Invalid weight profile rank={rank} owner={owner}")
        decoded.append(
            {
                "rank": rank,
                "core": owner + 32,
                "send_ms": (words[1] - words[0]) / 1e6,
                "receive_ms": (words[2] - words[1]) / 1e6,
                "total_ms": (words[2] - words[0]) / 1e6,
                "scan_ms": words[3] / 1e6,
                "put_ms": words[4] / 1e6,
                "wait_ms": words[5] / 1e6,
                "expand_ms": words[6] / 1e6,
                "sent_entries": words[9],
                "received_entries": words[10],
                "raw_ticks": words,
            }
        )
    return decoded


def decode_detail(records, rank, batch, mode):
    if len(records) != 64:
        raise ValueError("Expected 64 detail records")
    counters = {
        0: "data_wqes",
        1: "data_dbs",
        2: "short_batches",
        3: "sq_wraps",
        4: "so_submissions",
        5: "planned_wqes",
        6: "planned_dbs",
        24: "relay_wqes",
        25: "direct_wqes",
        26: "relay_dbs",
        27: "direct_dbs",
    }
    times = {
        7: "route_load",
        8: "build_and_fences",
        9: "sq_copy",
        10: "db_helper",
        11: "peer_ready",
        12: "finish_quiet",
        13: "local_join",
        14: "done_publish",
        15: "remote_done_wait",
        16: "final_join",
    }
    out = []
    for core, w in enumerate(records):
        if (
            len(w) != 32
            or w[17] != batch
            or w[18] != mode
            or (w[20:23] != [0x44504454, rank, core] or w[23] not in (1, 2))
        ):
            raise ValueError("Stale detail profile or wrong batch/mode")
        if min(w) < 0 or w[0] != w[5] or w[1] != w[6] or w[0] != w[24] + w[25] or w[1] != w[26] + w[27]:
            raise ValueError("Actual WQE/DB counts disagree with plan")
        if w[2] > w[1] or w[3] > w[1] or w[0] > w[1] * batch or (w[1] and w[0] < w[1]):
            raise ValueError("Invalid batch counts")
        if mode == 1 and any(w[i] for i in times):
            raise ValueError("Counter-only detail unexpectedly read clocks")
        clos_end = 20
        if w[23] == 2:
            if w[28] not in (2, 4, 8, 16, 32, 64, 128) or w[29] not in (4, 8):
                raise ValueError("Invalid detail worker schedule")
            clos_end = 16 + w[29]
        if not 8 <= core < clos_end and (w[0] or w[1] or w[4]):
            raise ValueError("Unexpected Clos owner")
        out.append(
            dict(
                rank=rank,
                core=core,
                counters={name: w[i] for i, name in counters.items()},
                phases_ms={name + "_ms": w[i] / 1e6 for i, name in times.items()},
                raw_words=w,
            )
        )
    return out


def summarize(data):
    if not data:
        raise ValueError("No rank results")
    world = data[0]["world"]
    if len(data) != world or {r["rank"] for r in data} != set(range(world)) or not all(r["passed"] for r in data):
        raise ValueError("Complete successful rank results required")
    for key in ("world", "tokens", "topk", "experts", "dtype", "weights", "backend", "source_sha256"):
        if len({r[key] for r in data}) != 1:
            raise ValueError(f"Mixed configuration: {key}")
    for key in ("hidden", "scales", "rotation_wqes", "route_profile", "route_seed", "route_sha256", "fm_transport"):
        if any(key in r for r in data) and (not all(key in r for r in data) or len({r[key] for r in data}) != 1):
            raise ValueError(f"Mixed configuration: {key}")
    count = len(data[0]["clean_device_ms"])
    if count < 1 or any(len(r["clean_device_ms"]) != count for r in data):
        raise ValueError("Incomplete clean device samples")
    if any(
        len(sample) != 4 or any(not math.isfinite(v) or v <= 0 for v in sample)
        for rank in data
        for sample in rank["clean_device_ms"]
    ):
        raise ValueError("Invalid Event sample")
    result = {
        "passed_ranks": world,
        "tokens": data[0]["tokens"],
        "clean_samples": count,
        "backend": data[0]["backend"],
        "source_sha256": data[0]["source_sha256"],
    }
    result.update(
        {
            key: data[0][key]
            for key in (
                "experts",
                "topk",
                "hidden",
                "scales",
                "weights",
                "route_profile",
                "route_seed",
                "route_sha256",
                "fm_transport",
            )
            if key in data[0]
        }
    )
    for index, name in enumerate(("input_copy", "kernel", "output_copy", "device_total")):
        values = [max(r["clean_device_ms"][i][index] for r in data) for i in range(count)]
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"Invalid Event time: {name}")
        result[name] = {
            "rank_max_samples_ms": values,
            "median_ms": median(values),
            "mean_ms": mean(values),
            "p95_ms": sorted(values)[math.ceil(len(values) * 0.95) - 1],
            "min_ms": min(values),
            "max_ms": max(values),
        }
    t, k = data[0]["tokens"], data[0]["topk"]
    fp8 = data[0]["dtype"] == "fp8"
    hidden, scales = t * k * 7168 * (1 if fp8 else 2), t * k * 224 if fp8 else 0
    ms = result["kernel"]["mean_ms"]
    result["algorithm_bandwidth"] = {
        "logical_hidden_bytes_per_rank": hidden,
        "logical_scale_bytes_per_rank": scales,
        "logical_total_bytes_per_rank": hidden + scales,
        "GBps_per_rank": (hidden + scales) / (ms * 1e6),
        "GBps_all_ranks_logical": world * (hidden + scales) / (ms * 1e6),
        "definition": "T*K*(hidden_bytes+scale_bytes) / mean(rank-MAX clean kernel Event ms); decimal GB/s. "
        "Includes local logical routes, excludes padding/protocol/duplicate physical hops and weight bytes; "
        "kernel time includes weights when enabled; not link bandwidth.",
    }
    profiles = len(data[0]["kernel_profiles"])
    if (profiles < 1 and not all(r.get("event_only", False) for r in data)) or any(
        len(r["kernel_profiles"]) != profiles for r in data
    ):
        raise ValueError("Incomplete diagnostic samples")
    groups = {}
    for sample in range(profiles):
        maxima = {}
        for rank in data:
            diagnostics = rank["kernel_profiles"][sample]
            if diagnostics.get("detail_cores"):
                decode_detail(
                    [c["raw_words"] for c in diagnostics["detail_cores"]],
                    rank["rank"],
                    rank["rotation_wqes"],
                    diagnostics["detail_mode"],
                )
            weights = rank["kernel_profiles"][sample].get("weight_cores", [])
            if rank["weights"]:
                if len(weights) != 32 or {c["core"] for c in weights} != set(range(32, 64)):
                    raise ValueError("Incomplete weight worker records")
                for record in weights:
                    if record["rank"] != rank["rank"]:
                        raise ValueError("Mismatched weight profile rank")
                    for phase in ("send", "receive", "total", "scan", "put", "wait", "expand"):
                        value = record[phase + "_ms"]
                        if not math.isfinite(value) or value < 0:
                            raise ValueError("Invalid weight phase time")
                        key = "weight/" + phase + "_ms"
                        if key not in maxima or value > maxima[key]["ms"]:
                            maxima[key] = {"ms": value, "rank": record["rank"], "core": record["core"]}
            records = rank["kernel_profiles"][sample]["cores"]
            if len(records) != 64:
                raise ValueError("Incomplete per-core records")
            for record in records:
                for name, value in record["phases_ms"].items():
                    key = record["role"] + "/" + name
                    if key not in maxima or value > maxima[key]["ms"]:
                        maxima[key] = {"ms": value, "rank": record["rank"], "core": record["core"]}
        for key, value in maxima.items():
            groups.setdefault(key, []).append(value)
    result["profile_phase_maxima"] = {
        key: {"samples": values, "median_ms": median(v["ms"] for v in values)} for key, values in groups.items()
    }
    routing_count = len(data[0].get("routing_device_ms", []))
    if any(len(r.get("routing_device_ms", [])) != routing_count for r in data):
        raise ValueError("Incomplete routing samples")
    if routing_count:
        result["routing"] = {}
        plan_locations = {
            bool(sample.get("plan_in_notify", False)) for rank in data for sample in rank["routing_device_ms"]
        }
        if len(plan_locations) != 1:
            raise ValueError("Cannot mix routing timing boundaries from different implementations")
        plan_in_notify = plan_locations.pop()
        for field, names in (
            (
                "notify_device_ms",
                ("kernel", "post_barrier_and_plan" if plan_in_notify else "post_barrier", "count_copy", "total"),
            ),
            ("prepare_device_ms", ("clear", "kernel", "event_gap", "total")),
        ):
            for rank in data:
                for sample in rank["routing_device_ms"]:
                    values = sample.get(field, [])
                    if len(values) != 4 or any(not math.isfinite(v) or v < 0 for v in values):
                        raise ValueError("Invalid routing Event sample")
            for column, name in enumerate(names):
                values = [max(r["routing_device_ms"][i][field][column] for r in data) for i in range(routing_count)]
                result["routing"][field.removesuffix("_device_ms") + "/" + name] = {
                    "rank_max_samples_ms": values,
                    "mean_ms": mean(values),
                    "median_ms": median(values),
                }
        result["routing_note"] = (
            "Separate fresh-handle passes, warmup excluded, no added pre-timing barrier. "
            + (
                "Notify total includes preparation; prepare timings are nested, not additional. "
                if plan_in_notify
                else ""
            )
            + "Notify kernel excludes its post barrier and count copy; prepare kernel includes its internal "
            "admission barriers and forward-plan construction, excludes clear and host status readback. "
            "Phase rank maxima must not be added to infer end-to-end latency."
        )
    result["profile_note"] = (
        "Separate instrumented passes; phases overlap across cores and their maxima must not be summed."
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    summary = summarize([json.loads(p.read_text()) for p in sorted(args.results.glob("rank-*.json"))])
    output = args.results.parent / "kernel-summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    with (args.results.parent / "kernel-phases.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["role/phase", "sample", "max_ms", "rank", "core"])
        for key, phase in sorted(summary["profile_phase_maxima"].items()):
            for index, sample in enumerate(phase["samples"]):
                writer.writerow([key, index, sample["ms"], sample["rank"], sample["core"]])
    print(json.dumps({k: v for k, v in summary.items() if k != "profile_phase_maxima"}, indent=2))
