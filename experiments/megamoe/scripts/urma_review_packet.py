#!/usr/bin/env python3
"""One comment-safe coarse+fine packet, <=60000 ASCII characters/bytes; no Torch."""

import argparse
import json
import math
from pathlib import Path
import urma_shared_trace_to_text as shared
import urma_rank_compare as pair

STAGES = [
    ("route", {62}, None),
    ("summary", {63}, None),
    ("routewait", {64}, None),
    ("pack", {44, 0, 54}, None),
    ("gmm", {46, 15, 16, 19, 21, 27, 33, 34, 35, 36, 55, 56, 58, 59}, 0),
    ("dequant", {47, 17, 20, 37, 38, 55, 56, 58, 59}, 1),
    ("dispatch", {48, 53, 1, 2, 4}, None),
    ("combine", {49, 61, 10, 30}, None),
    ("activation", {50, 23, 57}, None),
    ("reorder", {40}, None),
    ("reduce", {51, 25, 32}, None),
    ("sync", {24, 26, 45, 52}, None),
]


def render(coarse, fine=None, label="", revision="unknown", table_us=None, limit=60000, context=None):
    if not 16000 <= limit <= 60000:
        raise ValueError("max-chars must be16000..60000")
    data = shared.split_ranks(coarse)
    declared = coarse.get("metadata", {}).get("rank_count")
    if declared is not None and declared != len(data):
        raise ValueError("coarse ranks missing")
    lines = [
        "URMA_REVIEW_V1 " + json.dumps(dict(label=label, revision=revision, table_us=table_us), ensure_ascii=True),
        "NOTE coarse/fine are separate invocations; ranks have independent clocks. No cross-file/rank timestamp alignment.",
        "NOTE PUT scopes are submissions, not wire bandwidth. Non-GMM portions may contain communication/Pack/Reduce.",
        "FIELDS stage=start/end/max_lane_union_us/count; times in us. Missing=-. No event-count assumptions.",
    ]
    if context:
        lines.append("CAPTURE_CONTEXT " + json.dumps(context, ensure_ascii=True, separators=(",", ":")))

    def f(x):
        return "-" if x is None else f"{x:.2f}"

    def select(rows, ids, engine):
        return [e for e in rows if e["id"] in ids and (engine is None or e["tid"] % 3 == engine)]

    def edge(es):
        return (min(e["ts"] for e in es), max(e["ts"] + e["dur"] for e in es)) if es else (None, None)

    def stats(es):
        if not es:
            return "-"
        lanes = {}
        for e in es:
            lanes.setdefault(e["tid"], []).append(e)
        a, b = edge(es)
        return f"{f(a)}/{f(b)}/{f(max(shared.base.intervals(v) for v in lanes.values()))}/{len(es)}"

    first = True
    for r, (meta, rows, _, status) in sorted(data.items()):
        if meta.get("dropped", 0):
            raise ValueError(f"rank{r}: dropped records")
        if meta.get("instrumentation") == "fine":
            raise ValueError("--coarse points at fine data")
        if first:
            lines.append(
                "MODEL "
                + json.dumps(
                    {
                        k: meta.get(k)
                        for k in ("bs", "world", "model", "cv_rows", "compute_schedule", "communication_protocol")
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
            )
            first = False
        lines.append(
            f"R{r} diag={f(meta.get('envelope_us'))} status={status} hidden={meta.get('hidden_wait_count', 0)} n={len(rows)}"
        )
        lines.append(" ".join(name + "=" + stats(select(rows, ids, engine)) for name, ids, engine in STAGES))
        g = edge(select(rows, STAGES[4][1], 0))
        red = edge(select(rows, {51, 25, 32}, None))
        start = min(e["ts"] for e in rows)
        lines.append(
            "observed_pre_gmm="
            + f(g[0] - start if g[0] is not None else None)
            + " reduce_end_minus_gmm_end="
            + f(red[1] - g[1] if red[1] is not None and g[1] is not None else None)
        )
    coarse_text = "\n".join(lines) + "\n"
    coarse_text = coarse_text.encode("ascii", "backslashreplace").decode("ascii")
    if len(coarse_text) > limit - 4096:
        raise ValueError("coarse summary exceeds reserved budget")
    if fine is not None:
        fine_data = shared.split_ranks(fine)
        if set(fine_data) != set(data):
            raise ValueError("coarse/fine rank sets differ")
        for r in data:
            cm, fm = data[r][0], fine_data[r][0]
            if fm.get("instrumentation") == "coarse":
                raise ValueError("--fine points at coarse data")
            for key in ("bs", "world", "model", "cv_rows"):
                if cm.get(key) is not None and fm.get(key) is not None and cm[key] != fm[key]:
                    raise ValueError(f"coarse/fine {key} mismatch on rank{r}")
        # Keep route preparation in the fastest/slowest stage comparison too.
        extra = [("RoutePrepare", {62}, None), ("RouteSummary", {63}, None), ("RouteWait", {64}, None)]
        original = pair.STAGES
        try:
            pair.STAGES = extra + original
            tail = pair.render(fine, source="fine/allrank.json", limit=limit - len(coarse_text) - 64)
        finally:
            pair.STAGES = original
        result = coarse_text + "\n[FINE_FAST_SLOW]\n" + tail
    else:
        result = coarse_text + "NO_FINE: fine diagnostics not supplied.\n"
    assert len(result.encode("utf-8")) == len(result) <= limit
    return result


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def export_run(root, output=None, expected_ep=None, expected_schedule=None, limit=60000):
    root = root.resolve()
    paths = sorted(root.glob("bs_*/coarse/allrank.json"), key=lambda p: int(p.parent.parent.name[3:]))
    if not paths:
        raise ValueError("No bs_*/coarse/allrank.json found; pass the EP output directory")
    manifests = sorted(root.glob("rank_*/manifest.json"))
    if not manifests and (root / "manifest.json").exists():
        manifests = [root / "manifest.json"]
    headers = []
    for path in manifests:
        m = read_json(path)
        headers.append(
            {k: m.get(k) for k in ("rank", "world", "compute_schedule", "native_urma_protocol", "routing", "cv_rows")}
        )
    schedules = {m["compute_schedule"] for m in headers if m["compute_schedule"]}
    if len(schedules) > 1:
        raise ValueError("Mixed native schedules in capture manifests")
    if expected_schedule and (not headers or any(m["compute_schedule"] != expected_schedule for m in headers)):
        raise ValueError(
            f"Capture schedule is {sorted(schedules) or 'unknown'}, expected {expected_schedule}; refusing to mix versions"
        )
    formal = read_json(root / "summary.json") if (root / "summary.json").exists() else []
    if not isinstance(formal, list):
        raise ValueError("Expected a list in run summary.json")
    destination = output or root / "comment_packets"
    packets = []
    # Validate and render the whole batch before writing any partial output.
    for path in paths:
        bs = int(path.parent.parent.name[3:])
        coarse = read_json(path)
        ranks = shared.split_ranks(coarse)
        ep = len(ranks)
        if expected_ep and ep != expected_ep:
            raise ValueError(f"BS{bs}: found {ep} ranks, expected EP{expected_ep}")
        for rank, (meta, _, _, _) in ranks.items():
            if meta.get("bs") is not None and meta["bs"] != bs:
                raise ValueError(f"BS directory/metadata mismatch: rank{rank}")
            if meta.get("world") is not None and meta["world"] != ep:
                raise ValueError(f"Missing ranks or world mismatch: rank{rank}")
        fine_path = path.parent.parent / "fine/allrank.json"
        matches = [x for x in formal if x.get("bs") == bs]
        if len(matches) > 1:
            raise ValueError(f"Duplicate BS{bs} in summary.json")
        profile = (matches[0].get("profile") or {}) if matches else {}
        context = dict(
            native_schedule=next(iter(schedules), "unknown"),
            manifests=headers,
            formal_profile={k: profile.get(k) for k in ("rank_max_mean_us", "rank_min_mean_us", "rounds")},
            formal_source="summary.json" if matches else "unavailable",
            revision="not inferred from current checkout",
        )
        text = render(
            coarse,
            read_json(fine_path) if fine_path.exists() else None,
            label=f"EP{ep}_BS{bs}",
            limit=limit,
            context=context,
        )
        packets.append((destination / f"ep{ep}_bs{bs}.txt", text))
    for path, text in packets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="ascii")
    return packets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path, nargs="?", help="EP output directory; export all BS into comment_packets/")
    p.add_argument("--coarse", type=Path)
    p.add_argument("--fine", type=Path)
    p.add_argument("--expected-ep", type=int, choices=(4, 8, 16))
    p.add_argument(
        "--expected-schedule", help="Require the native schedule recorded during capture, not the current checkout"
    )
    p.add_argument("--label", default="")
    p.add_argument("--revision", default="unknown", help="revision that PRODUCED the trace")
    p.add_argument("--table-us", type=float)
    p.add_argument("--max-chars", type=int, default=60000)
    p.add_argument("--output", type=Path, help="File in single mode; output directory in batch mode")
    a = p.parse_args()
    try:
        if a.root:
            if a.coarse or a.fine:
                raise ValueError("Use root OR --coarse/--fine")
            for path, text in export_run(a.root, a.output, a.expected_ep, a.expected_schedule, a.max_chars):
                print(f"{path}: {len(text)} chars / bytes; cap={a.max_chars}")
            return
        if not a.coarse or not a.output:
            raise ValueError("Single mode requires --coarse and --output")
        if a.table_us is not None and (not math.isfinite(a.table_us) or a.table_us <= 0):
            raise ValueError("invalid table-us")
        if any(a.output.resolve() == x.resolve() for x in (a.coarse, a.fine) if x):
            raise ValueError("output equals input")

        def read(x):
            return json.loads(x.read_text(encoding="utf-8-sig"))

        text = render(read(a.coarse), read(a.fine) if a.fine else None, a.label, a.revision, a.table_us, a.max_chars)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text, encoding="ascii")
    except (OSError, ValueError, KeyError, TypeError) as e:
        p.exit(2, f"Export failed: {e}\n")
    print(f"{a.output}: {len(text)} chars / bytes; cap={a.max_chars}")


if __name__ == "__main__":
    main()
