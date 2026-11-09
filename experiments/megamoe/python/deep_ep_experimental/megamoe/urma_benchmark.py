"""Compatibility module; all CLI options come from the main benchmark."""

from pathlib import Path
import json


def export_trace_collective(group, raw, case, *, bs, clock_mhz, include_waits=False, model=None):
    """All ranks exchange exporter errors before any rank raises on overflow."""
    from .urma_trace import export_trace

    case = Path(case)
    status = dict(
        rank=group.rank(),
        error=None,
        metadata=None,
        raw_path=str(case / "urma_trace.bin"),
        chrome_path=str(case / "urma_trace.json"),
    )
    try:
        raw.astype("<i8", copy=False).tofile(case / "urma_trace.bin")
        status["metadata"] = export_trace(
            raw, case / "urma_trace.json", rank=group.rank(), clock_mhz=clock_mhz, include_waits=include_waits
        )
        if model is not None:
            trace_path = case / "urma_trace.json"
            trace_doc = json.loads(trace_path.read_text(encoding="utf-8"))
            trace_doc["metadata"].update(bs=bs, world=group.size(), model=dict(model))
            trace_path.write_text(json.dumps(trace_doc, separators=(",", ":")), encoding="utf-8")
            status["metadata"] = trace_doc["metadata"]
    except Exception as error:
        status["error"] = f"{type(error).__name__}: {error}"
    try:
        (case / "trace_metadata.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    except OSError as error:
        status["error"] = (status["error"] or "") + f"; failed to save trace metadata: {error}"
    records = group.all_gather(status, f"URMA trace export BS={bs}")
    failures = [f"rank={record['rank']}: {record['error']}" for record in records if record["error"]]
    if failures:
        raise RuntimeError("URMA detailed trace failed: " + "; ".join(failures))
    return records


if __name__ == "__main__":
    from .benchmark import main

    main()
