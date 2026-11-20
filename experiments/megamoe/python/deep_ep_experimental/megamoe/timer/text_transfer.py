# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Lossless, checksummed timer transfer through a size-limited text clipboard."""

import argparse
import base64
import binascii
import csv
import hashlib
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import re
import sys

from . import protocol
from .parse_timer_csv import core_info


FORMAT = "megamoe_trace_text_v1"
COLLECTION_FORMAT = "megamoe_trace_collection_v1"
COMPACT_FORMAT = "megamoe_trace_compact_v1"
BEGIN = "-----BEGIN MEGAMOE TRACE TEXT V1-----"
END = "-----END MEGAMOE TRACE TEXT V1-----"
DEFAULT_MAX_BYTES = 60000
CLIPBOARD_LIMIT = 65535
LINE_WIDTH = 76
MAX_BUNDLE_BYTES = 512 * 1024 * 1024
CSV_NAME = re.compile(r"rank_(\d+)_npu_time_parsed\.csv")
SHA256 = re.compile(r"[0-9a-f]{64}")
NOTES = [
    "Parsed CSV files are preserved byte for byte, including WAIT, raw integer cycles and task metadata.",
    "Only the listed ranks and the captured invocation are included; no events are sampled or filtered here.",
    "Device clocks are rank-local. Raw cycles from different ranks must not be subtracted as network latency.",
    "The trace is instrumented. Use a separate timer-off perf result for end-to-end performance.",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def parse_ranks(value):
    from .__main__ import parse_ranks as parse

    try:
        return sorted(parse(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def find_timer_directory(source):
    source = Path(source).resolve()
    if any(source.glob("rank_*_npu_time_parsed.csv")):
        return source
    candidates = [
        path
        for pattern in ("timer", "bs_*/timer")
        for path in source.glob(pattern)
        if any(path.glob("rank_*_npu_time_parsed.csv"))
    ]
    if len(candidates) != 1:
        raise ValueError("Specify one timer directory, BS directory, or run with exactly one parsed timer directory")
    return candidates[0]


def validate_csv(data, name):
    """Check the complete rectangular export, including idle cores and task counters."""
    stream = io.StringIO(data.decode("utf-8-sig"), newline="")
    reader = csv.reader(stream, strict=True)
    fields = next(reader, [])
    if fields[:4] != ["core_id", "group_id", "core_type", "sub_id"] or len(set(fields)) != len(fields):
        raise ValueError(f"{name}: invalid or duplicate parsed CSV columns")
    positions = {field: index for index, field in enumerate(fields)}
    starts = [field for field in fields if field.endswith("_start_cycle")]
    counts = [field for field in fields if field.endswith("_task_count")]
    if (
        not starts
        or not counts
        or any(field.removesuffix("_start_cycle") + "_end_cycle" not in positions for field in starts)
    ):
        raise ValueError(f"{name}: missing timer timestamps or task counters; export the full parsed CSV")
    cores = set()
    for row in reader:
        if len(row) != len(fields):
            raise ValueError(f"{name}: truncated CSV row {reader.line_num}")
        core = int(row[0])
        if not 0 <= core < protocol.N_CORE_COUNT or core in cores:
            raise ValueError(f"{name}: duplicate or invalid core {core}")
        if (int(row[1]), row[2], int(row[3])) != core_info(core):
            raise ValueError(f"{name}: inconsistent core identity at core {core}")
        cores.add(core)
        for field in starts:
            int(row[positions[field]])
            int(row[positions[field.removesuffix("_start_cycle") + "_end_cycle"]])
        for field in counts:
            count = int(row[positions[field]])
            if not 0 <= count <= protocol.MAX_DYNAMIC_ITER:
                raise ValueError(f"{name}: invalid task count at core {core}: {field}={count}")
            family = field.removesuffix("_task_count")
            if any(f"{family}_{slot}_start_cycle" not in positions for slot in range(count)):
                raise ValueError(f"{name}: missing recorded slots for core {core}: {field}={count}")
    if len(cores) != protocol.N_CORE_COUNT:
        raise ValueError(f"{name}: incomplete core coverage: {len(cores)}/{protocol.N_CORE_COUNT}")


def collect_bundle(source, ranks=None, attachments=(), label=None):
    timer = find_timer_directory(source)
    available = {
        int(match[1]): path
        for path in timer.glob("rank_*_npu_time_parsed.csv")
        if (match := CSV_NAME.fullmatch(path.name))
    }
    ranks = sorted(available if ranks is None else ranks)
    if (
        not ranks
        or len(set(ranks)) != len(ranks)
        or any(not 0 <= rank < 128 or rank not in available for rank in ranks)
    ):
        raise ValueError(f"Requested ranks are missing or duplicated; available parsed ranks: {sorted(available)}")
    files = []
    total_bytes = 0

    def add(path, name, *, timer_csv=False):
        nonlocal total_bytes
        total_bytes += path.stat().st_size
        if total_bytes > MAX_BUNDLE_BYTES:
            raise ValueError("Transfer input exceeds 512 MiB; export fewer ranks per bundle")
        data = path.read_bytes()
        if timer_csv:
            validate_csv(data, name)
        files.append(dict(name=name, bytes=len(data), sha256=digest(data), text=data.decode("utf-8")))

    for rank in ranks:
        add(available[rank], f"timer/{available[rank].name}", timer_csv=True)
    # Only known run context is collected. Do not sweep logs, environment files,
    # machine-local launch settings, or unrelated neighboring runs.
    if timer.name == "timer" and re.fullmatch(r"bs_\d+", timer.parent.name):
        run = timer.parent.parent
        for filename in ("case.json", "run.json"):
            if (run / filename).is_file():
                add(run / filename, f"context/{filename}")
        for rank in ranks:
            for relative in (
                Path(f"rank_{rank}/manifest.json"),
                Path(f"rank_{rank}/{timer.parent.name}/resources.json"),
            ):
                if (run / relative).is_file():
                    add(run / relative, f"context/{relative.as_posix()}")
    for index, attachment in enumerate(attachments, 1):
        path = Path(attachment)
        add(path, f"attachments/{index:03d}_{path.name}")
    return dict(
        format=FORMAT,
        label=label if label is not None else f"{timer.parent.name}/{timer.name}",
        ranks=ranks,
        cycles_per_us=protocol.CYCLE_TO_US,
        notes=NOTES,
        files=files,
    )


def render_part(bundle_hash, bundle_size, index, total, payload):
    encoded = base64.b64encode(payload).decode("ascii")
    lines = [
        BEGIN,
        f"bundle-sha256: {bundle_hash}",
        f"bundle-bytes: {bundle_size}",
        f"part: {index:06d}/{total:06d}",
        f"chunk-sha256: {digest(payload)}",
        "",
        *(encoded[offset : offset + LINE_WIDTH] for offset in range(0, len(encoded), LINE_WIDTH)),
        END,
        "",
    ]
    return "\n".join(lines)


def text_sizes(text):
    # The terminal/editor may change LF to CRLF. Both complete representations,
    # including envelope, line endings and the final newline, fit the limit.
    return len(text.encode("ascii")), len(text.replace("\n", "\r\n").encode("ascii"))


def csv_columns(fields):
    cycles = [i for i, field in enumerate(fields) if field.endswith("_cycle")]
    positions = {field: i for i, field in enumerate(fields)}
    durations = [
        (
            i,
            positions[field.removesuffix("_duration(us)") + "_start_cycle"],
            positions[field.removesuffix("_duration(us)") + "_end_cycle"],
        )
        for i, field in enumerate(fields)
        if field.endswith("_duration(us)")
    ]
    return cycles, durations


def duration_text(row, start, end):
    start, end = int(row[start]), int(row[end])
    return f"{(end - start) / protocol.CYCLE_TO_US if end > start else 0.0:.3f}"


def write_csv(rows, line_ending):
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator=line_ending).writerows(rows)
    return stream.getvalue()


def compact_csv(text):
    """Encode cycles as deltas and omit only exactly reconstructible durations.

    Noncanonical CSV dialects and numeric spellings stay as original text. The
    full byte comparison also protects quoting, line endings and future fields.
    """
    rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    if len(rows) != protocol.N_CORE_COUNT + 1 or any(len(row) != len(rows[0]) for row in rows):
        return None
    line_ending = "\r\n" if text.endswith("\r\n") else "\n"
    if write_csv(rows, line_ending) != text:
        return None
    try:
        cycles, durations = csv_columns(rows[0])
    except KeyError:
        return None
    if not cycles or any(str(int(row[i])) != row[i] for row in rows[1:] for i in cycles):
        return None
    # Empty duration cells are reserved for the derived value in this codec.
    if any(not row[i] for row in rows[1:] for i, _, _ in durations):
        return None
    base = min((int(row[i]) for row in rows[1:] for i in cycles if int(row[i])), default=0)
    for row in rows[1:]:
        for i, start, end in durations:
            if row[i] == duration_text(row, start, end):
                row[i] = ""
        previous = base
        for i in cycles:
            value = int(row[i])
            row[i] = str(value - previous) if value else ""
            if value:
                previous = value
    packed = dict(base=base, line_ending=line_ending, text=write_csv(rows, line_ending))
    return packed if expand_csv(packed) == text else None


def expand_csv(packed):
    if packed["line_ending"] not in ("\n", "\r\n") or type(packed["base"]) is not int:
        raise ValueError("Invalid compact CSV layout")
    rows = list(csv.reader(io.StringIO(packed["text"], newline=""), strict=True))
    if len(rows) != protocol.N_CORE_COUNT + 1 or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Invalid compact CSV dimensions")
    cycles, durations = csv_columns(rows[0])
    for row in rows[1:]:
        previous = packed["base"]
        for i in cycles:
            if row[i]:
                previous += int(row[i])
                row[i] = str(previous)
            else:
                row[i] = "0"
        for i, start, end in durations:
            if not row[i]:
                row[i] = duration_text(row, start, end)
    return write_csv(rows, packed["line_ending"])


def map_bundle_files(bundle, transform):
    if bundle.get("format") == COLLECTION_FORMAT:
        return dict(
            bundle,
            captures=[
                dict(capture, bundle=map_bundle_files(capture["bundle"], transform)) for capture in bundle["captures"]
            ],
        )
    return dict(bundle, files=[transform(record) for record in bundle["files"]])


def compact_bundle(bundle):
    def pack(record):
        if record["name"].startswith("timer/") and CSV_NAME.fullmatch(record["name"][6:]):
            packed = compact_csv(record["text"])
            if packed is not None:
                return {**{key: value for key, value in record.items() if key != "text"}, "csv_delta": packed}
        return record

    return dict(format=COMPACT_FORMAT, bundle=map_bundle_files(bundle, pack))


def expand_bundle(bundle):
    remaining = MAX_BUNDLE_BYTES

    def unpack(record):
        nonlocal remaining
        if type(record["bytes"]) is not int or not 0 <= record["bytes"] <= remaining:
            raise ValueError("Invalid or oversized restored file")
        remaining -= record["bytes"]
        if "csv_delta" in record:
            if "text" in record:
                raise ValueError("Ambiguous compact CSV encoding")
            text = expand_csv(record["csv_delta"])
            data = text.encode("utf-8")
            if len(data) != record["bytes"] or digest(data) != record["sha256"]:
                raise ValueError(f"File size/SHA-256 mismatch: {record['name']}")
            return {**{key: value for key, value in record.items() if key != "csv_delta"}, "text": text}
        return record

    return map_bundle_files(bundle, unpack)


def make_parts(bundle, max_bytes=DEFAULT_MAX_BYTES):
    if not 1024 <= max_bytes <= CLIPBOARD_LIMIT:
        raise ValueError("--max-bytes must be between 1024 and 65535, including headers and CRLF")
    raw = json_bytes(bundle)
    if len(raw) > MAX_BUNDLE_BYTES:
        raise ValueError("Unpacked transfer exceeds 512 MiB; export fewer ranks per bundle")
    compressed = lzma.compress(raw, preset=6 | lzma.PRESET_EXTREME)
    compact = json_bytes(compact_bundle(bundle)) if "files" in bundle or "captures" in bundle else raw
    if compact != raw and len(compact) <= MAX_BUNDLE_BYTES:
        candidate = lzma.compress(compact, preset=6 | lzma.PRESET_EXTREME)
        if len(candidate) < len(compressed):
            compressed = candidate
    if len(compressed) > MAX_BUNDLE_BYTES:
        raise ValueError("Compressed transfer exceeds 512 MiB; export fewer ranks per bundle")
    bundle_hash = digest(compressed)
    # Find the largest raw chunk whose complete envelope fits, even with CRLF.
    # Fixed-width part numbers make this bound independent of the part count.
    low, high = 1, max_bytes
    while low < high:
        middle = (low + high + 1) // 2
        trial = render_part(bundle_hash, len(compressed), 999999, 999999, bytes(middle))
        if text_sizes(trial)[1] <= max_bytes:
            low = middle
        else:
            high = middle - 1
    chunk_bytes = low
    total = (len(compressed) + chunk_bytes - 1) // chunk_bytes
    if total > 999999:
        raise ValueError("Too many transfer parts")
    parts = [
        render_part(bundle_hash, len(compressed), index + 1, total, compressed[offset : offset + chunk_bytes])
        for index, offset in enumerate(range(0, len(compressed), chunk_bytes))
    ]
    if any(text_sizes(part)[1] > max_bytes for part in parts):
        raise ValueError("Internal error: encoded part exceeds the clipboard limit")
    return bundle_hash, len(compressed), parts


def export_text(source, output=None, *, ranks=None, attachments=(), label=None, max_bytes=DEFAULT_MAX_BYTES):
    bundle = collect_bundle(source, ranks, attachments, label)
    return export_bundle(bundle, output, find_timer_directory(source), max_bytes)


def bundle_files(bundle):
    if bundle["format"] == FORMAT:
        return bundle["files"]
    return [
        dict(record, name=f"captures/{capture['name']}/{record['name']}")
        for capture in bundle["captures"]
        for record in capture["bundle"]["files"]
    ]


def export_bundle(bundle, output, parent, max_bytes):
    bundle_hash, compressed_bytes, parts = make_parts(bundle, max_bytes)
    output = Path(output) if output is not None else Path(parent) / "text-transfer" / bundle_hash[:16]
    index = dict(
        format=FORMAT,
        label=bundle["label"],
        ranks=bundle["ranks"],
        bundle_sha256=bundle_hash,
        bundle_bytes=compressed_bytes,
        max_bytes=max_bytes,
        files=[{key: value for key, value in record.items() if key != "text"} for record in bundle_files(bundle)],
        parts=[],
    )
    output.mkdir(parents=True, exist_ok=False)
    for number, text in enumerate(parts, 1):
        filename = f"part-{number:06d}.txt"
        data = text.encode("ascii")
        (output / filename).write_bytes(data)
        lf_bytes, crlf_bytes = text_sizes(text)
        index["parts"].append(dict(file=filename, sha256=digest(data), lf_bytes=lf_bytes, crlf_bytes=crlf_bytes))
    # Written last: an interrupted export cannot be used by show as a complete bundle.
    (output / "manifest.json").write_bytes(json_bytes(index))
    return output, index


def show_part(directory, number):
    directory = Path(directory)
    index = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if index.get("format") != FORMAT or not 1 <= number <= len(index["parts"]):
        raise ValueError("Invalid transfer format or part number")
    record = index["parts"][number - 1]
    # Derive the filename rather than trusting a path from a manifest.
    data = (directory / f"part-{number:06d}.txt").read_bytes()
    if digest(data) != record["sha256"]:
        raise ValueError(f"Part {number} changed since export; export again")
    text = data.decode("ascii")
    if text_sizes(text)[1] > min(index["max_bytes"], CLIPBOARD_LIMIT):
        raise ValueError("Part exceeds the clipboard limit")
    return text


def parse_parts(texts):
    """Accept reordered/repeated blocks, CRLF, and wrapping from a text clipboard."""
    identity = None
    chunks = {}
    for text in texts:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        blocks = re.findall(re.escape(BEGIN) + r"\n(.*?)\n" + re.escape(END), text, flags=re.S)
        if not blocks or len(blocks) != text.count(BEGIN) or len(blocks) != text.count(END):
            raise ValueError("Missing or truncated BEGIN/END envelope; copy one complete text block")
        for block in blocks:
            header, separator, body = block.partition("\n\n")
            if not separator:
                raise ValueError("Missing blank line before Base64 payload")
            fields = {}
            for line in header.splitlines():
                key, colon, value = line.partition(": ")
                if not colon or key in fields:
                    raise ValueError("Invalid or duplicate text block header")
                fields[key] = value
            if set(fields) != {"bundle-sha256", "bundle-bytes", "part", "chunk-sha256"}:
                raise ValueError("Invalid text block header fields")
            if not SHA256.fullmatch(fields["bundle-sha256"]) or not SHA256.fullmatch(fields["chunk-sha256"]):
                raise ValueError("Invalid SHA-256 header")
            match = re.fullmatch(r"(\d{6})/(\d{6})", fields["part"])
            if not match or not 1 <= int(match[1]) <= int(match[2]):
                raise ValueError("Invalid part number")
            number, total = map(int, match.groups())
            size = int(fields["bundle-bytes"])
            if not 0 < size <= MAX_BUNDLE_BYTES:
                raise ValueError("Invalid compressed bundle size")
            current = fields["bundle-sha256"], size, total
            if identity is not None and identity != current:
                raise ValueError("Mixed bundles: keep different exports/hosts in separate restore commands")
            identity = current
            # Whitespace changes are harmless. Other edits must fail, not be guessed.
            encoded = re.sub(r"[ \t\n]+", "", body)
            try:
                payload = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as error:
                raise ValueError(f"Part {number}: invalid Base64; copy it again") from error
            if not payload or digest(payload) != fields["chunk-sha256"]:
                raise ValueError(f"Part {number}: SHA-256 mismatch; copy it again")
            if number in chunks and chunks[number] != payload:
                raise ValueError(f"Part {number}: conflicting duplicate")
            chunks[number] = payload
    if identity is None:
        raise ValueError("No transfer text was supplied")
    bundle_hash, size, total = identity
    missing = sorted(set(range(1, total + 1)) - chunks.keys())
    status = dict(
        bundle_sha256=bundle_hash, bundle_bytes=size, total_parts=total, received_parts=len(chunks), missing=missing
    )
    if not missing:
        compressed = b"".join(chunks[index] for index in range(1, total + 1))
        if len(compressed) != size or digest(compressed) != bundle_hash:
            raise ValueError("Complete bundle size/SHA-256 mismatch")
    else:
        compressed = None
    return status, compressed


def unpack_bundle(compressed):
    decoder = lzma.LZMADecompressor(memlimit=256 * 1024 * 1024)
    raw = decoder.decompress(compressed, max_length=MAX_BUNDLE_BYTES + 1)
    if len(raw) > MAX_BUNDLE_BYTES or not decoder.eof or decoder.unused_data:
        raise ValueError("Invalid or oversized compressed bundle")
    bundle = json.loads(raw)
    if bundle.get("format") == COMPACT_FORMAT:
        bundle = expand_bundle(bundle["bundle"])
    return validate_bundle(bundle)


def validate_bundle(bundle):
    if bundle.get("format") == COLLECTION_FORMAT:
        captures = bundle.get("captures", [])
        if not 1 <= len(captures) <= 32 or bundle.get("cycles_per_us") != protocol.CYCLE_TO_US:
            raise ValueError("Invalid trace collection")
        names, ranks, files = set(), set(), {}
        for capture in captures:
            name = capture["name"]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name) or name in names:
                raise ValueError("Invalid or duplicate capture path")
            names.add(name)
            if capture["bundle"].get("format") != FORMAT:
                raise ValueError("Nested collections are not supported")
            child, child_files = validate_bundle(capture["bundle"])
            ranks.update(child["ranks"])
            files.update({f"captures/{name}/{path}": data for path, data in child_files.items()})
        if bundle.get("ranks") != sorted(ranks):
            raise ValueError("Invalid collection rank coverage")
        return bundle, files
    if bundle.get("format") != FORMAT or bundle.get("cycles_per_us") != protocol.CYCLE_TO_US:
        raise ValueError("Unsupported transfer format or timer clock conversion")
    files = {}
    for record in bundle["files"]:
        name = record["name"]
        path = PurePosixPath(name)
        # Restore plain UTF-8 data only, under the selected output directory.
        if (
            "\\" in name
            or ":" in name
            or path.is_absolute()
            or str(path) != name
            or any(part in ("", ".", "..") for part in path.parts)
            or len(path.parts) < 2
            or path.parts[0] not in ("timer", "context", "attachments")
            or name in files
        ):
            raise ValueError(f"Invalid or duplicate bundle path: {name}")
        data = record["text"].encode("utf-8")
        if len(data) != record["bytes"] or digest(data) != record["sha256"]:
            raise ValueError(f"File size/SHA-256 mismatch: {name}")
        files[name] = data
    ranks = bundle["ranks"]
    if not ranks or any(type(rank) is not int or not 0 <= rank < 128 for rank in ranks) or sorted(set(ranks)) != ranks:
        raise ValueError("Invalid bundle rank coverage")
    expected = {f"timer/rank_{rank}_npu_time_parsed.csv" for rank in ranks}
    if {name for name in files if name.startswith("timer/")} != expected:
        raise ValueError("Timer files differ from the declared rank coverage")
    for name in expected:
        validate_csv(files[name], name)
    return bundle, files


def restore_text(texts, output, *, render=True):
    status, compressed = parse_parts(texts)
    if status["missing"]:
        raise ValueError(f"Missing parts: {status['missing']}; no output was written")
    bundle, files = unpack_bundle(compressed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (output / "transfer_manifest.json").write_bytes(
        json_bytes(
            dict(
                **status,
                format=bundle["format"],
                label=bundle["label"],
                ranks=bundle["ranks"],
                notes=bundle["notes"],
                files=[
                    {key: value for key, value in record.items() if key != "text"} for record in bundle_files(bundle)
                ],
            )
        )
    )
    if render:
        from .__main__ import main as render_timer

        captures = (
            [(output / "captures" / capture["name"], capture["bundle"]) for capture in bundle["captures"]]
            if bundle["format"] == COLLECTION_FORMAT
            else [(output, bundle)]
        )
        for directory, capture in captures:
            render_timer(
                [str(directory / "timer"), "--ranks", ",".join(map(str, capture["ranks"])), "--include-wait", "--merge"]
            )
    return bundle


def read_text_inputs(paths):
    if "-" in paths:
        if len(paths) != 1:
            raise ValueError("Use '-' alone to read copied text from stdin")
        return [sys.stdin.read()]
    return [Path(path).read_text(encoding="utf-8-sig") for path in paths]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Package full parsed CSVs into clipboard-sized text blocks")
    export.add_argument("source", type=Path, help="Timer directory, BS directory, or single-BS run directory")
    export.add_argument("--output-dir", type=Path, help="New directory (default: timer/text-transfer/<bundle id>)")
    export.add_argument("--ranks", type=parse_ranks, help="Default: every parsed rank found in this timer directory")
    export.add_argument(
        "--attach", action="append", type=Path, default=[], help="Include an explicit UTF-8 file, e.g. perf JSON"
    )
    export.add_argument("--label", help="Description identifying this capture/host")
    export.add_argument(
        "--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help="Whole block limit including CRLF; 1024..65535"
    )
    export.add_argument("--print-parts", action="store_true", help="Print every complete block after export")
    show = commands.add_parser("show", help="Print exactly one block, with no other stdout")
    show.add_argument("directory", type=Path, help="Directory returned by export")
    show.add_argument("--part", type=int, required=True, help="One-based part number")
    inspect = commands.add_parser("inspect", help="Check copied blocks and list missing parts")
    inspect.add_argument("texts", nargs="+", help="Copied UTF-8 text files, or '-' for stdin")
    restore = commands.add_parser("restore", help="Verify, restore original CSVs and render traces with WAIT")
    restore.add_argument("texts", nargs="+", help="Copied UTF-8 text files, or '-' for stdin")
    restore.add_argument("--output-dir", type=Path, required=True, help="New restoration directory")
    restore.add_argument("--no-render", action="store_true", help="Restore and verify files without rendering JSON")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            output, index = export_text(
                args.source,
                args.output_dir,
                ranks=args.ranks,
                attachments=args.attach,
                label=args.label,
                max_bytes=args.max_bytes,
            )
            print(
                f"[TEXT] label={index['label']} ranks={index['ranks']} compressed_bytes={index['bundle_bytes']} parts={len(index['parts'])}",
                flush=True,
            )
            print(f"[TEXT] output={output}", flush=True)
            print(
                f"[TEXT] Copy one complete BEGIN-through-END block at a time; each <= {args.max_bytes} bytes including CRLF.",
                flush=True,
            )
            if args.print_parts:
                for number in range(1, len(index["parts"]) + 1):
                    print(f"[TEXT PART {number}/{len(index['parts'])}] {index['label']}", flush=True)
                    print(show_part(output, number), end="", flush=True)
            else:
                print("[TEXT] Use 'show <output> --part 1', then 2, 3, ...")
        elif args.command == "show":
            sys.stdout.write(show_part(args.directory, args.part))
        elif args.command == "inspect":
            status, _ = parse_parts(read_text_inputs(args.texts))
            print(json.dumps(status, indent=2))
        else:
            bundle = restore_text(read_text_inputs(args.texts), args.output_dir, render=not args.no_render)
            print(f"[TEXT] Verified and restored ranks={bundle['ranks']} to {args.output_dir}")
    except (ValueError, OSError, KeyError, TypeError, csv.Error, lzma.LZMAError) as error:
        parser.exit(1, f"[TEXT ERROR] {error}\n")


if __name__ == "__main__":
    main()
