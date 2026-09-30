# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Record or verify a prebuilt SDK; never download/build dependencies.

Header checks establish the expected public API, not binary provenance. The SDK
provider must attest that the library was built from the pinned commit.
"""

import argparse
import hashlib
import json
from pathlib import Path


def record(sdk, revision, *, build_info=None):
    sdk = Path(sdk).resolve()
    lock = json.loads((Path(__file__).resolve().parents[1] / "dependencies.lock.json").read_text())
    expected = lock["shmem"]
    if revision != expected["revision"]:
        raise ValueError("SDK source revision must match dependencies.lock.json")
    for relative, digest in expected["header_sha256_lf"].items():
        header = sdk / "include" / relative
        normalized = header.read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(normalized).hexdigest() != digest:
            raise ValueError(f"SDK public header mismatch: {relative}")
    files = sorted((sdk / "include").rglob("*.h")) + [sdk / "lib/libshmem.so"]
    hashes = {}
    for path in files:
        if not path.resolve().is_relative_to(sdk) or not path.is_file():
            raise ValueError("SDK files must stay inside the selected root")
        hashes[path.relative_to(sdk).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = sdk / "deepep-sdk.json"
    if manifest.exists():
        raise FileExistsError("SDK manifest already exists; do not overwrite an existing SDK identity")
    identity = {"revision": revision, "sha256": hashes}
    if build_info is not None:
        identity["build"] = build_info
    manifest.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    return manifest


def check(sdk):
    """Validate an existing SDK identity without changing its files."""
    sdk = Path(sdk).resolve()
    lock = json.loads((Path(__file__).resolve().parents[1] / "dependencies.lock.json").read_text())["shmem"]
    manifest = sdk / "deepep-sdk.json"
    recorded = json.loads(manifest.read_text())
    if recorded.get("revision") != lock["revision"]:
        raise ValueError("SDK source revision must match dependencies.lock.json")
    for relative, expected in lock["header_sha256_lf"].items():
        data = (sdk / "include" / relative).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"SDK public header mismatch: {relative}")
    hashes = recorded.get("sha256", {})
    required = {"lib/libshmem.so", *("include/" + name for name in lock["header_sha256_lf"])}
    if not required.issubset(hashes):
        raise ValueError("SDK manifest must cover the locked headers and libshmem.so")
    for relative, expected in hashes.items():
        path = (sdk / relative).resolve()
        if not path.is_relative_to(sdk) or not path.is_file():
            raise ValueError(f"Invalid SDK manifest path: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"SDK checksum mismatch: {relative}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--revision", help="SDK provider's source revision attestation")
    mode.add_argument("--check", action="store_true", help="verify an existing SDK identity without modifying it")
    args = parser.parse_args()
    print(check(args.sdk) if args.check else record(args.sdk, args.revision))
