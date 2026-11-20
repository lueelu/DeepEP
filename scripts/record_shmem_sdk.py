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
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_support import shmem_sdk_inputs, validate_shmem_headers, validate_shmem_sdk


def record(sdk, revision, *, build_info=None):
    sdk = Path(sdk).resolve()
    lock = json.loads((Path(__file__).resolve().parents[1] / "dependencies.lock.json").read_text())
    expected = lock["shmem"]
    if revision != expected["revision"]:
        raise ValueError("SDK source revision must match dependencies.lock.json")
    metadata_mode = validate_shmem_headers(sdk, expected)
    if build_info is not None and build_info.get("metadata_mode", metadata_mode) != metadata_mode:
        raise ValueError("Installed SHMEM metadata mode differs from requested build")
    files = shmem_sdk_inputs(sdk)
    hashes = {}
    for path in files:
        if not path.resolve().is_relative_to(sdk) or not path.is_file():
            raise ValueError("SDK files must stay inside the selected root")
        hashes[path.relative_to(sdk).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = sdk / "deepep-sdk.json"
    if manifest.exists():
        raise FileExistsError("SDK manifest already exists; do not overwrite an existing SDK identity")
    identity = {"revision": revision, "metadata_mode": metadata_mode, "sha256": hashes}
    if build_info is not None:
        identity["build"] = build_info
    manifest.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    return manifest


def check(sdk):
    """Validate an existing SDK identity without changing its files."""
    sdk = Path(sdk).resolve()
    lock = json.loads((Path(__file__).resolve().parents[1] / "dependencies.lock.json").read_text())["shmem"]
    manifest = sdk / "deepep-sdk.json"
    recorded = validate_shmem_sdk(sdk, lock)
    required = {"lib/libshmem.so", *("include/" + name for name in lock["header_sha256_lf"])}
    if not required.issubset(recorded["sha256"]):
        raise ValueError("SDK manifest must cover the locked headers and libshmem.so")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--revision", help="SDK provider's source revision attestation")
    mode.add_argument("--check", action="store_true", help="verify an existing SDK identity without modifying it")
    args = parser.parse_args()
    print(check(args.sdk) if args.check else record(args.sdk, args.revision))
