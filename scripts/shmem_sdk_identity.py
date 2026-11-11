"""Build identity admission for locked SHMEM SDKs with explicit capability and source identity."""

from pathlib import Path
import json
import re
import shlex


def compiled_hns_1825_pi_on_chip(build_root):
    """Read the SDK device target's actual flags, never infer its firmware mode.

    Upstream's auto-selected value is a normal CMake variable, not necessarily a
    cache entry. Inspect the generated Make/Ninja compile rules after a successful
    build; a missing, conflicting, or unsupported definition cannot be attested.
    """
    build_root = Path(build_root)
    commands = []
    for path in build_root.glob("**/CMakeFiles/aclshmem_device.dir/flags.make"):
        fields = re.findall(r"^\w+_(?:DEFINES|FLAGS)\s*=\s*(.*)$", path.read_text(), re.MULTILINE)
        commands.append(" ".join(fields))
    graphs = [build_root / "build.ninja", *build_root.glob("CMakeFiles/impl-*.ninja")]
    for path in graphs:
        if not path.is_file():
            continue
        for block in re.split(r"\n(?=build )", path.read_text()):
            header, _, body = block.partition("\n")
            if not re.search(r"^build .*CMakeFiles/aclshmem_device\.dir/.*: \w+_COMPILER", header):
                continue
            fields = re.findall(r"^\s+(?:DEFINES|FLAGS)\s*=\s*(.*)$", body, re.MULTILINE)
            commands.append(" ".join(fields))
    modes = set()
    for command in commands:
        tokens = shlex.split(command)
        prefix = "-DACLSHMEMI_HNS_1825_PI_ON_CHIP"
        values = [token[len(prefix):] for token in tokens if token.startswith(prefix)]
        if len(values) != 1 or values[0] not in {"=0", "=1"}:
            raise RuntimeError("Cannot attest SHMEM HNS1825 doorbell mode from device compile flags")
        modes.add(int(values[0][1:]))
    if len(modes) != 1:
        raise RuntimeError("Missing or inconsistent SHMEM HNS1825 doorbell mode in device build graph")
    return modes.pop()


def validate_build_identity(build, require_rdma=False):
    required = {"cann_version", "cann_timestamp", "target", "source_patches"}
    if (not isinstance(build, dict) or not required.issubset(build)
            or set(build) - required - {"rdma_backend", "hns_1825_pi_on_chip"}
            or not isinstance(build.get("source_patches"), list)
            or build.get("target") != "Ascend950"
            or not all(isinstance(build.get(k), str) and build[k] for k in required - {"source_patches"})
            or ("rdma_backend" in build and build["rdma_backend"] != "HNS_1825")
            or (require_rdma and build.get("rdma_backend") != "HNS_1825")
            or (build.get("rdma_backend") == "HNS_1825" and
                (type(build.get("hns_1825_pi_on_chip")) is not int
                 or build["hns_1825_pi_on_chip"] not in (0, 1)))
            or ("hns_1825_pi_on_chip" in build and build.get("rdma_backend") != "HNS_1825")):
        raise RuntimeError("Require a current native SHMEM SDK build identity; stop SDK users and rebuild explicitly with --rebuild")


def recorded_hns_1825_pi_on_chip(sdk_root):
    identity = json.loads((Path(sdk_root) / "deepep-sdk.json").read_text(encoding="utf-8"))
    validate_build_identity(identity.get("build"), require_rdma=True)
    return identity["build"]["hns_1825_pi_on_chip"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Print the selected SDK's attested HNS1825 doorbell mode.")
    parser.add_argument("sdk", type=Path)
    print(recorded_hns_1825_pi_on_chip(parser.parse_args().sdk))
