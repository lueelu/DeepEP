# Copyright (c) 2026, Lu Lu
# Modified by SimpleBrightMan 2026

"""Normalize project device headers for the runtime DeepJIT compile.

DeepJIT only parses angle-bracket includes whose name starts with a configured
prefix (``kernels/``); any quoted include in a hashed header is rejected. Two
views are produced because shared headers name ``epilogue/...`` and
``catlass_gmm/...`` without a backend prefix and must resolve through the
backend directory that comes first on that backend's device include path.

Resolution order for a quoted include: the including file's directory, the
backend directory, the ``kernels`` root, then the ``csrc`` root. Unresolved
names are external headers (CATLASS, SHMEM, CANN) and keep their spelling in
angle brackets; they are resolved through the installed include roots.
"""

import argparse
import re
from pathlib import Path
import shutil
import sys

# view name -> backend directory below kernels/
BACKENDS = (("mte", "mega_moe"), ("urma", "mega_moe_urma"))
INCLUDE = re.compile(r'(?P<lead>\s*#\s*include\s+)"(?P<name>[^"\n]+)"(?P<tail>\s*(?://[^\n]*)?)')

# External headers are not shipped with unit tests, so names that resolve from
# an installed include root are recognized by their top-level directory.
EXTERNAL_PREFIXES = (
    "kernel_operator.h",
    "kernel_operator_list_tensor_intf.h",
    "kernel_tiling/",
    "lib/",
    "op_common/",
    "adv_api/",
    "basic_api/",
    "c_api/",
    "tensor_api/",
    "version/",
    "catlass/",
    "tla/",
    "shmem.h",
)


def normalize_file(text, source, csrc, backend, external_roots):
    lines = text.splitlines(keepends=True)
    normalized = []
    for line in lines:
        match = INCLUDE.match(line)
        if match is None:
            normalized.append(line)
            continue
        name = match.group("name")
        target = resolve(name, source, csrc, backend, external_roots)
        if target is None:
            if name.startswith(EXTERNAL_PREFIXES):
                # CANN headers are resolved from the active installation.
                target = name
            else:
                raise RuntimeError(f"unresolved device include in {source}: {name}")
        normalized.append(f'{match.group("lead")}<{target}>{match.group("tail")}')
    return "".join(normalized)


def resolve(name, source, csrc, backend, external_roots):
    if name.startswith("kernels/"):
        candidate = csrc / name
        return name if candidate.is_file() else None
    for base in (source.parent, csrc / "kernels" / backend, csrc / "kernels", csrc):
        candidate = base / name
        if candidate.is_file():
            resolved = candidate.resolve()
            if not resolved.is_relative_to(csrc):
                return name
            # Candidates below csrc always live under the kernels/ include root.
            return resolved.relative_to(csrc).as_posix()
    for root in external_roots:
        if (root / name).is_file():
            return name
    return None


def prepare(csrc, output, catlass, shmem):
    csrc = csrc.resolve()
    kernels = csrc / "kernels"
    if not kernels.is_dir():
        raise RuntimeError(f"missing device headers: {kernels}")
    external_roots = []
    if catlass is not None:
        catlass = catlass.resolve()
        external_roots.extend([catlass / "include", catlass / "examples/common"])
        for root in external_roots:
            if not root.is_dir():
                raise RuntimeError(f"missing CATLASS include root: {root}")
    if shmem is not None:
        shmem = shmem.resolve()
        for sub in ("include", "src"):
            if not (shmem / sub).is_dir():
                raise RuntimeError(f"missing SHMEM include root: {shmem / sub}")
            external_roots.append(shmem / sub)
    headers = sorted(
        path for path in kernels.rglob("*") if path.is_file() and path.suffix in {".h", ".hpp"}
    )
    if not headers:
        raise RuntimeError(f"no device headers under {kernels}")
    output = output.resolve()
    for view, backend in BACKENDS:
        view_root = output / view / "kernels"
        if view_root.parent.exists() and not view_root.parent.is_dir():
            raise RuntimeError(f"refusing to replace non-directory staging: {view_root.parent}")
        if view_root.exists():
            shutil.rmtree(view_root)
        for path in headers:
            relative = path.relative_to(kernels)
            target = view_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            text = path.read_text(encoding="utf-8")
            target.write_text(normalize_file(text, path, csrc, backend, external_roots), encoding="utf-8", newline="\n")
        verify(view_root)
    print(f"Prepared device header views for {len(headers)} headers under {output}")


def verify(view_root):
    """Reject a view that still quotes includes or names a missing project header."""
    for path in view_root.rglob("*"):
        if not path.is_file() or path.suffix not in {".h", ".hpp"}:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if INCLUDE.match(line):
                raise RuntimeError(f"quoted include survived normalization in {path}: {line.strip()}")
            name = re.match(r'\s*#\s*include\s+<(?P<name>[^>\n]+)>', line)
            if name and name.group("name").startswith("kernels/"):
                # Normalized names are relative to the view root's parent.
                if not (view_root.parent / name.group("name")).is_file():
                    raise RuntimeError(f"missing normalized include in {path}: {name.group('name')}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csrc", required=True, type=Path, help="project csrc directory")
    parser.add_argument("--catlass", type=Path, help="CATLASS checkout with include and examples/common")
    parser.add_argument("--shmem", type=Path, help="installed SHMEM SDK with include and src")
    parser.add_argument("--output", required=True, type=Path, help="destination for the mte/ and urma/ views")
    args = parser.parse_args(argv)
    prepare(args.csrc, args.output, args.catlass, args.shmem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
