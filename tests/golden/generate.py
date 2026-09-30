# Copyright (c) 2026, Lu Lu
# Modified by nino888 2026

"""Regenerate the checked-in deterministic golden manifest."""

import json
from pathlib import Path

from tests.utils.golden import build_manifest


def main():
    target = Path(__file__).with_name("deepep_reference_v2.json")
    target.write_text(
        json.dumps(build_manifest(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(target)


if __name__ == "__main__":
    main()
