#!/usr/bin/env python3
# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026
"""Merge an existing run's URMA rank traces; no Torch, NPU, or installation needed."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from deep_ep_experimental.megamoe.urma_trace_merge import main  # noqa: E402

if __name__ == "__main__":
    main()
