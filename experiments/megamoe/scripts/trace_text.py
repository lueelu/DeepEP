#!/usr/bin/env python3
# Copyright (c) 2026, Lu Lu

"""Run the checkout's offline clipboard transfer tool without rebuilding/installing."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "experiments/megamoe/python"), str(ROOT)]

from deep_ep_experimental.megamoe.timer.text_transfer import main  # noqa: E402


if __name__ == "__main__":
    main()
