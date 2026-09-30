# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Tests run from source without an installed experimental extension."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/megamoe/python"))
