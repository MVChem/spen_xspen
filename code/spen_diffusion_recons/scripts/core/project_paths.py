"""Code, input and output locations for the migrated SPEN project.

Importing this module never creates directories. External inputs are opt-in.
"""
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CORE = PACKAGE_ROOT / "scripts/core"
PRIOR96 = PACKAGE_ROOT / "scripts/prior96"
PRIOR192 = PACKAGE_ROOT / "scripts/prior192"
RUNS = PACKAGE_ROOT / "runs"
DATA_ROOT = Path(os.environ.get("SPEN_DATA_ROOT", PACKAGE_ROOT / "data")).expanduser().resolve()
REFERENCE_ROOT = Path(os.environ.get("SPEN_REFERENCE_ROOT", DATA_ROOT / "legacy_spen_recons")).expanduser().resolve()
PRIOR96_DATA = Path(os.environ.get("SPEN_PRIOR96_DATA", DATA_ROOT / "prior96")).expanduser().resolve()
PRIOR192_DATA = Path(os.environ.get("SPEN_PRIOR192_DATA", DATA_ROOT / "prior192")).expanduser().resolve()
MOUSE_RAW = Path(os.environ.get("SPEN_MOUSE_RAW", DATA_ROOT / "mouse_raw")).expanduser().resolve()
RAT_SPLIT = Path(os.environ.get("SPEN_RAT_SPLIT", REFERENCE_ROOT / "log/0710_primary_strict_source_robust_scannerop_split20260710_seed7/split.json")).expanduser().resolve()
