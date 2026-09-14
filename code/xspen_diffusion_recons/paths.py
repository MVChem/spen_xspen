"""Explicit external inputs; generated artifacts default to this checkout.

Setting an input path never changes where preparation or reconstruction writes.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def input_path(variable, default):
    return Path(os.environ.get(variable, str(default))).expanduser().resolve()


IXI_SOURCE = input_path('XSPEN_IXI_SOURCE', PROJECT_ROOT / 'data/raw/ixi')
IXI_DATA = input_path('XSPEN_IXI_DATA', PROJECT_ROOT / 'data/ixi128')
RAW_SCANNER = input_path('XSPEN_RAW_SCANNER', PROJECT_ROOT / 'data/raw/siemens')
SCANNER_DATA = input_path('XSPEN_SCANNER_DATA', PROJECT_ROOT / 'scanner')
