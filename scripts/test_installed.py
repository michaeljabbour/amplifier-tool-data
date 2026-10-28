#!/usr/bin/env python3
"""Test the installed native wheel, refusing source-checkout/old editable imports.

Install the wheel and pytest in a clean virtual environment first. This imports
and verifies the installed package before pytest can add source paths.
"""
from pathlib import Path
import sys
import amplifier_data

ROOT=Path(__file__).resolve().parents[1]
origin=Path(amplifier_data.__file__).resolve()
if not origin.is_relative_to(Path(sys.prefix).resolve()) or origin.is_relative_to(ROOT):
    raise SystemExit('Install the wheel into a clean venv; this command refuses source/editable imports.')
if not amplifier_data.RUST_AVAILABLE:
    raise SystemExit('Installed wheel has no native kernel; durable tests cannot be claimed.')
import pytest
raise SystemExit(pytest.main(['-q',str(ROOT/'tests')]))
