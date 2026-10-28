"""Smoke test: examples/scope_and_access.py exits 0.

TDD step 2 (RED): this test must fail first because the example file does not
yet exist, producing a non-zero return code (FileNotFoundError → OSError from
subprocess, or python itself exiting non-zero if the path resolves to nothing).

TDD step 3 (GREEN): create examples/scope_and_access.py; the test turns green.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_scope_and_access_example_exits_zero() -> None:
    """examples/scope_and_access.py must run to completion and exit with code 0."""
    example = Path(__file__).parent.parent / "examples" / "scope_and_access.py"
    proc = subprocess.run(
        [sys.executable, str(example)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"example exited with return code {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
