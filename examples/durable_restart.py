"""Durable restart story — user-facing demonstration of AmplifierStore persistence.

Demonstrates the E1 restart-survival property: events written to a path-backed
``AmplifierStore`` survive a close/reopen cycle exactly as if the process had
exited and restarted.

Run::

    python examples/durable_restart.py

What this script proves
-----------------------
Session 1  Write four event types to an on-disk log, verify all cells are
           readable, then call ``close()`` to flush and release the file handle.
           This simulates a process writing data and then exiting cleanly.

Session 2  Reopen ``AmplifierStore(path=...)`` on the **same** path (simulating
           a process restart).  ``AmplifierStore`` replays the on-disk log and
           reconstructs full kernel state automatically.  Every ``get_cell()``
           call must succeed and return a ``Cell`` whose ``payload`` and
           ``interpreters`` are value-equal to what was written in Session 1.

The assertions mirror the ``test_e1_survives_restart`` killer gate in
``tests/test_durable_restart.py``.  The example exits with code 0 on success
and code 1 on any assertion failure, so it can be used as a smoke test.
"""

from __future__ import annotations

import os
import sys
import tempfile

from amplifier_data import RUST_AVAILABLE
from amplifier_data.store import AmplifierStore


def _step(msg: str) -> None:
    """Print a single progress line to stdout."""
    print(f"  {msg}")


def main() -> int:
    """Run the durable-restart demonstration.  Returns 0 on success, 1 on failure.

    Raises
    ------
    SystemExit(2)
        If the Rust extension is not available (path= is not supported without it).
    """
    if not RUST_AVAILABLE:
        print(
            "SKIP: AmplifierStore(path=...) requires the compiled Rust extension.\n"
            "Build it with `maturin develop` and retry.",
            file=sys.stderr,
        )
        return 2

    # A syntactically valid 64-hex-char interpreter reference — mimics a real hash.
    interp_ref = "a" * 64

    # Use a temporary directory so the demo is self-cleaning and repeatable.
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "demo.log")

        print(f"\nDurable restart demo  (log: {path})\n")

        # ------------------------------------------------------------------ #
        # Session 1 — write events and verify pre-close.                      #
        # ------------------------------------------------------------------ #
        print("Session 1: open, write, verify, close")

        store1 = AmplifierStore(path=path)
        try:
            # Event A: plain CellWriteEvent — no interpreter list.
            ref_alpha = store1.write_cell(b"alpha payload")
            _step(f"wrote cell_alpha  ({ref_alpha[:16]}...)")

            # Event B: CellWriteEvent carrying one interpreter reference.
            ref_beta = store1.write_cell(b"beta payload", interpreters=(interp_ref,))
            _step(f"wrote cell_beta   ({ref_beta[:16]}...)")

            # Event C: RelationshipEvent — grows cell_alpha's provenance.
            store1.relate(ref_alpha, ref_beta, "references")
            _step("wrote relationship  alpha -> beta")

            # Events D + E: ToolResultEvent + InterpreterOutputEvent (atomic batch).
            derived_ref = store1.write_interpretation(
                model="gpt-4o",
                input_hash=ref_alpha,
                tool_results=(b"tool result bytes",),
                output=b"derived output",
            )
            _step(f"wrote derived cell ({derived_ref[:16]}...)")

            # Pre-close verification — all cells must be readable before flush.
            cell_pre_alpha = store1.get_cell(ref_alpha)
            assert cell_pre_alpha.payload == b"alpha payload", (
                "pre-close: cell_alpha payload mismatch"
            )
            assert cell_pre_alpha.interpreters == (), (
                "pre-close: cell_alpha must have no interpreters"
            )
            _step("verified cell_alpha pre-close  ✓")

            cell_pre_beta = store1.get_cell(ref_beta)
            assert cell_pre_beta.payload == b"beta payload", "pre-close: cell_beta payload mismatch"
            assert cell_pre_beta.interpreters == (interp_ref,), (
                "pre-close: cell_beta interpreter list mismatch"
            )
            _step("verified cell_beta pre-close   ✓")

            cell_pre_derived = store1.get_cell(derived_ref)
            assert cell_pre_derived.payload == b"derived output", (
                "pre-close: derived cell payload mismatch"
            )
            _step("verified derived cell pre-close ✓")

        finally:
            store1.close()
            _step("closed store (flush + release file handle)")

        # ------------------------------------------------------------------ #
        # Session 2 — reopen on the same path (simulated restart).            #
        # ------------------------------------------------------------------ #
        print("\nSession 2: reopen on same path (simulated restart), verify, close")

        store2 = AmplifierStore(path=path)
        try:
            # E1 core — every cell from Session 1 must be present and value-equal.
            cell_alpha = store2.get_cell(ref_alpha)
            assert cell_alpha.payload == b"alpha payload", (
                "E1: cell_alpha payload must be value-equal after restart"
            )
            assert cell_alpha.interpreters == (), (
                "E1: cell_alpha must have no interpreters after restart"
            )
            _step(
                f"cell_alpha   payload={cell_alpha.payload!r}  "
                f"interpreters={cell_alpha.interpreters!r}  ✓"
            )

            cell_beta = store2.get_cell(ref_beta)
            assert cell_beta.payload == b"beta payload", (
                "E1: cell_beta payload must be value-equal after restart"
            )
            assert cell_beta.interpreters == (interp_ref,), (
                "E1: cell_beta interpreter list must be value-equal after restart"
            )
            _step(
                f"cell_beta    payload={cell_beta.payload!r}  "
                f"interpreters=({cell_beta.interpreters[0][:8]}...)  ✓"
            )

            cell_derived = store2.get_cell(derived_ref)
            assert cell_derived.payload == b"derived output", (
                "E1: derived cell payload must be value-equal after restart"
            )
            _step(f"derived cell payload={cell_derived.payload!r}  ✓")

        finally:
            store2.close()
            _step("closed store")

    print(
        "\nSUCCESS: E1 restart-survival demonstrated — all cells survived "
        "close + reopen with payload and interpreter list value-equal.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
