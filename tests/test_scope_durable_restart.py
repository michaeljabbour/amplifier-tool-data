"""Killer Gate — scope membership survives a durable close/reopen.

This gate proves that scope() + query_relational(scope=) results are durable
across a close() + reopen of AmplifierStore(path=...).

The durability is inherited for free: scope() appends a
RelationshipEvent(type=SCOPED_TO) via AmplifierStore.relate().
RelationshipEvents are persisted by the existing durable RustFileKernel, so
RelationalLens + fold_scope recover the full scope index on the second open
purely by re-folding the on-disk log.

TDD note
--------
This gate was written BEFORE the SCOPED_TO edge durability was confirmed
end-to-end.  Before the durable opt-in path was complete, reopening a new
AmplifierStore on the same path returned an empty kernel; the first
query_relational(scope=scope) call in session 2 returned an empty result
instead of the pre-close membership set.

After the durable path was wired (DurableKernel + RustFileKernel, Tasks 4-5),
the gate passes because the CellWriteEvent and RelationshipEvent(SCOPED_TO)
appended by scope() both survive the close/reopen cycle.

Without maturin develop, AmplifierStore(path=...) raises:
  RuntimeError: AmplifierStore(path=...) requires the compiled Rust extension ...

Naming
------
test_gate_scope_survives_durable_restart — scope membership survives durable restart
"""

from __future__ import annotations

import os
import tempfile

from amplifier_data.store import AmplifierStore


def test_gate_scope_survives_durable_restart() -> None:
    """Gate: query_relational(scope=) returns identical membership after close + reopen.

    Write sequence (session 1):
      - inside   CellWriteEvent (b'inside'), scoped to scope_cell
      - outside  CellWriteEvent (b'outside'), intentionally unscoped
      - scope    CellWriteEvent (b'scope'), the scope anchor cell

    Pre-close sanity: query_relational(scope=scope) must return {inside} only.

    After close + reopen (session 2), query_relational(scope=scope) must return
    the identical set {inside} — same membership, recovered purely by re-folding
    the on-disk log.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "scope_gate.log")

        # ------------------------------------------------------------------
        # Session 1: write cells, scope one, verify pre-close membership, close.
        # Simulates a process that writes data and then exits cleanly.
        # ------------------------------------------------------------------
        store1 = AmplifierStore(path=path)
        try:
            inside = store1.write_cell(b"inside")
            store1.write_cell(b"outside")  # intentionally unscoped — must not appear
            scope = store1.write_cell(b"scope")

            # Associate only 'inside' with the scope; 'outside' is deliberately excluded.
            store1.scope(inside, scope)

            before = {r["cell_ref"] for r in store1.query_relational(scope=scope).output}

            # Pre-close sanity: only 'inside' is directly scoped to scope.
            assert before == {inside}, f"pre-close: expected {{inside}} but got {before!r}"
        finally:
            store1.close()  # Simulates process exit: flush + release file handle.

        # ------------------------------------------------------------------
        # Session 2 (simulated restart): reopen at the same path.
        #
        # AmplifierStore(path=...) must replay the on-disk log and reconstruct
        # the full scope index (via fold_scope over RelationshipEvents) without
        # any external assistance.
        # ------------------------------------------------------------------
        store2 = AmplifierStore(path=path)
        try:
            after = {r["cell_ref"] for r in store2.query_relational(scope=scope).output}
        finally:
            store2.close()

        assert after == before, (
            "Gate: scope membership must survive durable restart — "
            "query_relational(scope=scope) must return the identical set after "
            f"close + reopen; expected {before!r}, got {after!r}"
        )
