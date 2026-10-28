"""Killer Gate 3 — embeddings survive a durable close/reopen.

This gate proves that add_embedding / query_vector results are durable across
a close() + reopen of AmplifierStore(path=...).

The durability is inherited for free: add_embedding appends a CellWriteEvent
(packed f32-LE vector) plus a RelationshipEvent(type=EMBEDDING_OF).  Both
event types are persisted by the existing durable RustFileKernel, so VectorLens
recovers the full index on the second open purely by re-folding the on-disk log.

TDD note
--------
This gate was written BEFORE the vector embedding events were confirmed to
survive a durable restart.  Before the durable opt-in path was complete
end-to-end, reopening a new AmplifierStore on the same path returned an
empty kernel; the first query_vector call in session 2 returned [] instead
of the pre-close results.

After the durable path was wired (DurableKernel + RustFileKernel, Tasks 7-8),
the gate passes because the CellWriteEvent and RelationshipEvent appended by
add_embedding both survive the close/reopen cycle.

Without maturin develop, AmplifierStore(path=...) raises:
  RuntimeError: AmplifierStore(path=...) requires the compiled Rust extension ...

Naming
------
test_gate3_embeddings_survive_durable_restart — Gate 3: embeddings survive durable restart
"""

from __future__ import annotations

import os
import tempfile

from amplifier_data.store import AmplifierStore


def test_gate3_embeddings_survive_durable_restart() -> None:
    """Gate 3: query_vector returns identical top-k after close + reopen.

    Write sequence (session 1):
      - near   CellWriteEvent (b'near target'), embedding (1, 0, 0)
      - far    CellWriteEvent (b'far target'),  embedding (0, 1, 0)

    Pre-close sanity: query_vector((1,0,0), k=2) must rank near before far
    (cosine similarity 1.0 vs 0.0).

    After close + reopen (session 2), query_vector must return the identical
    list of (cell_ref, score) pairs — same ordering, same floating-point scores.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "vector_gate.log")

        # ------------------------------------------------------------------
        # Session 1: write embeddings, verify pre-close ranking, then close.
        # Simulates a process that writes data and then exits cleanly.
        # ------------------------------------------------------------------
        store1 = AmplifierStore(path=path)
        try:
            near = store1.write_cell(b"near target")
            far = store1.write_cell(b"far target")
            store1.add_embedding(near, (1, 0, 0))
            store1.add_embedding(far, (0, 1, 0))

            before = store1.query_vector((1, 0, 0), k=2)

            # Pre-close ranking sanity: near (cosine=1.0) must outrank far (cosine=0.0).
            assert [ref for ref, _ in before] == [near, far], "pre-close ranking sanity"
        finally:
            store1.close()  # Simulates process exit: flush + release file handle.

        # ------------------------------------------------------------------
        # Session 2 (simulated restart): reopen at the same path.
        #
        # AmplifierStore(path=...) must replay the on-disk log and reconstruct
        # the full VectorLens index without any external assistance.
        # ------------------------------------------------------------------
        store2 = AmplifierStore(path=path)
        try:
            after = store2.query_vector((1, 0, 0), k=2)
        finally:
            store2.close()

        assert after == before, (
            "Gate 3: embeddings must survive durable restart — query_vector must return "
            "the identical top-k (cell_ref, score) after close + reopen"
        )
