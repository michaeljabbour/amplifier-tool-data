"""Killer-gate durability tests at the Python / AmplifierStore layer.

Contains two killer gates:

E1 — restart-survival (test_e1_survives_restart)
    Mirrors gate_e1_restart_survival_byte_identical_regeneration in
    crates/amplifier-data-core/tests/durable_gates.rs at the Python/binding
    layer (Task T8, cargo gate T4 / E1 equivalent).

E2 — torn-tail recovery and clean re-append (test_recovers_after_torn_write)
    Mirrors gate_e2_cargo_torn_tail_recovery_and_clean_reappend in
    crates/amplifier-data-core/tests/durable_gates.rs at the Python/binding
    layer (Task T9, cargo gate T5 / E2 equivalent).

TDD note (E1)
-------------
This gate was written BEFORE the durable opt-in path was complete end-to-end.
Before ``DurableKernel`` (wrapping ``RustFileKernel``) replayed the on-disk
log, reopening a new ``AmplifierStore`` on the same path returned an empty
kernel; the first ``get_cell`` call in session 2 would raise:

  amplifier_data.kernel.NotFound: ref not found: '<ref>'

After the durable path was wired (``DurableKernel`` + ``RustFileKernel``
via T7), the gate passes.

TDD note (E2)
-------------
This gate was written BEFORE the torn-tail truncation path was confirmed
end-to-end at the Python layer.  Without the truncation performed inside
``RustFileKernel.open``, the post-recovery append lands after the torn bytes
and is invisible on the next reopen (T3 failure).  After ``FileKernel::open``
in Rust was wired to call ``file.set_len(valid_bytes)`` (T5 / cargo gate E2),
the gate passes at both the Rust and Python layers.

Naming
------
``test_e1_survives_restart``        — E1: events survive a clean restart
``test_recovers_after_torn_write``  — E2: torn-tail recovery + durable re-append
"""

from __future__ import annotations

import os
import struct
import tempfile

from amplifier_data.store import AmplifierStore


def test_e1_survives_restart() -> None:
    """E1 gate: events appended before close are present and value-equal after restart.

    Write sequence (session 1):
      - cell_alpha   CellWriteEvent, no interpreters
      - cell_beta    CellWriteEvent, one interpreter reference
      - edge         RelationshipEvent from cell_alpha → cell_beta
      - derived      ToolResultEvent + InterpreterOutputEvent (write_interpretation)

    All four cells (cell_alpha, cell_beta, derived) are verified readable
    *before* ``close()``.  After close + reopen (simulated restart) in
    session 2, each ``get_cell`` call must succeed and return a Cell whose
    ``payload`` and ``interpreters`` are value-equal to the originals.

    Without the durable opt-in path this gate fails at the first
    ``store2.get_cell(ref_alpha)`` call in session 2:

      amplifier_data.kernel.NotFound: ref not found: '...'
    """
    # A syntactically valid 64-hex-char interpreter reference.
    interp_ref = "a" * 64

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "e1_gate.log")

        # ------------------------------------------------------------------
        # Session 1: write events, verify pre-close, then close.
        # Simulates a process that writes data and then exits cleanly.
        # ------------------------------------------------------------------
        store1 = AmplifierStore(path=path)
        try:
            # Event A: plain CellWriteEvent — no interpreter list.
            ref_alpha = store1.write_cell(b"alpha payload")

            # Event B: CellWriteEvent carrying an interpreter reference.
            ref_beta = store1.write_cell(b"beta payload", interpreters=(interp_ref,))

            # Event C: RelationshipEvent — amends cell_alpha's provenance.
            store1.relate(ref_alpha, ref_beta, "references")

            # Events D + E: ToolResultEvent + InterpreterOutputEvent.
            derived_ref = store1.write_interpretation(
                model="gpt-4o",
                input_hash=ref_alpha,
                tool_results=(b"tool result bytes",),
                output=b"derived output",
            )

            # Pre-close sanity: all cells must be readable before closing.
            cell_pre_alpha = store1.get_cell(ref_alpha)
            assert cell_pre_alpha.payload == b"alpha payload", (
                "E1 pre-close: cell_alpha payload must be b'alpha payload'"
            )
            assert cell_pre_alpha.interpreters == (), (
                "E1 pre-close: cell_alpha must have no interpreters"
            )

            cell_pre_beta = store1.get_cell(ref_beta)
            assert cell_pre_beta.payload == b"beta payload", (
                "E1 pre-close: cell_beta payload must be b'beta payload'"
            )
            assert cell_pre_beta.interpreters == (interp_ref,), (
                "E1 pre-close: cell_beta must carry the interpreter ref"
            )

            cell_pre_derived = store1.get_cell(derived_ref)
            assert cell_pre_derived.payload == b"derived output", (
                "E1 pre-close: derived cell payload must be b'derived output'"
            )
        finally:
            store1.close()  # Simulates process exit: flush + release file handle.

        # ------------------------------------------------------------------
        # Session 2: reopen at the same path (simulates process restart).
        #
        # AmplifierStore(path=...) must replay the on-disk log and
        # reconstruct the full kernel state without any external assistance.
        # ------------------------------------------------------------------
        store2 = AmplifierStore(path=path)
        try:
            # E1 core: cell_alpha — payload and empty interpreter list value-equal.
            cell_alpha = store2.get_cell(ref_alpha)
            assert cell_alpha.payload == b"alpha payload", (
                "E1: cell_alpha payload must be value-equal after restart"
            )
            assert cell_alpha.interpreters == (), (
                "E1: cell_alpha must have no interpreters after restart"
            )

            # E1 core: cell_beta — payload and interpreter list value-equal.
            cell_beta = store2.get_cell(ref_beta)
            assert cell_beta.payload == b"beta payload", (
                "E1: cell_beta payload must be value-equal after restart"
            )
            assert cell_beta.interpreters == (interp_ref,), (
                "E1: cell_beta interpreter list must be value-equal after restart"
            )

            # E1 core: derived cell — payload from InterpreterOutputEvent value-equal.
            cell_derived = store2.get_cell(derived_ref)
            assert cell_derived.payload == b"derived output", (
                "E1: derived cell payload must be value-equal after restart"
            )
        finally:
            store2.close()


def test_recovers_after_torn_write() -> None:
    """E2 gate: torn-tail recovery and durable re-append at the Python/AmplifierStore layer.

    Mirrors gate_e2_cargo_torn_tail_recovery_and_clean_reappend from
    crates/amplifier-data-core/tests/durable_gates.rs at the Python/binding layer.

    What this gate proves
    ---------------------

    | Label | Property |
    |-------|----------|
    | T2    | ``AmplifierStore(path=...)`` after a truncated final record recovers all
              complete cells without error; torn tail silently discarded.             |
    | T3    | A ``write_cell`` immediately after torn-tail recovery produces a durable
              record that survives a further reopen.                                  |

    Failure mode without truncation on open
    ----------------------------------------
    Without truncating the torn tail during ``RustFileKernel.open``, subsequent
    ``write_cell`` calls land *after* the torn bytes.  On the next reopen the
    reader stops at the corrupted record, so the newly-written cell is invisible::

        T3: post-recovery cell must survive further reopen
          amplifier_data.kernel.NotFound: ref not found: '...'

    This gate was written before ``FileKernel::open`` in Rust was confirmed to
    call ``file.set_len(valid_bytes)`` to truncate the torn tail.  After that path
    was wired (T5 / cargo gate E2), the gate passes at both the Rust and Python layers.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "e2_torn_tail_gate.log")

        # ------------------------------------------------------------------
        # Session 1: write 3 complete events, verify pre-close, then close.
        # After this session the on-disk log contains exactly 3 valid records.
        # ------------------------------------------------------------------
        store1 = AmplifierStore(path=path)
        try:
            ref_0 = store1.write_cell(b"payload-zero")
            ref_1 = store1.write_cell(b"payload-one")
            ref_2 = store1.write_cell(b"payload-two")

            # Pre-close sanity: all 3 cells must be readable before close.
            assert store1.get_cell(ref_0).payload == b"payload-zero", (
                "E2 pre-close: cell_0 payload must be b'payload-zero'"
            )
            assert store1.get_cell(ref_1).payload == b"payload-one", (
                "E2 pre-close: cell_1 payload must be b'payload-one'"
            )
            assert store1.get_cell(ref_2).payload == b"payload-two", (
                "E2 pre-close: cell_2 payload must be b'payload-two'"
            )
        finally:
            store1.close()  # Simulates process exit: flush + release file handle.

        # ------------------------------------------------------------------
        # Inject a torn tail: append a partial (incomplete) record to disk.
        #
        # We construct a well-formed CellWrite record, then write only the
        # first half of its bytes — exactly what a crash mid-append leaves behind.
        #
        # Wire format for a CellWrite event (from crates/amplifier-data-core/src/codec.rs):
        #   body: [tag=0x00][u32_LE payload_len][payload_bytes][u32_LE interp_count=0]
        #   sandwich record: [u32_LE body_len][body][u32_LE body_len]
        # ------------------------------------------------------------------
        torn_payload = b"this-write-was-interrupted-mid-append"
        body = bytearray()
        body.append(0x00)  # CellWrite variant tag
        body.extend(struct.pack("<I", len(torn_payload)))  # length-prefixed payload
        body.extend(torn_payload)
        body.extend(struct.pack("<I", 0))  # 0 interpreters (length-prefixed count)
        body_len_bytes = struct.pack("<I", len(body))
        full_torn_record = body_len_bytes + bytes(body) + body_len_bytes  # sandwich
        # Take only the first half — definitely an incomplete record.
        torn_partial = full_torn_record[: len(full_torn_record) // 2]
        # Sanity: the partial write must include at least the 4-byte leading length.
        assert len(torn_partial) >= 4, (
            "E2: partial torn record must be ≥ 4 bytes to include the leading length field"
        )

        with open(path, "ab") as f:
            f.write(torn_partial)

        # Verify torn bytes are present on disk before proceeding.
        actual_disk_size = os.path.getsize(path)
        assert actual_disk_size > (actual_disk_size - len(torn_partial)), (
            "E2: on-disk file must be larger after torn-write injection"
        )

        # ------------------------------------------------------------------
        # Session 2: reopen after torn tail.
        #
        # AmplifierStore(path=...) must:
        #   (a) succeed — no exception raised (T2)
        #   (b) recover all 3 complete cells (T2)
        #   (c) truncate the torn bytes so the subsequent write_cell lands at
        #       the correct offset and is recoverable on the next reopen (T3)
        # ------------------------------------------------------------------
        store2 = AmplifierStore(path=path)
        try:
            # T2: all 3 complete cells recovered; torn tail silently discarded.
            cell_0 = store2.get_cell(ref_0)
            assert cell_0.payload == b"payload-zero", (
                "T2: cell_0 payload must be value-equal after torn-tail recovery"
            )

            cell_1 = store2.get_cell(ref_1)
            assert cell_1.payload == b"payload-one", (
                "T2: cell_1 payload must be value-equal after torn-tail recovery"
            )

            cell_2 = store2.get_cell(ref_2)
            assert cell_2.payload == b"payload-two", (
                "T2: cell_2 payload must be value-equal after torn-tail recovery"
            )

            # T3 (write): append one new cell immediately after torn-tail recovery.
            # This write must land at the correct offset (after the 3 clean records,
            # NOT after the torn bytes).  Correct placement is guaranteed only if
            # open() truncated the torn tail before returning.
            ref_post = store2.write_cell(b"post-recovery-event")

            # New cell immediately visible in memory.
            cell_post_pre_close = store2.get_cell(ref_post)
            assert cell_post_pre_close.payload == b"post-recovery-event", (
                "T3: post-recovery cell must be immediately readable after write"
            )
        finally:
            store2.close()

        # ------------------------------------------------------------------
        # T3 durability: reopen a third time and verify the post-recovery
        # write survives.
        #
        # Critical: without torn-tail truncation in session 2, ref_post would
        # have been written after the torn bytes.  On this third reopen the
        # reader would stop at the torn record and raise NotFound for ref_post.
        # With correct truncation the log is clean and all 4 cells are present.
        # ------------------------------------------------------------------
        store3 = AmplifierStore(path=path)
        try:
            # All 3 original cells must still be present.
            assert store3.get_cell(ref_0).payload == b"payload-zero", (
                "T3: cell_0 must survive further reopen after post-recovery append"
            )
            assert store3.get_cell(ref_1).payload == b"payload-one", (
                "T3: cell_1 must survive further reopen after post-recovery append"
            )
            assert store3.get_cell(ref_2).payload == b"payload-two", (
                "T3: cell_2 must survive further reopen after post-recovery append"
            )

            # The post-recovery cell must also survive.
            cell_post = store3.get_cell(ref_post)
            assert cell_post.payload == b"post-recovery-event", (
                "T3: post-recovery cell must survive further reopen"
            )
        finally:
            store3.close()
