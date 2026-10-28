"""E3 proof: reflexive self-indexing as an explicit static fold, no mutable side-car.

E3 traces the full lifecycle of a reflexively self-indexed store:

  1. Synthetic workload — write cells and perform public reads, which append
     AccessEvents to the kernel log (D4 read-vs-fold boundary).
  2. IndexLens static fold — ``IndexLens.project()`` folds the AccessEvents into
     an IndexCell and returns a ``Cell`` whose provenance carries the
     ``IndexCellEvent`` (the defining event).
  3. Defining event appended — the ``IndexCellEvent`` is appended to the kernel
     so the index is a FIRST-CLASS CELL in the log (reflexive self-indexing):
     the same store that was indexed now contains its own index as a Cell.
  4. IndexCell materializes — ``kernel.resolve(index_cell_ref)`` regenerates the
     materialized Cell from the log.
  5. drop_materialized + regenerate — evicting the cache and re-folding from the
     log alone yields a BYTE-IDENTICAL IndexCell with provenance present.
  6. No mutable side-car — ``IndexLens`` has no mutable instance state;
     the index is a Cell in the kernel log, not an external dict or file.

Never calls a live model.
Plan 1 tests (all existing tests) remain green.
"""

from __future__ import annotations

import json

from amplifier_data.kernel import InMemoryKernel
from amplifier_data.lenses.index import IndexLens
from amplifier_data.models import AccessEvent, Cell, IndexCellEvent
from amplifier_data.replay import regenerate
from amplifier_data.store import AmplifierStore


def test_e3_static_index() -> None:
    """E3: reflexive self-indexing as an explicit static fold, no mutable side-car.

    Full lifecycle proof:
    AccessEvents (via public reads)
    → IndexLens static fold
    → defining event appended (reflexive: index is a Cell in the kernel)
    → IndexCell materializes
    → drop_materialized + regenerate yields an IDENTICAL IndexCell with provenance
    → ASSERT provenance present
    → ASSERT no mutable side-car (the index is a Cell, not an external dict/file)
    """
    # ------------------------------------------------------------------ setup
    # Use AmplifierStore so public reads append AccessEvents (D4).
    # Explicitly pass InMemoryKernel so this test can access its internal
    # test-surface attributes (_cell_refs, _materialized).  The Protocol
    # (StorageKernel) exposes only the five-verb public API; private attributes
    # are an intentional test seam for InMemoryKernel specifically.
    kernel = InMemoryKernel()
    store = AmplifierStore(kernel)

    # Synthetic workload: three cells (no agents, no live model calls).
    payload_a = b"Sample drawer A: deploy key lives in 1Password vault Eng"
    payload_b = b"Sample drawer B: vault rotated 2026-06-01"
    payload_c = b"Sample drawer C: staging host config (never read)"

    ref_a = store.write_cell(payload_a)
    ref_b = store.write_cell(payload_b)
    ref_c = store.write_cell(payload_c)

    # ------------------------------------------- public reads → AccessEvents
    # Each store.get_cell() appends exactly one AccessEvent (D4 boundary).
    # Cell A read 3 times, cell B read once, cell C never read.
    store.get_cell(ref_a)
    store.get_cell(ref_a)
    store.get_cell(ref_a)
    store.get_cell(ref_b)

    # Confirm AccessEvents are in the kernel log.
    access_events = [ev for _, ev in kernel.all_events() if isinstance(ev, AccessEvent)]
    assert len(access_events) == 4, (
        f"expected 4 AccessEvents from public reads; got {len(access_events)}"
    )
    assert sum(1 for ae in access_events if ae.cell_ref == ref_a) == 3, (
        "cell A must have 3 AccessEvents"
    )
    assert sum(1 for ae in access_events if ae.cell_ref == ref_b) == 1, (
        "cell B must have 1 AccessEvent"
    )
    assert sum(1 for ae in access_events if ae.cell_ref == ref_c) == 0, (
        "unread cell C must have 0 AccessEvents"
    )

    # ---------------------------------------------- IndexLens static fold
    # project() is a PURE fold — it reads kernel.all_events() and does NOT
    # append any new events to the kernel.
    lens = IndexLens()
    log_len_before_fold = len(kernel.all_events())
    index_cell: Cell = lens.project(kernel)
    log_len_after_fold = len(kernel.all_events())

    assert log_len_after_fold == log_len_before_fold, (
        "IndexLens.project() must not mutate the kernel (pure static fold); "
        f"log length changed from {log_len_before_fold} to {log_len_after_fold}"
    )

    # Cell payload is the canonical JSON access count table.
    counts: dict[str, int] = json.loads(index_cell.payload.decode("utf-8"))
    assert counts.get(ref_a) == 3, f"expected count 3 for ref_a; got {counts.get(ref_a)!r}"
    assert counts.get(ref_b) == 1, f"expected count 1 for ref_b; got {counts.get(ref_b)!r}"
    assert ref_c not in counts, "unread cell C must not appear in the index"

    # Cell provenance carries the defining IndexCellEvent (D2a).
    assert isinstance(index_cell, Cell)
    assert len(index_cell.provenance) == 1, (
        f"IndexCell provenance must have exactly 1 event; got {len(index_cell.provenance)}"
    )
    ice: IndexCellEvent = index_cell.provenance[0]  # type: ignore[assignment]
    assert isinstance(ice, IndexCellEvent), (
        f"provenance[0] must be an IndexCellEvent (the defining event); got {type(ice).__name__}"
    )
    # The defining event's cell_ref equals its own derived identity (no circularity).
    assert ice.cell_ref == IndexCellEvent.derive_ref(
        access_summary=ice.access_summary,
        output=ice.output,
    ), "IndexCellEvent.cell_ref must equal its content-addressed identity"

    # ----------------------- defining event appended (reflexive self-indexing)
    # Append the IndexCellEvent to the kernel so the index is a FIRST-CLASS
    # CELL in the log — not just a transient in-memory computation.
    # This is the "reflexive" part: the store contains its own index as a Cell.
    index_cell_ref = ice.cell_ref
    kernel.append_batch([ice])

    # The kernel now recognises the index cell ref.
    assert index_cell_ref in kernel._cell_refs, (
        "after appending the defining event, index_cell_ref must be in kernel._cell_refs"
    )

    # ----------------------------------------------- IndexCell materializes
    # kernel.resolve() regenerates the Cell from the log on first access.
    resolved = kernel.resolve(index_cell_ref)
    assert isinstance(resolved, Cell), (
        "kernel.resolve(index_cell_ref) must return a Cell after the defining event is appended"
    )
    assert resolved.payload == index_cell.payload, "materialized payload must equal the fold output"
    assert isinstance(resolved.provenance[0], IndexCellEvent), (
        "materialized provenance must carry the IndexCellEvent"
    )

    # --------------------------------- drop_materialized + regenerate cycle
    # Evict the materialized cache — the next resolve() must re-fold from the log.
    kernel.drop_materialized(index_cell_ref)

    # After eviction the cell is absent from the cache.
    assert index_cell_ref not in kernel._materialized, (
        "drop_materialized must evict index_cell_ref from the materialized cache"
    )

    # Re-fold from the log alone → must be BYTE-IDENTICAL.
    regenerated = kernel.resolve(index_cell_ref)
    assert isinstance(regenerated, Cell)
    assert regenerated.payload == index_cell.payload, (
        "regenerated payload must be BYTE-IDENTICAL to the original fold output "
        "(deterministic static fold)"
    )
    assert regenerated.provenance == index_cell.provenance, (
        "regenerated provenance must be identical to the original fold provenance"
    )

    # ------------------------------------------ ASSERT provenance present
    # The defining IndexCellEvent is accessible via provenance after regeneration.
    assert len(regenerated.provenance) >= 1, "regenerated IndexCell must have provenance"
    regen_ice: IndexCellEvent = regenerated.provenance[0]  # type: ignore[assignment]
    assert isinstance(regen_ice, IndexCellEvent), (
        "regenerated provenance[0] must be the IndexCellEvent"
    )
    assert regen_ice.access_summary == ice.access_summary, (
        "access_summary ('why I indexed this' provenance) must survive regeneration"
    )
    # The access_summary is itself the canonical count JSON — round-trip check.
    summary_counts: dict[str, int] = json.loads(regen_ice.access_summary.decode("utf-8"))
    assert summary_counts.get(ref_a) == 3, "access_summary must encode 3 reads for ref_a"
    assert summary_counts.get(ref_b) == 1, "access_summary must encode 1 read for ref_b"

    # ---------------------- ASSERT NO mutable side-car anywhere
    # 1. IndexLens has ZERO mutable instance state — it is a pure stateless fold.
    #    (No cached view, no per-instance dict, no external file.)
    assert vars(lens) == {}, (
        "IndexLens must have no mutable instance variables (no side-car); "
        f"found vars(lens) = {vars(lens)!r}"
    )

    # 2. A FRESH IndexLens instance over the same kernel yields the same counts.
    #    This proves there is no shared mutable state between lens instances.
    fresh_lens = IndexLens()
    fresh_cell = fresh_lens.project(kernel)
    fresh_counts: dict[str, int] = json.loads(fresh_cell.payload.decode("utf-8"))
    # IndexCellEvent (just appended) is NOT an AccessEvent, so counts are unchanged.
    assert fresh_counts.get(ref_a) == 3, (
        "fresh IndexLens must yield the same count for ref_a (no shared mutable state)"
    )
    assert fresh_counts.get(ref_b) == 1, (
        "fresh IndexLens must yield the same count for ref_b (no shared mutable state)"
    )

    # 3. The index is a Cell IN the kernel log — not an external container.
    #    Proof: drop the cache again; the kernel regenerates from the log alone.
    kernel.drop_materialized(index_cell_ref)
    third_resolved = kernel.resolve(index_cell_ref)
    assert isinstance(third_resolved, Cell)
    assert third_resolved.payload == index_cell.payload, (
        "third materialization must be byte-identical — the index lives in the log, "
        "not in any external mutable container"
    )

    # 4. regenerate() directly (bypassing kernel.resolve()) is also byte-identical —
    #    proving the fold is deterministic and the result is fully log-derivable.
    direct_regen: Cell = regenerate(kernel.all_events(), index_cell_ref)
    assert direct_regen.payload == index_cell.payload, (
        "direct regenerate() must produce byte-identical output (deterministic static fold)"
    )
    assert isinstance(direct_regen.provenance[0], IndexCellEvent), (
        "direct regenerate() provenance must carry the IndexCellEvent"
    )

    # 5. The CellWriteEvents for the original cells remain intact and unaffected.
    #    (Appending the IndexCellEvent must not corrupt any other cell.)
    assert store.get_cell(ref_a).payload == payload_a, "ref_a payload must be unchanged"
    assert store.get_cell(ref_b).payload == payload_b, "ref_b payload must be unchanged"
    assert store.get_cell(ref_c).payload == payload_c, "ref_c payload must be unchanged"
