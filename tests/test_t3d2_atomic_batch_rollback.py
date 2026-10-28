"""T3D-2 killer gates — atomic multi-fact batch + multi-call rollback envelope.

Two failure models, two gates:

* :class:`WriteBatch` — importance + category + source must co-land in ONE
  ``append_batch`` (all-or-nothing). A malformed batch must leave NO half-write.
* :class:`Transaction` — a multi-call flow that raises must be compensated: every
  fact written through the envelope is reversed, restoring pre-transaction state.

(memory T1-MEM-2 depends on the all-or-nothing batch.)
"""

from __future__ import annotations

import pytest

from amplifier_data import AmplifierStore
from amplifier_data.kernel import InMemoryKernel
from amplifier_data.models import CellWriteEvent, Event, RelationshipEvent, SeqPos


def _all_events(store: AmplifierStore) -> list[tuple[SeqPos, Event]]:
    kernel: InMemoryKernel = store.kernel  # type: ignore[assignment]
    return kernel.all_events()


def _log_len(store: AmplifierStore) -> int:
    return len(_all_events(store))


def test_three_facts_co_land_in_one_append_batch() -> None:
    """KILLER GATE: importance + category + source land all-or-nothing in one call.

    The memory cell plus its three facts are five staged events (1 cell + 3
    relationship facts + 0 causal here). After commit the log grows by exactly
    that many at a contiguous SeqPos run.
    """
    store = AmplifierStore()
    importance = store.write_cell(b"high")
    category = store.write_cell(b"decision")
    source = store.write_cell(b"interaction-42")
    before = _log_len(store)

    batch = store.write_batch()
    mem = batch.write_cell(b"a memory")
    batch.assert_fact(mem, "importance", importance)
    batch.assert_fact(mem, "category", category)
    batch.assert_fact(mem, "source", source)
    staged = len(batch)
    positions = batch.commit()

    assert staged == 4  # 1 cell write + 3 facts
    assert positions == list(range(before, before + 4))  # contiguous, single call
    assert _log_len(store) == before + 4
    facts = {f.predicate for f in store.query_facts(subject=mem).output}
    assert facts == {"importance", "category", "source"}


def test_partial_failure_leaves_no_half_write() -> None:
    """KILLER GATE: a malformed batch writes NOTHING (no half-write).

    The kernel validates the whole batch before mutating, so staging a non-Event
    aborts the entire commit. The log length is unchanged and no fact lands.
    """
    store = AmplifierStore()
    mem = store.write_cell(b"victim")
    importance = store.write_cell(b"high")
    before = _log_len(store)

    batch = store.write_batch()
    batch.assert_fact(mem, "importance", importance)
    batch.add_event("not-an-event")  # type: ignore[arg-type]  # corrupt the batch
    batch.assert_fact(mem, "category", importance)

    with pytest.raises(ValueError):
        batch.commit()

    assert _log_len(store) == before  # NOTHING was appended
    assert store.query_facts(subject=mem).output == []  # the importance fact did not land


def test_transaction_clean_exit_persists_writes() -> None:
    """A transaction that exits cleanly leaves its facts asserted."""
    store = AmplifierStore()
    importance = store.write_cell(b"high")
    category = store.write_cell(b"note")

    with store.transaction() as txn:
        mem = txn.write_cell(b"memory", source_interaction_id="int-7")
        txn.assert_fact(mem, "importance", importance)
        txn.assert_fact(mem, "category", category)

    # The user-facing facts persist. Causal edges live in the reserved "@causal:"
    # predicate namespace, so consumers filter them out; here we assert the user
    # facts are present rather than requiring exact set equality.
    facts = {f.predicate for f in store.query_facts(subject=mem).output}
    assert {"importance", "category"} <= facts
    assert store.writes_by_interaction("int-7") == [mem]  # attribution survived too


def test_transaction_exception_compensates_all_facts() -> None:
    """KILLER GATE: an exception mid-transaction rolls back via compensating writes.

    Two facts are asserted, then the body raises. On exit every fact is
    invalidated, so the externally-visible fact set returns to empty — even
    though the append-only log still contains the (now-superseded) events.
    """
    store = AmplifierStore()
    importance = store.write_cell(b"high")
    category = store.write_cell(b"note")
    mem = store.write_cell(b"memory")

    boom = RuntimeError("plasticity write failed downstream")
    with pytest.raises(RuntimeError) as caught:
        with store.transaction() as txn:
            txn.assert_fact(mem, "importance", importance)
            txn.assert_fact(mem, "category", category)
            raise boom

    assert caught.value is boom  # original exception propagates, not swallowed
    assert store.query_facts(subject=mem).output == []  # both facts compensated away


def test_relationship_compensator_tombstones_edge() -> None:
    """A non-fact edge written in a transaction is tombstoned on rollback.

    The compensator appends an ``__invalidate__:``-prefixed reverse edge; this
    gate asserts that exactly that tombstone exists for the original edge type.
    """
    from amplifier_data.lenses.temporal import INVALIDATE_PREFIX

    store = AmplifierStore()
    a = store.write_cell(b"A")
    b = store.write_cell(b"B")

    with pytest.raises(RuntimeError):
        with store.transaction() as txn:
            txn.relate(a, b, "links_to")
            raise RuntimeError("fail")

    edge_types = [
        ev.type
        for _pos, ev in _all_events(store)
        if isinstance(ev, RelationshipEvent) and ev.from_ref == a and ev.to_ref == b
    ]
    assert "links_to" in edge_types
    assert INVALIDATE_PREFIX + "links_to" in edge_types  # compensating tombstone present


def test_batch_does_not_change_cell_identity() -> None:
    """A cell staged in a batch has the same content-address it would have alone."""
    store = AmplifierStore()
    batch = store.write_batch()
    ref = batch.write_cell(b"identity-stable")
    batch.commit()
    assert ref == CellWriteEvent(payload=b"identity-stable").cell_ref()
