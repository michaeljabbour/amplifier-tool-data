"""T3E killer gates — autonomic self-indexing (the reflexive loop, made self-driving).

E3 already proved the *manual* reflexive index: reads emit AccessEvents and an
indexer can fold them into a regenerable IndexCell that explains why. What was
missing is the **autonomic** part — the substrate deciding, unprompted, to
materialize that index when access pressure warrants it.

``AutonomicIndexer`` closes that loop:

* It subscribes to the kernel as a READ-ONLY observer that only *counts*
  AccessEvents — it never writes from inside the observer callback, so the
  kernel's ordered post-commit notification contract is untouched.
* ``run_once()`` is where the WRITE happens: when accumulated accesses cross a
  threshold, it folds a fresh IndexCell (via the existing ``IndexLens``) and
  appends its ``IndexCellEvent`` to the log. The cell is regenerable (E1) and
  self-explaining (its ``access_summary`` is the "why").
* An optional background thread (``start``/``stop``) makes it truly autonomic;
  ``run_once`` is exposed so the loop can also be driven deterministically.

These gates prove the threshold gate, materialization + E1 regeneration, the
"explain why" provenance, idempotence, the read-only/no-self-trigger invariant,
and the background-thread autonomic path.
"""

from __future__ import annotations

import time

from amplifier_data import AmplifierStore, AutonomicIndexer
from amplifier_data.kernel import InMemoryKernel
from amplifier_data.models import AccessEvent, Event, IndexCellEvent, SeqPos


def _all_events(store: AmplifierStore) -> list[tuple[SeqPos, Event]]:
    kernel: InMemoryKernel = store.kernel  # type: ignore[assignment]
    return kernel.all_events()


def _index_cells(store: AmplifierStore) -> list[IndexCellEvent]:
    return [ev for _pos, ev in _all_events(store) if isinstance(ev, IndexCellEvent)]


def _count_access_events(store: AmplifierStore) -> int:
    return sum(1 for _pos, ev in _all_events(store) if isinstance(ev, AccessEvent))


def test_threshold_must_be_positive() -> None:
    store = AmplifierStore()
    try:
        AutonomicIndexer(store, threshold=0)
    except ValueError:
        return
    raise AssertionError("threshold=0 should raise ValueError")


def test_does_not_index_below_threshold() -> None:
    """Below the access threshold, nothing materializes."""
    store = AmplifierStore()
    indexer = AutonomicIndexer(store, threshold=5)
    a = store.write_cell(b"a")
    for _ in range(4):  # only 4 reads, threshold is 5
        store.get_cell(a)

    assert indexer.pending == 4
    assert indexer.run_once() is None  # threshold not reached
    assert _index_cells(store) == []  # no index cell in the log
    assert indexer.materializations == 0


def test_materializes_when_threshold_crossed() -> None:
    """KILLER GATE: crossing the threshold self-builds a regenerable index cell."""
    store = AmplifierStore()
    indexer = AutonomicIndexer(store, threshold=3)
    a = store.write_cell(b"a")
    b = store.write_cell(b"b")
    store.get_cell(a)
    store.get_cell(a)
    store.get_cell(b)  # 3 reads -> threshold reached

    ref = indexer.run_once()
    assert ref is not None
    assert indexer.materializations == 1
    assert indexer.pending == 0  # counter reset after materialization

    # The index cell is a real, resolvable cell whose payload is the count table.
    cell = store.get_cell(ref, record_access=False)
    import json

    counts = json.loads(cell.payload.decode("utf-8"))
    assert counts == {a: 2, b: 1}


def test_index_cell_regenerates_from_log_e1() -> None:
    """E1: the autonomically-built index cell regenerates byte-identically."""
    store = AmplifierStore()
    indexer = AutonomicIndexer(store, threshold=2)
    a = store.write_cell(b"a")
    store.get_cell(a)
    store.get_cell(a)
    ref = indexer.run_once()
    assert ref is not None

    before = store.get_cell(ref, record_access=False).payload
    store.drop_materialized(ref)  # evict the cache; force a fresh fold from the log
    after = store.regenerate(ref, record_access=False).payload
    assert before == after


def test_explain_returns_why_provenance() -> None:
    """The index is self-explaining: explain() is the folded access summary."""
    import json

    store = AmplifierStore()
    indexer = AutonomicIndexer(store, threshold=2)
    a = store.write_cell(b"a")
    store.get_cell(a)
    store.get_cell(a)
    indexer.run_once()

    why = indexer.explain()
    assert why is not None
    assert json.loads(why.decode("utf-8")) == {a: 2}


def test_no_new_accesses_is_idempotent() -> None:
    """KILLER GATE: forcing a re-index with no new reads does NOT duplicate.

    Same access state -> same content-addressed index ref -> no second append.
    """
    store = AmplifierStore()
    indexer = AutonomicIndexer(store, threshold=2)
    a = store.write_cell(b"a")
    store.get_cell(a)
    store.get_cell(a)
    first = indexer.run_once()
    assert first is not None
    n_index_cells = len(_index_cells(store))

    # Force another pass with NO new reads through the indexer's counter.
    again = indexer.run_once(force=True)
    assert again is None  # unchanged -> nothing appended
    assert len(_index_cells(store)) == n_index_cells


def test_materializing_emits_no_access_events_and_does_not_self_trigger() -> None:
    """KILLER GATE: the index write is not an AccessEvent and cannot re-trigger.

    Materialization folds via all_events() (D4: no AccessEvents) and appends only
    the IndexCellEvent. That append must not bump the indexer's pending counter,
    or the loop would run away.
    """
    store = AmplifierStore()
    indexer = AutonomicIndexer(store, threshold=3)
    a = store.write_cell(b"a")
    store.get_cell(a)
    store.get_cell(a)
    store.get_cell(a)
    access_before = _count_access_events(store)

    indexer.run_once()

    assert _count_access_events(store) == access_before  # folding emitted no reads
    assert indexer.pending == 0  # the IndexCellEvent append did NOT self-trigger


def test_background_thread_is_autonomic() -> None:
    """KILLER GATE: with the background loop running, reads alone trigger indexing.

    No explicit run_once() call — the daemon thread materializes on its own once
    enough reads accumulate. This is the 'autonomic' guarantee.
    """
    store = AmplifierStore(InMemoryKernel())
    indexer = AutonomicIndexer(store, threshold=2)
    indexer.start(interval=0.01)
    try:
        a = store.write_cell(b"a")
        store.get_cell(a)
        store.get_cell(a)  # 2 reads -> threshold; nobody calls run_once

        deadline = time.time() + 3.0
        while time.time() < deadline and indexer.materializations == 0:
            time.sleep(0.02)

        assert indexer.materializations >= 1
        assert len(_index_cells(store)) >= 1
    finally:
        indexer.stop()


def test_detach_stops_observing() -> None:
    """After detach(), new reads no longer feed the indexer."""
    store = AmplifierStore()
    indexer = AutonomicIndexer(store, threshold=100)
    a = store.write_cell(b"a")
    store.get_cell(a)
    assert indexer.pending == 1

    indexer.detach()
    store.get_cell(a)
    store.get_cell(a)
    assert indexer.pending == 1  # frozen — observer detached


def test_store_factory_wires_indexer() -> None:
    """store.autonomic_indexer(...) returns an indexer bound to this store."""
    store = AmplifierStore()
    indexer = store.autonomic_indexer(threshold=2)
    assert isinstance(indexer, AutonomicIndexer)
    a = store.write_cell(b"a")
    store.get_cell(a)
    store.get_cell(a)
    assert indexer.run_once() is not None
    indexer.detach()
