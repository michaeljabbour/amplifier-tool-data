"""T3D-3 killer gates — the causal tag (write_interaction_id).

A tagged write records WHICH interaction caused it / WHICH hook performed it, and
that attribution is queryable ("show all writes from interaction X"). The encoding
is a reserved-type provenance edge, so it must be fully backward compatible: an
untagged write is byte-identical to the pre-feature behaviour, and tagging never
changes a cell's content-addressed identity.
"""

from __future__ import annotations

from amplifier_data import AmplifierStore
from amplifier_data.kernel import InMemoryKernel
from amplifier_data.models import CellWriteEvent, Event, RelationshipEvent, SeqPos


def _all_events(store: AmplifierStore) -> list[tuple[SeqPos, Event]]:
    kernel: InMemoryKernel = store.kernel  # type: ignore[assignment]
    return kernel.all_events()


def _count_events(store: AmplifierStore) -> int:
    return len(_all_events(store))


def test_write_is_attributable_to_interaction() -> None:
    """KILLER GATE: 'show all writes from interaction X' returns exactly those writes."""
    store = AmplifierStore()
    a = store.write_cell(b"first", source_interaction_id="int-X")
    b = store.write_cell(b"second", source_interaction_id="int-X")
    other = store.write_cell(b"unrelated", source_interaction_id="int-Y")

    assert store.writes_by_interaction("int-X") == [a, b]  # in log order
    assert store.writes_by_interaction("int-Y") == [other]
    assert store.writes_by_interaction("never") == []


def test_write_is_attributable_to_hook() -> None:
    """A write tagged with a hook name is queryable by that hook."""
    store = AmplifierStore()
    ref = store.write_cell(b"captured", written_by_hook="hooks-memory-capture")
    assert store.writes_by_hook("hooks-memory-capture") == [ref]
    assert store.writes_by_hook("some-other-hook") == []


def test_both_tags_on_one_write() -> None:
    """interaction + hook can be recorded together and queried independently."""
    store = AmplifierStore()
    ref = store.write_cell(
        b"both",
        source_interaction_id="int-7",
        written_by_hook="hooks-memory-capture",
    )
    assert store.writes_by_interaction("int-7") == [ref]
    assert store.writes_by_hook("hooks-memory-capture") == [ref]


def test_tagging_does_not_change_cell_identity() -> None:
    """Backward compat: a tagged cell has the SAME ref as the untagged equivalent.

    Identity is the content address of the CellWriteEvent payload (+interpreters);
    causal edges land in provenance only, so they cannot move the ref.
    """
    store = AmplifierStore()
    tagged = store.write_cell(b"same-bytes", source_interaction_id="int-1")
    plain_ref = CellWriteEvent(payload=b"same-bytes").cell_ref()
    assert tagged == plain_ref
    assert store.get_cell(tagged).payload == b"same-bytes"


def test_untagged_write_is_single_event_unchanged() -> None:
    """An untagged write appends exactly ONE event — no behavioural drift."""
    store = AmplifierStore()
    before = _count_events(store)
    store.write_cell(b"plain")
    assert _count_events(store) == before + 1  # one CellWriteEvent, no causal edges


def test_tagged_write_lands_atomically_with_its_edges() -> None:
    """A write with both tags appends the cell + 2 edges in ONE contiguous run.

    Proves the attribution co-lands with the cell (all-or-nothing), not in a
    separate, racy follow-up call.
    """
    store = AmplifierStore()
    before = _count_events(store)
    ref = store.write_cell(
        b"atomic",
        source_interaction_id="int-1",
        written_by_hook="h",
    )
    after_events = _all_events(store)
    assert _count_events(store) == before + 3  # 1 cell + 2 causal edges

    # The two trailing events are the causal edges sourced from the new cell.
    tail = [ev for _pos, ev in after_events[before:]]
    edges = [ev for ev in tail if isinstance(ev, RelationshipEvent)]
    assert len(edges) == 2
    assert all(ev.from_ref == ref for ev in edges)


def test_causal_edges_appear_in_cell_provenance() -> None:
    """The attribution is part of the cell's regenerable provenance (auditability)."""
    from amplifier_data.causal import CAUSED_BY_INTERACTION

    store = AmplifierStore()
    ref = store.write_cell(b"audited", source_interaction_id="int-42")
    cell = store.get_cell(ref)
    edge_types = {ev.type for ev in cell.provenance if isinstance(ev, RelationshipEvent)}
    assert CAUSED_BY_INTERACTION in edge_types
