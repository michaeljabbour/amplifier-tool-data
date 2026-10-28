"""Tests for AmplifierStore.scope() write method.

Design Decision D4 (read-vs-fold boundary)
-------------------------------------------
``scope()`` is a WRITE-only operation — it appends exactly ONE
``RelationshipEvent(from_ref=cell_ref, to_ref=scope_ref, type=SCOPED_TO)``
and must NOT emit any ``AccessEvent``.

Coverage
--------
* test_scope_appends_one_scoped_to_edge: after store.scope(cell, scope),
  exactly one SCOPED_TO RelationshipEvent exists with from_ref==cell,
  to_ref==scope, type==SCOPED_TO.
* test_scope_emits_no_access_event: after store.scope(...), there are
  zero AccessEvents in the kernel log.
"""

from __future__ import annotations

from amplifier_data.lenses._scope import SCOPED_TO
from amplifier_data.models import AccessEvent, RelationshipEvent
from amplifier_data.store import AmplifierStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_relationship_events(store: AmplifierStore) -> list[RelationshipEvent]:
    """Return all RelationshipEvents currently in the kernel log."""
    return [ev for _, ev in store.kernel.all_events() if isinstance(ev, RelationshipEvent)]  # type: ignore[attr-defined]


def _access_events(store: AmplifierStore) -> list[AccessEvent]:
    """Return all AccessEvents currently in the kernel log."""
    return [ev for _, ev in store.kernel.all_events() if isinstance(ev, AccessEvent)]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Test: scope() appends exactly one SCOPED_TO RelationshipEvent
# ---------------------------------------------------------------------------


def test_scope_appends_one_scoped_to_edge() -> None:
    """After store.scope(cell, scope), exactly one SCOPED_TO RelationshipEvent exists.

    The event must have from_ref==cell, to_ref==scope, type==SCOPED_TO.
    """
    store = AmplifierStore()
    cell = store.write_cell(b"content cell")
    scope = store.write_cell(b"scope cell")

    store.scope(cell, scope)

    rel_events = _all_relationship_events(store)
    scoped_to_events = [e for e in rel_events if e.type == SCOPED_TO]

    assert len(scoped_to_events) == 1
    edge = scoped_to_events[0]
    assert edge.from_ref == cell
    assert edge.to_ref == scope
    assert edge.type == SCOPED_TO


# ---------------------------------------------------------------------------
# Test: scope() emits no AccessEvent (D4)
# ---------------------------------------------------------------------------


def test_scope_emits_no_access_event() -> None:
    """After store.scope(...), there are zero AccessEvents in the kernel log.

    scope() is write-only — it must not touch get_cell or regenerate,
    which are the only public store methods that emit AccessEvents (D4).
    """
    store = AmplifierStore()
    cell = store.write_cell(b"some content")
    scope = store.write_cell(b"some scope")

    store.scope(cell, scope)

    assert len(_access_events(store)) == 0
