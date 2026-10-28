"""Tests for AmplifierStore record_access instance-default parameter (Task 8).

Spec: Add keyword-only record_access: bool = True to AmplifierStore.__init__.
The parameter is stored as self._record_access. No read behaviour changes yet.

Acceptance criteria
--------------------
* test_record_access_default_is_true
    AmplifierStore()._record_access is True   (default = True)
* test_record_access_instance_default_false_stored
    AmplifierStore(record_access=False)._record_access is False
* All pre-existing access-append tests still pass with default True.

Task 9 — tri-state per-call record_access on get_cell
------------------------------------------------------
* test_get_cell_explicit_false_suppresses_access_event
    instance default True, get_cell(ref, record_access=False) -> zero AccessEvents
* test_get_cell_explicit_true_forces_access_event
    instance default False, record_access=True -> 1 AccessEvent
* test_get_cell_none_defers_to_instance_default_true
    default True, get_cell(ref) -> 1 AccessEvent
* test_get_cell_none_defers_to_instance_default_false
    default False, get_cell(ref) -> zero
* test_get_cell_still_returns_cell_when_access_suppressed
    cell.payload == b"payload" with default False

Task 10 — tri-state record_access forwarded through regenerate()
----------------------------------------------------------------
* test_regenerate_explicit_false_suppresses
    instance default True, regenerate(ref, record_access=False) -> zero AccessEvents
* test_regenerate_none_defers_to_instance_default_false
    instance default False, regenerate(ref) -> zero AccessEvents
* test_regenerate_default_records_when_instance_default_true
    instance default True, regenerate(ref) -> 1 AccessEvent
"""

from __future__ import annotations

import pytest

from amplifier_data.models import AccessEvent
from amplifier_data.store import AmplifierStore

# ---------------------------------------------------------------------------
# RED phase validation: the parameter must NOT exist before implementation.
# Step 2 must first produce: TypeError: unexpected keyword argument 'record_access'
# ---------------------------------------------------------------------------


def test_record_access_default_is_true() -> None:
    """AmplifierStore()._record_access defaults to True."""
    store = AmplifierStore()
    assert store._record_access is True  # noqa: SLF001


def test_record_access_instance_default_false_stored() -> None:
    """AmplifierStore(record_access=False)._record_access is False."""
    store = AmplifierStore(record_access=False)
    assert store._record_access is False  # noqa: SLF001


def test_record_access_explicit_true_stored() -> None:
    """AmplifierStore(record_access=True) stores True explicitly."""
    store = AmplifierStore(record_access=True)
    assert store._record_access is True  # noqa: SLF001


def test_record_access_is_keyword_only() -> None:
    """record_access must be keyword-only; positional use raises TypeError."""
    with pytest.raises(TypeError):
        AmplifierStore(None, None, False)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Task 9: Helpers
# ---------------------------------------------------------------------------


def _access_events(store: AmplifierStore) -> list[AccessEvent]:
    """Return all AccessEvents currently in the kernel log."""
    all_events = store.kernel.all_events()  # type: ignore[attr-defined]
    return [ev for _, ev in all_events if isinstance(ev, AccessEvent)]


# ---------------------------------------------------------------------------
# Task 9: tri-state per-call record_access on get_cell
# ---------------------------------------------------------------------------


def test_get_cell_explicit_false_suppresses_access_event() -> None:
    """Instance default True, get_cell(ref, record_access=False) -> zero AccessEvents."""
    store = AmplifierStore(record_access=True)
    ref = store.write_cell(b"payload")

    store.get_cell(ref, record_access=False)

    assert len(_access_events(store)) == 0


def test_get_cell_explicit_true_forces_access_event() -> None:
    """Instance default False, get_cell(ref, record_access=True) -> 1 AccessEvent."""
    store = AmplifierStore(record_access=False)
    ref = store.write_cell(b"payload")

    store.get_cell(ref, record_access=True)

    events = _access_events(store)
    assert len(events) == 1
    assert events[0].cell_ref == ref


def test_get_cell_none_defers_to_instance_default_true() -> None:
    """Instance default True, get_cell(ref) with no per-call override -> 1 AccessEvent."""
    store = AmplifierStore(record_access=True)
    ref = store.write_cell(b"payload")

    store.get_cell(ref)

    events = _access_events(store)
    assert len(events) == 1
    assert events[0].cell_ref == ref


def test_get_cell_none_defers_to_instance_default_false() -> None:
    """Instance default False, get_cell(ref) with no per-call override -> zero AccessEvents."""
    store = AmplifierStore(record_access=False)
    ref = store.write_cell(b"payload")

    store.get_cell(ref)

    assert len(_access_events(store)) == 0


def test_get_cell_still_returns_cell_when_access_suppressed() -> None:
    """get_cell returns the Cell even when access recording is suppressed."""
    store = AmplifierStore(record_access=False)
    ref = store.write_cell(b"payload")

    cell = store.get_cell(ref)

    assert cell.payload == b"payload"
    assert len(_access_events(store)) == 0


# ---------------------------------------------------------------------------
# Task 10: tri-state record_access forwarded through regenerate()
# ---------------------------------------------------------------------------


def test_regenerate_explicit_false_suppresses() -> None:
    """Instance default True, regenerate(ref, record_access=False) -> zero AccessEvents."""
    store = AmplifierStore(record_access=True)
    ref = store.write_cell(b"payload")

    store.regenerate(ref, record_access=False)

    assert len(_access_events(store)) == 0


def test_regenerate_none_defers_to_instance_default_false() -> None:
    """Instance default False, regenerate(ref) with no per-call override -> zero AccessEvents."""
    store = AmplifierStore(record_access=False)
    ref = store.write_cell(b"payload")

    store.regenerate(ref)

    assert len(_access_events(store)) == 0


def test_regenerate_default_records_when_instance_default_true() -> None:
    """Instance default True, regenerate(ref) -> exactly 1 AccessEvent."""
    store = AmplifierStore(record_access=True)
    ref = store.write_cell(b"payload")

    store.regenerate(ref)

    events = _access_events(store)
    assert len(events) == 1
    assert events[0].cell_ref == ref
