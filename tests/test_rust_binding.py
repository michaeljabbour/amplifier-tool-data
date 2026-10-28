"""Smoke test and marshaling tests for the PyO3 extension module.

Task 9: Python↔Rust marshaling helpers.
- _rust._roundtrip_event(ev) → an equal Python Event object (tests both directions)
- _rust._fold(events, cell_ref) → a Python Cell (tests Rust fold + Rust Cell → Python Cell)

Task 10: RustKernel pyclass.
- RustKernel wraps InMemoryKernel (Rust) with parallel py_events Vec<Py<PyAny>>
  so identity semantics are preserved: resolve() returns the ORIGINAL Python
  event objects, not reconstructions.
"""

import pytest

from amplifier_data import _rust  # type: ignore[import]
from amplifier_data.kernel import NotFound, StorageKernel
from amplifier_data.models import (
    AccessEvent,
    Cell,
    CellWriteEvent,
    IndexCellEvent,
    InterpreterOutputEvent,
    RelationshipEvent,
    ToolResultEvent,
)
from amplifier_data.replay import MissingDefiningEvent
from amplifier_data.store import AmplifierStore

# RustKernel is implemented in Task 10; use getattr so a missing attribute
# gives a failing test rather than a collection-time ImportError that would
# break all Task 9 tests.
RustKernel = getattr(_rust, "RustKernel", None)


# ─── Existing smoke test ──────────────────────────────────────────────────────


def test_extension_imports() -> None:
    """amplifier_data._rust must import successfully after maturin develop."""
    assert _rust is not None
    assert _rust.__built() is True


# ─── Round-trip: Python Event → Rust Event → Python Event ───────────────────
#
# _roundtrip_event converts a Python Event subclass to a Rust Event then back
# to a Python Event.  The result must compare == to the original.


def test_roundtrip_cell_write_event_no_interpreters() -> None:
    ev = CellWriteEvent(payload=b"hello", interpreters=())
    assert _rust._roundtrip_event(ev) == ev


def test_roundtrip_cell_write_event_with_interpreters() -> None:
    ev = CellWriteEvent(payload=b"data", interpreters=("a" * 64,))
    assert _rust._roundtrip_event(ev) == ev


def test_roundtrip_relationship_event() -> None:
    ev = RelationshipEvent(from_ref="a" * 64, to_ref="b" * 64, type="uses")
    assert _rust._roundtrip_event(ev) == ev


def test_roundtrip_tool_result_event() -> None:
    ev = ToolResultEvent(output=b"tool output")
    assert _rust._roundtrip_event(ev) == ev


def test_roundtrip_interpreter_output_event() -> None:
    ev = InterpreterOutputEvent(
        cell_ref="c" * 64,
        model="gpt-4o",
        input_hash="d" * 64,
        tool_results_consumed=("e" * 64,),
        output=b"result",
    )
    assert _rust._roundtrip_event(ev) == ev


def test_roundtrip_interpreter_output_event_empty_tools() -> None:
    ev = InterpreterOutputEvent(
        cell_ref="c" * 64,
        model="gpt-4o",
        input_hash="d" * 64,
        tool_results_consumed=(),
        output=b"derived",
    )
    assert _rust._roundtrip_event(ev) == ev


def test_roundtrip_access_event() -> None:
    ev = AccessEvent(cell_ref="f" * 64)
    assert _rust._roundtrip_event(ev) == ev


def test_roundtrip_index_cell_event() -> None:
    ev = IndexCellEvent(
        cell_ref="g" * 64,
        access_summary=b"summary",
        output=b"index",
    )
    assert _rust._roundtrip_event(ev) == ev


def test_roundtrip_rejects_non_event_string() -> None:
    """Unknown types raise ValueError mentioning 'Event'."""
    with pytest.raises(ValueError, match="Event"):
        _rust._roundtrip_event("not an event")


def test_roundtrip_rejects_plain_dict() -> None:
    """A dict is not an Event subclass."""
    with pytest.raises(ValueError):
        _rust._roundtrip_event({"payload": b"x"})


def test_roundtrip_rejects_none() -> None:
    """None is not an Event."""
    with pytest.raises(ValueError):
        _rust._roundtrip_event(None)


# ─── Fold: Python Events → Rust regenerate → Python Cell ────────────────────
#
# _fold(events, cell_ref) runs the Rust regenerate() fold and returns a Python Cell.
# The provenance tuple must contain the EXACT SAME Python objects from the input.


def test_fold_single_cell_write_payload() -> None:
    ev = CellWriteEvent(payload=b"hello", interpreters=())
    cell_ref = ev.cell_ref()
    result = _rust._fold([ev], cell_ref)
    assert isinstance(result, Cell)
    assert result.payload == b"hello"


def test_fold_single_cell_write_interpreters() -> None:
    ev = CellWriteEvent(payload=b"hello", interpreters=("a" * 64,))
    cell_ref = ev.cell_ref()
    result = _rust._fold([ev], cell_ref)
    assert result.interpreters == ("a" * 64,)


def test_fold_single_cell_write_provenance_length() -> None:
    ev = CellWriteEvent(payload=b"hello", interpreters=())
    cell_ref = ev.cell_ref()
    result = _rust._fold([ev], cell_ref)
    assert len(result.provenance) == 1


def test_fold_single_cell_write_provenance_value() -> None:
    ev = CellWriteEvent(payload=b"hello", interpreters=())
    cell_ref = ev.cell_ref()
    result = _rust._fold([ev], cell_ref)
    assert result.provenance[0] == ev


def test_fold_provenance_uses_original_python_objects() -> None:
    """Provenance must contain the exact same Python object (identity check)."""
    ev = CellWriteEvent(payload=b"data", interpreters=())
    cell_ref = ev.cell_ref()
    result = _rust._fold([ev], cell_ref)
    # 'is' tests object identity, not equality
    assert result.provenance[0] is ev


def test_fold_interpreter_output_payload() -> None:
    """InterpreterOutputEvent folds: payload = output field."""
    output = b"derived payload"
    model = "gpt-4o"
    input_hash = "0" * 64
    cell_ref = InterpreterOutputEvent.derive_ref(
        model=model,
        input_hash=input_hash,
        tool_results_consumed=(),
        output=output,
    )
    ev = InterpreterOutputEvent(
        cell_ref=cell_ref,
        model=model,
        input_hash=input_hash,
        tool_results_consumed=(),
        output=output,
    )
    result = _rust._fold([ev], cell_ref)
    assert result.payload == output


def test_fold_interpreter_output_provenance() -> None:
    output = b"derived payload"
    model = "gpt-4o"
    input_hash = "0" * 64
    cell_ref = InterpreterOutputEvent.derive_ref(
        model=model,
        input_hash=input_hash,
        tool_results_consumed=(),
        output=output,
    )
    ev = InterpreterOutputEvent(
        cell_ref=cell_ref,
        model=model,
        input_hash=input_hash,
        tool_results_consumed=(),
        output=output,
    )
    result = _rust._fold([ev], cell_ref)
    assert result.provenance == (ev,)


def test_fold_relationship_included_in_from_ref_provenance() -> None:
    """RelationshipEvent is included in the from_ref cell's provenance."""
    cw = CellWriteEvent(payload=b"src", interpreters=())
    from_ref = cw.cell_ref()
    to_ref = "b" * 64
    rel = RelationshipEvent(from_ref=from_ref, to_ref=to_ref, type="uses")
    result = _rust._fold([cw, rel], from_ref)
    assert result.payload == b"src"
    assert len(result.provenance) == 2
    assert result.provenance[0] is cw
    assert result.provenance[1] is rel


def test_fold_relationship_not_in_to_ref_provenance() -> None:
    """RelationshipEvent does NOT appear in the to_ref cell's provenance."""
    target_cw = CellWriteEvent(payload=b"target", interpreters=())
    to_ref = target_cw.cell_ref()
    source_cw = CellWriteEvent(payload=b"src", interpreters=())
    from_ref = source_cw.cell_ref()
    rel = RelationshipEvent(from_ref=from_ref, to_ref=to_ref, type="uses")
    # Fold for to_ref should only see target_cw
    result = _rust._fold([target_cw, source_cw, rel], to_ref)
    assert result.provenance == (target_cw,)


def test_fold_access_event_not_in_provenance() -> None:
    """AccessEvent is provenance-neutral — never included in provenance."""
    cw = CellWriteEvent(payload=b"data", interpreters=())
    cell_ref = cw.cell_ref()
    access = AccessEvent(cell_ref=cell_ref)
    # AccessEvent has target_ref == None, so it's excluded
    result = _rust._fold([cw, access], cell_ref)
    assert result.provenance == (cw,)


# ─── Exception mapping ────────────────────────────────────────────────────────


def test_fold_empty_log_raises_missing_defining_event() -> None:
    """Empty log raises MissingDefiningEvent (not NotFound, not RuntimeError)."""
    fake_ref = "0" * 64
    with pytest.raises(MissingDefiningEvent):
        _rust._fold([], fake_ref)


def test_fold_missing_defining_event_carries_cell_ref() -> None:
    """The raised exception has the cell_ref attribute."""
    fake_ref = "0" * 64
    with pytest.raises(MissingDefiningEvent) as exc_info:
        _rust._fold([], fake_ref)
    assert exc_info.value.cell_ref == fake_ref


def test_fold_access_only_raises_missing_defining_event() -> None:
    """AccessEvent alone doesn't define a cell — raises MissingDefiningEvent."""
    cell_ref = "0" * 64
    ev = AccessEvent(cell_ref=cell_ref)
    with pytest.raises(MissingDefiningEvent):
        _rust._fold([ev], cell_ref)


def test_fold_missing_defining_event_is_correct_python_type() -> None:
    """The exception is exactly amplifier_data.replay.MissingDefiningEvent."""
    from amplifier_data.replay import MissingDefiningEvent as PyMDE

    fake_ref = "1" * 64
    with pytest.raises(PyMDE) as exc_info:
        _rust._fold([], fake_ref)
    # Must be the Python class, not a RuntimeError wrapper
    assert type(exc_info.value).__name__ == "MissingDefiningEvent"
    assert exc_info.value.cell_ref == fake_ref


# ─── RustKernel: Task 10 ──────────────────────────────────────────────────────
#
# RustKernel wraps InMemoryKernel (Rust) with a parallel py_events Vec<Py<PyAny>>
# so that identity semantics are preserved: resolve() returns the ORIGINAL Python
# event objects, not reconstructions.
#
# Tests exercise via AmplifierStore(RustKernel()) to validate the full contract.
#
# RustKernel is imported as `getattr(_rust, "RustKernel", None)` at the top of
# this module so that a missing attribute gives a failing test (not a
# collection-time ImportError that would break Task 9 tests).


def _require_rust_kernel() -> type:
    """Return RustKernel or fail the test if not yet implemented."""
    if RustKernel is None:
        pytest.fail(
            "RustKernel not yet implemented in amplifier_data._rust; "
            "implement the #[pyclass] RustKernel in crates/amplifier-data-py/src/lib.rs"
        )
    return RustKernel  # type: ignore[return-value]


def test_rust_kernel_satisfies_storage_kernel_protocol() -> None:
    """isinstance(RustKernel(), StorageKernel) must be True.

    StorageKernel is @runtime_checkable; isinstance only checks that all
    five protocol methods exist on the object.
    """
    K = _require_rust_kernel()
    assert isinstance(K(), StorageKernel)


def test_rust_kernel_write_cell_payload_roundtrips() -> None:
    """AmplifierStore(RustKernel()) write_cell + get_cell round-trips payload."""
    K = _require_rust_kernel()
    store = AmplifierStore(K())
    ref = store.write_cell(b"hello")
    cell = store.get_cell(ref)
    assert cell.payload == b"hello"


def test_rust_kernel_resolve_unknown_raises_not_found() -> None:
    """resolve(unknown_ref) raises amplifier_data.kernel.NotFound."""
    K = _require_rust_kernel()
    k = K()
    with pytest.raises(NotFound):
        k.resolve("0" * 64)


def test_rust_kernel_resolve_tool_result_is_original_object() -> None:
    """resolve(tool_hash) returns the exact Python object passed to append_batch.

    Identity is preserved via the parallel py_events Vec<Py<PyAny>> — no
    reconstruction from Rust fields.
    """
    K = _require_rust_kernel()
    k = K()
    ev = ToolResultEvent(output=b"tool output")
    k.append_batch([ev])
    tool_hash = ev.hash
    result = k.resolve(tool_hash)
    # 'is' tests object identity, not equality
    assert result is ev


def test_rust_kernel_all_events_returns_original_python_objects() -> None:
    """all_events() yields the exact Python event objects in insertion order.

    Each entry must be the same object (identity) as was passed to append_batch,
    and must satisfy isinstance for the concrete event subclass.
    """
    K = _require_rust_kernel()
    k = K()
    cw = CellWriteEvent(payload=b"data", interpreters=())
    tr = ToolResultEvent(output=b"tool")
    k.append_batch([cw, tr])
    events = k.all_events()
    assert len(events) == 2
    pos0, ev0 = events[0]
    pos1, ev1 = events[1]
    assert pos0 == 0
    assert pos1 == 1
    # Identity checks — must be the exact same Python objects
    assert ev0 is cw
    assert ev1 is tr
    # isinstance checks per variant
    assert isinstance(ev0, CellWriteEvent)
    assert isinstance(ev1, ToolResultEvent)


def test_rust_kernel_relate_edge_appears_in_source_provenance() -> None:
    """An amending relate(...) makes the edge appear in the source cell's provenance.

    Uses AmplifierStore(RustKernel()) to exercise the full write → relate → get_cell
    path.  The returned Cell's provenance must contain a RelationshipEvent.
    """
    K = _require_rust_kernel()
    store = AmplifierStore(K())
    from_ref = store.write_cell(b"source")
    to_ref = store.write_cell(b"target")
    store.relate(from_ref, to_ref, "uses")
    cell = store.get_cell(from_ref)
    # Provenance must include the write event + the relationship event
    assert len(cell.provenance) == 2
    assert any(isinstance(p, RelationshipEvent) for p in cell.provenance)


# ── RustKernel: Task 11 (observers) ──────────────────────────────────────────
#
# Mirrors the observer test-cases in tests/test_kernel.py but against
# RustKernel instead of InMemoryKernel.
#
# subscribe(observer) returns a Subscription-like object with unsubscribe().
# observer_errors is a readable list-like attribute.


def test_rust_kernel_subscribe_notifies_post_commit_in_seqpos_order() -> None:
    """Observers are called post-commit in ascending SeqPos order."""
    K = _require_rust_kernel()
    k = K()
    seen: list[tuple[int, object]] = []
    k.subscribe(lambda pos, ev: seen.append((pos, ev)))
    a = CellWriteEvent(payload=b"a")
    b = CellWriteEvent(payload=b"b")
    k.append_batch([a, b])
    assert [pos for pos, _ in seen] == [0, 1]
    assert [ev for _, ev in seen] == [a, b]


def test_rust_kernel_observer_failure_does_not_roll_back_append() -> None:
    """Observer raising an exception is recorded in observer_errors; commit stands."""
    K = _require_rust_kernel()
    k = K()

    def boom(pos: int, ev: object) -> None:
        raise RuntimeError("observer kaboom")

    k.subscribe(boom)
    positions = k.append_batch([CellWriteEvent(payload=b"a")])
    assert positions == [0]  # append committed despite observer error
    assert len(k.all_events()) == 1  # the log grew (log is truth)
    assert len(k.observer_errors) == 1  # error recorded, NOT silently swallowed


def test_rust_kernel_unsubscribe_detaches_observer() -> None:
    """Calling unsubscribe() stops all future notifications to that observer."""
    K = _require_rust_kernel()
    k = K()
    seen: list[int] = []
    sub = k.subscribe(lambda pos, ev: seen.append(pos))
    k.append_batch([CellWriteEvent(payload=b"a")])
    sub.unsubscribe()
    k.append_batch([CellWriteEvent(payload=b"b")])
    assert seen == [0]  # only the first append was observed


def test_rust_kernel_unsubscribe_is_idempotent() -> None:
    """Calling unsubscribe() multiple times must not raise."""
    K = _require_rust_kernel()
    k = K()
    seen: list[int] = []
    sub = k.subscribe(lambda pos, ev: seen.append(pos))
    sub.unsubscribe()
    sub.unsubscribe()  # must not raise
    k.append_batch([CellWriteEvent(payload=b"a")])
    assert seen == []  # observer was already detached; no notification


def test_rust_kernel_observer_can_unsubscribe_itself_without_skipping_others() -> None:
    """Re-entrant self-unsubscribe must not skip subsequent observers (snapshot semantics)."""
    K = _require_rust_kernel()
    k = K()
    seen: list[str] = []
    sub_a: object = None

    def observer_a(pos: int, ev: object) -> None:
        seen.append("a")
        sub_a.unsubscribe()  # type: ignore[union-attr]  # mutate list mid-notification

    def observer_b(pos: int, ev: object) -> None:
        seen.append("b")

    sub_a = k.subscribe(observer_a)
    k.subscribe(observer_b)
    k.append_batch([CellWriteEvent(payload=b"x")])
    assert seen == ["a", "b"]  # b must NOT be skipped despite a unsubscribing itself


# ── Phase 2: switchover wiring (Task 12) ─────────────────────────────────────
#
# AmplifierStore() must default to the Rust kernel when RUST_AVAILABLE is True.
# The pure-Python InMemoryKernel must remain importable and usable as a fallback.


def test_store_defaults_to_rust() -> None:
    """AmplifierStore() defaults to the Rust kernel when RUST_AVAILABLE is True.

    Phase 2 acceptance test:
    - RUST_AVAILABLE is exported from amplifier_data and is True
    - AmplifierStore() with no kernel argument uses the Rust kernel
      (isinstance check against _rust.RustKernel)
    - The pure-Python InMemoryKernel is still importable and usable:
      AmplifierStore(InMemoryKernel()) constructs without error
    """
    from amplifier_data import RUST_AVAILABLE

    assert RUST_AVAILABLE is True

    # AmplifierStore() with no kernel arg must use the Rust kernel
    default_store = AmplifierStore()
    from amplifier_data._rust import RustKernel  # type: ignore[import]

    assert isinstance(default_store.kernel, RustKernel), (
        f"Expected AmplifierStore() to use RustKernel, "
        f"but got {type(default_store.kernel)!r}. "
        "Check that store.py _default_kernel() returns RustKernel() when RUST_AVAILABLE."
    )

    # The pure-Python InMemoryKernel must still be importable and usable
    from amplifier_data.kernel import InMemoryKernel

    py_store = AmplifierStore(InMemoryKernel())
    assert isinstance(py_store.kernel, InMemoryKernel)
