"""Tests for AmplifierStore AccessEvent appending, lens query accessors, and interpret().

Design Decision D4 (read-vs-fold boundary)
-------------------------------------------
* ONLY public store read methods (get_cell, regenerate) append AccessEvents.
* Lens internals and the index fold call kernel.all_events() directly, NOT
  get_cell, so folding the log produces NO access noise.
* The access record is therefore meaningful (real reads) and the static fold
  (E3) is deterministic (it never sees AccessEvents it generated itself).

Coverage
--------
* get_cell appends exactly one AccessEvent{cell_ref} per call.
* regenerate (which delegates to get_cell) also appends one AccessEvent.
* Multiple reads produce multiple AccessEvents — one per call.
* Lens folds (query_relational, query_grouped, query_index) generate ZERO
  AccessEvents: they pass self._kernel directly to the lens, which reads
  all_events() without touching get_cell.
* store.query_relational / query_grouped / query_index run the respective
  lens over self.kernel and return the expected QueryResult.
* store.interpret(cell_ref, spawn=...) constructs an Interpreter and calls
  interpret(cell_ref), returning the derived ref (or InterpretFailure).
"""

from __future__ import annotations

from amplifier_data.interpreter import InterpretFailure
from amplifier_data.models import AccessEvent, Hash
from amplifier_data.store import AmplifierStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _access_events(store: AmplifierStore) -> list[AccessEvent]:
    """Return all AccessEvents currently in the kernel log."""
    all_events = store.kernel.all_events()  # type: ignore[attr-defined]
    return [ev for _, ev in all_events if isinstance(ev, AccessEvent)]


def _fake_spawn(cell_ref: Hash, payload: bytes) -> tuple[str, bytes, tuple[bytes, ...]]:
    """Deterministic fake spawn: always succeeds with fixed output."""
    return ("fake-model", b"interpreted output", (b"tool result",))


# ---------------------------------------------------------------------------
# D4: get_cell appends exactly ONE AccessEvent per public read
# ---------------------------------------------------------------------------


def test_get_cell_appends_exactly_one_access_event() -> None:
    """Single get_cell call appends exactly one AccessEvent."""
    store = AmplifierStore()
    ref = store.write_cell(b"hello world")

    store.get_cell(ref)

    events = _access_events(store)
    assert len(events) == 1
    assert events[0].cell_ref == ref


def test_get_cell_access_event_cell_ref_matches_read_ref() -> None:
    """AccessEvent.cell_ref equals the ref that was read."""
    store = AmplifierStore()
    ref_a = store.write_cell(b"cell A")
    ref_b = store.write_cell(b"cell B")

    store.get_cell(ref_a)

    events = _access_events(store)
    assert len(events) == 1
    assert events[0].cell_ref == ref_a
    assert events[0].cell_ref != ref_b


def test_get_cell_two_calls_append_two_access_events() -> None:
    """Each get_cell call produces its own AccessEvent — one per call."""
    store = AmplifierStore()
    ref = store.write_cell(b"repeated")

    store.get_cell(ref)
    store.get_cell(ref)

    events = _access_events(store)
    assert len(events) == 2
    assert all(ae.cell_ref == ref for ae in events)


def test_get_cell_different_refs_each_produce_one_access_event() -> None:
    """Reads on different cells each append their own AccessEvent."""
    store = AmplifierStore()
    ref_a = store.write_cell(b"A")
    ref_b = store.write_cell(b"B")

    store.get_cell(ref_a)
    store.get_cell(ref_b)

    events = _access_events(store)
    assert len(events) == 2
    cell_refs = {ae.cell_ref for ae in events}
    assert ref_a in cell_refs
    assert ref_b in cell_refs


def test_write_cell_does_not_append_access_event() -> None:
    """write_cell is a write operation, not a read — no AccessEvent appended."""
    store = AmplifierStore()
    store.write_cell(b"no read yet")
    assert len(_access_events(store)) == 0


def test_no_access_events_before_any_get_cell() -> None:
    """Fresh store has zero AccessEvents in the kernel log."""
    store = AmplifierStore()
    assert len(_access_events(store)) == 0


# ---------------------------------------------------------------------------
# D4: regenerate (delegates to get_cell) also appends one AccessEvent
# ---------------------------------------------------------------------------


def test_regenerate_appends_access_event_via_get_cell() -> None:
    """regenerate() calls get_cell internally, so it also appends one AccessEvent."""
    store = AmplifierStore()
    ref = store.write_cell(b"regenerate me")

    store.regenerate(ref)

    events = _access_events(store)
    assert len(events) == 1
    assert events[0].cell_ref == ref


# ---------------------------------------------------------------------------
# D4: Lens folds via kernel.all_events() generate ZERO AccessEvents
# ---------------------------------------------------------------------------


def test_query_relational_generates_no_access_events() -> None:
    """Folding the relational lens does NOT append any AccessEvent."""
    store = AmplifierStore()
    store.write_cell(b"data 1")
    store.write_cell(b"data 2")

    store.query_relational()

    assert len(_access_events(store)) == 0


def test_query_grouped_generates_no_access_events() -> None:
    """Folding the grouped lens does NOT append any AccessEvent."""
    store = AmplifierStore()
    store.write_cell(b"data")

    store.query_grouped(group="cell_write")

    assert len(_access_events(store)) == 0


def test_query_index_generates_no_access_events() -> None:
    """Folding the index lens does NOT append any AccessEvent."""
    store = AmplifierStore()
    ref = store.write_cell(b"data")
    store.get_cell(ref)  # one real read to populate the index

    before = len(_access_events(store))
    store.query_index()  # fold should NOT add more AccessEvents
    after = len(_access_events(store))

    assert after == before


def test_lens_fold_after_reads_sees_correct_access_count() -> None:
    """Index lens fold reflects AccessEvents that were appended by real reads."""
    store = AmplifierStore()
    ref = store.write_cell(b"data")

    # Three real reads → three AccessEvents
    store.get_cell(ref)
    store.get_cell(ref)
    store.get_cell(ref)

    result = store.query_index()
    assert result.success
    # The index should report 3 accesses for this cell_ref
    assert result.output.get(ref) == 3


# ---------------------------------------------------------------------------
# Lens query accessor surface: query_relational
# ---------------------------------------------------------------------------


def test_query_relational_returns_successful_result() -> None:
    """query_relational() runs RelationalLens.query over self.kernel."""
    store = AmplifierStore()
    store.write_cell(b"payload bytes")

    result = store.query_relational()

    assert result.success
    assert len(result.output) == 1
    assert result.output[0]["payload"] == b"payload bytes"
    assert result.output[0]["event_type"] == "cell_write"


def test_query_relational_with_filter_kwarg() -> None:
    """query_relational accepts filter keyword argument."""
    store = AmplifierStore()
    store.write_cell(b"keep me")
    store.write_cell(b"drop me")

    result = store.query_relational(filter=lambda r: b"keep" in r["payload"])

    assert result.success
    assert len(result.output) == 1
    assert result.output[0]["payload"] == b"keep me"


def test_query_relational_with_project_kwarg() -> None:
    """query_relational accepts project keyword argument for column selection."""
    store = AmplifierStore()
    store.write_cell(b"hello")

    result = store.query_relational(project=["cell_ref", "payload"])

    assert result.success
    assert len(result.output) == 1
    assert set(result.output[0].keys()) == {"cell_ref", "payload"}


def test_query_relational_empty_store_returns_empty_output() -> None:
    """query_relational on an empty store returns an empty list."""
    store = AmplifierStore()
    result = store.query_relational()
    assert result.success
    assert result.output == []


# ---------------------------------------------------------------------------
# Lens query accessor surface: query_grouped
# ---------------------------------------------------------------------------


def test_query_grouped_returns_successful_result() -> None:
    """query_grouped() runs GroupedLens.query over self.kernel."""
    store = AmplifierStore()
    store.write_cell(b"payload bytes")

    result = store.query_grouped()

    assert result.success
    assert "cell_write" in result.output


def test_query_grouped_with_group_kwarg() -> None:
    """query_grouped accepts group keyword argument."""
    store = AmplifierStore()
    store.write_cell(b"data")

    result = store.query_grouped(group="cell_write")

    assert result.success
    assert "cell_write" in result.output


def test_query_grouped_empty_store_returns_empty_output() -> None:
    """query_grouped on an empty store returns an empty dict."""
    store = AmplifierStore()
    result = store.query_grouped()
    assert result.success
    assert result.output == {}


# ---------------------------------------------------------------------------
# Lens query accessor surface: query_index
# ---------------------------------------------------------------------------


def test_query_index_returns_successful_result() -> None:
    """query_index() runs IndexLens.query over self.kernel."""
    store = AmplifierStore()
    result = store.query_index()
    assert result.success


def test_query_index_empty_when_no_reads() -> None:
    """query_index on a store with no reads returns an empty counts dict."""
    store = AmplifierStore()
    store.write_cell(b"not read yet")

    result = store.query_index()

    assert result.success
    assert result.output == {}


def test_query_index_counts_single_read() -> None:
    """query_index counts one AccessEvent as one read."""
    store = AmplifierStore()
    ref = store.write_cell(b"data")
    store.get_cell(ref)

    result = store.query_index()

    assert result.success
    assert result.output == {ref: 1}


def test_query_index_with_cell_ref_kwarg_filters_to_one_cell() -> None:
    """query_index accepts cell_ref keyword argument to filter output."""
    store = AmplifierStore()
    ref_a = store.write_cell(b"A")
    ref_b = store.write_cell(b"B")
    store.get_cell(ref_a)
    store.get_cell(ref_b)
    store.get_cell(ref_b)

    result = store.query_index(cell_ref=ref_b)

    assert result.success
    # Only ref_b in the output, with count 2
    assert ref_b in result.output
    assert result.output[ref_b] == 2
    assert ref_a not in result.output


# ---------------------------------------------------------------------------
# interpret() convenience method
# ---------------------------------------------------------------------------


def test_interpret_convenience_returns_derived_hash() -> None:
    """store.interpret() returns the derived cell ref on success."""
    store = AmplifierStore()
    source_ref = store.write_cell(b"source bytes")

    derived_ref = store.interpret(source_ref, spawn=_fake_spawn)

    assert isinstance(derived_ref, str)
    # Derived cell must be resolvable
    cell = store.get_cell(derived_ref)  # type: ignore[arg-type]
    assert cell.payload == b"interpreted output"


def test_interpret_convenience_records_atomic_batch() -> None:
    """store.interpret() writes one atomic batch (tool results + output event)."""
    store = AmplifierStore()
    source_ref = store.write_cell(b"source bytes")

    before = len(store.kernel.all_events())  # type: ignore[attr-defined]
    store.interpret(source_ref, spawn=_fake_spawn)
    after = len(store.kernel.all_events())  # type: ignore[attr-defined]

    # 1 ToolResultEvent + 1 InterpreterOutputEvent = 2 events, plus the
    # AccessEvent from getting the source cell inside Interpreter.interpret()
    assert after > before


def test_interpret_convenience_failure_returns_interpret_failure() -> None:
    """store.interpret() returns InterpretFailure when spawn raises."""
    store = AmplifierStore()
    source_ref = store.write_cell(b"source bytes")

    def failing_spawn(cell_ref: Hash, payload: bytes) -> tuple[str, bytes, tuple[bytes, ...]]:
        raise RuntimeError("spawn failed")

    result = store.interpret(source_ref, spawn=failing_spawn)

    assert isinstance(result, InterpretFailure)
    assert "spawn failed" in result.reason


def test_interpret_convenience_uses_injected_spawn() -> None:
    """store.interpret() calls the injected spawn (not a hard-coded model)."""
    store = AmplifierStore()
    source_ref = store.write_cell(b"source")
    call_log: list[tuple[Hash, bytes]] = []

    def tracking_spawn(cell_ref: Hash, payload: bytes) -> tuple[str, bytes, tuple[bytes, ...]]:
        call_log.append((cell_ref, payload))
        return ("model", b"output", ())

    store.interpret(source_ref, spawn=tracking_spawn)

    assert len(call_log) == 1
    assert call_log[0][0] == source_ref
    assert call_log[0][1] == b"source"
