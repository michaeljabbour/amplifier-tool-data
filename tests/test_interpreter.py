"""Tests for Interpreter (Plan 2: meaning-on-spawn with injectable spawn callable).

TDD RED pass — all tests reference amplifier_data.interpreter which does not yet exist.
"""

from __future__ import annotations

from amplifier_data.interpreter import Interpreter, InterpretFailure
from amplifier_data.models import InterpreterOutputEvent, ToolResultEvent
from amplifier_data.store import AmplifierStore

# ---------------------------------------------------------------------------
# Test doubles (deterministic fakes — no real model ever loaded here)
# ---------------------------------------------------------------------------


class DeterministicFakeSpawn:
    """Fake spawn callable for tests: returns fixed outputs, tracks calls."""

    def __init__(
        self,
        model: str = "test-model",
        output: bytes = b"test output",
        tool_results: tuple[bytes, ...] = (),
    ) -> None:
        self.model = model
        self.output = output
        self.tool_results = tool_results
        self.calls: list[tuple[str, bytes]] = []

    def __call__(self, cell_ref: str, payload: bytes) -> tuple[str, bytes, tuple[bytes, ...]]:
        self.calls.append((cell_ref, payload))
        return (self.model, self.output, self.tool_results)


class FailingSpawn:
    """Spawn that always raises the configured exception."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc: Exception = exc or RuntimeError("spawn failed")

    def __call__(self, cell_ref: str, payload: bytes) -> tuple[str, bytes, tuple[bytes, ...]]:
        raise self._exc


class NeverCallSpawn:
    """Spawn that must never be called — proves F1 (replay skips spawn)."""

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("spawn was invoked — F1 violated")


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_interpret_returns_derived_hash() -> None:
    """interpret returns a Hash (str) identifying the derived cell."""
    store = AmplifierStore()
    cell_ref = store.write_cell(b"source payload")
    spawn = DeterministicFakeSpawn(model="m", output=b"output")
    interpreter = Interpreter(store, spawn)

    result = interpreter.interpret(cell_ref)

    assert isinstance(result, str), "expected a Hash (str), got failure"


def test_interpret_calls_spawn_with_cell_ref_and_payload() -> None:
    """spawn receives the exact cell_ref and source payload."""
    store = AmplifierStore()
    cell_ref = store.write_cell(b"source payload")
    spawn = DeterministicFakeSpawn(model="m", output=b"out")
    interpreter = Interpreter(store, spawn)

    interpreter.interpret(cell_ref)

    assert len(spawn.calls) == 1
    assert spawn.calls[0] == (cell_ref, b"source payload")


def test_interpret_records_exactly_one_atomic_batch() -> None:
    """write_interpretation appends [ToolResultEvent..., InterpreterOutputEvent] atomically.

    Event count breakdown (D4: AccessEvents added by public reads):
      1 CellWriteEvent (write_cell)
    + 1 AccessEvent    (get_cell for source cell inside Interpreter.interpret)
    + 2 ToolResultEvents (write_interpretation batch)
    + 1 InterpreterOutputEvent (write_interpretation batch)
    + 1 AccessEvent    (regenerate → get_cell for derived cell inside Interpreter.interpret)
    = 6 total events
    """
    store = AmplifierStore()
    cell_ref = store.write_cell(b"source")
    spawn = DeterministicFakeSpawn(
        model="m",
        output=b"out",
        tool_results=(b"r1", b"r2"),
    )
    interpreter = Interpreter(store, spawn)

    interpreter.interpret(cell_ref)

    kernel = store.kernel
    assert hasattr(kernel, "all_events")
    # 1 (source cell) + 1 (access: read source) + 2 (tool results) + 1 (interp output)
    # + 1 (access: read derived via regenerate) = 6
    assert len(kernel.all_events()) == 6  # type: ignore[attr-defined]


def test_interpret_derived_cell_has_correct_payload() -> None:
    """The derived Cell's payload equals spawn's output bytes."""
    store = AmplifierStore()
    cell_ref = store.write_cell(b"source")
    spawn = DeterministicFakeSpawn(model="m", output=b"interpreter conclusion")
    interpreter = Interpreter(store, spawn)

    derived_ref = interpreter.interpret(cell_ref)

    assert isinstance(derived_ref, str)
    cell = store.get_cell(derived_ref)
    assert cell.payload == b"interpreter conclusion"


def test_interpret_derived_cell_provenance_is_single_interpreter_output_event() -> None:
    """Derived Cell's provenance is exactly one InterpreterOutputEvent (F1)."""
    store = AmplifierStore()
    cell_ref = store.write_cell(b"source")
    spawn = DeterministicFakeSpawn(
        model="test-model",
        output=b"output",
        tool_results=(b"tool result",),
    )
    interpreter = Interpreter(store, spawn)

    derived_ref = interpreter.interpret(cell_ref)

    assert isinstance(derived_ref, str)
    cell = store.get_cell(derived_ref)
    assert len(cell.provenance) == 1
    interp_event = cell.provenance[0]
    assert isinstance(interp_event, InterpreterOutputEvent)
    assert interp_event.model == "test-model"


def test_interpret_tool_results_referenced_by_hash() -> None:
    """Consumed tool results are referenced by hash and resolvable, not inlined."""
    store = AmplifierStore()
    cell_ref = store.write_cell(b"source")
    spawn = DeterministicFakeSpawn(
        model="m",
        output=b"output",
        tool_results=(b"tool result A", b"tool result B"),
    )
    interpreter = Interpreter(store, spawn)

    derived_ref = interpreter.interpret(cell_ref)

    assert isinstance(derived_ref, str)
    cell = store.get_cell(derived_ref)
    interp_event = cell.provenance[0]
    assert isinstance(interp_event, InterpreterOutputEvent)
    assert len(interp_event.tool_results_consumed) == 2
    for tool_hash in interp_event.tool_results_consumed:
        resolved = store.kernel.resolve(tool_hash)
        assert isinstance(resolved, ToolResultEvent)


def test_interpret_no_tool_results() -> None:
    """interpret works with zero tool results."""
    store = AmplifierStore()
    cell_ref = store.write_cell(b"source")
    spawn = DeterministicFakeSpawn(model="m", output=b"output", tool_results=())
    interpreter = Interpreter(store, spawn)

    derived_ref = interpreter.interpret(cell_ref)

    assert isinstance(derived_ref, str)
    cell = store.get_cell(derived_ref)
    assert cell.payload == b"output"
    interp_event = cell.provenance[0]
    assert isinstance(interp_event, InterpreterOutputEvent)
    assert interp_event.tool_results_consumed == ()


# ---------------------------------------------------------------------------
# F1 contract: replay folds the log, never re-invokes spawn
# ---------------------------------------------------------------------------


def test_f1_regenerate_does_not_invoke_spawn_again() -> None:
    """After interpret, regenerate re-folds from the log without calling spawn."""
    store = AmplifierStore()
    cell_ref = store.write_cell(b"source")
    spawn = DeterministicFakeSpawn(model="m", output=b"recorded output")
    interpreter = Interpreter(store, spawn)

    derived_ref = interpreter.interpret(cell_ref)
    assert isinstance(derived_ref, str)
    calls_after_interpret = len(spawn.calls)

    # Evict materialized cache; re-fold from log only — spawn must not be called
    cell = store.regenerate(derived_ref)

    assert cell.payload == b"recorded output"
    assert len(spawn.calls) == calls_after_interpret  # no additional call


def test_f1_replay_payload_equals_recorded_interpreter_output() -> None:
    """After cache eviction, folded payload equals the verbatim recorded output."""
    store = AmplifierStore()
    cell_ref = store.write_cell(b"source")
    spawn = DeterministicFakeSpawn(model="m", output=b"verbatim conclusion")
    interpreter = Interpreter(store, spawn)

    derived_ref = interpreter.interpret(cell_ref)
    assert isinstance(derived_ref, str)

    store.drop_materialized(derived_ref)
    cell = store.get_cell(derived_ref)
    assert cell.payload == b"verbatim conclusion"


# ---------------------------------------------------------------------------
# Spawn failure: atomic — nothing appended, typed failure returned
# ---------------------------------------------------------------------------


def test_spawn_failure_appends_nothing() -> None:
    """If spawn raises, no interpretation batch is appended (ToolResult + InterpreterOutput).

    Design Decision D4: get_cell inside Interpreter.interpret reads the source cell,
    so ONE AccessEvent IS appended before spawn is even called.  The guarantee is that
    the interpretation batch (ToolResultEvent... + InterpreterOutputEvent) is NOT
    appended on failure — not that the log is byte-for-byte identical.
    """
    from amplifier_data.models import AccessEvent, InterpreterOutputEvent, ToolResultEvent

    store = AmplifierStore()
    cell_ref = store.write_cell(b"source")
    spawn = FailingSpawn(RuntimeError("model unavailable"))
    interpreter = Interpreter(store, spawn)

    kernel = store.kernel
    assert hasattr(kernel, "all_events")

    result = interpreter.interpret(cell_ref)

    assert isinstance(result, InterpretFailure)

    # D4: exactly ONE AccessEvent is appended (reading the source cell before spawn).
    all_events = kernel.all_events()  # type: ignore[attr-defined]
    access_events = [ev for _, ev in all_events if isinstance(ev, AccessEvent)]
    assert len(access_events) == 1
    assert access_events[0].cell_ref == cell_ref

    # Interpretation batch NOT appended on failure.
    assert not any(isinstance(ev, ToolResultEvent) for _, ev in all_events)
    assert not any(isinstance(ev, InterpreterOutputEvent) for _, ev in all_events)


def test_spawn_failure_returns_typed_interpret_failure() -> None:
    """InterpretFailure carries the original exception and a reason string."""
    store = AmplifierStore()
    cell_ref = store.write_cell(b"x")
    exc = ValueError("spawn exploded")
    spawn = FailingSpawn(exc)
    interpreter = Interpreter(store, spawn)

    result = interpreter.interpret(cell_ref)

    assert isinstance(result, InterpretFailure)
    assert result.exception is exc
    assert "spawn exploded" in result.reason


def test_spawn_failure_reason_is_nonempty_string() -> None:
    """InterpretFailure.reason is a non-empty string."""
    store = AmplifierStore()
    cell_ref = store.write_cell(b"x")
    spawn = FailingSpawn(RuntimeError("oops"))
    interpreter = Interpreter(store, spawn)

    result = interpreter.interpret(cell_ref)

    assert isinstance(result, InterpretFailure)
    assert isinstance(result.reason, str)
    assert result.reason  # non-empty


# ---------------------------------------------------------------------------
# Injectability: tests use deterministic fakes; interpreter never hard-codes model
# ---------------------------------------------------------------------------


def test_spawn_is_injectable_different_fakes_produce_different_outputs() -> None:
    """Two interpreters with different fake spawns produce different derived cells."""
    store = AmplifierStore()
    cell_ref = store.write_cell(b"source")

    spawn_a = DeterministicFakeSpawn(model="model-a", output=b"output from a")
    spawn_b = DeterministicFakeSpawn(model="model-b", output=b"output from b")

    ref_a = Interpreter(store, spawn_a).interpret(cell_ref)
    ref_b = Interpreter(store, spawn_b).interpret(cell_ref)

    assert isinstance(ref_a, str)
    assert isinstance(ref_b, str)
    assert ref_a != ref_b

    assert store.get_cell(ref_a).payload == b"output from a"
    assert store.get_cell(ref_b).payload == b"output from b"


def test_interpreter_does_not_import_live_model() -> None:
    """The Interpreter class is constructible without any live model dependency.

    This is a structural test: if Interpreter imported a live model at module
    level, the import itself would fail in a test environment with no API keys.
    Constructing Interpreter with a fake spawn proves the seam is respected.
    """
    store = AmplifierStore()
    spawn = DeterministicFakeSpawn()

    # Should not raise regardless of environment
    interpreter = Interpreter(store, spawn)
    assert interpreter is not None
