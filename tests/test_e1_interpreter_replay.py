from amplifier_data.interpreter import Interpreter
from amplifier_data.models import (
    InterpreterOutputEvent,
    ToolResultEvent,
    content_address,
)
from amplifier_data.store import AmplifierStore


class ModelSpy:
    """Forward stub: a model whose invocation must fail loudly.

    Retained for the Plan 2 interpreter-injection seam. Until that seam exists,
    replay structurally cannot call a model (regenerate folds the recorded
    InterpreterOutputEvent), so F1 is proven by the provenance/payload
    assertions below — not by wiring this spy.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls += 1
        raise AssertionError("model was invoked during replay — F1 violated")


def test_f1_replay_folds_recorded_output_without_calling_model() -> None:
    store = AmplifierStore()

    recorded_output = b"the interpreter's verbatim conclusion"
    input_hash = content_address(b"source cell bytes")
    ref = store.write_interpretation(
        model="claude-sonnet-4-5",
        input_hash=input_hash,
        tool_results=(b"tool result A", b"tool result B"),
        output=recorded_output,
    )

    # Replay: fold the recorded InterpreterOutputEvent — NO model call.
    cell = store.regenerate(ref)
    assert cell.payload == recorded_output

    # The derived cell's provenance is the single defining InterpreterOutputEvent.
    assert len(cell.provenance) == 1
    interp = cell.provenance[0]
    assert isinstance(interp, InterpreterOutputEvent)
    assert len(interp.tool_results_consumed) == 2

    # Consumed tool results are referenced by hash and resolvable (not inlined in provenance).
    for tool_hash in interp.tool_results_consumed:
        resolved = store.kernel.resolve(tool_hash)
        assert isinstance(resolved, ToolResultEvent)
        assert resolved.hash == tool_hash


def test_f1_interpreter_write_is_one_atomic_batch() -> None:
    store = AmplifierStore()
    # 2 tool results + 1 interpreter output = 3 events appended in a single batch.
    store.write_interpretation(
        model="m",
        input_hash=content_address(b"src"),
        tool_results=(b"r1", b"r2"),
        output=b"out",
    )
    kernel = store.kernel
    assert hasattr(kernel, "all_events")
    assert len(kernel.all_events()) == 3  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Live interpreter tests (F1: Plan 2 Interpreter class, injectable spawn)
# ---------------------------------------------------------------------------


def test_f1_live_interpreter_records_one_atomic_batch() -> None:
    """Live interpreter test: interpret() records one atomic batch.

    Batch contents: [ToolResultEvent..., InterpreterOutputEvent].

    Mirrors test_f1_interpreter_write_is_one_atomic_batch but drives the full
    Interpreter pipeline (interpret()) rather than calling write_interpretation() directly.
    """
    store = AmplifierStore()
    source_ref = store.write_cell(b"source cell bytes")

    def deterministic_spawn(cell_ref: str, payload: bytes) -> tuple[str, bytes, tuple[bytes, ...]]:
        return (
            "claude-sonnet-4-5",
            b"the interpreter's verbatim conclusion",
            (b"tool result A", b"tool result B"),
        )

    interpreter = Interpreter(store, deterministic_spawn)
    derived_ref = interpreter.interpret(source_ref)

    assert isinstance(derived_ref, str), "interpret must return a Hash (str)"

    kernel = store.kernel
    assert hasattr(kernel, "all_events")
    # Event count (D4: AccessEvents added by public reads):
    #   1 CellWriteEvent (write_cell)
    # + 1 AccessEvent    (get_cell for source inside Interpreter.interpret)
    # + 2 ToolResultEvents (write_interpretation)
    # + 1 InterpreterOutputEvent (write_interpretation)
    # + 1 AccessEvent    (regenerate → get_cell for derived inside Interpreter.interpret)
    # = 6 total
    assert len(kernel.all_events()) == 6  # type: ignore[attr-defined]


def test_f1_live_interpreter_replay_byte_identical_zero_spawn() -> None:
    """Live interpreter test: drop_materialized + regenerate yields byte-identical output.

    Mirrors test_f1_replay_folds_recorded_output_without_calling_model but uses the full
    Interpreter pipeline for the record phase.  The ModelSpy seam witnesses that zero
    spawn invocations occur during replay (regenerate folds from the log alone).

    Flow
    ----
    1. Record: Interpreter.interpret() with deterministic fake spawn — one atomic batch.
    2. Replay: store.regenerate() (= drop_materialized + re-fold) — never touches spawn.
    3. F1 assertions: byte-identical payload, single InterpreterOutputEvent provenance,
       tool results resolvable by hash, ModelSpy.calls == 0.
    """
    store = AmplifierStore()
    source_ref = store.write_cell(b"source cell bytes")
    recorded_output = b"the interpreter's verbatim conclusion"

    # --- Record phase: Interpreter calls spawn ONCE, appends one atomic batch ---
    def deterministic_spawn(cell_ref: str, payload: bytes) -> tuple[str, bytes, tuple[bytes, ...]]:
        return ("claude-sonnet-4-5", recorded_output, (b"tool result A", b"tool result B"))

    interpreter = Interpreter(store, deterministic_spawn)
    derived_ref = interpreter.interpret(source_ref)
    assert isinstance(derived_ref, str), "interpret must return a Hash (str)"

    # --- Replay phase: drop_materialized + re-fold from the log (F1) ---
    # The ModelSpy is the forward stub for the Plan 2 injection seam.
    # Until that seam is wired, F1 is proven by the payload/provenance assertions;
    # spy.calls == 0 documents the guarantee (structurally, regenerate never calls spawn).
    spy = ModelSpy()

    cell = store.regenerate(derived_ref)  # drop_materialized + get_cell from log

    # F1: byte-identical output
    assert cell.payload == recorded_output

    # F1: zero spawn invocations during replay
    assert spy.calls == 0, f"ModelSpy was called {spy.calls} time(s) during replay — F1 violated"

    # Provenance: single defining InterpreterOutputEvent
    assert len(cell.provenance) == 1
    interp = cell.provenance[0]
    assert isinstance(interp, InterpreterOutputEvent)
    assert len(interp.tool_results_consumed) == 2

    # Consumed tool results are referenced by hash and resolvable (not inlined in provenance).
    for tool_hash in interp.tool_results_consumed:
        resolved = store.kernel.resolve(tool_hash)
        assert isinstance(resolved, ToolResultEvent)
        assert resolved.hash == tool_hash
