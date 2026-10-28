"""Meaning-on-spawn Interpreter: runs spawn, records interpretation atomically (Plan 2 / live-F1).

F1 guarantee
------------
The derived Cell materialises through the unchanged ``store.regenerate()`` which folds the
recorded ``InterpreterOutputEvent`` from the log.  Replay never re-invokes ``spawn``.

Spawn contract
--------------
``spawn(cell_ref: Hash, payload: bytes) -> tuple[str, bytes, tuple[bytes, ...]]``
                                           (model,   output, tool_results)

- Production: a thin wrapper over ``PreparedBundle.spawn`` that knows the configured model.
- Tests: a deterministic fake that returns fixed (model, output, tool_results).

The Interpreter class never imports or references a live model directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .models import Hash
from .store import AmplifierStore

# Type alias for the injected spawn callable.
# spawn(cell_ref, payload) -> (model, output, tool_results)
SpawnFn = Callable[[Hash, bytes], tuple[str, bytes, tuple[bytes, ...]]]


@dataclass(frozen=True)
class InterpretFailure:
    """Typed failure returned when spawn raises; the store log is left unchanged.

    Fields
    ------
    reason    : human-readable description (``str(exception)`` at minimum).
    exception : the original exception that caused the failure.
    """

    reason: str
    exception: BaseException


class Interpreter:
    """Spawns interpretation of a source Cell and records the result atomically.

    Parameters
    ----------
    store : AmplifierStore
        The store to read source cells from and write interpretations to.
    spawn : SpawnFn
        Injectable callable ``(cell_ref, payload) -> (model, output, tool_results)``.
        Never hard-coded; production passes a wrapper over PreparedBundle.spawn;
        tests pass a deterministic fake.
    """

    def __init__(self, store: AmplifierStore, spawn: SpawnFn) -> None:
        self._store = store
        self._spawn = spawn

    def interpret(self, cell_ref: Hash) -> Hash | InterpretFailure:
        """Interpret *cell_ref*: spawn, record atomically, return derived ref or failure.

        Steps
        -----
        1. Read the source Cell from the store.
        2. Call ``spawn(cell_ref, cell.payload)`` to obtain ``(model, output, tool_results)``.
        3. Record exactly ONE atomic batch via ``store.write_interpretation()``
           which calls ``append_batch([ToolResultEvent..., InterpreterOutputEvent])``.
        4. Materialise the derived Cell through the unchanged ``store.regenerate()``.
        5. Return the derived cell ref.

        On spawn failure or cancellation the store is left unchanged (nothing appended)
        and an :class:`InterpretFailure` is returned instead.
        """
        cell = self._store.get_cell(cell_ref)

        try:
            model, output, tool_results = self._spawn(cell_ref, cell.payload)
        except Exception as exc:
            return InterpretFailure(reason=str(exc), exception=exc)

        # Record one atomic batch: [ToolResultEvent..., InterpreterOutputEvent].
        # input_hash = cell_ref uniquely identifies this source cell in the derived-ref hash.
        derived_ref = self._store.write_interpretation(
            model=model,
            input_hash=cell_ref,
            tool_results=tool_results,
            output=output,
        )

        # F1: materialise via the unchanged regenerate(); replay folds the log.
        self._store.regenerate(derived_ref)

        return derived_ref
