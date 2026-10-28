"""The storage kernel — mechanism only: the floor, content-addressing, resolve."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .models import (
    Cell,
    Event,
    Hash,
    SeqPos,
    ToolResultEvent,
    content_address,
)
from .replay import is_defining, regenerate, target_ref

logger = logging.getLogger(__name__)

# A read-only, post-commit reactor notified as observer(seq_pos, event).
Observer = Callable[[SeqPos, Event], None]


class NotFound(Exception):
    """Raised by resolve() for an unknown ref — never a silent None."""

    def __init__(self, ref: Hash) -> None:
        super().__init__(f"ref not found: {ref!r}")
        self.ref = ref


@dataclass
class Subscription:
    """Handle returned by subscribe(); call unsubscribe() to detach the observer."""

    _unsubscribe: Callable[[], None]

    def unsubscribe(self) -> None:
        self._unsubscribe()


@runtime_checkable
class StorageKernel(Protocol):
    """The five-verb floor. Backends are swappable behind this seam."""

    def append_batch(self, events: list[Event]) -> list[SeqPos]: ...
    def content_address(self, payload: bytes) -> Hash: ...
    def resolve(self, ref: Hash) -> Event | Cell: ...
    def subscribe(self, observer: Observer) -> Subscription: ...
    def drop_materialized(self, ref: Hash) -> None: ...


class InMemoryKernel:
    """In-memory proof-build kernel. One unified, append-only, totally-ordered log."""

    def __init__(self) -> None:
        self._log: list[Event] = []  # index == SeqPos
        self._cell_refs: set[Hash] = set()  # defining cell refs seen so far
        self._by_hash: dict[Hash, Event] = {}  # content-addressed events (tool results)
        self._materialized: dict[Hash, Cell] = {}
        self._observers: list[Observer] = []
        self.observer_errors: list[tuple[SeqPos, Event, Exception]] = []
        # Serializes append_batch so concurrent writers (e.g. via the T3D-1 async
        # seam, which offloads to a threadpool) cannot interleave SeqPos assignment.
        # Re-entrant so a same-thread observer may itself append. The sync API and
        # its single-threaded behaviour are unchanged.
        self._lock = threading.RLock()

    def content_address(self, payload: bytes) -> Hash:
        return content_address(payload)

    def all_events(self) -> list[tuple[SeqPos, Event]]:
        """Read accessor for the fold engine: (SeqPos, Event) pairs in order."""
        return list(enumerate(self._log))

    def resolve(self, ref: Hash) -> Event | Cell:
        if ref in self._materialized:
            return self._materialized[ref]
        if ref in self._cell_refs:
            cell = regenerate(self.all_events(), ref)
            self._materialized[ref] = cell
            return cell
        if ref in self._by_hash:
            return self._by_hash[ref]
        raise NotFound(ref)

    def subscribe(self, observer: Observer) -> Subscription:
        self._observers.append(observer)

        def _unsubscribe() -> None:
            if observer in self._observers:
                self._observers.remove(observer)

        return Subscription(_unsubscribe)

    def append_batch(self, events: list[Event]) -> list[SeqPos]:
        # Validate the WHOLE batch before mutating anything (atomicity).
        if not events:
            raise ValueError("append_batch requires a non-empty list of events")
        if not all(isinstance(e, Event) for e in events):
            raise ValueError("append_batch requires every item to be an Event")

        with self._lock:
            # Commit: assign contiguous SeqPos and append.
            start = len(self._log)
            positions = list(range(start, start + len(events)))
            self._log.extend(events)
            for ev in events:
                ref = target_ref(ev)
                if ref is not None:
                    if is_defining(ev, ref):
                        self._cell_refs.add(ref)
                    else:
                        # Amendment (e.g. RelationshipEvent): evict stale materialized
                        # view so the next resolve() re-folds with the new event.
                        self._materialized.pop(ref, None)
                if isinstance(ev, ToolResultEvent):
                    self._by_hash[ev.hash] = ev

            # Post-commit, ordered, synchronous observer notification. Held under
            # the same lock so the (SeqPos, Event) stream a synchronous observer
            # sees stays totally ordered even with concurrent writers.
            for pos, ev in zip(positions, events, strict=True):
                for obs in list(self._observers):  # snapshot: re-entrant sub/unsub must not skip
                    try:
                        obs(pos, ev)
                    except Exception as exc:  # noqa: BLE001 - observer failure must not roll back
                        self.observer_errors.append((pos, ev, exc))
                        logger.error("observer failed at SeqPos %d: %r", pos, exc)
            return positions

    async def append_batch_async(self, events: list[Event]) -> list[SeqPos]:
        """Async seam (T3D-1): offload the sync :meth:`append_batch` to a worker
        thread via :func:`asyncio.to_thread`. The thread-safe :meth:`append_batch`
        keeps concurrent writers correctly serialized."""
        return await asyncio.to_thread(self.append_batch, events)

    def drop_materialized(self, ref: Hash) -> None:
        with self._lock:
            self._materialized.pop(ref, None)  # idempotent: discard if present


class DurableKernel:
    """Python adapter wrapping ``RustFileKernel`` to satisfy the ``StorageKernel`` protocol.

    Bridges the PyO3-bound ``RustFileKernel`` (``open`` / ``append`` / ``all_events`` /
    ``close``) to the five-verb ``StorageKernel`` surface expected by the rest of the
    system.  Maintains a Python-side materialised-cell cache and observer list; all
    event storage and ordering is delegated to the underlying Rust file kernel.

    Use :meth:`open` (or ``AmplifierStore(path=...)``) — do not call ``__init__``
    directly.
    """

    def __init__(self, rust_file_kernel: object) -> None:
        self._fk = rust_file_kernel
        self._materialized: dict[Hash, Cell] = {}
        self._observers: list[Observer] = []
        # Serializes the durable append + observer fan-out so the T3D-1 async
        # seam (threadpool offload) cannot interleave two writers. Re-entrant.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, path: str) -> DurableKernel:
        """Open or create the durable log file at *path*.

        Delegates to ``RustFileKernel.open(path)`` and wraps the result.
        Raises :exc:`ImportError` if the Rust extension is not compiled.
        """
        from . import _rust  # type: ignore[import]

        fk = _rust.RustFileKernel.open(path)
        return cls(fk)

    # ------------------------------------------------------------------
    # StorageKernel protocol
    # ------------------------------------------------------------------

    def append_batch(self, events: list[Event]) -> list[SeqPos]:
        if not events:
            raise ValueError("append_batch requires a non-empty list of events")

        with self._lock:
            positions: list[SeqPos] = self._fk.append(events)  # type: ignore[assignment]

            # Evict stale materialized views for any amendment events.
            for ev in events:
                ref = target_ref(ev)
                if ref is not None and not is_defining(ev, ref):
                    self._materialized.pop(ref, None)

            # Post-commit, ordered observer notification (mirrors InMemoryKernel).
            for pos, ev in zip(positions, events, strict=True):
                for obs in list(self._observers):
                    try:
                        obs(pos, ev)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("observer failed at SeqPos %d: %r", pos, exc)

            return positions

    async def append_batch_async(self, events: list[Event]) -> list[SeqPos]:
        """Async seam (T3D-1): offload the thread-safe sync :meth:`append_batch`
        to a worker thread via :func:`asyncio.to_thread`."""
        return await asyncio.to_thread(self.append_batch, events)

    def content_address(self, payload: bytes) -> Hash:
        return content_address(payload)

    def resolve(self, ref: Hash) -> Event | Cell:
        """Resolve *ref* to a Cell or Event; raises :exc:`NotFound` if unknown."""
        if ref in self._materialized:
            return self._materialized[ref]

        all_evs: list[tuple[SeqPos, Event]] = self._fk.all_events()  # type: ignore[assignment]

        # Check for a defining event (cell refs: CellWriteEvent, InterpreterOutputEvent, …)
        for _, ev in all_evs:
            ev_ref = target_ref(ev)
            if ev_ref == ref and is_defining(ev, ref):
                cell = regenerate(all_evs, ref)
                self._materialized[ref] = cell
                return cell

        # Fall back to hash-addressed ToolResultEvent lookup.
        for _, ev in all_evs:
            if isinstance(ev, ToolResultEvent) and ev.hash == ref:
                return ev

        raise NotFound(ref)

    def subscribe(self, observer: Observer) -> Subscription:
        self._observers.append(observer)

        def _unsubscribe() -> None:
            if observer in self._observers:
                self._observers.remove(observer)

        return Subscription(_unsubscribe)

    def drop_materialized(self, ref: Hash) -> None:
        self._materialized.pop(ref, None)

    # ------------------------------------------------------------------
    # Extra surface used by lenses (mirrors InMemoryKernel.all_events)
    # ------------------------------------------------------------------

    def all_events(self) -> list[tuple[SeqPos, Event]]:
        """Return ``[(SeqPos, Event)]`` in log order.

        Delegates to ``RustFileKernel.all_events()``; events are value-equal
        (``==``) to the originals but are NOT the same Python objects
        (value-equality durable-read identity).
        """
        return self._fk.all_events()  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and release the underlying file handle (idempotent)."""
        self._fk.close()  # type: ignore[attr-defined]
