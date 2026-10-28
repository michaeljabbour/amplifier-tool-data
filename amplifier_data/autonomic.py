"""Autonomic self-indexing — the reflexive index loop, made self-driving.

Background
----------
E3 proved the *manual* reflexive index: reads emit ``AccessEvent``s, and
:class:`~amplifier_data.lenses.index.IndexLens` folds them into a regenerable
``IndexCell`` that explains *why* it indexed (its ``access_summary``). What was
missing is the **autonomic** half — the substrate deciding, with no explicit
caller, to materialize that index when access pressure warrants it.

:class:`AutonomicIndexer` closes the loop with deliberately small machinery:

1. **Read-only observer.** It subscribes to the kernel and, on each committed
   event, only *counts* ``AccessEvent``s. It performs NO log writes inside the
   observer callback — so the kernel's ordered, post-commit notification contract
   (observers are read-only reactors) is preserved, and there is no re-entrant
   append that could reorder notifications for other observers.

2. **Write happens in ``run_once``.** When the accumulated access count crosses
   ``threshold``, ``run_once`` folds a fresh ``IndexCell`` via the existing
   ``IndexLens`` and appends its ``IndexCellEvent`` to the log — a normal,
   top-level write, never nested inside an observer notification. The appended
   cell is regenerable (E1) and self-explaining.

3. **Optional background runtime.** ``start``/``stop`` run a daemon thread that
   calls ``run_once`` on an interval — this is the truly "autonomic" mode. The
   loop is also exposed directly (``run_once``) so it can be driven
   deterministically in tests and by embedders that prefer to pump it themselves.

Invariants
----------
* **No runaway loop.** Materializing appends an ``IndexCellEvent``, not an
  ``AccessEvent``; the observer ignores it, so indexing never self-triggers.
* **Determinism preserved.** The index *content* is a pure content-addressed fold
  (E1 holds); only *when* it materializes is timing-dependent — and an unmaterialized
  index changes nothing about the log's truth.
* **Idempotent.** If nothing was read since the last checkpoint, the fold yields
  the same content-addressed ref and no duplicate event is appended.
* **Additive.** A standalone component attached via ``kernel.subscribe`` — no
  kernel or store-core change. Detach with :meth:`detach`.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from .lenses.index import IndexLens
from .models import AccessEvent, Event, Hash, IndexCellEvent, SeqPos

if TYPE_CHECKING:
    from .kernel import Subscription
    from .store import AmplifierStore

#: Default accesses-since-last-index that trigger a materialization.
DEFAULT_THRESHOLD = 64


class AutonomicIndexer:
    """Self-driving reflexive indexer for an :class:`AmplifierStore`.

    Construct one and it immediately begins observing reads. Call :meth:`run_once`
    to drive it deterministically, or :meth:`start` to let a background thread
    drive it autonomically. Always :meth:`detach` (or :meth:`stop`) when done.
    """

    def __init__(self, store: AmplifierStore, *, threshold: int = DEFAULT_THRESHOLD) -> None:
        if threshold < 1:
            raise ValueError(f"threshold must be >= 1, got {threshold}")
        self._store = store
        self._threshold = threshold
        self._lens = IndexLens()

        # Counter state (touched by writer threads via the observer AND by
        # run_once); guarded by _count_lock.
        self._count_lock = threading.Lock()
        self._pending = 0  # AccessEvents seen since the last materialization
        self._total_seen = 0

        # Materialization state; _materialize_lock ensures at most one concurrent
        # fold+append (e.g. background thread vs a manual run_once).
        self._materialize_lock = threading.Lock()
        self._last_index_ref: Hash | None = None
        self._materializations = 0

        # Background runtime.
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_error: Exception | None = None

        # Begin observing immediately (read-only).
        self._subscription: Subscription = store.kernel.subscribe(self._observe)

    # ------------------------------------------------------------------
    # Read-only observer — counts AccessEvents, never writes to the log
    # ------------------------------------------------------------------

    def _observe(self, _seq_pos: SeqPos, event: Event) -> None:
        if isinstance(event, AccessEvent):
            with self._count_lock:
                self._pending += 1
                self._total_seen += 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> int:
        return self._threshold

    @property
    def pending(self) -> int:
        """AccessEvents observed since the last materialization."""
        with self._count_lock:
            return self._pending

    @property
    def total_seen(self) -> int:
        """AccessEvents observed over this indexer's lifetime."""
        with self._count_lock:
            return self._total_seen

    @property
    def last_index_ref(self) -> Hash | None:
        """Content-addressed ref of the most recently materialized index cell."""
        return self._last_index_ref

    @property
    def materializations(self) -> int:
        """How many distinct index checkpoints this indexer has appended."""
        return self._materializations

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def explain(self) -> bytes | None:
        """Return the access summary ('why I indexed this') of the latest checkpoint.

        ``None`` if nothing has been materialized yet. Reads with access recording
        suppressed so introspection does not itself feed the indexer.
        """
        ref = self._last_index_ref
        if ref is None:
            return None
        cell = self._store.get_cell(ref, record_access=False)
        event = cell.provenance[0]
        assert isinstance(event, IndexCellEvent)  # defining event of an index cell
        return event.access_summary

    # ------------------------------------------------------------------
    # The materialization step (the only place that writes)
    # ------------------------------------------------------------------

    def run_once(self, *, force: bool = False) -> Hash | None:
        """Materialize a fresh index checkpoint if warranted; return its ref or None.

        Materializes when ``pending >= threshold`` (or ``force=True``). The fold is
        pure (``IndexLens.project`` reads via ``all_events`` — D4: emits no
        AccessEvents); the only write is appending the resulting ``IndexCellEvent``.
        If the access state is unchanged since the last checkpoint the content-
        addressed ref is identical and nothing is appended (idempotent).
        """
        with self._count_lock:
            if not force and self._pending < self._threshold:
                return None
            self._pending = 0  # consume the accumulated pressure

        # Fold + append outside the counter lock so reads are never blocked on I/O.
        # The materialize lock serializes concurrent run_once callers.
        with self._materialize_lock:
            index_cell = self._lens.project(self._store.kernel)
            defining_event = index_cell.provenance[0]
            assert isinstance(defining_event, IndexCellEvent)
            new_ref = defining_event.cell_ref

            if new_ref == self._last_index_ref:
                return None  # nothing changed → no duplicate checkpoint

            # Top-level write (NOT inside an observer callback): no ordering hazard.
            self._store.kernel.append_batch([defining_event])
            self._last_index_ref = new_ref
            self._materializations += 1
            return new_ref

    # ------------------------------------------------------------------
    # Optional autonomic runtime — a daemon thread that pumps run_once
    # ------------------------------------------------------------------

    def start(self, *, interval: float = 0.5) -> None:
        """Start a background thread that calls :meth:`run_once` every *interval* s.

        Idempotent: a no-op if already running. This is what makes indexing truly
        autonomic — reads alone cause the index to materialize, with no caller.
        """
        if self.running:
            return
        self._stop_event.clear()
        thread = threading.Thread(
            target=self._loop,
            args=(interval,),
            name="amplifier-data-autonomic-indexer",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def _loop(self, interval: float) -> None:
        # wait() returns True when stop() is signalled, False on timeout → tick.
        while not self._stop_event.wait(interval):
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001 - a bad tick must not kill the loop
                self._last_error = exc

    def stop(self) -> None:
        """Stop the background thread (idempotent); blocks until it has joined."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
            self._thread = None

    @property
    def last_error(self) -> Exception | None:
        """The most recent exception raised inside the background loop, if any."""
        return self._last_error

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def detach(self) -> None:
        """Stop the runtime and unsubscribe from the kernel (idempotent)."""
        self.stop()
        self._subscription.unsubscribe()
