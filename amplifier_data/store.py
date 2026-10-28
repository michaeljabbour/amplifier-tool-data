"""AmplifierStore — composes the kernel + fold engine into the user-facing API."""

from __future__ import annotations

import asyncio
import os
import struct
import threading
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from .compaction import CompactionReport
from .embedding import Embedder
from .kernel import DurableKernel, InMemoryKernel, StorageKernel
from .lenses._scope import SCOPED_TO
from .lenses.graph import GraphLens
from .lenses.graph import QueryResult as GraphQueryResult
from .lenses.grouped import GroupedLens
from .lenses.grouped import QueryResult as GroupedQueryResult
from .lenses.index import IndexLens
from .lenses.index import QueryResult as IndexQueryResult
from .lenses.relational import QueryResult as RelationalQueryResult
from .lenses.relational import RelationalLens
from .lenses.temporal import INVALIDATE_PREFIX, TemporalLens, TimelineEntry
from .lenses.temporal import QueryResult as TemporalQueryResult
from .lenses.vector import EMBEDDING_OF, VectorLens
from .models import (
    AccessEvent,
    Cell,
    CellWriteEvent,
    Hash,
    InterpreterOutputEvent,
    RelationshipEvent,
    SeqPos,
    ToolResultEvent,
)
from .query import QueryResult as QueryIRResult
from .query import run_query

if TYPE_CHECKING:
    from .autonomic import AutonomicIndexer
    from .envelope import Transaction, WriteBatch
    from .integrity import TensionRecord
    from .intent import IntentCompiler, PlanRef, PlanRun
    from .interpreter import InterpretFailure, SpawnFn


def _default_kernel() -> StorageKernel:
    """Return the best available kernel: Rust when RUST_AVAILABLE, else InMemoryKernel."""
    from . import RUST_AVAILABLE

    if RUST_AVAILABLE:
        from ._rust import RustKernel  # type: ignore

        return RustKernel()
    return InMemoryKernel()


class AmplifierStore:
    """Write cells/edges/interpretations as events; read them back as regenerable Cells.

    Design Decision D4 — read-vs-fold boundary
    -------------------------------------------
    * Public read methods (:meth:`get_cell`, :meth:`regenerate`) append exactly
      ONE :class:`~amplifier_data.models.AccessEvent` per call.
    * Lens internals (:meth:`query_relational`, :meth:`query_grouped`,
      :meth:`query_index`) pass ``self._kernel`` directly to the lens, which
      reads via ``kernel.all_events()`` — never through ``get_cell`` — so
      folding the log generates **zero** new AccessEvents.  This keeps the
      E3 static fold deterministic and the access record meaningful.

    Durable opt-in
    --------------
    Pass ``path=...`` to select the durable :class:`~amplifier_data.kernel.DurableKernel`
    backed by ``RustFileKernel``.  Omitting ``path`` (the default) preserves the existing
    in-memory behaviour unchanged.  Call :meth:`close` to flush and release the file handle
    when a durable store is no longer needed; ``close()`` is a safe no-op on in-memory stores.
    """

    def __init__(
        self,
        kernel: StorageKernel | None = None,
        *,
        path: os.PathLike[str] | str | None = None,
        record_access: bool = True,
    ) -> None:
        self._record_access = record_access
        # Serializes the T3D-1 async write seam at the store boundary so concurrent
        # async writers are correct regardless of which kernel backs this store.
        # The synchronous API does not take this lock — its behaviour is unchanged.
        self._write_lock = threading.Lock()
        if path is not None:
            # Additive durable opt-in: path= selects the RustFileKernel-backed DurableKernel.
            from . import RUST_AVAILABLE

            if not RUST_AVAILABLE:
                raise RuntimeError(
                    "AmplifierStore(path=...) requires the compiled Rust extension "
                    "(RUST_AVAILABLE is False).  Build the extension with `maturin develop` "
                    "and reinstall, then retry."
                )
            self._kernel: StorageKernel = DurableKernel.open(str(path))
        else:
            self._kernel = kernel if kernel is not None else _default_kernel()

    @property
    def kernel(self) -> StorageKernel:
        return self._kernel

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and release the underlying durable kernel, if any.

        Safe to call on in-memory stores (no-op) and idempotently on durable
        :class:`~amplifier_data.kernel.DurableKernel` stores.
        """
        if hasattr(self._kernel, "close"):
            self._kernel.close()  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    def write_cell(
        self,
        payload: bytes,
        interpreters: tuple[Hash, ...] = (),
        *,
        source_interaction_id: str | None = None,
        written_by_hook: str | None = None,
    ) -> Hash:
        """Append a CellWriteEvent; return the content-addressed cell ref.

        Causal tagging (T3D-3, both keyword-only, both backward compatible)
        ------------------------------------------------------------------
        * ``source_interaction_id`` — the interaction that caused this write.
        * ``written_by_hook`` — the hook that performed it.

        When either is supplied, reserved-type provenance edges are written in the
        SAME atomic ``append_batch`` as the cell (so the cell and its attribution
        co-land, all-or-nothing). Omitting both reproduces the original
        single-event behaviour exactly. Query later with
        :meth:`writes_by_interaction` / :meth:`writes_by_hook`.
        """
        from .causal import causal_edges

        event = CellWriteEvent(payload=payload, interpreters=interpreters)
        cell_ref = event.cell_ref()
        edges = causal_edges(
            cell_ref,
            source_interaction_id=source_interaction_id,
            written_by_hook=written_by_hook,
        )
        self._kernel.append_batch([event, *edges])
        return cell_ref

    def relate(self, from_ref: Hash, to_ref: Hash, rel_type: str) -> None:
        """Append a RelationshipEvent; the edge lands in from_ref's provenance only."""
        self._kernel.append_batch(
            [RelationshipEvent(from_ref=from_ref, to_ref=to_ref, type=rel_type)]
        )

    def write_interpretation(
        self,
        model: str,
        input_hash: Hash,
        tool_results: tuple[bytes, ...],
        output: bytes,
    ) -> Hash:
        """Record an interpreter run as ONE atomic batch: tool results + output (F1)."""
        tool_events = [ToolResultEvent(output=o) for o in tool_results]
        consumed = tuple(e.hash for e in tool_events)
        cell_ref = InterpreterOutputEvent.derive_ref(
            model=model, input_hash=input_hash, tool_results_consumed=consumed, output=output
        )
        interp = InterpreterOutputEvent(
            cell_ref=cell_ref,
            model=model,
            input_hash=input_hash,
            tool_results_consumed=consumed,
            output=output,
        )
        self._kernel.append_batch([*tool_events, interp])
        return cell_ref

    def add_embedding(self, target_ref: Hash, vector: Sequence[float]) -> Hash:
        """Store a vector embedding and link it to *target_ref*.

        Packs *vector* as a little-endian f32 sequence, writes it as a new cell,
        and relates it to *target_ref* with the :data:`~amplifier_data.lenses.vector.EMBEDDING_OF`
        edge type.  Returns the content-addressed ref of the new embedding cell.

        This is a WRITE-only operation — no :class:`~amplifier_data.models.AccessEvent`
        is emitted (D4: read-vs-fold boundary).

        Parameters
        ----------
        target_ref:
            Hash of the cell this embedding represents.
        vector:
            Sequence of float values to pack as little-endian f32.

        Returns
        -------
        Hash
            Content-addressed ref of the embedding cell
            (a :class:`~amplifier_data.models.CellWriteEvent`).
        """
        packed = struct.pack(f"<{len(vector)}f", *vector)
        emb_ref = self.write_cell(packed)
        self.relate(emb_ref, target_ref, EMBEDDING_OF)
        return emb_ref

    def scope(self, cell_ref: Hash, scope_ref: Hash) -> None:
        """Associate *cell_ref* with the scope identified by *scope_ref*.

        Appends exactly one
        ``RelationshipEvent(from_ref=cell_ref, to_ref=scope_ref, type=SCOPED_TO)``
        via :meth:`relate`.

        This is a WRITE-only operation — no
        :class:`~amplifier_data.models.AccessEvent` is emitted
        (D4: read-vs-fold boundary).  Multiple calls with different
        *scope_ref* values are allowed; the substrate records each as an
        independent direct edge and performs no scope-to-scope traversal
        (F3: direct edges only).

        Parameters
        ----------
        cell_ref:
            Hash of the content cell being scoped.
        scope_ref:
            Hash of the scope cell this content belongs to.
        """
        self.relate(cell_ref, scope_ref, SCOPED_TO)

    #: Alias for :meth:`scope`. Matches ``RemoteStore.add_to_scope`` and the
    #: name used in docs/CONSUMER_INTEGRATION.md, so the in-process and remote
    #: surfaces are identical.
    add_to_scope = scope

    # ------------------------------------------------------------------
    # Async write seam (T3D-1) — sync API above is unchanged
    # ------------------------------------------------------------------
    #
    # The whole store is synchronous by design. Rather than thread async all the
    # way down (which would fork every method), the async seam is a thin, honest
    # wrapper: each coroutine offloads the corresponding sync write to a worker
    # thread via ``asyncio.to_thread`` while holding ``self._write_lock`` so
    # concurrent async writers serialize correctly on ANY kernel backend. The
    # event loop stays unblocked; the log stays totally ordered.

    async def write_cell_async(
        self,
        payload: bytes,
        interpreters: tuple[Hash, ...] = (),
        *,
        source_interaction_id: str | None = None,
        written_by_hook: str | None = None,
    ) -> Hash:
        """Async counterpart of :meth:`write_cell` (offloaded via asyncio.to_thread)."""

        def _do() -> Hash:
            with self._write_lock:
                return self.write_cell(
                    payload,
                    interpreters,
                    source_interaction_id=source_interaction_id,
                    written_by_hook=written_by_hook,
                )

        return await asyncio.to_thread(_do)

    async def relate_async(self, from_ref: Hash, to_ref: Hash, rel_type: str) -> None:
        """Async counterpart of :meth:`relate`."""

        def _do() -> None:
            with self._write_lock:
                self.relate(from_ref, to_ref, rel_type)

        await asyncio.to_thread(_do)

    async def add_embedding_async(self, target_ref: Hash, vector: Sequence[float]) -> Hash:
        """Async counterpart of :meth:`add_embedding`."""

        def _do() -> Hash:
            with self._write_lock:
                return self.add_embedding(target_ref, vector)

        return await asyncio.to_thread(_do)

    async def commit_batch_async(self, batch: WriteBatch) -> list[SeqPos]:
        """Commit a staged :class:`~amplifier_data.envelope.WriteBatch` off-thread.

        The whole batch lands in one atomic ``append_batch`` (T3D-2 semantics);
        this only moves the blocking commit off the event loop.
        """

        def _do() -> list[SeqPos]:
            with self._write_lock:
                return batch.commit()

        return await asyncio.to_thread(_do)

    # ------------------------------------------------------------------
    # Atomic batch + rollback envelopes (T3D-2)
    # ------------------------------------------------------------------

    def write_batch(self) -> WriteBatch:
        """Open a :class:`~amplifier_data.envelope.WriteBatch`.

        Stage several writes (e.g. a memory cell plus its importance + category +
        source facts) and ``commit()`` them in ONE all-or-nothing append_batch.
        """
        from .envelope import WriteBatch

        return WriteBatch(self)

    def transaction(self) -> Transaction:
        """Open a :class:`~amplifier_data.envelope.Transaction` rollback envelope.

        For multi-call flows: on a clean exit the writes stand; on an exception
        every fact/edge written through the envelope is compensated away.
        """
        from .envelope import Transaction

        return Transaction(self)

    def autonomic_indexer(self, *, threshold: int = 64) -> AutonomicIndexer:
        """Attach an :class:`~amplifier_data.autonomic.AutonomicIndexer` to this store.

        It begins observing reads immediately. Drive it with ``run_once()`` or
        ``start()`` (background thread); always ``detach()`` when done. See
        :mod:`amplifier_data.autonomic` for the reflexive self-indexing loop.
        """
        from .autonomic import AutonomicIndexer

        return AutonomicIndexer(self, threshold=threshold)

    # ------------------------------------------------------------------
    # Causal-tag queries (T3D-3) — read-only, no AccessEvents (D4)
    # ------------------------------------------------------------------

    def writes_by_interaction(self, interaction_id: str) -> list[Hash]:
        """Return the cell refs written under *interaction_id*, in log order.

        Folds the reserved ``@causal:interaction`` provenance edges directly from
        the kernel — emits no AccessEvents (D4 read-vs-fold boundary).
        """
        from .causal import CAUSED_BY_INTERACTION, interaction_ref

        target = interaction_ref(interaction_id)
        return self._writes_targeting(CAUSED_BY_INTERACTION, target)

    def writes_by_hook(self, hook: str) -> list[Hash]:
        """Return the cell refs written by *hook*, in log order (read-only, D4)."""
        from .causal import WRITTEN_BY_HOOK, hook_ref

        return self._writes_targeting(WRITTEN_BY_HOOK, hook_ref(hook))

    def _writes_targeting(self, rel_type: str, target: Hash) -> list[Hash]:
        refs: list[Hash] = []
        for _pos, ev in self._kernel.all_events():  # type: ignore[attr-defined]
            if isinstance(ev, RelationshipEvent) and ev.type == rel_type and ev.to_ref == target:
                refs.append(ev.from_ref)
        return refs

    # ------------------------------------------------------------------
    # Public read methods — each appends ONE AccessEvent (D4)
    # ------------------------------------------------------------------

    def get_cell(self, ref: Hash, *, record_access: bool | None = None) -> Cell:
        """Resolve ref to a Cell; raises NotFound if unknown, TypeError if the ref is an Event.

        Tri-state access recording (D4: read-vs-fold boundary):

        * ``record_access=None`` (default) — defers to the instance default
          set at construction (``self._record_access``).
        * ``record_access=True`` — forces one :class:`~amplifier_data.models.AccessEvent`
          even when the instance default is ``False``.
        * ``record_access=False`` — suppresses the AccessEvent even when the
          instance default is ``True``.

        Exactly one AccessEvent with ``cell_ref=ref`` is appended after a
        successful resolution when the effective flag is truthy.
        """
        result = self._kernel.resolve(ref)
        if isinstance(result, Cell):
            should_record = self._record_access if record_access is None else record_access
            if should_record:
                # Record the read AFTER successful resolution (D3: provenance-neutral).
                self._kernel.append_batch([AccessEvent(cell_ref=ref)])
            return result
        raise TypeError(f"ref {ref!r} resolved to an Event, not a Cell")

    def regenerate(self, ref: Hash, *, record_access: bool | None = None) -> Cell:
        """Force a fresh deterministic fold from the log (E1) by dropping cache then re-folding.

        Delegates to :meth:`get_cell` and forwards the tri-state *record_access* so
        regeneration can opt out of access recording exactly as a plain read can.

        Tri-state access recording (D4: read-vs-fold boundary):

        * ``record_access=None`` (default) — defers to the instance default
          set at construction (``self._record_access``).
        * ``record_access=True`` — forces one :class:`~amplifier_data.models.AccessEvent`
          even when the instance default is ``False``.
        * ``record_access=False`` — suppresses the AccessEvent even when the
          instance default is ``True``.
        """
        self._kernel.drop_materialized(ref)
        return self.get_cell(ref, record_access=record_access)

    def drop_materialized(self, ref: Hash) -> None:
        """Evict ref from the materialized cache; idempotent."""
        self._kernel.drop_materialized(ref)

    # ------------------------------------------------------------------
    # Lens query accessor surface (D4: reads kernel directly — no AccessEvents)
    # ------------------------------------------------------------------

    def query_relational(self, **kwargs: Any) -> RelationalQueryResult:
        """Run :class:`~amplifier_data.lenses.relational.RelationalLens` over the kernel.

        Passes ``kernel=self._kernel`` plus any additional keyword arguments
        to :meth:`~amplifier_data.lenses.relational.RelationalLens.query`.
        The lens reads ``kernel.all_events()`` directly — **no AccessEvents
        are appended** (D4 read-vs-fold boundary).

        Keyword arguments
        -----------------
        filter : Callable[[dict], bool] | None
            Optional row predicate.
        project : Sequence[str] | None
            Optional column projection.

        Returns
        -------
        RelationalQueryResult
            ToolResult-shaped result from :class:`~amplifier_data.lenses.relational.RelationalLens`.
        """
        return RelationalLens().query(kernel=self._kernel, **kwargs)

    def query_grouped(self, **kwargs: Any) -> GroupedQueryResult:
        """Run :class:`~amplifier_data.lenses.grouped.GroupedLens` over the kernel.

        Passes ``kernel=self._kernel`` plus any additional keyword arguments
        to :meth:`~amplifier_data.lenses.grouped.GroupedLens.query`.
        The lens reads ``kernel.all_events()`` directly — **no AccessEvents
        are appended** (D4 read-vs-fold boundary).

        Keyword arguments
        -----------------
        group : str | None
            Optional group key (``\"cell_write\"`` or ``\"interpreter_output\"``).

        Returns
        -------
        GroupedQueryResult
            ToolResult-shaped result from :class:`~amplifier_data.lenses.grouped.GroupedLens`.
        """
        return GroupedLens().query(kernel=self._kernel, **kwargs)

    def query_index(self, **kwargs: Any) -> IndexQueryResult:
        """Run :class:`~amplifier_data.lenses.index.IndexLens` over the kernel.

        Passes ``kernel=self._kernel`` plus any additional keyword arguments
        to :meth:`~amplifier_data.lenses.index.IndexLens.query`.
        The lens reads ``kernel.all_events()`` directly — **no AccessEvents
        are appended** (D4 read-vs-fold boundary).

        Keyword arguments
        -----------------
        cell_ref : Hash | None
            Optional cell reference to restrict output to a single cell.

        Returns
        -------
        IndexQueryResult
            ToolResult-shaped result from :class:`~amplifier_data.lenses.index.IndexLens`.
        """
        return IndexLens().query(kernel=self._kernel, **kwargs)

    def query_vector(
        self,
        vector: Sequence[float],
        k: int,
        scope: Hash | None = None,
    ) -> list[tuple[Hash, float]]:
        """Top-k nearest-neighbor (cosine) over folded embeddings.

        Delegates to a freshly-instantiated :class:`~amplifier_data.lenses.vector.VectorLens`,
        mirroring the :meth:`query_index` delegation shape. The lens reads
        ``kernel.all_events()`` directly \u2014 **no AccessEvents are appended** (D4 read-vs-fold
        boundary). Returns ``[(target_ref, score)]`` sorted by descending similarity.

        Parameters
        ----------
        vector:
            Query vector as a sequence of floats.
        k:
            Maximum number of results to return.
        scope:
            When provided, only embeddings whose target cell is directly in *scope*
            are scored (filter-before-score, F4); an unknown scope returns an empty
            list (F6).  Reads the kernel directly \u2014 emits no AccessEvents (D4).

        Returns
        -------
        list[tuple[Hash, float]]
            ``[(target_ref, score)]`` sorted by descending cosine similarity.
        """
        result = VectorLens().query(kernel=self._kernel, vector=vector, k=k, scope=scope)
        return result.output

    def query_graph(
        self, start: Hash, max_hops: int, rel_type: str | None = None
    ) -> GraphQueryResult:
        """Traverse RelationshipEvent edges from *start* up to *max_hops* (BFS).

        Delegates to :class:`~amplifier_data.lenses.graph.GraphLens` over the
        kernel directly — **no AccessEvents are appended** (D4 read-vs-fold
        boundary). *rel_type* optionally restricts the walk to one edge type.
        ``output`` is the sorted list of reachable refs.
        """
        return GraphLens().query(
            kernel=self._kernel, start=start, max_hops=max_hops, rel_type=rel_type
        )

    def graph_neighbors(self, ref: Hash, rel_type: str | None = None) -> list[Hash]:
        """Return the direct out-neighbors of *ref* (optionally filtered by *rel_type*).

        Reads the kernel directly via :class:`~amplifier_data.lenses.graph.GraphLens`
        — **no AccessEvents are appended** (D4).
        """
        return GraphLens().neighbors(self._kernel, ref, rel_type=rel_type)

    def assert_fact(self, subject: Hash, predicate: str, object: Hash) -> None:
        """Assert the KG fact (subject, predicate, object) — sugar over relate()."""
        self.relate(subject, object, predicate)

    def invalidate_fact(self, subject: Hash, predicate: str, object: Hash) -> None:
        """Invalidate the KG fact (subject, predicate, object) at the current logical time.

        Appends a tombstone RelationshipEvent; the fact becomes invalid until
        (and unless) it is asserted again. Validity is by SeqPos (logical) time.
        """
        self.relate(subject, object, INVALIDATE_PREFIX + predicate)

    def query_facts(
        self, subject: Hash | None = None, predicate: str | None = None
    ) -> TemporalQueryResult:
        """Return the currently-valid KG facts (latest event per triple is an assertion).

        Reads the kernel directly via TemporalLens — **no AccessEvents** (D4).
        Re-assertion after invalidation makes a fact valid again.
        """
        return TemporalLens().query(kernel=self._kernel, subject=subject, predicate=predicate)

    def timeline(self, subject: Hash) -> list[TimelineEntry]:
        """Return *subject*'s SeqPos-ordered assert/invalidate history (D4 read)."""
        return TemporalLens().timeline(self._kernel, subject)

    def query(self, ir: list[dict[str, Any]]) -> QueryIRResult:
        """Run a unified QueryIR pipeline over this store (Plan 3g).

        See :mod:`amplifier_data.query`. The pipeline composes the existing
        lenses (scan/filter/traverse/project/order_by/limit/aggregate) and
        appends no AccessEvents (D4). Equivalent to ``run_query(self, ir)``.
        """
        return run_query(self, ir)

    # ------------------------------------------------------------------
    # Intent-compiled queries (Plan 4a) -- mechanism, not policy
    # ------------------------------------------------------------------

    def compile_intent(
        self,
        intent: str,
        compiler: IntentCompiler,
        *,
        compiler_id: str | None = None,
        catalog: dict[str, Any] | None = None,
    ) -> PlanRef:
        """Compile *intent* with the injected *compiler*; store the plan as a Cell.

        Validates the emitted IR BEFORE appending anything
        (:class:`~amplifier_data.intent.InvalidPlanError` on failure; compiler
        exceptions propagate; the log is unchanged either way). On success
        appends ONE atomic batch (intent cell, plan cell, ``@plan:`` edges)
        and returns the content-addressed plan ref.
        """
        from .intent import (
            CATALOG,
            COMPILED_BY,
            COMPILED_FROM,
            InvalidPlanError,
            compiler_ref,
            plan_payload,
            validate_ir,
        )

        cat = CATALOG if catalog is None else catalog
        ir = compiler.compile(intent, cat)
        errors = validate_ir(ir, cat)
        if errors:
            raise InvalidPlanError("invalid plan IR: " + "; ".join(errors))

        intent_event = CellWriteEvent(payload=intent.encode("utf-8"))
        intent_ref = intent_event.cell_ref()
        plan_event = CellWriteEvent(payload=plan_payload(ir))
        plan_ref = plan_event.cell_ref()
        edges = [RelationshipEvent(from_ref=plan_ref, to_ref=intent_ref, type=COMPILED_FROM)]
        if compiler_id is not None:
            edges.append(
                RelationshipEvent(
                    from_ref=plan_ref, to_ref=compiler_ref(compiler_id), type=COMPILED_BY
                )
            )
        self._kernel.append_batch([intent_event, plan_event, *edges])
        return plan_ref

    def execute_plan(self, plan_ref: PlanRef, *, record_result: bool = False) -> PlanRun:
        """Deterministically execute a stored plan cell.

        Resolves the plan via :meth:`get_cell` (one AccessEvent -- the execution
        trace), decodes + validates the envelope
        (:class:`~amplifier_data.intent.InvalidPlanError` if not a plan cell;
        :class:`~amplifier_data.kernel.NotFound` propagates for unknown refs),
        and runs the Plan 3g executor. ``record_result=True`` and
        ``result.success``: appends ONE atomic batch (result cell,
        ``@plan:result_of`` edge) and sets ``PlanRun.result_ref``.
        """
        from .intent import RESULT_OF, PlanRun, decode_plan, result_payload

        cell = self.get_cell(plan_ref)
        ir = decode_plan(cell.payload)
        result = run_query(self, ir)

        result_ref: Hash | None = None
        if record_result and result.success:
            result_event = CellWriteEvent(payload=result_payload(result.output))
            result_ref = result_event.cell_ref()
            edge = RelationshipEvent(from_ref=result_ref, to_ref=plan_ref, type=RESULT_OF)
            self._kernel.append_batch([result_event, edge])

        return PlanRun(result=result, result_ref=result_ref)

    def query_plans(self) -> list[tuple[PlanRef, Hash]]:
        """Return ``[(plan_ref, intent_ref)]`` in first-compiled log order, deduplicated.

        Folds ``@plan:compiled_from`` RelationshipEvents directly from the
        kernel -- D4: no AccessEvents (mirrors :meth:`writes_by_interaction`).
        """
        from .intent import COMPILED_FROM

        seen: set[Hash] = set()
        out: list[tuple[Hash, Hash]] = []
        for _pos, ev in self._kernel.all_events():  # type: ignore[attr-defined]
            if isinstance(ev, RelationshipEvent) and ev.type == COMPILED_FROM:
                if ev.from_ref not in seen:
                    seen.add(ev.from_ref)
                    out.append((ev.from_ref, ev.to_ref))
        return out

    # ------------------------------------------------------------------
    # Convergent integrity (Plan 4b, E4) -- mechanism, not policy
    # ------------------------------------------------------------------

    def query_tensions(self, *, unresolved_only: bool = False) -> list[TensionRecord]:
        """All materialized tensions, in first-detected log order.

        Folds ``@integrity:tension`` edges (to discover every materialized
        tension and its first-detected position) and ``@integrity:resolves``
        validity directly from the kernel -- D4: no AccessEvents for the
        enumeration (payload decode uses ``get_cell(record_access=False)``,
        the autonomic ``explain()`` precedent). ``resolved`` reflects the
        TEMPORAL fold: an invalidated resolves fact (a reopened tension)
        reads as unresolved again. ``unresolved_only=True`` filters to
        ``resolved=False``.
        """
        from .integrity import RESOLVES, TENSION, TensionRecord, decode_tension

        # 1. Discover every tension ref, ordered by its first @integrity:tension edge.
        first_seen: dict[Hash, SeqPos] = {}
        for pos, ev in self._kernel.all_events():  # type: ignore[attr-defined]
            if isinstance(ev, RelationshipEvent) and ev.type == TENSION:
                first_seen.setdefault(ev.from_ref, pos)
        ordered_refs = sorted(first_seen, key=lambda r: first_seen[r])

        # 2. Fold @integrity:resolves validity (temporal: latest event per
        #    (resolution_ref, tension_ref) pair wins), preserving log order.
        history: dict[tuple[Hash, Hash], list[tuple[SeqPos, str]]] = {}
        first_res_seen: dict[tuple[Hash, Hash], SeqPos] = {}
        invalidate_resolves = INVALIDATE_PREFIX + RESOLVES
        for pos, ev in self._kernel.all_events():  # type: ignore[attr-defined]
            if not isinstance(ev, RelationshipEvent):
                continue
            if ev.type == RESOLVES:
                key = (ev.from_ref, ev.to_ref)
                history.setdefault(key, []).append((pos, "assert"))
                first_res_seen.setdefault(key, pos)
            elif ev.type == invalidate_resolves:
                key = (ev.from_ref, ev.to_ref)
                history.setdefault(key, []).append((pos, "invalidate"))

        resolutions_by_tension: dict[Hash, list[Hash]] = {}
        for (res_ref, tens_ref), events in history.items():
            if max(events, key=lambda e: e[0])[1] == "assert":
                resolutions_by_tension.setdefault(tens_ref, []).append(res_ref)
        for tens_ref, refs in resolutions_by_tension.items():
            refs.sort(key=lambda r: first_res_seen[(r, tens_ref)])

        records: list[TensionRecord] = []
        for ref in ordered_refs:
            cell = self.get_cell(ref, record_access=False)
            t = decode_tension(cell.payload)
            resolution_refs = tuple(resolutions_by_tension.get(ref, ()))
            record = TensionRecord(
                ref=ref,
                kind=t.kind,
                parties=t.parties,
                resolved=bool(resolution_refs),
                resolution_refs=resolution_refs,
            )
            if unresolved_only and record.resolved:
                continue
            records.append(record)
        return records

    def resolve_tension(
        self, tension_ref: Hash, resolution: str, *, resolved_by: str | None = None
    ) -> Hash:
        """Record a resolution for *tension_ref*; return the resolution cell ref.

        Validates FIRST (:class:`~amplifier_data.kernel.NotFound` propagates
        for unknown refs; :class:`~amplifier_data.integrity.NotATensionError`
        if the cell is not a decodable tension; the resolution is appended
        only after both checks succeed), then appends ONE atomic batch:
        ``CellWrite(resolution_payload(...))`` + ``Rel(resolution_ref ->
        tension_ref, "@integrity:resolves")``. Never mutates the conflicting
        facts -- picking a winner (e.g. via :meth:`invalidate_fact` on the
        losing side) is the CALLER's separate, explicit act. Resolving an
        already-resolved tension is allowed: conflicting resolutions are all
        recorded; the store picks between none of them.
        """
        from .integrity import RESOLVES, decode_tension, resolution_payload

        cell = self.get_cell(tension_ref)  # NotFound / TypeError propagate
        decode_tension(cell.payload)  # NotATensionError propagates; log unchanged

        res_event = CellWriteEvent(payload=resolution_payload(tension_ref, resolution, resolved_by))
        res_ref = res_event.cell_ref()
        edge = RelationshipEvent(from_ref=res_ref, to_ref=tension_ref, type=RESOLVES)
        self._kernel.append_batch([res_event, edge])
        return res_ref

    def to_dot(self, cell_refs: list[Hash] | None = None, *, name: str = "amplifier_data") -> str:
        """Render this store (or a subset of cells) as a Graphviz DOT string (Plan 3h)."""
        from .projection import build_projection, to_dot

        return to_dot(build_projection(self, cell_refs), name=name)

    def to_html(self, cell_refs: list[Hash] | None = None, *, title: str = "amplifier-data") -> str:
        """Render this store (or a subset of cells) as a self-contained HTML explorer (Plan 3h)."""
        from .projection import build_projection, to_html

        return to_html(build_projection(self, cell_refs), title=title)

    def iter_embeddings(self) -> list[tuple[Hash, list[float]]]:
        """Return ``[(target_ref, vector)]`` folded from embedding events (Plan 3i).

        Read-only (no AccessEvents, D4). The log is the source of truth; a pluggable
        ``VectorBackend`` (see :mod:`amplifier_data.embedding`) is a rebuildable
        accelerator built from this — ``BruteForceBackend.build_from_store`` and any
        ANN consumer repopulate from here, preserving E1.
        """
        index = VectorLens().project(self._kernel)
        return [(e.target_ref, list(e.vector)) for e in index.entries]

    def add_text_embedding(self, target_ref: Hash, text: str, embedder: Embedder) -> Hash:
        """Embed *text* with *embedder* (bring-your-own) and store it for *target_ref*."""
        return self.add_embedding(target_ref, embedder.embed(text))

    def query_text(
        self, text: str, k: int, embedder: Embedder, scope: Hash | None = None
    ) -> list[tuple[Hash, float]]:
        """Embed *text* (bring-your-own embedder) and run a scoped top-k vector query."""
        return self.query_vector(embedder.embed(text), k, scope)

    def compact_into(self, target: AmplifierStore) -> CompactionReport:
        """Re-materialize this store's current state into *target* (Plan 3j).

        Drops AccessEvent history, invalidated facts + tombstones, and superseded
        cell history while preserving refs of carried ``write_cell`` cells (content
        addressing → E1 holds). See :mod:`amplifier_data.compaction`.
        """
        from .compaction import compact

        return compact(self, target)

    def compact_to(self, path: str) -> CompactionReport:
        """Write a compact durable copy of the current state to *path*; return the report.

        The original log is untouched (archive it). Reopen *path* to use the compact
        store.
        """
        from .compaction import compact

        target = AmplifierStore(path=path)
        report = compact(self, target)
        target.close()
        return report

    # ------------------------------------------------------------------
    # interpret() convenience — constructs Interpreter with injected spawn
    # ------------------------------------------------------------------

    def interpret(
        self,
        cell_ref: Hash,
        spawn: SpawnFn | Callable[..., Any],
    ) -> Hash | InterpretFailure:
        """Interpret *cell_ref* using the injected *spawn* callable.

        Convenience wrapper that constructs an
        :class:`~amplifier_data.interpreter.Interpreter` bound to this store
        and the caller-supplied *spawn* function, then calls
        :meth:`~amplifier_data.interpreter.Interpreter.interpret`.

        Parameters
        ----------
        cell_ref:
            Hash of the source cell to interpret.
        spawn:
            Injectable callable
            ``(cell_ref: Hash, payload: bytes) -> (model, output, tool_results)``.
            Production code passes a wrapper over ``PreparedBundle.spawn``;
            tests pass a deterministic fake.

        Returns
        -------
        Hash
            The derived cell ref on success.
        InterpretFailure
            A typed failure object when *spawn* raises; the store log is left
            unchanged.
        """
        # Import here to avoid a module-level circular dependency
        # (interpreter.py imports AmplifierStore from this module).
        from .interpreter import Interpreter

        return Interpreter(self, spawn).interpret(cell_ref)
