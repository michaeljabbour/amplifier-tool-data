"""VectorLens \u2014 pure-fold view over embedding cells (Plan 3c).

Design Decision D1 compliance
------------------------------
``project()`` folds ``kernel.all_events()`` on every call and persists **no**
mutated state between calls.  Two ``VectorLens`` instances over the same kernel
are fully independent.

EMBEDDING_OF
------------
The canonical relationship-type string for ``RelationshipEvent`` edges that
link an embedding cell (the packed f32-LE vector bytes) back to the source
cell it represents.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from amplifier_data.lenses._scope import fold_scope

# ---------------------------------------------------------------------------
# Canonical relationship-type constant
# ---------------------------------------------------------------------------

EMBEDDING_OF: str = "embedding_of"
"""Edge type: *from* the embedding cell *to* the cell it represents.

Used in :meth:`AmplifierStore.add_embedding` and consumed by :class:`VectorLens`.
"""

# ---------------------------------------------------------------------------
# Lens-local dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VectorEntry:
    """One folded embedding: the embedding cell, its target, and the decoded vector.

    ``raw`` is the verbatim f32-LE payload as stored \u2014 used for byte-identical
    canonical serialisation; ``vector`` is the decoded float tuple used for cosine.
    """

    emb_ref: str
    target_ref: str
    vector: tuple[float, ...]
    raw: bytes


@dataclass(frozen=True)
class VectorIndex:
    """Immutable, canonically ordered fold of all embeddings in the log.

    ``entries`` are sorted by ``(emb_ref, target_ref)`` for a deterministic fold order.
    """

    entries: tuple[VectorEntry, ...]

    def canonical_bytes(self) -> bytes:
        """Stable serialisation for the regeneration gate.

        Concatenates ``emb_ref || target_ref || raw f32-LE bytes`` over entries already
        sorted by ``(emb_ref, target_ref)``. Hash refs are fixed-width 64-char sha256
        hex, so the concatenation is unambiguous.
        """
        out = bytearray()
        for e in self.entries:
            out.extend(e.emb_ref.encode("utf-8"))
            out.extend(e.target_ref.encode("utf-8"))
            out.extend(e.raw)
        return bytes(out)


@dataclass(frozen=True)
class VectorQueryResult:
    """ToolResult-shaped result from :meth:`VectorLens.query` (reuse, not reinvent)."""

    success: bool
    output: list[tuple[str, float]]
    error: str | None = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors.

    Returns ``0.0`` when either vector has zero magnitude (cosine is otherwise
    undefined) so queries stay total and deterministic. Raises ``ValueError`` on a
    dimension mismatch.
    """
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Lens
# ---------------------------------------------------------------------------


class VectorLens:
    """Static pure fold of embeddings into a cosine nearest-neighbor index.

    For each ``RelationshipEvent(type=EMBEDDING_OF, from_ref=emb_ref, to_ref=target_ref)``
    in the log, the corresponding ``CellWriteEvent`` payload is decoded as a
    packed f32-LE sequence and stored as a :class:`VectorEntry`.

    Design Decision D1
    ------------------
    * ``project()`` is a pure fold: reads ``kernel.all_events()`` fresh each time.
    * No mutable state is held between calls.
    """

    @property
    def name(self) -> str:
        return "vector"

    def project(self, kernel: Any) -> VectorIndex:
        """Fold ``kernel.all_events()`` into a deterministic, canonically ordered VectorIndex.

        Pure: appends no events and caches no state between calls.

        * Pass 1 \u2014 every ``CellWriteEvent`` contributes ``cell_ref -> payload``.
        * Pass 2 \u2014 every ``RelationshipEvent`` with ``type == EMBEDDING_OF`` contributes an
          ``(emb_ref, target_ref)`` edge.
        * Join \u2014 for each edge whose ``emb_ref`` has a payload, unpack the f32-LE bytes into a
          vector. Dangling edges (no matching cell) are silently skipped, keeping the fold total.

        Entries are sorted by ``(emb_ref, target_ref)`` for a canonical fold order.
        """
        from amplifier_data.models import CellWriteEvent, RelationshipEvent

        cell_payloads: dict[str, bytes] = {}
        edges: list[tuple[str, str]] = []
        for _seq_pos, event in kernel.all_events():
            if isinstance(event, CellWriteEvent):
                cell_payloads[event.cell_ref()] = event.payload
            elif isinstance(event, RelationshipEvent) and event.type == EMBEDDING_OF:
                edges.append((event.from_ref, event.to_ref))

        entries: list[VectorEntry] = []
        for emb_ref, target_ref in edges:
            raw = cell_payloads.get(emb_ref)
            if raw is None:
                continue  # dangling edge \u2014 skip (cannot occur via add_embedding)
            n = len(raw) // 4
            vector = struct.unpack(f"<{n}f", raw)
            entries.append(
                VectorEntry(emb_ref=emb_ref, target_ref=target_ref, vector=vector, raw=raw)
            )

        entries.sort(key=lambda e: (e.emb_ref, e.target_ref))
        return VectorIndex(entries=tuple(entries))

    def query(self, **typed_args: Any) -> VectorQueryResult:
        """Top-k nearest-neighbor (cosine) over the folded embeddings.

        Keyword arguments
        -----------------
        kernel : Any (required)   \u2014 the storage kernel; ``project()`` runs fresh each call.
        vector : Sequence[float] (required) \u2014 the query vector.
        k : int | None            \u2014 max results; ``None`` \u2192 all (capped at N).
        scope : Hash | None       \u2014 when given, only entries whose ``target_ref`` has a
                                    direct ``SCOPED_TO`` edge to this scope ref are
                                    considered.  Filtering happens **before** cosine
                                    scoring and top-k selection (F4: filter-before-score),
                                    so k results are k-within-scope and never under-filled
                                    by post-filtering.  An unknown scope (no member cells)
                                    yields an empty output with ``success=True`` (F6).

        Returns a ToolResult-shaped ``VectorQueryResult`` whose ``output`` is
        ``[(target_ref, score)]`` sorted by descending similarity, ties broken by ``target_ref``.

        Raises
        ------
        TypeError   \u2014 if ``kernel`` or ``vector`` is absent.
        ValueError  \u2014 if a stored vector's dimension differs from the query vector's
                      (the message names the offending ``emb_ref``).
        """
        if "kernel" not in typed_args:
            raise TypeError("VectorLens.query() requires a 'kernel' keyword argument")
        if "vector" not in typed_args:
            raise TypeError("VectorLens.query() requires a 'vector' keyword argument")

        kernel = typed_args["kernel"]
        query_vec: Sequence[float] = typed_args["vector"]
        k: int | None = typed_args.get("k")
        scope: str | None = typed_args.get("scope")

        index = self.project(kernel)

        # F4: filter-before-score — restrict entries by scope BEFORE cosine scoring
        # and top-k selection so that k results are k-within-scope (never under-filled).
        # Scope membership lives on the content cell (target_ref), NOT the embedding cell.
        entries = index.entries
        if scope is not None:
            scope_index = fold_scope(kernel)
            entries = tuple(e for e in entries if scope_index.is_in_scope(e.target_ref, scope))

        scored: list[tuple[str, float]] = []
        for entry in entries:
            if len(entry.vector) != len(query_vec):
                raise ValueError(
                    f"dimension mismatch for embedding {entry.emb_ref}: "
                    f"query has {len(query_vec)}, stored has {len(entry.vector)}"
                )
            score = cosine_similarity(query_vec, entry.vector)
            scored.append((entry.target_ref, score))

        # Descending score; deterministic tie-break by target_ref.
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        output = scored if k is None else scored[: max(k, 0)]
        return VectorQueryResult(success=True, output=output)
