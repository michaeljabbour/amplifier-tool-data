"""GraphLens — directed traversal over the RelationshipEvent log (pure fold).

Mirrors ``lenses/_scope.py``: a pure fold over ``kernel.all_events()`` with no
persisted state and no AccessEvents appended (D1/D4). Folds ``RelationshipEvent``
entries into a directed adjacency (``from_ref --rel_type--> to_ref``) and answers
reachability questions — direct ``neighbors`` and a bounded-depth ``traverse``.

Generic = mechanism: the lens walks ALL edge types by default; an optional
``rel_type`` filter restricts the walk. The consumer decides what edge types
mean (delegation trees, KG relations, scope edges, ...). This is policy-free.

Determinism: ``traverse`` returns a SORTED list, so the result is independent of
adjacency / BFS ordering and byte-stable across regenerations of the same log.
Cycles terminate via a ``visited`` set; ``max_hops`` bounds the depth.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QueryResult:
    """ToolResult-shaped result from :meth:`GraphLens.query` (reuse, not reinvent)."""

    success: bool
    output: list[str]
    error: str | None = None


Result = QueryResult


class GraphLens:
    """Directed traversal over RelationshipEvent edges implementing the Lens Protocol."""

    @property
    def name(self) -> str:
        """Human-readable identifier for this lens type."""
        return "graph"

    def _adjacency(self, kernel: Any) -> dict[str, list[tuple[str, str]]]:
        """Fold the log into ``{from_ref: [(to_ref, rel_type), ...]}`` in SeqPos order.

        Pure (D1/D4): reads ``kernel.all_events()`` fresh, persists nothing, and
        appends no events. Only ``RelationshipEvent`` entries contribute.
        """
        from amplifier_data.models import RelationshipEvent

        adj: dict[str, list[tuple[str, str]]] = {}
        for _seq_pos, event in kernel.all_events():
            if isinstance(event, RelationshipEvent):
                adj.setdefault(event.from_ref, []).append((event.to_ref, event.type))
        return adj

    def neighbors(self, kernel: Any, ref: str, rel_type: str | None = None) -> list[str]:
        """Return the directly-connected ``to_ref`` values of *ref* (SeqPos order, deduped).

        When *rel_type* is given, only edges of that type are followed. An *ref*
        with no outgoing edges yields an empty list (never raises).
        """
        adj = self._adjacency(kernel)
        seen: set[str] = set()
        out: list[str] = []
        for to_ref, etype in adj.get(ref, []):
            if rel_type is not None and etype != rel_type:
                continue
            if to_ref not in seen:
                seen.add(to_ref)
                out.append(to_ref)
        return out

    def traverse(
        self, kernel: Any, start: str, max_hops: int, rel_type: str | None = None
    ) -> list[str]:
        """Return the SORTED set of refs reachable from *start* within *max_hops* (BFS).

        Cycle-safe via a ``visited`` set; ``max_hops`` bounds the depth
        (``max_hops=1`` → direct neighbors only). When *rel_type* is given, only
        edges of that type are traversed. *start* itself is included in the
        result only if it is genuinely reachable via a cycle.
        """
        adj = self._adjacency(kernel)
        reached: set[str] = set()
        visited: set[str] = {start}
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for to_ref, etype in adj.get(node, []):
                if rel_type is not None and etype != rel_type:
                    continue
                reached.add(to_ref)
                if to_ref not in visited:
                    visited.add(to_ref)
                    queue.append((to_ref, depth + 1))
        return sorted(reached)

    def query(self, **typed_args: Any) -> QueryResult:
        """Typed surface over :meth:`traverse`.

        Keyword arguments
        -----------------
        kernel : Any  (required)  — the storage kernel to fold.
        start : str   (required)  — the ref to traverse from.
        max_hops : int            — depth bound (default 1 → direct neighbors).
        rel_type : str | None     — optional edge-type filter.

        Returns a ToolResult-shaped :class:`QueryResult` whose ``output`` is the
        sorted list of reachable refs.
        """
        if "kernel" not in typed_args:
            raise TypeError("GraphLens.query() requires a 'kernel' keyword argument")
        if "start" not in typed_args:
            raise TypeError("GraphLens.query() requires a 'start' keyword argument")
        kernel = typed_args["kernel"]
        start: str = typed_args["start"]
        max_hops: int = typed_args.get("max_hops", 1)
        rel_type: str | None = typed_args.get("rel_type")
        reached = self.traverse(kernel, start, max_hops, rel_type=rel_type)
        return QueryResult(success=True, output=reached)
