"""Unified composable query layer over the existing lenses (Plan 3g).

Today a consumer must call six separate lens methods and stitch results by hand.
This module unifies them into ONE composable pipeline with two faces:

  * **QueryIR** — pure JSON-serializable data (a list of operation dicts). The
    *agentic* surface: an agent emits the IR, it executes. Predicates are DATA,
    never Python callables, so a query is storable, diffable, and transportable.
  * **Query** — a fluent builder. The *human* surface. It compiles to the SAME
    IR a human could hand-write (``to_ir()`` / ``from_ir()`` round-trip).

The executor runs over the EXISTING lenses (relational / grouped / graph /
temporal) reading the kernel through the store's lens accessors — it appends NO
AccessEvents (D4 read-vs-fold boundary) and is deterministic.

Operations (a pipeline always begins with ``scan``):

  scan      {op:'scan', lens:'relational'|'grouped'|'facts', scope?, subject?, predicate?}
  filter    {op:'filter', field, cmp:'eq'|'ne'|'contains'|'gt'|'lt', value}   (value is JSON data)
  traverse  {op:'traverse', max_hops, rel_type?, from_field?='cell_ref'}
  project   {op:'project', fields:[...]}
  order_by  {op:'order_by', field, desc?}
  limit     {op:'limit', n}
  aggregate {op:'aggregate', group_by:[...], agg:'count'}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Row = dict[str, Any]
IR = list[dict[str, Any]]


@dataclass(frozen=True)
class QueryResult:
    """ToolResult-shaped result from :func:`run_query` (reuse, not reinvent)."""

    success: bool
    output: list[Row]
    error: str | None = None


# ---------------------------------------------------------------------------
# Operation implementations (pure, deterministic)
# ---------------------------------------------------------------------------


def _scan(store: Any, op: dict[str, Any]) -> list[Row]:
    lens = op.get("lens", "relational")
    if lens == "relational":
        return [dict(r) for r in store.query_relational(scope=op.get("scope")).output]
    if lens == "grouped":
        result = store.query_grouped(scope=op.get("scope"))
        rows: list[Row] = []
        for items in result.output.values():
            rows.extend(dict(r) for r in items)
        return rows
    if lens == "facts":
        result = store.query_facts(subject=op.get("subject"), predicate=op.get("predicate"))
        return [
            {"subject": f.subject, "predicate": f.predicate, "object": f.object}
            for f in result.output
        ]
    raise ValueError(f"unknown scan lens: {lens!r}")


def _filter(rows: list[Row], op: dict[str, Any]) -> list[Row]:
    field = op["field"]
    cmp = op["cmp"]
    value = op["value"]

    def keep(row: Row) -> bool:
        fv = row.get(field)
        if fv is None:
            return False
        if cmp == "eq":
            return bool(fv == value)
        if cmp == "ne":
            return bool(fv != value)
        if cmp == "contains":
            if isinstance(fv, bytes):
                needle = value.encode("utf-8") if isinstance(value, str) else value
                return needle in fv
            return value in fv
        if cmp == "gt":
            return bool(fv > value)
        if cmp == "lt":
            return bool(fv < value)
        raise ValueError(f"unknown filter cmp: {cmp!r}")

    return [r for r in rows if keep(r)]


def _traverse(store: Any, rows: list[Row], op: dict[str, Any]) -> list[Row]:
    max_hops = op["max_hops"]
    rel_type = op.get("rel_type")
    from_field = op.get("from_field", "cell_ref")
    seen: set[str] = set()
    out: list[Row] = []
    for row in rows:
        start = row.get(from_field)
        if start is None:
            continue
        for reached in store.query_graph(start, max_hops, rel_type).output:
            if reached not in seen:
                seen.add(reached)
                out.append({"cell_ref": reached})
    return out


def _sort_key(value: Any) -> tuple[bool, Any]:
    # None sorts last; otherwise natural order within a homogeneous column.
    return (value is None, value)


def _aggregate(rows: list[Row], op: dict[str, Any]) -> list[Row]:
    group_by: list[str] = op["group_by"]
    counts: dict[tuple[Any, ...], int] = {}
    for row in rows:
        key = tuple(row.get(f) for f in group_by)
        counts[key] = counts.get(key, 0) + 1
    out: list[Row] = []
    for key in sorted(counts, key=lambda k: tuple("" if x is None else str(x) for x in k)):
        row = {f: key[i] for i, f in enumerate(group_by)}
        row["count"] = counts[key]
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


def run_query(store: Any, ir: IR) -> QueryResult:
    """Execute *ir* against *store* (an ``AmplifierStore``); return a QueryResult.

    Data errors (unknown op/lens/cmp, bad field) are returned as
    ``QueryResult(success=False, error=...)`` — never raised into the caller.
    A pipeline MUST begin with a ``scan`` op.
    """
    if not ir or ir[0].get("op") != "scan":
        return QueryResult(False, [], "query must begin with a 'scan' op")
    try:
        rows: list[Row] = []
        for op in ir:
            name = op.get("op")
            if name == "scan":
                rows = _scan(store, op)
            elif name == "filter":
                rows = _filter(rows, op)
            elif name == "traverse":
                rows = _traverse(store, rows, op)
            elif name == "project":
                fields = op["fields"]
                rows = [{k: r[k] for k in fields if k in r} for r in rows]
            elif name == "order_by":
                field = op["field"]
                rows = sorted(
                    rows, key=lambda r: _sort_key(r.get(field)), reverse=op.get("desc", False)
                )
            elif name == "limit":
                rows = rows[: op["n"]]
            elif name == "aggregate":
                rows = _aggregate(rows, op)
            else:
                return QueryResult(False, [], f"unknown op: {name!r}")
        return QueryResult(True, rows, None)
    except (KeyError, TypeError, ValueError) as exc:
        return QueryResult(False, [], f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Fluent builder (the human surface)
# ---------------------------------------------------------------------------


class Query:
    """Fluent builder that compiles to QueryIR. ``execute(store)`` runs it."""

    def __init__(self) -> None:
        self._ops: IR = []

    @classmethod
    def scan(
        cls,
        lens: str = "relational",
        *,
        scope: str | None = None,
        subject: str | None = None,
        predicate: str | None = None,
    ) -> Query:
        """Start a pipeline by scanning a lens (relational / grouped / facts)."""
        q = cls()
        op: dict[str, Any] = {"op": "scan", "lens": lens}
        if scope is not None:
            op["scope"] = scope
        if subject is not None:
            op["subject"] = subject
        if predicate is not None:
            op["predicate"] = predicate
        q._ops.append(op)
        return q

    def filter(self, field: str, cmp: str, value: Any) -> Query:
        """Keep rows where ``row[field] <cmp> value`` (value is JSON data)."""
        self._ops.append({"op": "filter", "field": field, "cmp": cmp, "value": value})
        return self

    def traverse(
        self, max_hops: int, rel_type: str | None = None, from_field: str = "cell_ref"
    ) -> Query:
        """Expand each row's ref via the graph lens (bounded BFS)."""
        op: dict[str, Any] = {"op": "traverse", "max_hops": max_hops, "from_field": from_field}
        if rel_type is not None:
            op["rel_type"] = rel_type
        self._ops.append(op)
        return self

    def project(self, fields: list[str]) -> Query:
        """Keep only *fields* in each row."""
        self._ops.append({"op": "project", "fields": list(fields)})
        return self

    def order_by(self, field: str, desc: bool = False) -> Query:
        """Stable-sort rows by *field*."""
        self._ops.append({"op": "order_by", "field": field, "desc": desc})
        return self

    def limit(self, n: int) -> Query:
        """Truncate to the first *n* rows."""
        self._ops.append({"op": "limit", "n": n})
        return self

    def aggregate(self, group_by: list[str], agg: str = "count") -> Query:
        """Group by *group_by* and count (deterministically ordered)."""
        self._ops.append({"op": "aggregate", "group_by": list(group_by), "agg": agg})
        return self

    def to_ir(self) -> IR:
        """Return the JSON-serializable QueryIR for this query."""
        return [dict(o) for o in self._ops]

    @classmethod
    def from_ir(cls, ir: IR) -> Query:
        """Rebuild a Query from a QueryIR (the agentic ↔ human bridge)."""
        q = cls()
        q._ops = [dict(o) for o in ir]
        return q

    def execute(self, store: Any) -> QueryResult:
        """Run this query against *store*."""
        return run_query(store, self.to_ir())
