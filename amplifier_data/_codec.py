"""Primitive encode/decode helpers for the companion server codec layer.

Pure-stdlib (``base64``, ``json``), additive, touches no kernel/models/Rust code.
Used by every higher codec layer.
"""

from __future__ import annotations

import base64

from amplifier_data.models import (
    AccessEvent,
    Cell,
    CellWriteEvent,
    Event,
    IndexCellEvent,
    InterpreterOutputEvent,
    RelationshipEvent,
    ToolResultEvent,
)

# A sha256 hexdigest string — type alias used across the codebase.
Hash = str


def encode_bytes(b: bytes) -> str:
    """Encode raw bytes to a base64 ASCII string."""
    return base64.b64encode(b).decode("ascii")


def decode_bytes(s: str) -> bytes:
    """Decode a base64 ASCII string back to raw bytes."""
    return base64.b64decode(s.encode("ascii"))


def encode_interpreters(t: tuple[Hash, ...]) -> list[str]:
    """Encode a tuple of Hash (hex) strings as a list of strings.

    Hex strings pass through unchanged; the container changes from tuple to list.
    """
    return list(t)


def decode_interpreters(xs: list[str]) -> tuple[Hash, ...]:
    """Decode a list of hex strings back to a tuple of Hash strings."""
    return tuple(xs)


# ---------------------------------------------------------------------------
# Tagged Event encode/decode table
# ---------------------------------------------------------------------------


def encode_event(ev: Event) -> dict:
    """Encode a single Event to a JSON-serialisable dict with a ``_event`` tag.

    * ``bytes`` fields (``payload``, ``output``, ``access_summary``) are
      base64-encoded via :func:`encode_bytes`.
    * ``tuple[Hash, ...]`` fields (``interpreters``, ``tool_results_consumed``)
      are serialised as plain ``list[str]``.
    * All other fields (strings, Hash aliases) pass through as-is.
    """
    tag = type(ev).__name__

    if isinstance(ev, CellWriteEvent):
        return {
            "_event": tag,
            "payload": encode_bytes(ev.payload),
            "interpreters": list(ev.interpreters),
        }
    if isinstance(ev, RelationshipEvent):
        return {
            "_event": tag,
            "from_ref": ev.from_ref,
            "to_ref": ev.to_ref,
            "type": ev.type,
        }
    if isinstance(ev, ToolResultEvent):
        return {
            "_event": tag,
            "output": encode_bytes(ev.output),
        }
    if isinstance(ev, InterpreterOutputEvent):
        return {
            "_event": tag,
            "cell_ref": ev.cell_ref,
            "model": ev.model,
            "input_hash": ev.input_hash,
            "tool_results_consumed": list(ev.tool_results_consumed),
            "output": encode_bytes(ev.output),
        }
    if isinstance(ev, AccessEvent):
        return {
            "_event": tag,
            "cell_ref": ev.cell_ref,
        }
    if isinstance(ev, IndexCellEvent):
        return {
            "_event": tag,
            "cell_ref": ev.cell_ref,
            "access_summary": encode_bytes(ev.access_summary),
            "output": encode_bytes(ev.output),
        }
    # Fallback: unknown Event subtype
    raise ValueError(f"Unknown event type: {tag!r}")


def decode_event(d: dict) -> Event:
    """Reconstruct a frozen Event dataclass from a tagged dict produced by
    :func:`encode_event`.

    Raises :class:`ValueError` if the ``_event`` tag is unrecognised.
    """
    tag = d.get("_event")

    if tag == "CellWriteEvent":
        return CellWriteEvent(
            payload=decode_bytes(d["payload"]),
            interpreters=tuple(d["interpreters"]),
        )
    if tag == "RelationshipEvent":
        return RelationshipEvent(
            from_ref=d["from_ref"],
            to_ref=d["to_ref"],
            type=d["type"],
        )
    if tag == "ToolResultEvent":
        return ToolResultEvent(
            output=decode_bytes(d["output"]),
        )
    if tag == "InterpreterOutputEvent":
        return InterpreterOutputEvent(
            cell_ref=d["cell_ref"],
            model=d["model"],
            input_hash=d["input_hash"],
            tool_results_consumed=tuple(d["tool_results_consumed"]),
            output=decode_bytes(d["output"]),
        )
    if tag == "AccessEvent":
        return AccessEvent(
            cell_ref=d["cell_ref"],
        )
    if tag == "IndexCellEvent":
        return IndexCellEvent(
            cell_ref=d["cell_ref"],
            access_summary=decode_bytes(d["access_summary"]),
            output=decode_bytes(d["output"]),
        )
    raise ValueError(f"Unknown event tag: {tag!r}")


# ---------------------------------------------------------------------------
# Cell encode/decode
# ---------------------------------------------------------------------------


def encode_cell(cell: Cell) -> dict:
    """Encode a :class:`~amplifier_data.models.Cell` to a JSON-serialisable dict.

    Schema::

        {
            "payload":      <base64 str>,
            "provenance":   [<encode_event(e)>, ...],
            "interpreters": [<hex str>, ...],
        }
    """
    return {
        "payload": encode_bytes(cell.payload),
        "provenance": [encode_event(e) for e in cell.provenance],
        "interpreters": list(cell.interpreters),
    }


def decode_cell(d: dict) -> Cell:
    """Reconstruct a :class:`~amplifier_data.models.Cell` from the dict produced
    by :func:`encode_cell`.
    """
    return Cell(
        payload=decode_bytes(d["payload"]),
        provenance=tuple(decode_event(e) for e in d["provenance"]),
        interpreters=tuple(d["interpreters"]),
    )


# ---------------------------------------------------------------------------
# Recursive value encode/decode (base64-tags bytes, passes primitives through)
# ---------------------------------------------------------------------------

_B64_KEY = "__b64__"


def encode_value(v: object) -> object:
    """Recursively encode a value for JSON transport.

    * ``bytes``         → ``{"__b64__": "<base64-ascii>"}``
    * ``dict``          → encode each value recursively (keys unchanged)
    * ``list``/``tuple``→ encode each item recursively (always emits a list)
    * ``int``, ``str``, ``float``, ``bool``, ``None`` → pass through unchanged
    """
    if isinstance(v, bytes):
        return {_B64_KEY: base64.b64encode(v).decode("ascii")}
    if isinstance(v, dict):
        return {k: encode_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [encode_value(item) for item in v]
    return v


def decode_value(v: object) -> object:
    """Inverse of :func:`encode_value`.

    * ``{"__b64__": "..."}`` → ``bytes``
    * other ``dict``         → decode each value recursively
    * ``list``               → decode each item recursively (stays a list)
    * primitives             → pass through unchanged
    """
    if isinstance(v, dict):
        if list(v.keys()) == [_B64_KEY]:
            return base64.b64decode(v[_B64_KEY].encode("ascii"))
        return {k: decode_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [decode_value(item) for item in v]
    return v


# ---------------------------------------------------------------------------
# QueryResult encode/decode
# ---------------------------------------------------------------------------


def encode_query_result(qr: object) -> dict:
    """Encode a QueryResult to a JSON-serialisable dict with a ``_qr`` tag.

    Handles ``RelationalQueryResult``, ``GroupedQueryResult``, and
    ``IndexQueryResult``.  Nested ``bytes`` values (e.g. row ``payload``
    fields) are base64-tagged via :func:`encode_value`.

    Schema::

        {
            "_qr":     "<relational|grouped|index>",
            "success": <bool>,
            "output":  <output with all nested bytes base64-tagged>,
            "error":   <str | null>,
        }
    """
    # Import here to avoid circular-import issues; these are pure data classes.
    from amplifier_data.lenses.grouped import QueryResult as _Grouped
    from amplifier_data.lenses.index import QueryResult as _Index
    from amplifier_data.lenses.relational import QueryResult as _Relational

    if isinstance(qr, _Relational):
        tag = "relational"
    elif isinstance(qr, _Grouped):
        tag = "grouped"
    elif isinstance(qr, _Index):
        tag = "index"
    else:
        raise ValueError(f"Unknown QueryResult type: {type(qr)!r}")

    return {
        "_qr": tag,
        "success": qr.success,  # type: ignore[union-attr]
        "output": encode_value(qr.output),  # type: ignore[union-attr]
        "error": qr.error,  # type: ignore[union-attr]
    }


def decode_query_result(d: dict) -> object:
    """Reconstruct a QueryResult from the dict produced by :func:`encode_query_result`.

    Returns the matching concrete type
    (``RelationalQueryResult``, ``GroupedQueryResult``, or ``IndexQueryResult``)
    determined by the ``_qr`` tag.

    Raises :class:`ValueError` for unrecognised tags.
    """
    from amplifier_data.lenses.grouped import QueryResult as _Grouped
    from amplifier_data.lenses.index import QueryResult as _Index
    from amplifier_data.lenses.relational import QueryResult as _Relational

    tag: str = d["_qr"]
    success: bool = d["success"]
    output = decode_value(d["output"])
    error: str | None = d.get("error")

    if tag == "relational":
        return _Relational(success=success, output=output, error=error)  # type: ignore[arg-type]
    if tag == "grouped":
        return _Grouped(success=success, output=output, error=error)  # type: ignore[arg-type]
    if tag == "index":
        return _Index(success=success, output=output, error=error)  # type: ignore[arg-type]
    raise ValueError(f"Unknown _qr tag: {tag!r}")


# ---------------------------------------------------------------------------
# Vector-hit encode/decode
# ---------------------------------------------------------------------------


def encode_vector_hits(hits: list[tuple[Hash, float]]) -> list[list]:
    """Encode a list of ``(Hash, score)`` tuples to a JSON-serialisable list of lists.

    ``[(ref, score), ...]`` → ``[[ref, score], ...]``
    """
    return [[ref, score] for ref, score in hits]


def decode_vector_hits(xs: list) -> list[tuple[Hash, float]]:
    """Decode a list of ``[ref, score]`` pairs back to ``list[tuple[Hash, float]]``.

    ``[[ref, score], ...]`` → ``[(ref, score), ...]``
    """
    return [(item[0], item[1]) for item in xs]
