"""Projection / visualization layer (Plan 3h).

Turns an ``AmplifierStore`` (or a subset of its cells) into a *visual* artifact:

  * :func:`to_dot` — a Graphviz **DOT** string (the canonical, deterministic
    visualization; render with ``dot -Tsvg`` or the ecosystem ``dot_graph`` tool).
  * :func:`to_html` — a **self-contained, dependency-free HTML** explorer: an
    inline-SVG node-link diagram (deterministic circular layout) plus tables of
    cells / edges / facts / scopes. Open it directly in a browser — no server,
    no JS libraries, no network.

Both consume a :class:`Projection` — a pure, deterministic fold of the store:
content cells (nodes), relationship edges, currently-valid KG facts, and scope
memberships. Reserved internal edge types (``scoped_to`` -> scopes,
``embedding_of`` -> omitted, ``__invalidate__:`` tombstones -> omitted) are not
drawn as generic edges. Building a projection appends NO AccessEvents (D4).
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Any

from .lenses._scope import SCOPED_TO
from .lenses.temporal import INVALIDATE_PREFIX
from .lenses.vector import EMBEDDING_OF

_PREVIEW_LEN = 32


@dataclass(frozen=True)
class Projection:
    """A deterministic, render-agnostic view of a store (or a subset of cells)."""

    nodes: list[tuple[str, str]]  # (cell_ref, payload preview)
    edges: list[tuple[str, str, str]]  # (from_ref, to_ref, rel_type)
    facts: list[tuple[str, str, str]]  # (subject, predicate, object) — currently valid
    scopes: list[tuple[str, str]]  # (cell_ref, scope_ref)


def _preview(payload: bytes) -> str:
    text = payload[:_PREVIEW_LEN].decode("utf-8", "replace")
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
    if len(payload) > _PREVIEW_LEN:
        text += "..."
    return text


def _short(ref: str) -> str:
    return ref[:8] if len(ref) > 8 else ref


def build_projection(store: Any, cell_refs: list[str] | None = None) -> Projection:
    """Fold *store* into a :class:`Projection` (optionally restricted to *cell_refs*).

    Pure / deterministic (sorted) and read-only (no AccessEvents).
    """
    from .models import RelationshipEvent

    keep: set[str] | None = set(cell_refs) if cell_refs is not None else None

    previews: dict[str, str] = {}
    for row in store.query_relational().output:
        ref = row["cell_ref"]
        if keep is None or ref in keep:
            previews[ref] = _preview(row.get("payload", b"") or b"")

    edges: list[tuple[str, str, str]] = []
    scopes: list[tuple[str, str]] = []
    for _seq, event in store.kernel.all_events():
        if not isinstance(event, RelationshipEvent):
            continue
        etype = event.type
        if etype.startswith(INVALIDATE_PREFIX) or etype == EMBEDDING_OF:
            continue
        if etype == SCOPED_TO:
            if keep is None or event.from_ref in keep:
                scopes.append((event.from_ref, event.to_ref))
            continue
        if keep is not None and (event.from_ref not in keep or event.to_ref not in keep):
            continue
        edges.append((event.from_ref, event.to_ref, etype))

    facts = [(f.subject, f.predicate, f.object) for f in store.query_facts().output]

    return Projection(
        nodes=sorted(previews.items()),
        edges=sorted(set(edges)),
        facts=sorted(set(facts)),
        scopes=sorted(set(scopes)),
    )


# ---------------------------------------------------------------------------
# DOT
# ---------------------------------------------------------------------------


def _dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def to_dot(projection: Projection, *, name: str = "amplifier_data") -> str:
    """Render *projection* as a deterministic Graphviz DOT string."""
    lines: list[str] = [
        f"digraph {name} {{",
        "  rankdir=LR;",
        "  node [shape=box, style=rounded];",
    ]
    for ref, preview in projection.nodes:
        text = f"{_short(ref)}\n{preview}" if preview else _short(ref)
        lines.append(f'  "{ref}" [label="{_dot_escape(text)}"];')
    for frm, to, etype in projection.edges:
        lines.append(f'  "{frm}" -> "{to}" [label="{_dot_escape(etype)}"];')
    for cell, scope in projection.scopes:
        label = _dot_escape(scope)
        lines.append(f'  "{cell}" -> "{label}" [label="scoped_to", style=dashed, color=gray];')
    for subj, pred, obj in projection.facts:
        s, o = f"fact:{subj}", f"fact:{obj}"
        lines.append(f'  "{s}" [label="{_dot_escape(subj)}", shape=ellipse, color=blue];')
        lines.append(f'  "{o}" [label="{_dot_escape(obj)}", shape=ellipse, color=blue];')
        lines.append(f'  "{s}" -> "{o}" [label="{_dot_escape(pred)}", color=blue];')
    lines.append("}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# HTML (self-contained, dependency-free)
# ---------------------------------------------------------------------------

_HTML_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#fafafa;color:#222}
header{padding:12px 20px;background:#1f2933;color:#fff}
header h1{margin:0;font-size:16px;font-weight:600}
.wrap{display:flex;flex-wrap:wrap;gap:16px;padding:16px}
.panel{background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:12px}
svg{background:#fff;border-radius:8px}
circle.node{fill:#3b82f6;stroke:#1e3a8a;stroke-width:1.5}
text.node{font-size:10px;fill:#111}
line.edge{stroke:#9ca3af;stroke-width:1.2}
table{border-collapse:collapse;font-size:12px}
th,td{border:1px solid #e3e3e3;padding:3px 8px;text-align:left}
th{background:#f3f4f6}
.muted{color:#6b7280;font-size:11px}
"""


def _rows_html(headers: list[str], rows: list[tuple[str, ...]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def to_html(projection: Projection, *, title: str = "amplifier-data") -> str:
    """Render *projection* as a self-contained, dependency-free HTML explorer.

    Deterministic circular layout computed in Python (no JS libraries, no
    network). Open the file directly in a browser.
    """
    n = len(projection.nodes)
    size = 640
    cx = cy = size / 2
    radius = size / 2 - 60
    pos: dict[str, tuple[float, float]] = {}
    for i, (ref, _p) in enumerate(projection.nodes):
        angle = (2 * math.pi * i / n) - math.pi / 2 if n else 0.0
        pos[ref] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

    svg: list[str] = [f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">']
    for frm, to, _etype in projection.edges:
        if frm in pos and to in pos:
            x1, y1 = pos[frm]
            x2, y2 = pos[to]
            svg.append(
                f'<line class="edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>'
            )
    for ref, preview in projection.nodes:
        x, y = pos[ref]
        tip = html.escape(f"{ref}: {preview}")
        svg.append(
            f'<circle class="node" cx="{x:.1f}" cy="{y:.1f}" r="7"><title>{tip}</title></circle>'
        )
        svg.append(
            f'<text class="node" x="{x + 9:.1f}" y="{y + 3:.1f}">{html.escape(_short(ref))}</text>'
        )
    svg.append("</svg>")

    node_data = [(_short(r), p) for r, p in projection.nodes]
    edge_data = [(_short(a), _short(b), t) for a, b, t in projection.edges]
    scope_data = [(_short(c), s) for c, s in projection.scopes]
    cells_tbl = _rows_html(["ref", "preview"], node_data)
    edge_tbl = _rows_html(["from", "to", "type"], edge_data)
    fact_tbl = _rows_html(["subject", "predicate", "object"], list(projection.facts))
    scope_tbl = _rows_html(["cell", "scope"], scope_data)
    esc_title = html.escape(title)
    summary = (
        f"{n} cells &middot; {len(projection.edges)} edges &middot; {len(projection.facts)} facts"
    )

    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>{esc_title}</title><style>{_HTML_CSS}</style></head><body>",
        f"<header><h1>{esc_title} &mdash; amplifier-data projection</h1></header>",
        "<div class='wrap'>",
        f"<div class='panel'><div class='muted'>{summary}</div>",
        "".join(svg),
        "</div>",
        f"<div class='panel'><b>Cells</b>{cells_tbl}</div>",
        f"<div class='panel'><b>Edges</b>{edge_tbl}</div>",
        f"<div class='panel'><b>Facts</b>{fact_tbl}</div>",
        f"<div class='panel'><b>Scopes</b>{scope_tbl}</div>",
        "</div></body></html>",
    ]
    return "".join(parts)
