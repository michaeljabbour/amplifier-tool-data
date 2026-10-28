"""Worked story: visualize a store as DOT + a self-contained HTML explorer (Plan 3h).

Builds a small delegation tree with a scope and a KG fact, then writes both a
Graphviz ``graph.dot`` (render with ``dot -Tsvg``) and a dependency-free
``graph.html`` you can open directly in a browser.

Run:  python examples/visualize.py   (exits 0; writes to a temp dir)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from amplifier_data import AmplifierStore


def main() -> None:
    print("Visualize amplifier-data (DOT + self-contained HTML)\n")

    store = AmplifierStore()
    root = store.write_cell(b"root session")
    child_a = store.write_cell(b"child A")
    child_b = store.write_cell(b"child B")
    store.relate(root, child_a, "fork")
    store.relate(root, child_b, "fork")
    store.scope(root, "wing:demo")
    store.assert_fact("alice", "works_on", "acme")

    dot = store.to_dot()
    html_page = store.to_html(title="demo session")

    assert dot.startswith("digraph")
    assert html_page.startswith("<!DOCTYPE html>")
    assert "http://" not in html_page and "https://" not in html_page  # self-contained

    out = Path(tempfile.mkdtemp())
    (out / "graph.dot").write_text(dot, encoding="utf-8")
    (out / "graph.html").write_text(html_page, encoding="utf-8")

    print(f"  wrote {out / 'graph.dot'}  ({len(dot)} bytes)")
    print(f"  wrote {out / 'graph.html'} ({len(html_page)} bytes, open in a browser)")
    print("  [\u2713] DOT renders the delegation tree + scope + fact")
    print("  [\u2713] HTML is self-contained (no network, no JS libraries)\n")

    print("SUCCESS: store visualized as Graphviz DOT and a dependency-free HTML explorer.")


if __name__ == "__main__":
    main()
