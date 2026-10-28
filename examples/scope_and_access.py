"""Scope and access control story.

What this script proves
-----------------------
Step 1  (setup): Four cells are written — a project scope, two in-scope
         documents, and one out-of-scope document.  doc_a and doc_b are
         explicitly scoped to the project; doc_c is not.

Step 2  (scoped relational query): ``store.query_relational(scope=project)``
         returns exactly the in-scope cells (doc_a, doc_b); doc_c is excluded.

Step 3  (scoped vector query): ``store.query_vector((1.0, 0.0), k=5, scope=project)``
         returns only embeddings whose target cell is in scope.  doc_c has the
         identical embedding (1.0, 0.0) but is excluded because it is not
         scoped to the project.

Step 4  (access opt-out): ``store.get_cell(doc_c, record_access=False)``
         resolves the cell payload correctly but appends **zero** AccessEvents.
         The before/after count proves the suppression.

Step 5  (normal access): A plain ``store.get_cell(doc_a)`` increments the
         AccessEvent count by exactly 1, confirming the default remains active.

Run::

    python examples/scope_and_access.py   # or:  uv run python examples/scope_and_access.py

Exit code 0 on success; 1 on any assertion failure.
"""

from __future__ import annotations

import sys

from amplifier_data.models import AccessEvent
from amplifier_data.store import AmplifierStore


def _access_count(store: AmplifierStore) -> int:
    """Sum all AccessEvents currently in the kernel log."""
    return sum(1 for _, ev in store.kernel.all_events() if isinstance(ev, AccessEvent))  # type: ignore[attr-defined]


def main() -> int:
    """Run the scope-and-access demonstration.  Returns 0 on success, 1 on failure."""

    # ------------------------------------------------------------------ #
    # Step 1 — setup: write cells and establish scope                     #
    # ------------------------------------------------------------------ #
    print("\nStep 1: write cells and establish scope\n")

    store = AmplifierStore()

    project = store.write_cell(b"project-x")
    doc_a = store.write_cell(b"doc-a")
    doc_b = store.write_cell(b"doc-b")
    doc_c = store.write_cell(b"doc-c")  # intentionally out-of-scope

    # Scope doc_a and doc_b into the project; leave doc_c unscoped.
    store.scope(doc_a, project)
    store.scope(doc_b, project)

    # Add embeddings: doc_a ≈ (1,0), doc_b ≈ (0,1), doc_c ≈ (1,0) (out-of-scope).
    store.add_embedding(doc_a, (1.0, 0.0))
    store.add_embedding(doc_b, (0.0, 1.0))
    store.add_embedding(doc_c, (1.0, 0.0))

    print(f"  project : {project[:16]}")
    print(f"  doc_a   : {doc_a[:16]}  (in-scope, embedding (1,0))")
    print(f"  doc_b   : {doc_b[:16]}  (in-scope, embedding (0,1))")
    print(f"  doc_c   : {doc_c[:16]}  (out-of-scope, embedding (1,0))")

    # ------------------------------------------------------------------ #
    # Step 2 — scoped relational query                                    #
    # ------------------------------------------------------------------ #
    print("\nStep 2: scoped relational query\n")

    result = store.query_relational(scope=project)
    assert result.success, f"query_relational failed: {result.error}"

    members = sorted(r["cell_ref"] for r in result.output)
    print(f"  Scoped members (sorted): {[m[:16] for m in members]}")

    assert doc_a in members, "doc_a must be in scoped relational results"
    assert doc_b in members, "doc_b must be in scoped relational results"
    assert doc_c not in members, "doc_c (out-of-scope) must NOT appear in scoped results"

    print("  [✓] doc_a and doc_b present; doc_c excluded")

    # ------------------------------------------------------------------ #
    # Step 3 — scoped vector query                                        #
    # ------------------------------------------------------------------ #
    print("\nStep 3: scoped vector query\n")

    hits = store.query_vector((1.0, 0.0), k=5, scope=project)
    print("  Vector hits (ref[:16] / score):")
    for ref, score in hits:
        print(f"    {ref[:16]}  score={score:.6f}")

    hit_refs = [ref for ref, _ in hits]
    assert doc_c not in hit_refs, "doc_c has embedding (1,0) but must be excluded by scope filter"
    assert doc_a in hit_refs, "doc_a (in-scope, embedding (1,0)) must appear in hits"

    print("  [✓] doc_c excluded despite matching embedding; doc_a included")

    # ------------------------------------------------------------------ #
    # Step 4 — access opt-out: record_access=False                       #
    # ------------------------------------------------------------------ #
    print("\nStep 4: access opt-out  (record_access=False)\n")

    before = _access_count(store)
    cell = store.get_cell(doc_c, record_access=False)
    after = _access_count(store)
    delta = after - before

    print(f"  doc_c payload       : {cell.payload}")
    print(f"  AccessEvent count before : {before}")
    print(f"  AccessEvent count after  : {after}")
    print(f"  delta (expected 0)       : {delta}")

    assert cell.payload == b"doc-c", f"payload mismatch: {cell.payload!r}"
    assert after == before, f"record_access=False must suppress AccessEvent; got delta={delta}"

    print("  [✓] record_access=False: payload resolved, no AccessEvent appended")

    # ------------------------------------------------------------------ #
    # Step 5 — normal access (records exactly 1 AccessEvent)             #
    # ------------------------------------------------------------------ #
    print("\nStep 5: normal access (record_access default=True)\n")

    before2 = _access_count(store)
    _ = store.get_cell(doc_a)
    after2 = _access_count(store)

    print(f"  AccessEvent count after normal get_cell : {after2}")
    print(f"  Total AccessEvents recorded in log       : {after2}")
    assert after2 == before2 + 1, (
        f"normal get_cell must record exactly 1 AccessEvent; delta={after2 - before2}"
    )

    print("  [✓] Normal access recorded exactly 1 AccessEvent")

    # ------------------------------------------------------------------ #
    # Done                                                                 #
    # ------------------------------------------------------------------ #
    print(
        "\nSUCCESS: scope-and-access story complete — "
        "scoped relational query, scoped vector query, "
        "access opt-out, and normal access recording.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
