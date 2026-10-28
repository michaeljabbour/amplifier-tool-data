"""Vector lens story — append embeddings, query, regenerate byte-identically, restart.

What this script proves
-----------------------
Part 1  (in-memory): Three targets with known unit vectors are written and
         linked to embeddings.  A top-3 cosine query returns them in the
         expected order: the near-duplicate ranks first, the orthogonal
         vector ranks last.

Part 2  (byte-identical regeneration): The VectorIndex produced by
         ``VectorLens.project()`` serialises to identical bytes regardless of
         how many fresh ``VectorLens`` instances are instantiated over the
         same kernel.  Proves the Design Decision D1 pure-fold guarantee.

Part 3  (durable restart, only when RUST_AVAILABLE): Embeddings written to a
         path-backed ``AmplifierStore`` survive a ``close()`` / reopen cycle
         with the same ``query_vector`` output, proving E1 restart-survival for
         the vector layer.

Run::

    python examples/vector_lens.py   # or:  uv run python examples/vector_lens.py

Exit code 0 on success; 1 on any assertion failure.
"""

from __future__ import annotations

import os
import sys
import tempfile

from amplifier_data import RUST_AVAILABLE
from amplifier_data.kernel import InMemoryKernel
from amplifier_data.lenses.vector import VectorLens
from amplifier_data.store import AmplifierStore


def main() -> int:
    """Run the vector-lens demonstration.  Returns 0 on success, 1 on failure."""

    # ------------------------------------------------------------------ #
    # Part 1 — in-memory cosine ranking                                    #
    # ------------------------------------------------------------------ #
    print("\nPart 1: in-memory cosine ranking\n")

    store = AmplifierStore(kernel=InMemoryKernel())

    # Write three target cells with distinct semantic content.
    near = store.write_cell(b"the cat sat on the mat")
    diag = store.write_cell(b"a dog barked loudly")
    orth = store.write_cell(b"quarterly revenue projections")

    # Link each target to an embedding vector.
    # near  ≈ (1, 0, 0) — same direction as the query
    # diag  ≈ (1, 1, 0) — diagonal, cosine ≈ 0.707 with query
    # orth  ≈ (0, 1, 0) — orthogonal, cosine = 0.0 with query
    store.add_embedding(near, (1, 0, 0))
    store.add_embedding(diag, (1, 1, 0))
    store.add_embedding(orth, (0, 1, 0))

    # Query with the unit vector (1, 0, 0).
    results = store.query_vector((1, 0, 0), k=3)

    print("  Top-3 results (ref[:16] / score):")
    for ref, score in results:
        print(f"    {ref[:16]}  score={score:.6f}")

    assert results[0][0] == near, "near-duplicate must rank first"
    assert results[-1][0] == orth, "orthogonal vector must rank last"

    print("\n  [✓] near-duplicate ranked first")
    print("  [✓] orthogonal vector ranked last")

    # ------------------------------------------------------------------ #
    # Part 2 — byte-identical regeneration                                 #
    # ------------------------------------------------------------------ #
    print("\nPart 2: byte-identical regeneration\n")

    lens = VectorLens()
    bytes_a = lens.project(store.kernel).canonical_bytes()
    bytes_b = VectorLens().project(store.kernel).canonical_bytes()

    print(f"  bytes_a length: {len(bytes_a)}")
    print(f"  bytes_b length: {len(bytes_b)}")
    assert bytes_a == bytes_b, "canonical_bytes must be byte-identical across re-folds"

    print("  [✓] canonical_bytes byte-identical across independent VectorLens instances")

    # ------------------------------------------------------------------ #
    # Part 3 — durable restart (Rust extension required)                   #
    # ------------------------------------------------------------------ #
    print("\nPart 3: durable restart\n")

    if RUST_AVAILABLE:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "vector_demo.log")

            # Session 1: write a target + embedding, capture query result.
            s1 = AmplifierStore(path=path)
            try:
                t = s1.write_cell(b"durable target")
                s1.add_embedding(t, (1, 0, 0))
                before = s1.query_vector((1, 0, 0), k=1)
                print(f"  before (session 1): {before}")
            finally:
                s1.close()

            # Session 2: reopen the same path (simulated restart).
            s2 = AmplifierStore(path=path)
            try:
                after = s2.query_vector((1, 0, 0), k=1)
                print(f"  after  (session 2): {after}")
            finally:
                s2.close()

        assert after == before, "embeddings must survive durable restart"
        print("  [✓] embeddings survived durable restart (before == after)")
    else:
        print(
            "  SKIPPED — durable restart requires the compiled Rust extension "
            "(RUST_AVAILABLE is False).  Build it with `maturin develop` to enable."
        )

    # ------------------------------------------------------------------ #
    # Done                                                                 #
    # ------------------------------------------------------------------ #
    print(
        "\nSUCCESS: vector-lens story complete — "
        "cosine ranking, byte-identical regeneration"
        + (", and durable restart" if RUST_AVAILABLE else " (durable restart skipped)")
        + ".\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
