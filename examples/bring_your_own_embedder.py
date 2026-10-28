"""Worked story: bring-your-own embedder + pluggable vector backend (Plan 3i).

amplifier-data ships NO embedding model and NO ANN — by design. It ships the
*studs*: an ``Embedder`` protocol (bring your own model) and a ``VectorBackend``
protocol (bring your own ANN: hnswlib / FAISS / a hosted vector database). The log stays the
source of truth; a backend is a rebuildable accelerator over the embedding events.

This demo uses the dependency-free reference ``HashingEmbedder`` and shows the
reference ``BruteForceBackend`` agreeing exactly with ``query_vector``.

Run:  python examples/bring_your_own_embedder.py   (exits 0 on success)
"""

from __future__ import annotations

from amplifier_data import AmplifierStore
from amplifier_data.embedding import BruteForceBackend, HashingEmbedder


def main() -> None:
    print("Bring-your-own embedder + pluggable vector backend\n")

    store = AmplifierStore()
    embedder = HashingEmbedder(dim=48)  # swap in a real model for production semantics

    fox = store.write_cell(b"doc: the quick brown fox")
    dog = store.write_cell(b"doc: lazy dog sleeps")
    jump = store.write_cell(b"doc: quick foxes jump high")
    store.add_text_embedding(fox, "the quick brown fox jumps", embedder)
    store.add_text_embedding(dog, "a lazy dog sleeps all day", embedder)
    store.add_text_embedding(jump, "quick brown foxes jump high", embedder)

    # Query by TEXT — the embedder is injected (bring-your-own).
    ranked = store.query_text("quick brown fox", 3, embedder)
    print(f"  query_text('quick brown fox') -> {[r[:8] for r, _ in ranked]}")
    assert ranked[0][0] != dog  # the dog doc is least similar
    print("  [\u2713] most-similar document ranked first\n")

    # The reference brute-force backend, rebuilt from the log, agrees exactly with
    # the lens — it is the ground truth a consumer's ANN backend is measured against.
    backend = BruteForceBackend.build_from_store(store)
    qv = embedder.embed("quick brown fox")
    assert backend.search(qv, 3) == store.query_vector(qv, 3)
    print("  [\u2713] BruteForceBackend.search == store.query_vector (rebuilt from the log)\n")

    print(
        "SUCCESS: text in -> bring-your-own embedder -> stored as regenerable events; "
        "a pluggable VectorBackend is a rebuildable accelerator over the log."
    )


if __name__ == "__main__":
    main()
