"""Tests for the embedder + vector-backend protocols (Plan 3i)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from amplifier_data import AmplifierStore
from amplifier_data.embedding import (
    BruteForceBackend,
    Embedder,
    HashingEmbedder,
    VectorBackend,
)
from amplifier_data.kernel import InMemoryKernel


def test_hashing_embedder_deterministic_and_normalized() -> None:
    emb = HashingEmbedder(dim=32)
    v1 = emb.embed("the quick brown fox")
    v2 = emb.embed("the quick brown fox")
    assert v1 == v2  # deterministic (sha1, not salted hash())
    assert len(v1) == 32
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-9  # L2-normalized
    assert emb.embed("totally different text") != v1


def test_reference_impls_satisfy_protocols() -> None:
    assert isinstance(HashingEmbedder(), Embedder)
    assert isinstance(BruteForceBackend(), VectorBackend)


def test_backend_matches_query_vector() -> None:
    store = AmplifierStore()
    emb = HashingEmbedder(dim=32)
    a = store.write_cell(b"alpha")
    b = store.write_cell(b"beta")
    c = store.write_cell(b"gamma")
    store.add_text_embedding(a, "the quick brown fox", emb)
    store.add_text_embedding(b, "lazy dog sleeps all day", emb)
    store.add_text_embedding(c, "quick brown foxes jump", emb)

    qv = emb.embed("quick brown fox")
    backend = BruteForceBackend.build_from_store(store)
    assert backend.search(qv, 3) == store.query_vector(qv, 3)  # reference == ground truth


def test_query_text_ranks_similar_first() -> None:
    store = AmplifierStore()
    emb = HashingEmbedder(dim=48)
    fox = store.write_cell(b"fox")
    dog = store.write_cell(b"dog")
    store.add_text_embedding(fox, "the quick brown fox jumps", emb)
    store.add_text_embedding(dog, "lazy dog sleeps all day", emb)

    res = store.query_text("quick brown fox", 2, emb)
    assert res[0][0] == fox  # most similar ranked first


def test_custom_backend_plugs_in() -> None:
    store = AmplifierStore()
    emb = HashingEmbedder(dim=16)
    a = store.write_cell(b"a")
    store.add_text_embedding(a, "hello world", emb)

    class ReversedBackend:
        def __init__(self) -> None:
            self.refs: list[str] = []

        def add(self, ref: str, vector: Sequence[float]) -> None:
            self.refs.append(ref)

        def search(self, vector: Sequence[float], k: int) -> list[tuple[str, float]]:
            return [(r, 0.0) for r in reversed(self.refs)][:k]

    assert isinstance(ReversedBackend(), VectorBackend)  # structural typing — the seam is real
    rb = ReversedBackend()
    for ref, vec in store.iter_embeddings():
        rb.add(ref, vec)
    assert rb.search(emb.embed("hello"), 1) == [(a, 0.0)]


def test_iter_embeddings_no_access_events() -> None:
    store = AmplifierStore()
    emb = HashingEmbedder(dim=8)
    a = store.write_cell(b"a")
    store.add_text_embedding(a, "hello", emb)
    kernel: InMemoryKernel = store.kernel  # type: ignore[assignment]
    before = len(kernel.all_events())
    store.iter_embeddings()
    assert len(kernel.all_events()) == before


def test_durable_restart(tmp_path: Path) -> None:
    log = tmp_path / "emb.log"
    emb = HashingEmbedder(dim=16)
    store = AmplifierStore(path=str(log))
    a = store.write_cell(b"alpha")
    store.add_text_embedding(a, "quick brown fox", emb)
    before = store.query_text("quick brown fox", 1, emb)
    store.close()

    reopened = AmplifierStore(path=str(log))
    after = reopened.query_text("quick brown fox", 1, emb)
    reopened.close()
    assert after == before
    assert after[0][0] == a
