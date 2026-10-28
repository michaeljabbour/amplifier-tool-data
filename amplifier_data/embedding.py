"""Embedder + vector-backend protocols (Plan 3i) — mechanism, not policy.

memory (and any consumer) flagged two real gaps in the vector story: amplifier-data
ships **no embedding model** and only **brute-force** cosine (no ANN at scale). The
amplifier-way answer is NOT to bake in a specific model or ANN library — it is to
expose **studs** so a consumer plugs in their own:

  * :class:`Embedder` — ``embed(text) -> Sequence[float]``. Bring your own model.
    :class:`HashingEmbedder` is a deterministic, dependency-free reference used in
    tests/examples (NOT a semantic model — a hashing bag-of-features).
  * :class:`VectorBackend` — ``add`` / ``search``. The pluggable ANN seam. A consumer
    backs it with hnswlib / FAISS / a hosted vector database. :class:`BruteForceBackend` is the
    reference / ground-truth impl and agrees with ``AmplifierStore.query_vector``.

The log stays the source of truth: embeddings are ordinary events (regenerable, E1).
A :class:`VectorBackend` is a **rebuildable accelerator** over those events —
``BruteForceBackend.build_from_store`` (and a consumer's ANN equivalent) repopulates
it from ``store.iter_embeddings()``. The index is a cache; the log is truth.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .lenses.vector import cosine_similarity


@runtime_checkable
class Embedder(Protocol):
    """Turn text into a vector. Bring your own model; amplifier-data ships none."""

    def embed(self, text: str) -> Sequence[float]: ...


@runtime_checkable
class VectorBackend(Protocol):
    """Pluggable nearest-neighbor index (the ANN seam). Back it with hnswlib/FAISS/a vector DB."""

    def add(self, ref: str, vector: Sequence[float]) -> None: ...

    def search(self, vector: Sequence[float], k: int) -> list[tuple[str, float]]: ...


# ---------------------------------------------------------------------------
# Reference implementations (deterministic, dependency-free)
# ---------------------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    trigrams = [text[i : i + 3] for i in range(len(text) - 2)]
    return words + trigrams


class HashingEmbedder:
    """Deterministic, dependency-free reference Embedder (a hashing bag-of-features).

    NOT a semantic model — it hashes word + character-trigram tokens into ``dim``
    buckets and L2-normalizes. Deterministic (uses sha1, never the salted builtin
    ``hash``), good enough for tests, examples, and small exact use; swap in a real
    model for production semantics.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokens(text):
            digest = hashlib.sha1(token.encode("utf-8")).digest()[:8]  # noqa: S324 (non-crypto use)
            vec[int.from_bytes(digest, "big") % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return vec
        return [x / norm for x in vec]


class BruteForceBackend:
    """Reference VectorBackend: exact cosine. Ground truth for any ANN consumer plugs in."""

    def __init__(self) -> None:
        self._items: list[tuple[str, list[float]]] = []

    def add(self, ref: str, vector: Sequence[float]) -> None:
        self._items.append((ref, list(vector)))

    def search(self, vector: Sequence[float], k: int) -> list[tuple[str, float]]:
        scored = [(ref, cosine_similarity(vector, vec)) for ref, vec in self._items]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[: max(k, 0)]

    @classmethod
    def build_from_store(cls, store: object) -> BruteForceBackend:
        """Rebuild the index from the log via ``store.iter_embeddings()`` (E1: log is truth)."""
        backend = cls()
        for ref, vector in store.iter_embeddings():  # type: ignore[attr-defined]
            backend.add(ref, vector)
        return backend
