"""Unit tests for VectorLens — Plan3c (Task 9).

Tests target the lens directly (not via AmplifierStore), using InMemoryKernel
as the backing store.  Covers:

* ``VectorLens.name``
* ``VectorLens.project()``  — pure fold, D1 compliance
* ``VectorLens.query()``    — top-k cosine nearest-neighbour
* ``cosine_similarity()``   — helper
* ``VectorIndex.canonical_bytes()`` — deterministic serialisation
"""

from __future__ import annotations

import math
import struct

import pytest

from amplifier_data.kernel import InMemoryKernel
from amplifier_data.lenses.vector import (
    EMBEDDING_OF,
    VectorEntry,
    VectorIndex,
    VectorLens,
    VectorQueryResult,
    cosine_similarity,
)
from amplifier_data.models import CellWriteEvent, RelationshipEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kernel_with_embedding(
    kernel: InMemoryKernel,
    target_payload: bytes,
    vector: tuple[float, ...],
) -> tuple[str, str]:
    """Append a target cell and its embedding cell; return (target_ref, emb_ref)."""
    target_event = CellWriteEvent(payload=target_payload)
    target_ref = target_event.cell_ref()
    kernel.append_batch([target_event])

    packed = struct.pack(f"<{len(vector)}f", *vector)
    emb_event = CellWriteEvent(payload=packed)
    emb_ref = emb_event.cell_ref()
    kernel.append_batch([emb_event])

    rel_event = RelationshipEvent(type=EMBEDDING_OF, from_ref=emb_ref, to_ref=target_ref)
    kernel.append_batch([rel_event])

    return target_ref, emb_ref


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors() -> None:
    """Identical non-zero vectors have cosine similarity 1.0."""
    assert cosine_similarity((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    """Orthogonal vectors have cosine similarity 0.0."""
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors() -> None:
    """Opposite vectors have cosine similarity -1.0."""
    assert cosine_similarity((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_returns_zero() -> None:
    """Zero-magnitude vector returns 0.0 (undefined cosine ⇒ safe fallback)."""
    assert cosine_similarity((0.0, 0.0), (1.0, 1.0)) == 0.0


def test_cosine_similarity_dimension_mismatch_raises() -> None:
    """Mismatched dimensions raise ValueError."""
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity((1.0, 0.0), (0.0, 1.0, 0.0))


def test_cosine_similarity_normalised_value() -> None:
    """Cosine of 45° vectors: (1,1) and (1,0) → 1/√2 ≈ 0.7071."""
    expected = 1.0 / math.sqrt(2.0)
    assert cosine_similarity((1.0, 1.0), (1.0, 0.0)) == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# VectorLens.name
# ---------------------------------------------------------------------------


def test_vector_lens_name() -> None:
    """VectorLens.name must return 'vector'."""
    assert VectorLens().name == "vector"


# ---------------------------------------------------------------------------
# VectorLens.project
# ---------------------------------------------------------------------------


def test_project_empty_kernel_returns_empty_index() -> None:
    """project() on an empty kernel returns a VectorIndex with no entries."""
    kernel = InMemoryKernel()
    index = VectorLens().project(kernel)
    assert isinstance(index, VectorIndex)
    assert index.entries == ()


def test_project_single_embedding() -> None:
    """project() correctly folds one embedding into a single VectorEntry."""
    kernel = InMemoryKernel()
    vector = (1.0, 0.0, 0.0)
    target_ref, emb_ref = _kernel_with_embedding(kernel, b"target A", vector)

    index = VectorLens().project(kernel)
    assert len(index.entries) == 1
    entry = index.entries[0]
    assert isinstance(entry, VectorEntry)
    assert entry.target_ref == target_ref
    assert entry.emb_ref == emb_ref
    assert entry.vector == pytest.approx(vector)


def test_project_multiple_embeddings_sorted_canonically() -> None:
    """project() returns entries sorted by (emb_ref, target_ref) deterministically."""
    kernel = InMemoryKernel()
    t1, e1 = _kernel_with_embedding(kernel, b"first", (1.0, 0.0))
    t2, e2 = _kernel_with_embedding(kernel, b"second", (0.0, 1.0))

    index = VectorLens().project(kernel)
    assert len(index.entries) == 2

    # Canonical sort: (emb_ref, target_ref) — string ordering on hex refs.
    refs = [(e.emb_ref, e.target_ref) for e in index.entries]
    assert refs == sorted(refs), "entries must be sorted by (emb_ref, target_ref)"


def test_project_is_pure_two_instances_independent() -> None:
    """D1: two VectorLens instances over the same kernel produce equal results."""
    kernel = InMemoryKernel()
    _kernel_with_embedding(kernel, b"data", (0.5, 0.5))

    lens_a = VectorLens()
    lens_b = VectorLens()
    assert lens_a.project(kernel) == lens_b.project(kernel)


def test_project_reflects_new_events_immediately() -> None:
    """D1: project() reads fresh each time — adding an embedding shows up next call."""
    kernel = InMemoryKernel()
    lens = VectorLens()

    assert len(lens.project(kernel).entries) == 0

    _kernel_with_embedding(kernel, b"new", (1.0, 0.0))
    assert len(lens.project(kernel).entries) == 1


def test_project_dangling_edge_skipped_gracefully() -> None:
    """RelationshipEvent with no matching CellWriteEvent is silently skipped."""
    kernel = InMemoryKernel()
    # Append a relationship edge without the corresponding cell write.
    rel = RelationshipEvent(type=EMBEDDING_OF, from_ref="deadbeef" * 8, to_ref="cafebabe" * 8)
    kernel.append_batch([rel])

    index = VectorLens().project(kernel)
    assert index.entries == (), "dangling EMBEDDING_OF edge must be silently ignored"


# ---------------------------------------------------------------------------
# VectorIndex.canonical_bytes
# ---------------------------------------------------------------------------


def test_canonical_bytes_empty_index_is_empty() -> None:
    """canonical_bytes() of an empty VectorIndex returns b''."""
    index = VectorIndex(entries=())
    assert index.canonical_bytes() == b""


def test_canonical_bytes_is_deterministic() -> None:
    """canonical_bytes() returns the same bytes on repeated calls."""
    kernel = InMemoryKernel()
    _kernel_with_embedding(kernel, b"doc", (0.1, 0.2, 0.3))
    index = VectorLens().project(kernel)
    assert index.canonical_bytes() == index.canonical_bytes()


def test_canonical_bytes_changes_with_new_embedding() -> None:
    """canonical_bytes() changes after a new embedding is added."""
    kernel = InMemoryKernel()
    _kernel_with_embedding(kernel, b"v1", (1.0, 0.0))
    index_before = VectorLens().project(kernel).canonical_bytes()

    _kernel_with_embedding(kernel, b"v2", (0.0, 1.0))
    index_after = VectorLens().project(kernel).canonical_bytes()

    assert index_before != index_after


# ---------------------------------------------------------------------------
# VectorLens.query
# ---------------------------------------------------------------------------


def test_query_empty_store_returns_empty_result() -> None:
    """query() on a kernel with no embeddings returns a successful, empty result."""
    kernel = InMemoryKernel()
    result = VectorLens().query(kernel=kernel, vector=(1.0, 0.0))
    assert isinstance(result, VectorQueryResult)
    assert result.success is True
    assert result.output == []
    assert result.error is None


def test_query_top1_nearest_neighbor() -> None:
    """query(k=1) returns the single most-similar embedding."""
    kernel = InMemoryKernel()
    near, _ = _kernel_with_embedding(kernel, b"near", (1.0, 0.0))
    far, _ = _kernel_with_embedding(kernel, b"far", (0.0, 1.0))

    result = VectorLens().query(kernel=kernel, vector=(1.0, 0.0), k=1)
    assert result.success is True
    assert len(result.output) == 1
    assert result.output[0][0] == near
    assert result.output[0][1] == pytest.approx(1.0)


def test_query_sorted_by_descending_cosine() -> None:
    """query() returns all results sorted by descending cosine similarity."""
    kernel = InMemoryKernel()
    near, _ = _kernel_with_embedding(kernel, b"near", (1.0, 0.0))
    orth, _ = _kernel_with_embedding(kernel, b"orth", (0.0, 1.0))
    anti, _ = _kernel_with_embedding(kernel, b"anti", (-1.0, 0.0))

    result = VectorLens().query(kernel=kernel, vector=(1.0, 0.0), k=3)
    refs = [ref for ref, _ in result.output]
    assert refs.index(near) < refs.index(orth), "near must precede orth"
    assert refs.index(orth) < refs.index(anti), "orth must precede anti"


def test_query_k_none_returns_all() -> None:
    """query(k=None) returns all embeddings."""
    kernel = InMemoryKernel()
    for i in range(5):
        _kernel_with_embedding(kernel, f"doc{i}".encode(), (float(i), 0.0))

    result = VectorLens().query(kernel=kernel, vector=(1.0, 0.0), k=None)
    assert len(result.output) == 5


def test_query_k_zero_returns_empty() -> None:
    """query(k=0) returns an empty list (edge case)."""
    kernel = InMemoryKernel()
    _kernel_with_embedding(kernel, b"x", (1.0, 0.0))
    result = VectorLens().query(kernel=kernel, vector=(1.0, 0.0), k=0)
    assert result.output == []


def test_query_missing_kernel_arg_raises_type_error() -> None:
    """query() without 'kernel' raises TypeError."""
    with pytest.raises(TypeError, match="kernel"):
        VectorLens().query(vector=(1.0, 0.0))


def test_query_missing_vector_arg_raises_type_error() -> None:
    """query() without 'vector' raises TypeError."""
    kernel = InMemoryKernel()
    with pytest.raises(TypeError, match="vector"):
        VectorLens().query(kernel=kernel)


def test_query_dimension_mismatch_raises_value_error() -> None:
    """query() raises ValueError if stored embedding dim ≠ query vector dim."""
    kernel = InMemoryKernel()
    _kernel_with_embedding(kernel, b"2d", (1.0, 0.0))  # 2-dimensional
    with pytest.raises(ValueError, match="dimension mismatch"):
        VectorLens().query(kernel=kernel, vector=(1.0, 0.0, 0.0))  # 3-dimensional
