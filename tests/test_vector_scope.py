"""Tests for scope= keyword in VectorLens.query (Task 6 — F4/F6).

Design Decision F4: scope filter is filter-before-score — applied BEFORE cosine
scoring and top-k selection, ensuring k results are k-within-scope (never under-filled
by post-filtering).

Design Decision F6: unknown scope yields empty output + success=True, never raises.
"""

from __future__ import annotations

import struct

from amplifier_data.kernel import InMemoryKernel
from amplifier_data.lenses._scope import SCOPED_TO
from amplifier_data.lenses.vector import EMBEDDING_OF, VectorLens
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


def _append_scope_edge(kernel: InMemoryKernel, cell_ref: str, scope_ref: str) -> None:
    """Append a SCOPED_TO RelationshipEvent to the kernel."""
    event = RelationshipEvent(type=SCOPED_TO, from_ref=cell_ref, to_ref=scope_ref)
    kernel.append_batch([event])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_scope_filters_entries_by_target_ref() -> None:
    """scope= keeps only entries whose target_ref is directly SCOPED_TO the given scope.

    The scope lives on the content cell (target_ref), NOT the embedding cell (emb_ref).
    inside_ref must appear in output; outside_ref must not.
    """
    kernel = InMemoryKernel()
    scope_ref = "scope_ref_vector_aaa"

    inside_ref, _ = _kernel_with_embedding(kernel, b"inside", (1.0, 0.0))
    outside_ref, _ = _kernel_with_embedding(kernel, b"outside", (0.5, 0.5))

    # Add SCOPED_TO edge for inside only (on target_ref, not emb_ref)
    _append_scope_edge(kernel, inside_ref, scope_ref)
    # outside_ref has no SCOPED_TO edge — must be excluded

    result = VectorLens().query(kernel=kernel, vector=(1.0, 0.0), scope=scope_ref)

    assert result.success is True
    refs = {ref for ref, _ in result.output}
    assert inside_ref in refs, "in-scope target_ref must appear in output"
    assert outside_ref not in refs, "out-of-scope target_ref must NOT appear in output"


def test_filter_before_score_top_k_is_k_within_scope() -> None:
    """F4: scope= filters BEFORE cosine scoring, so k results are k-within-scope.

    Setup: 2 out-of-scope entries with *higher* cosine similarity than the
    2 in-scope entries. With k=2 and scope=, output must be exactly
    {in_a_ref, in_b_ref} — never under-filled by post-filtering.

    Step 1 (assert mismatch): Without scope=, top-2 includes out-of-scope entries
    (they have higher scores). This verifies the test scenario is meaningful.

    Step 2: With scope=, filter-before-score ensures k=2 is filled from
    the 2 in-scope entries, regardless of their lower scores.
    """
    kernel = InMemoryKernel()
    scope_ref = "scope_ref_f4_test"

    # Query direction: (1.0, 0.0)
    # Out-of-scope: high similarity (cosine=1.0 and ~0.99)
    out_a_ref, _ = _kernel_with_embedding(kernel, b"out_a", (1.0, 0.0))  # cosine=1.0
    out_b_ref, _ = _kernel_with_embedding(kernel, b"out_b", (0.9, 0.436))  # cosine~0.90

    # In-scope: lower similarity (~0.71 and ~0.39)
    in_a_ref, _ = _kernel_with_embedding(kernel, b"in_a", (0.5, 0.5))  # cosine~0.71
    in_b_ref, _ = _kernel_with_embedding(kernel, b"in_b", (0.3, 0.7))  # cosine~0.39

    # Only in-scope entries have SCOPED_TO edges on their target_ref
    _append_scope_edge(kernel, in_a_ref, scope_ref)
    _append_scope_edge(kernel, in_b_ref, scope_ref)
    # out_a_ref and out_b_ref are NOT scoped

    # Step 1: Without scope=, top-2 should include out-of-scope entries.
    # This proves that without filter, higher-scoring out-of-scope refs dominate top-k.
    no_scope_result = VectorLens().query(kernel=kernel, vector=(1.0, 0.0), k=2)
    no_scope_refs = {ref for ref, _ in no_scope_result.output}
    assert out_a_ref in no_scope_refs or out_b_ref in no_scope_refs, (
        "Without scope=, the higher-scoring out-of-scope refs must appear in top-2 "
        "(mismatch — confirms scenario is meaningful for F4 proof)"
    )

    # Step 2: With scope=, filter-before-score must yield exactly {in_a, in_b}.
    scoped_result = VectorLens().query(kernel=kernel, vector=(1.0, 0.0), k=2, scope=scope_ref)
    assert scoped_result.success is True
    scoped_refs = {ref for ref, _ in scoped_result.output}
    assert len(scoped_result.output) == 2, (
        "k=2 within scope must return exactly 2 results (not under-filled)"
    )
    assert scoped_refs == {in_a_ref, in_b_ref}, "Only in-scope target_refs must appear in output"
    assert out_a_ref not in scoped_refs, "out-of-scope ref must be excluded even with higher score"
    assert out_b_ref not in scoped_refs, "out-of-scope ref must be excluded even with higher score"


def test_unknown_scope_returns_empty_success() -> None:
    """F6 — unknown scope yields empty output with success=True, never raises."""
    kernel = InMemoryKernel()
    _kernel_with_embedding(kernel, b"some cell", (1.0, 0.0))

    result = VectorLens().query(kernel=kernel, vector=(1.0, 0.0), scope="nonexistent_scope_xyz")

    assert result.success is True
    assert result.output == []


def test_no_scope_kwarg_unchanged_behaviour() -> None:
    """Without scope=, query() returns all embedded entries — unchanged behaviour (1 entry)."""
    kernel = InMemoryKernel()
    target_ref, _ = _kernel_with_embedding(kernel, b"the entry", (1.0, 0.0))

    result = VectorLens().query(kernel=kernel, vector=(1.0, 0.0))

    assert result.success is True
    assert len(result.output) == 1
    refs = {ref for ref, _ in result.output}
    assert target_ref in refs
