"""Tests for scope= passthrough in store.query_relational() and store.query_vector().

Design Decision D4 (read-vs-fold boundary)
-------------------------------------------
Both query_relational and query_vector pass ``self._kernel`` directly to the
respective lens, which reads ``kernel.all_events()`` without touching
``get_cell``.  Therefore folding the log generates **zero** AccessEvents.

Coverage
--------
* test_query_relational_scope_passthrough
    store.query_relational(scope=scope) filters to inside cell only, zero AccessEvents.
* test_query_vector_scope_passthrough
    store.query_vector((1.0, 0.0), k=10, scope=scope) filters to inside cell only,
    zero AccessEvents.  VectorLens applies filter-before-score (F4).
* test_query_vector_without_scope_unchanged
    store.query_vector((1.0, 0.0), k=1) without scope= returns the embedded ref,
    confirming backward compatibility.
"""

from __future__ import annotations

from amplifier_data.models import AccessEvent
from amplifier_data.store import AmplifierStore

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _access_events(store: AmplifierStore) -> list[AccessEvent]:
    """Return all AccessEvents currently in the kernel log."""
    all_events = store.kernel.all_events()  # type: ignore[attr-defined]
    return [ev for _, ev in all_events if isinstance(ev, AccessEvent)]


# ---------------------------------------------------------------------------
# D4 + F3: query_relational forwards scope= kwarg to RelationalLens
# ---------------------------------------------------------------------------


def test_query_relational_scope_passthrough() -> None:
    """query_relational(scope=scope) filters to inside cell only; zero AccessEvents.

    query_relational(**kwargs) already forwards kwargs to RelationalLens.query(),
    which accepts scope= and applies F3 (direct SCOPED_TO edges only).
    Unknown scope returns empty output + success=True (F6).
    """
    store = AmplifierStore()
    scope = store.write_cell(b"scope cell")
    inside = store.write_cell(b"inside cell")
    outside = store.write_cell(b"outside cell")  # noqa: F841 — intentionally unscoped

    # Only 'inside' is directly scoped
    store.scope(inside, scope)

    result = store.query_relational(scope=scope)

    assert result.success is True
    refs = {row["cell_ref"] for row in result.output}
    assert refs == {inside}
    # D4: lens reads kernel directly — no AccessEvents
    assert len(_access_events(store)) == 0


# ---------------------------------------------------------------------------
# D4 + F4: query_vector forwards scope= kwarg to VectorLens (filter-before-score)
# ---------------------------------------------------------------------------


def test_query_vector_scope_passthrough() -> None:
    """query_vector((1.0, 0.0), k=10, scope=scope) filters to inside only; zero AccessEvents.

    store.query_vector must forward scope= to VectorLens.query(), which applies
    filter-before-score (F4): only embeddings whose target_ref has a direct
    SCOPED_TO edge to scope_ref are scored.  An unknown scope returns [].
    """
    store = AmplifierStore()
    scope = store.write_cell(b"scope cell")
    inside = store.write_cell(b"inside content")
    outside = store.write_cell(b"outside content")  # noqa: F841

    # Only 'inside' belongs to the scope
    store.scope(inside, scope)

    # Add identical embeddings for both cells — only the scoped one must appear
    store.add_embedding(inside, (1.0, 0.0))
    store.add_embedding(outside, (1.0, 0.0))

    result = store.query_vector((1.0, 0.0), k=10, scope=scope)

    refs = {ref for ref, _score in result}
    assert refs == {inside}
    # D4: lens reads kernel directly — no AccessEvents
    assert len(_access_events(store)) == 0


# ---------------------------------------------------------------------------
# Backward compatibility: query_vector without scope= is unchanged
# ---------------------------------------------------------------------------


def test_query_vector_without_scope_unchanged() -> None:
    """query_vector((1.0, 0.0), k=1) without scope= returns the embedded ref.

    Adding scope= as an optional kwarg must not break existing callers.
    """
    store = AmplifierStore()
    cell = store.write_cell(b"cell with embedding")
    store.add_embedding(cell, (1.0, 0.0))

    result = store.query_vector((1.0, 0.0), k=1)

    assert len(result) == 1
    ref, score = result[0]
    assert ref == cell
