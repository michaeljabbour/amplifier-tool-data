"""Causal tagging — attribute a write to the interaction / hook that caused it (T3D-3).

Why this exists
---------------
The behavioral-plasticity loop needs *attributable* writes: given an interaction
(or the hook that fired), it must be able to ask "show me every cell this
interaction wrote." amplifier-data is append-only and content-addressed, so the
cheapest backward-compatible way to record causality is a reserved-type
``RelationshipEvent`` that lands in the written cell's provenance.

Design choices
--------------
* **No model / serialization change.** We do NOT add a field to any event
  dataclass, so old logs replay byte-identically and the Rust kernel needs no
  change (Conventions: additive, not breaking; determinism is load-bearing).
* **Reserved ``@causal:`` rel_type namespace.** Edge types in this namespace are
  owned by this module; user code must not mint them. Because the edge lands in
  the *source* cell's provenance only (``target_ref`` → ``from_ref``), tagging a
  write never changes that cell's identity (its ref is the content address of the
  ``CellWriteEvent`` payload + interpreters, which is untouched).
* **Synthetic, content-addressed targets.** An interaction id / hook name is a
  free-form string, not a cell. We content-address it into a stable ``Hash`` so
  the edge has a well-formed ``to_ref`` and the reverse query is an exact match.
"""

from __future__ import annotations

from .models import Hash, RelationshipEvent, content_address

#: Edge type: written cell --> the interaction that caused the write.
CAUSED_BY_INTERACTION = "@causal:interaction"
#: Edge type: written cell --> the hook that performed the write.
WRITTEN_BY_HOOK = "@causal:hook"


def interaction_ref(interaction_id: str) -> Hash:
    """Stable synthetic ref for *interaction_id* (content-addressed, namespaced)."""
    return content_address(b"@causal:interaction:" + interaction_id.encode("utf-8"))


def hook_ref(hook: str) -> Hash:
    """Stable synthetic ref for the hook name *hook* (content-addressed, namespaced)."""
    return content_address(b"@causal:hook:" + hook.encode("utf-8"))


def causal_edges(
    cell_ref: Hash,
    *,
    source_interaction_id: str | None = None,
    written_by_hook: str | None = None,
) -> list[RelationshipEvent]:
    """Build the reserved-type provenance edges for a tagged write.

    Returns an empty list when no tag is supplied, so an untagged write produces
    exactly the events it produced before this feature existed (backward compat).
    """
    edges: list[RelationshipEvent] = []
    if source_interaction_id is not None:
        edges.append(
            RelationshipEvent(
                from_ref=cell_ref,
                to_ref=interaction_ref(source_interaction_id),
                type=CAUSED_BY_INTERACTION,
            )
        )
    if written_by_hook is not None:
        edges.append(
            RelationshipEvent(
                from_ref=cell_ref,
                to_ref=hook_ref(written_by_hook),
                type=WRITTEN_BY_HOOK,
            )
        )
    return edges
