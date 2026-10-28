"""Log compaction (Plan 3j) — bound the log without mutating it, preserving E1.

The append-only log is sacred (never rewritten in place). Compaction instead
**re-materializes the current state into a fresh, minimal log** — the
content-addressed-replay technique:

  * Every current ``write_cell`` cell is re-written into the target. Because refs
    are ``sha256`` of canonical bytes, the **ref is preserved** and the cell still
    regenerates byte-identically (E1 holds, per cell).
  * Current relationships / KG facts (``query_facts`` — invalidation already
    applied), scope memberships, and embeddings are re-emitted.
  * **Dropped:** ``AccessEvent`` history (the read-amplification culprit),
    invalidated facts + their tombstones, and superseded cell history.

Compaction is intentionally **lossy of history** — that is the point (cf. Kafka
log compaction / LSM tombstone collapse). The *original* log retains full history
if archived; the compact log is a smaller artifact that reproduces the current
state with identical refs.

Honest boundary: this carries verbatim ``write_cell`` cells (drawers, session
events, blobs — the bulk of real stores) plus relationships/facts/scopes/
embeddings. **Interpreter-output and index cells are NOT carried** (their refs
derive from interpreter inputs, not raw payload bytes); they are regenerable
*meaning* artifacts — recompute them on the compact store if needed. The
:class:`CompactionReport` names exactly what was carried and skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompactionReport:
    """Transparent accounting of a compaction run."""

    events_before: int
    events_after: int
    cells_carried: int
    derived_skipped: int
    facts: int
    scopes: int
    embeddings: int

    @property
    def events_dropped(self) -> int:
        return self.events_before - self.events_after


def compact(source: Any, target: Any) -> CompactionReport:
    """Re-materialize *source*'s current state into *target*; return a report.

    *target* should be a fresh (empty) ``AmplifierStore``. Refs of carried
    ``write_cell`` cells are preserved (content addressing). Read-only on *source*
    apart from the relational/fact/scope/embedding folds (no AccessEvents, D4).
    """
    from .projection import build_projection

    events_before = len(source.kernel.all_events())

    cells_carried = 0
    derived_skipped = 0
    for row in source.query_relational().output:
        if row["event_type"] == "cell_write":
            target.write_cell(row["payload"], row["interpreters"])
            cells_carried += 1
        else:  # interpreter_output / other derived cells — not ref-preserving via write_cell
            derived_skipped += 1

    scopes = build_projection(source).scopes
    for cell, scope in scopes:
        target.scope(cell, scope)

    embeddings = source.iter_embeddings()
    for target_ref, vector in embeddings:
        target.add_embedding(target_ref, vector)

    facts = source.query_facts().output
    for fact in facts:
        target.assert_fact(fact.subject, fact.predicate, fact.object)

    return CompactionReport(
        events_before=events_before,
        events_after=len(target.kernel.all_events()),
        cells_carried=cells_carried,
        derived_skipped=derived_skipped,
        facts=len(facts),
        scopes=len(scopes),
        embeddings=len(embeddings),
    )
