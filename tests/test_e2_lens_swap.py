"""E2 proof: policy ≠ structure — two shapes over identical bytes, concurrently.

The same recorded events in a single store can be projected through
RelationalLens (flat rows) and GroupedLens (grouped/inverted dict)
concurrently.  Both views carry the IDENTICAL underlying payload bytes —
the bytes are baked into the log, not into either schema.  This proves
that schema/policy is a swappable fold, not baked into the bytes.

Never calls a live model.
"""

from __future__ import annotations

from amplifier_data.lenses.grouped import GroupedLens
from amplifier_data.lenses.relational import RelationalLens
from amplifier_data.models import content_address
from amplifier_data.store import AmplifierStore


def test_e2_lens_swap() -> None:
    """E2: same recorded events produce both a flat RelationalLens view and a
    grouped GroupedLens view concurrently, both correct, presenting identical
    underlying payload bytes — proving policy ≠ structure.

    Setup: a synthetic workload that writes cells and a recorded interpreter
    output (no live model call).  RelationalLens and GroupedLens both project()
    over the *same* kernel at the same time.

    Assertions:
    1. Structural shape differs — relational is a flat list, grouped is a dict.
    2. Both cell-write events appear in each view.
    3. Payload bytes are IDENTICAL across both views (same bytes, two shapes).
    4. InterpreterOutputEvent payload is identical across both views.
    5. RelationshipEvent does NOT appear in either view (not cell-defining).
    6. Both views are held concurrently without interference.
    """
    # ------------------------------------------------------------------ setup
    store = AmplifierStore()

    payload_a = b"Sample drawer: 'the deploy key lives in 1Password vault Eng'"
    payload_b = b"Sample drawer: 'Eng vault rotated 2026-06-01'"
    interp_output = b"INTERPRETED: secret stored in 1Password vault 'Eng'"

    ref_a = store.write_cell(payload_a, interpreters=("summarizer-agent",))
    ref_b = store.write_cell(payload_b)

    # RelationshipEvent — must NOT appear in either lens view
    store.relate(ref_a, ref_b, "supersedes")

    # F1-style: record a previously-run interpretation — never calls a live model
    store.write_interpretation(
        model="claude-sonnet-4-5",
        input_hash=content_address(payload_a),
        tool_results=(b"1password://vault/Eng",),
        output=interp_output,
    )

    # ------------------------------------------- concurrent project() over same kernel
    kernel = store.kernel
    rel_lens = RelationalLens()
    grp_lens = GroupedLens()

    # Both lenses project over the same kernel in the same call scope — held concurrently.
    rel_view = rel_lens.project(kernel)  # list[dict]  — flat rows
    grp_view = grp_lens.project(kernel)  # dict[str, list[dict]]  — grouped by event_type

    # -------------------------------------------------------- 1. structural shape
    assert isinstance(rel_view, list), (
        "RelationalLens.project() must return a flat list (flat rows)"
    )
    assert isinstance(grp_view, dict), (
        "GroupedLens.project() must return a grouped dict (inverted shape)"
    )
    # A list and a dict are structurally different — the same bytes in two shapes.
    assert type(rel_view) is not type(grp_view), (
        "Both views must have different structural types (list vs dict)"
    )

    # ----------------------------------- 2. both cell-write events appear in each view
    rel_cell_writes = [r for r in rel_view if r["event_type"] == "cell_write"]
    grp_cell_writes = grp_view.get("cell_write", [])

    assert len(rel_cell_writes) == 2, (
        "RelationalLens must include both CellWriteEvents as flat rows"
    )
    assert len(grp_cell_writes) == 2, (
        "GroupedLens must include both CellWriteEvents in the 'cell_write' group"
    )

    # ---------------------------------------- 3. IDENTICAL payload bytes across both views
    rel_payloads_cw = {r["payload"] for r in rel_cell_writes}
    grp_payloads_cw = {e["payload"] for e in grp_cell_writes}

    assert rel_payloads_cw == grp_payloads_cw, (
        "Payload bytes for cell-write events must be IDENTICAL across both lenses.  "
        "Policy (schema fold) ≠ structure (bytes in the log)."
    )
    # The actual bytes must be the known payloads.
    assert payload_a in rel_payloads_cw, "payload_a must appear in the relational view"
    assert payload_b in rel_payloads_cw, "payload_b must appear in the relational view"
    assert payload_a in grp_payloads_cw, "payload_a must appear in the grouped view"
    assert payload_b in grp_payloads_cw, "payload_b must appear in the grouped view"

    # ----------------------- 4. InterpreterOutputEvent payload identical across both views
    rel_interp = [r for r in rel_view if r["event_type"] == "interpreter_output"]
    grp_interp = grp_view.get("interpreter_output", [])

    assert len(rel_interp) == 1, (
        "RelationalLens must include the InterpreterOutputEvent as one flat row"
    )
    assert len(grp_interp) == 1, (
        "GroupedLens must include the InterpreterOutputEvent in the 'interpreter_output' group"
    )
    assert rel_interp[0]["payload"] == grp_interp[0]["payload"], (
        "Interpreter output payload bytes must be identical across both lenses"
    )
    assert rel_interp[0]["payload"] == interp_output, (
        "Interpreter output payload bytes must equal the original recorded output"
    )

    # ----------------------- 5. RelationshipEvent absent from BOTH views (not cell-defining)
    rel_event_types = {r["event_type"] for r in rel_view}
    grp_event_type_keys = set(grp_view.keys())

    assert "relationship" not in rel_event_types, (
        "RelationshipEvent must NOT appear as a row in the relational view"
    )
    assert "relationship" not in grp_event_type_keys, (
        "RelationshipEvent must NOT appear as a group key in the grouped view"
    )

    # ----------------------------- 6. concurrent: both views alive; payloads still equal
    # Both rel_view and grp_view are in scope simultaneously — the lenses did not
    # interfere with each other.  Re-confirm payload byte equality while both are live.
    assert {r["payload"] for r in rel_cell_writes} == {e["payload"] for e in grp_cell_writes}, (
        "Concurrent views must present equal payload bytes "
        "(lens independence: no shared mutable state)"
    )

    # Bonus: the GroupedLens shape uses event_type as a GROUP KEY while the
    # RelationalLens shape uses event_type as a ROW FIELD — structurally inverted.
    assert "cell_write" in grp_view, "GroupedLens groups events under 'cell_write' key"
    assert all("event_type" in row for row in rel_view), (
        "RelationalLens rows expose event_type as a column, not a structural key"
    )
