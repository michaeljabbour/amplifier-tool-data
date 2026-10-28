"""Killer gates + unit tests for intent-compiled queries (Plan 4a).

KG1 - intent -> plan cell with provenance (atomic, one append_batch).
KG2 - deterministic + replayable across durable close/reopen.
KG3 - reflexivity: plans queryable as cells, via the query layer itself.
KG4 - invalid-IR rejection is loud; the log is untouched.
KG5 - zero kernel/Rust diff (verified at DoD via `git diff --stat crates/`, not here).
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import pytest

from amplifier_data import RUST_AVAILABLE, AmplifierStore
from amplifier_data.intent import (
    CATALOG,
    COMPILED_BY,
    COMPILED_FROM,
    RESULT_OF,
    IntentCompiler,
    IntentNotUnderstood,
    InvalidPlanError,
    RuleBasedCompiler,
    compiler_ref,
    decode_plan,
    plan_payload,
    validate_ir,
)
from amplifier_data.kernel import InMemoryKernel
from amplifier_data.models import Event, SeqPos


def _all_events(store: AmplifierStore) -> list[tuple[SeqPos, Event]]:
    kernel: InMemoryKernel = store.kernel  # type: ignore[assignment]
    return kernel.all_events()


def _count_events(store: AmplifierStore) -> int:
    return len(_all_events(store))


# ---------------------------------------------------------------------------
# KG1 - intent -> plan cell with provenance (atomic)
# ---------------------------------------------------------------------------


def test_kg1_compile_stores_plan_cell_with_provenance() -> None:
    store = AmplifierStore()
    before = _count_events(store)

    plan_ref = store.compile_intent(
        'cells containing "x"', RuleBasedCompiler(), compiler_id="rule-based-v1"
    )

    # (d) the batch's SeqPos run is contiguous: exactly 4 new events, in order.
    # Checked BEFORE any get_cell/query_facts calls below, since get_cell appends
    # its own AccessEvent (D4) and would otherwise pollute this count.
    after = _all_events(store)[before:]
    assert len(after) == 4
    kinds = [type(ev).__name__ for _pos, ev in after]
    assert kinds == ["CellWriteEvent", "CellWriteEvent", "RelationshipEvent", "RelationshipEvent"]

    expected_ir = [
        {"op": "scan", "lens": "relational"},
        {"op": "filter", "field": "payload", "cmp": "contains", "value": "x"},
        {"op": "project", "fields": ["cell_ref", "payload"]},
    ]

    # (a) plan cell payload is byte-exact canonical plan bytes.
    assert store.get_cell(plan_ref).payload == plan_payload(expected_ir)

    # (b) intent cell payload is the verbatim intent text.
    facts = store.query_facts(subject=plan_ref, predicate=COMPILED_FROM)
    assert len(facts.output) == 1
    intent_ref = facts.output[0].object
    assert store.get_cell(intent_ref).payload == b'cells containing "x"'

    # (c) provenance facts: exactly the intent-cell fact + the compiler fact.
    by_facts = store.query_facts(subject=plan_ref, predicate=COMPILED_BY)
    assert len(by_facts.output) == 1
    assert by_facts.output[0].object == compiler_ref("rule-based-v1")


def test_kg1_untagged_compile_has_no_compiled_by_edge() -> None:
    store = AmplifierStore()
    before = _count_events(store)
    plan_ref = store.compile_intent('cells containing "y"', RuleBasedCompiler())
    after = _all_events(store)[before:]
    assert len(after) == 3  # intent cell + plan cell + COMPILED_FROM edge only
    assert store.query_facts(subject=plan_ref, predicate=COMPILED_BY).output == []


# ---------------------------------------------------------------------------
# KG2 - deterministic + replayable across durable close/reopen
# ---------------------------------------------------------------------------


def test_kg2_execute_plan_deterministic_in_memory() -> None:
    """Same-process determinism: executing the same plan twice yields identical rows."""
    store = AmplifierStore()
    store.write_cell(b"alpha release shipped")
    store.write_cell(b"beta bug found")
    store.write_cell(b"alpha retro notes")

    seed_alpha_refs = {
        r["cell_ref"]
        for r in store.query([{"op": "scan", "lens": "relational"}]).output
        if b"alpha" in r["payload"]
    }
    plan_ref = store.compile_intent('cells containing "alpha"', RuleBasedCompiler())
    run1 = store.execute_plan(plan_ref)
    run2 = store.execute_plan(plan_ref)

    assert run1.result.success is True
    # Reflexivity note: the intent cell (verbatim text 'cells containing "alpha"')
    # and the plan cell (whose canonical JSON literally contains '"value":"alpha"')
    # are themselves ordinary relational-scannable cells written by compile_intent,
    # so they ALSO satisfy this filter -- correctly, not a bug. Assert the two
    # originally-seeded matches are present (a subset), and same-process
    # determinism holds across repeated execution of the same plan ref.
    result_refs = {r["cell_ref"] for r in run1.result.output}
    assert seed_alpha_refs <= result_refs
    assert run1.result.output == run2.result.output


@pytest.mark.skipif(not RUST_AVAILABLE, reason="requires the compiled Rust extension")
def test_kg2_execute_plan_survives_restart() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "intent_gate.log")

        store1 = AmplifierStore(path=path)
        try:
            store1.write_cell(b"alpha release shipped")
            store1.write_cell(b"beta bug found")
            store1.write_cell(b"alpha retro notes")
            plan_ref = store1.compile_intent('cells containing "alpha"', RuleBasedCompiler())
            run1 = store1.execute_plan(plan_ref)
            captured_rows = run1.result.output
            captured_plan_bytes = store1.get_cell(plan_ref).payload
        finally:
            store1.close()

        store2 = AmplifierStore(path=path)
        try:
            assert store2.regenerate(plan_ref).payload == captured_plan_bytes
            run2 = store2.execute_plan(plan_ref)
            assert run2.result.output == captured_rows
        finally:
            store2.close()


# ---------------------------------------------------------------------------
# KG3 - reflexivity: plans queryable as cells, via the query layer itself
# ---------------------------------------------------------------------------


def test_kg3_query_your_queries() -> None:
    store = AmplifierStore()
    store.write_cell(b"alpha release shipped")

    plan1 = store.compile_intent('cells containing "alpha"', RuleBasedCompiler())
    plan2 = store.compile_intent("count cells by event_type", RuleBasedCompiler())

    facts1 = store.query_facts(subject=plan1, predicate=COMPILED_FROM)
    facts2 = store.query_facts(subject=plan2, predicate=COMPILED_FROM)
    intent1 = facts1.output[0].object
    intent2 = facts2.output[0].object

    # (a) query_plans() in log order, deduplicated.
    assert store.query_plans() == [(plan1, intent1), (plan2, intent2)]

    # (b) the raw QueryIR scan over facts returns both plan refs as subjects.
    scanned = store.query([{"op": "scan", "lens": "facts", "predicate": COMPILED_FROM}])
    assert scanned.success is True
    assert {r["subject"] for r in scanned.output} == {plan1, plan2}

    # (c) a stored query that finds stored queries.
    plans_plan = store.compile_intent("show plans", RuleBasedCompiler())
    run = store.execute_plan(plans_plan)
    assert run.result.success is True
    assert any(r["subject"] == plan1 for r in run.result.output)


# ---------------------------------------------------------------------------
# KG4 - invalid-IR rejection is loud; the log is untouched
# ---------------------------------------------------------------------------


class _BadCompiler:
    def __init__(self, ir: Any) -> None:
        self._ir = ir

    def compile(self, intent: str, catalog: dict[str, Any]) -> Any:
        return self._ir


_BAD_IRS: list[Any] = [
    "not a list",
    [],
    [{"op": "filter", "field": "x", "cmp": "eq", "value": 1}],  # doesn't start with scan
    [{"op": "scan", "lens": "relational"}, {"op": "bogus"}],  # unknown op
    [{"op": "scan", "lens": "not-a-lens"}],  # unknown scan lens
    [
        {"op": "scan", "lens": "relational"},
        {"op": "filter", "field": "x", "cmp": "eq", "value": lambda: 1},
    ],  # non-JSON value (lambda)
]


@pytest.mark.parametrize("bad_ir", _BAD_IRS)
def test_kg4_invalid_ir_is_loud_and_log_untouched(bad_ir: Any) -> None:
    store = AmplifierStore()
    before = _count_events(store)
    with pytest.raises(InvalidPlanError):
        store.compile_intent("whatever", _BadCompiler(bad_ir))
    assert _count_events(store) == before


def test_kg4_rule_based_compiler_gibberish_is_loud_and_log_untouched() -> None:
    store = AmplifierStore()
    before = _count_events(store)
    with pytest.raises(IntentNotUnderstood):
        store.compile_intent("do something vague", RuleBasedCompiler())
    assert _count_events(store) == before


def test_kg4_execute_plan_on_non_plan_cell_is_loud() -> None:
    store = AmplifierStore()
    not_a_plan = store.write_cell(b"just a regular cell, not a plan envelope")
    with pytest.raises(InvalidPlanError):
        store.execute_plan(not_a_plan)


# ---------------------------------------------------------------------------
# Supporting unit tests: plan_payload / decode_plan
# ---------------------------------------------------------------------------


def test_plan_payload_decode_plan_round_trip() -> None:
    ir = [{"op": "scan", "lens": "relational"}]
    payload = plan_payload(ir)
    assert decode_plan(payload) == ir


def test_plan_payload_is_canonical_json() -> None:
    ir = [{"op": "scan", "lens": "relational"}]
    payload = plan_payload(ir)
    assert json.loads(payload.decode("utf-8")) == {"ir": ir, "v": 1}
    # Canonical: sorted keys, compact separators.
    assert b" " not in payload
    assert payload == json.dumps({"v": 1, "ir": ir}, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def test_decode_plan_rejects_non_json() -> None:
    with pytest.raises(InvalidPlanError):
        decode_plan(b"not json at all")


def test_decode_plan_rejects_missing_envelope_fields() -> None:
    with pytest.raises(InvalidPlanError):
        decode_plan(json.dumps({"ir": [{"op": "scan"}]}).encode("utf-8"))  # missing "v"
    with pytest.raises(InvalidPlanError):
        decode_plan(json.dumps({"v": 1}).encode("utf-8"))  # missing "ir"


def test_decode_plan_rejects_wrong_version() -> None:
    payload = json.dumps(
        {"ir": [{"op": "scan", "lens": "relational"}], "v": 2},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    with pytest.raises(InvalidPlanError):
        decode_plan(payload)


def test_decode_plan_rejects_invalid_ir() -> None:
    payload = json.dumps({"ir": [], "v": 1}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with pytest.raises(InvalidPlanError):
        decode_plan(payload)


# ---------------------------------------------------------------------------
# Supporting unit tests: validate_ir
# ---------------------------------------------------------------------------


def test_validate_ir_accepts_every_rule_based_compiler_output() -> None:
    compiler = RuleBasedCompiler()
    intents = [
        "show plans",
        "list plans",
        "count cells by event_type",
        'cells containing "hello"',
        "cells containing 'hello'",
        "facts with predicate likes",
        "first 3 cells",
        "neighbors of " + ("a" * 64) + " within 2 hops",
    ]
    for intent in intents:
        ir = compiler.compile(intent, CATALOG)
        assert validate_ir(ir, CATALOG) == [], f"unexpected errors for intent={intent!r}"


def test_validate_ir_accumulates_all_errors() -> None:
    ir = [{"op": "scan", "lens": "relational"}, {"op": "filter", "field": "x", "cmp": "bogus"}]
    errors = validate_ir(ir, CATALOG)
    # Missing 'value' AND unknown cmp -> at least two distinct problems reported.
    assert len(errors) >= 2


def test_validate_ir_empty_and_non_list() -> None:
    assert validate_ir([]) != []
    assert validate_ir("nope") != []
    assert validate_ir(None) != []


# ---------------------------------------------------------------------------
# Supporting unit tests: RuleBasedCompiler (one test per rule row)
# ---------------------------------------------------------------------------


def test_rule_1_show_list_plans() -> None:
    compiler = RuleBasedCompiler()
    expected = [{"op": "scan", "lens": "facts", "predicate": COMPILED_FROM}]
    assert compiler.compile("show plans", CATALOG) == expected
    assert compiler.compile("list plans", CATALOG) == expected
    assert compiler.compile("SHOW PLANS", CATALOG) == expected  # case-insensitivity


def test_rule_2_count_cells_by() -> None:
    compiler = RuleBasedCompiler()
    expected = [
        {"op": "scan", "lens": "relational"},
        {"op": "aggregate", "group_by": ["event_type"], "agg": "count"},
    ]
    assert compiler.compile("count cells by event_type", CATALOG) == expected


def test_rule_3_cells_containing() -> None:
    compiler = RuleBasedCompiler()
    expected = [
        {"op": "scan", "lens": "relational"},
        {"op": "filter", "field": "payload", "cmp": "contains", "value": "hello"},
        {"op": "project", "fields": ["cell_ref", "payload"]},
    ]
    assert compiler.compile('cells containing "hello"', CATALOG) == expected
    assert compiler.compile("cells containing 'hello'", CATALOG) == expected


def test_rule_4_facts_with_predicate() -> None:
    compiler = RuleBasedCompiler()
    expected = [{"op": "scan", "lens": "facts", "predicate": "likes"}]
    assert compiler.compile("facts with predicate likes", CATALOG) == expected


def test_rule_5_first_n_cells() -> None:
    compiler = RuleBasedCompiler()
    expected = [
        {"op": "scan", "lens": "relational"},
        {"op": "order_by", "field": "seq_pos"},
        {"op": "limit", "n": 3},
    ]
    assert compiler.compile("first 3 cells", CATALOG) == expected


def test_rule_6_neighbors_of_within_hops() -> None:
    compiler = RuleBasedCompiler()
    ref = "a" * 64
    expected = [
        {"op": "scan", "lens": "relational"},
        {"op": "filter", "field": "cell_ref", "cmp": "eq", "value": ref},
        {"op": "traverse", "max_hops": 2},
    ]
    assert compiler.compile(f"neighbors of {ref} within 2 hops", CATALOG) == expected

    expected_one_hop = [
        {"op": "scan", "lens": "relational"},
        {"op": "filter", "field": "cell_ref", "cmp": "eq", "value": ref},
        {"op": "traverse", "max_hops": 1},
    ]
    assert compiler.compile(f"neighbors of {ref} within 1 hop", CATALOG) == expected_one_hop


def test_rule_precedence_first_match_wins() -> None:
    """'show plans' matches rule 1 before it could ever be mistaken for another rule."""
    compiler = RuleBasedCompiler()
    assert compiler.compile("show plans", CATALOG) == compiler.compile("list plans", CATALOG)


def test_rule_based_compiler_unrecognized_intent_raises() -> None:
    compiler = RuleBasedCompiler()
    with pytest.raises(IntentNotUnderstood):
        compiler.compile("do something vague", CATALOG)


# ---------------------------------------------------------------------------
# Supporting unit tests: execute_plan(record_result=True)
# ---------------------------------------------------------------------------


def test_execute_plan_record_result_writes_cell_and_edge() -> None:
    store = AmplifierStore()
    store.write_cell(b"alpha release shipped")
    store.write_cell(b"beta bug found")
    plan_ref = store.compile_intent('cells containing "alpha"', RuleBasedCompiler())

    run = store.execute_plan(plan_ref, record_result=True)

    assert run.result.success is True
    assert run.result_ref is not None
    result_cell = store.get_cell(run.result_ref)
    decoded = json.loads(result_cell.payload.decode("utf-8"))
    assert decoded["v"] == 1
    assert isinstance(decoded["rows"], list)
    # payload is bytes in the underlying rows -> hex-encoded in the canonical cell.
    for row in decoded["rows"]:
        assert isinstance(row["payload"], str)
        bytes.fromhex(row["payload"])  # must be valid hex

    facts = store.query_facts(subject=run.result_ref, predicate=RESULT_OF)
    assert len(facts.output) == 1
    assert facts.output[0].object == plan_ref


def test_execute_plan_record_result_false_by_default() -> None:
    store = AmplifierStore()
    store.write_cell(b"alpha release shipped")
    plan_ref = store.compile_intent('cells containing "alpha"', RuleBasedCompiler())
    run = store.execute_plan(plan_ref)
    assert run.result_ref is None


def test_execute_plan_record_result_on_zero_rows_still_writes_cell() -> None:
    store = AmplifierStore()
    # "facts with predicate ..." scans the facts lens (not cell payloads), so it
    # has no self-collision risk the way "cells containing ..." does (the intent/
    # plan cell text would otherwise also satisfy a payload-substring filter).
    plan_ref = store.compile_intent("facts with predicate never-asserted", RuleBasedCompiler())
    run = store.execute_plan(plan_ref, record_result=True)
    assert run.result.success is True
    assert run.result.output == []
    assert run.result_ref is not None


class _FailingPlanCompiler:
    """Emits IR that is structurally valid (passes validate_ir) but fails at runtime.

    ``payload`` is bytes on every relational row; comparing bytes > str raises a
    TypeError inside the executor's filter, which run_query converts into a data
    error (success=False) rather than raising \u2014 exactly the case record_result
    must not cache.
    """

    def compile(self, intent: str, catalog: dict[str, Any]) -> Any:
        return [
            {"op": "scan", "lens": "relational"},
            {"op": "filter", "field": "payload", "cmp": "gt", "value": "alpha"},
        ]


def test_execute_plan_record_result_on_failure_writes_no_cell() -> None:
    store = AmplifierStore()
    store.write_cell(b"alpha")
    plan_ref = store.compile_intent("anything", _FailingPlanCompiler())
    before = _count_events(store)

    run = store.execute_plan(plan_ref, record_result=True)

    assert run.result.success is False
    assert run.result_ref is None
    # +1, not +0: get_cell(plan_ref) always appends its one AccessEvent (the
    # execution trace) regardless of outcome; only the result cell + edge are
    # withheld on failure.
    assert _count_events(store) == before + 1


# ---------------------------------------------------------------------------
# Supporting unit tests: IntentCompiler protocol satisfiability
# ---------------------------------------------------------------------------


def test_intent_compiler_protocol_satisfiability() -> None:
    assert isinstance(RuleBasedCompiler(), IntentCompiler)

    class _FakeLLMCompiler:
        def compile(self, intent: str, catalog: dict[str, Any]) -> list[dict[str, Any]]:
            return [{"op": "scan", "lens": "relational"}]

    assert isinstance(_FakeLLMCompiler(), IntentCompiler)


# ---------------------------------------------------------------------------
# query_plans() unit tests
# ---------------------------------------------------------------------------


def test_query_plans_empty_store() -> None:
    store = AmplifierStore()
    assert store.query_plans() == []


def test_query_plans_dedupes_identical_recompiles() -> None:
    store = AmplifierStore()
    plan1 = store.compile_intent("show plans", RuleBasedCompiler())
    plan2 = store.compile_intent("show plans", RuleBasedCompiler())
    assert plan1 == plan2  # identical intent + identical IR -> identical content address
    assert store.query_plans() == [(plan1, plan1)] or len(store.query_plans()) == 1
