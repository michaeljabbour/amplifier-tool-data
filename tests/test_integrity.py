"""Killer gates + unit tests for convergent integrity (Plan 4b, E4).

KG1 - contradictory facts => tension cell with both provenances, queryable.
KG2 - idempotent re-detection (no dupes), known set folded from the log.
KG3 - resolution is an event; tension shows resolved via query.
KG4 - the unresolvable stays queryable; never a silent winner.
KG5 - durable close/reopen preserves tensions (+ compaction bonus).
KG6 - zero kernel/Rust diff (verified at DoD via `git diff --stat crates/`, not here).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from typing import Any

import pytest

from amplifier_data import RUST_AVAILABLE, AmplifierStore
from amplifier_data.integrity import (
    CONFLICTS_WITH,
    IN_TENSION,
    RESOLVES,
    TENSION,
    Critic,
    IntegrityMonitor,
    InvalidTensionError,
    NotATensionError,
    TemporalContradictionCritic,
    Tension,
    decode_tension,
    resolution_payload,
    tension_payload,
    tension_ref,
)
from amplifier_data.kernel import InMemoryKernel, NotFound
from amplifier_data.models import Event, RelationshipEvent, SeqPos


def _all_events(store: AmplifierStore) -> list[tuple[SeqPos, Event]]:
    kernel: InMemoryKernel = store.kernel  # type: ignore[assignment]
    return kernel.all_events()


def _count_events(store: AmplifierStore) -> int:
    return len(_all_events(store))


def _seed_contradiction(store: AmplifierStore) -> tuple[str, str, str]:
    """Write S + two status objects, assert both -- both accepted, no gate."""
    s = store.write_cell(b"deploy #42")
    o1 = store.write_cell(b"succeeded")
    o2 = store.write_cell(b"failed")
    store.assert_fact(s, "status", o1)
    store.assert_fact(s, "status", o2)
    return s, o1, o2


# ---------------------------------------------------------------------------
# KG1 - contradiction becomes a queryable tension cell
# ---------------------------------------------------------------------------


def test_kg1_contradiction_becomes_queryable_tension_cell() -> None:
    store = AmplifierStore()
    critic = TemporalContradictionCritic(predicates=("status",))
    monitor = IntegrityMonitor(store, critic)
    try:
        s, o1, o2 = _seed_contradiction(store)
        p1, p2 = sorted((o1, o2))

        before = _all_events(store)
        refs = monitor.run_once(force=True)
        assert len(refs) == 1
        t_ref = refs[0]

        # (d) the batch's SeqPos run is contiguous: one CellWrite + 2 TENSION +
        # 2 IN_TENSION + 2 CONFLICTS_WITH edges, in that exact order.
        after = _all_events(store)[len(before) :]
        assert len(after) == 7
        kinds = [type(ev).__name__ for _pos, ev in after]
        assert kinds == ["CellWriteEvent"] + ["RelationshipEvent"] * 6
        rel_events: list[RelationshipEvent] = []
        for _pos, ev in after[1:]:
            assert isinstance(ev, RelationshipEvent)
            rel_events.append(ev)
        assert [(e.from_ref, e.to_ref, e.type) for e in rel_events] == [
            (t_ref, p1, TENSION),
            (t_ref, p2, TENSION),
            (p1, t_ref, IN_TENSION),
            (p2, t_ref, IN_TENSION),
            (p1, p2, CONFLICTS_WITH),
            (p2, p1, CONFLICTS_WITH),
        ]

        expected = Tension(
            kind="temporal_contradiction",
            parties=(p1, p2),
            nature=(
                f"functional predicate 'status' has 2 contradictory current objects for subject {s}"
            ),
            resolution_hint=(
                "invalidate all but one (subject, 'status', object) fact, then "
                "record a resolution via resolve_tension"
            ),
            detail={"subject": s, "predicate": "status"},
        )
        # (a) tension cell payload is byte-exact canonical tension bytes.
        cell = store.get_cell(t_ref, record_access=False)
        assert cell.payload == tension_payload(expected)
        decoded = decode_tension(cell.payload)
        assert decoded.parties == (p1, p2)
        assert decoded.detail == {"subject": s, "predicate": "status"}

        # (b) query_tensions() shows the one unresolved tension.
        records = store.query_tensions()
        assert len(records) == 1
        assert records[0].ref == t_ref
        assert records[0].kind == "temporal_contradiction"
        assert records[0].parties == (p1, p2)
        assert records[0].resolved is False
        assert records[0].resolution_refs == ()

        # (c) both TENSION and IN_TENSION facts are exactly the two parties.
        tension_facts = store.query_facts(subject=t_ref, predicate=TENSION)
        assert {f.object for f in tension_facts.output} == {o1, o2}
        in_tension_facts = store.query_facts(predicate=IN_TENSION)
        assert {(f.subject, f.object) for f in in_tension_facts.output} == {
            (o1, t_ref),
            (o2, t_ref),
        }

        # (e) both original facts remain current -- detection modified nothing.
        current = store.query_facts(subject=s, predicate="status")
        assert {f.object for f in current.output} == {o1, o2}
    finally:
        monitor.detach()


# ---------------------------------------------------------------------------
# KG2 - idempotent re-detection (no dupes)
# ---------------------------------------------------------------------------


def test_kg2_redetection_appends_nothing() -> None:
    store = AmplifierStore()
    critic = TemporalContradictionCritic(predicates=("status",))
    monitor = IntegrityMonitor(store, critic)
    try:
        _seed_contradiction(store)
        refs = monitor.run_once(force=True)
        assert len(refs) == 1

        before = _count_events(store)
        assert monitor.run_once(force=True) == []
        assert _count_events(store) == before

        # Fresh monitor instance, same store/critic config: known folded from LOG,
        # not from instance memory.
        fresh = IntegrityMonitor(store, TemporalContradictionCritic(predicates=("status",)))
        try:
            assert fresh.run_once(force=True) == []
            assert _count_events(store) == before
        finally:
            fresh.detach()

        # Unrelated writes between runs don't re-materialize the same tension.
        store.write_cell(b"unrelated")
        assert monitor.run_once(force=True) == []
        assert _count_events(store) == before + 1
    finally:
        monitor.detach()


# ---------------------------------------------------------------------------
# KG3 - resolution is an event; tension shows resolved via query
# ---------------------------------------------------------------------------


def test_kg3_resolution_recorded_and_visible() -> None:
    store = AmplifierStore()
    critic = TemporalContradictionCritic(predicates=("status",))
    monitor = IntegrityMonitor(store, critic)
    try:
        _seed_contradiction(store)
        [t] = monitor.run_once(force=True)

        r = store.resolve_tension(t, "picked O2", resolved_by="test")

        # (a) resolution cell payload is byte-exact.
        assert store.get_cell(r, record_access=False).payload == resolution_payload(
            t, "picked O2", "test"
        )

        # (b) the resolves fact names the tension.
        resolves_facts = store.query_facts(subject=r, predicate=RESOLVES)
        assert [f.object for f in resolves_facts.output] == [t]

        # (c) query_tensions shows resolved with the resolution ref.
        records = store.query_tensions()
        assert len(records) == 1
        assert records[0].resolved is True
        assert records[0].resolution_refs == (r,)
        assert store.query_tensions(unresolved_only=True) == []

        # (d) reopen via invalidate_fact -- unresolved again (temporal validity).
        store.invalidate_fact(r, RESOLVES, t)
        records2 = store.query_tensions()
        assert len(records2) == 1
        assert records2[0].resolved is False
        assert records2[0].resolution_refs == ()
        assert store.query_tensions(unresolved_only=True) == records2

        # Re-resolving re-asserts validity.
        store.assert_fact(r, RESOLVES, t)
        records3 = store.query_tensions()
        assert records3[0].resolved is True
    finally:
        monitor.detach()


# ---------------------------------------------------------------------------
# KG4 - the unresolvable stays queryable; never a silent winner
# ---------------------------------------------------------------------------


def test_kg4_unresolved_stays_queryable_never_picked() -> None:
    store = AmplifierStore()
    critic = TemporalContradictionCritic(predicates=("status",))
    monitor = IntegrityMonitor(store, critic)
    try:
        s, o1, o2 = _seed_contradiction(store)
        [t] = monitor.run_once(force=True)

        # (a) the store picked no winner.
        current = store.query_facts(subject=s, predicate="status")
        assert {f.object for f in current.output} == {o1, o2}

        # (b) the reflexive gate: contradictions found WITH a query.
        scanned = store.query([{"op": "scan", "lens": "facts", "predicate": IN_TENSION}])
        assert scanned.success is True
        assert t in {r["object"] for r in scanned.output}

        # (c) 1 hop from a party; 2 hops from the subject.
        assert t in store.query_graph(o1, max_hops=1).output
        assert t in store.query_graph(s, max_hops=2).output

        # (d) many later writes + another run_once later, still unresolved.
        for i in range(20):
            store.write_cell(f"noise-{i}".encode())
        assert monitor.run_once(force=True) == []
        records = store.query_tensions()
        assert len(records) == 1
        assert records[0].resolved is False
    finally:
        monitor.detach()


# ---------------------------------------------------------------------------
# KG5 - durable close/reopen preserves tensions
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUST_AVAILABLE, reason="requires the compiled Rust extension")
def test_kg5_tensions_survive_restart() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "integrity_gate.log")

        store1 = AmplifierStore(path=path)
        try:
            critic = TemporalContradictionCritic(predicates=("status",))
            monitor = IntegrityMonitor(store1, critic)
            try:
                _seed_contradiction(store1)
                [t] = monitor.run_once(force=True)
                store1.resolve_tension(t, "picked O2", resolved_by="test")
                captured = store1.query_tensions()
                captured_payload = store1.get_cell(t, record_access=False).payload
            finally:
                monitor.detach()
        finally:
            store1.close()

        store2 = AmplifierStore(path=path)
        try:
            assert store2.query_tensions() == captured
            assert store2.regenerate(t, record_access=False).payload == captured_payload
        finally:
            store2.close()

        # Bonus: compact_to a new path, reopen it, assert current state carried.
        compact_path = os.path.join(tmpdir, "integrity_compact.log")
        store3 = AmplifierStore(path=path)
        try:
            store3.compact_to(compact_path)
        finally:
            store3.close()

        store4 = AmplifierStore(path=compact_path)
        try:
            records = store4.query_tensions()
            assert len(records) == 1
            assert records[0].resolved is True
        finally:
            store4.close()


# ---------------------------------------------------------------------------
# Supporting unit tests: tension_payload / decode_tension
# ---------------------------------------------------------------------------


def test_tension_payload_decode_round_trip() -> None:
    t = Tension(
        kind="k",
        parties=("b" * 64, "a" * 64),
        nature="n",
        resolution_hint="h",
        detail={"x": 1},
    )
    payload = tension_payload(t)
    decoded = decode_tension(payload)
    assert decoded.kind == t.kind
    assert decoded.parties == tuple(sorted(t.parties))
    assert decoded.nature == t.nature
    assert decoded.resolution_hint == t.resolution_hint
    assert decoded.detail == t.detail


def test_tension_payload_party_order_insensitive() -> None:
    a, b = "a" * 64, "b" * 64
    t1 = Tension(kind="k", parties=(a, b), nature="n", resolution_hint="h")
    t2 = Tension(kind="k", parties=(b, a), nature="n", resolution_hint="h")
    assert tension_payload(t1) == tension_payload(t2)
    assert tension_ref(t1) == tension_ref(t2)


def test_tension_payload_is_canonical_json() -> None:
    a, b = "a" * 64, "b" * 64
    t = Tension(kind="k", parties=(a, b), nature="n", resolution_hint="h")
    payload = tension_payload(t)
    assert b" " not in payload
    obj = json.loads(payload.decode("utf-8"))
    assert obj == {
        "v": 1,
        "kind": "k",
        "parties": sorted((a, b)),
        "nature": "n",
        "resolution_hint": "h",
        "detail": {},
    }


def test_decode_tension_rejects_non_json() -> None:
    with pytest.raises(NotATensionError):
        decode_tension(b"not json at all")


def test_decode_tension_rejects_wrong_version() -> None:
    payload = json.dumps(
        {
            "v": 2,
            "kind": "k",
            "parties": ["a" * 64, "b" * 64],
            "nature": "n",
            "resolution_hint": "h",
            "detail": {},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    with pytest.raises(NotATensionError):
        decode_tension(payload)


def test_decode_tension_rejects_invalid_body() -> None:
    payload = json.dumps(
        {
            "v": 1,
            "kind": "k",
            "parties": ["only-one"],
            "nature": "n",
            "resolution_hint": "h",
            "detail": {},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    with pytest.raises(NotATensionError):
        decode_tension(payload)


def test_decode_tension_rejects_missing_fields() -> None:
    payload = json.dumps({"v": 1, "kind": "k"}, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    with pytest.raises(NotATensionError):
        decode_tension(payload)


# ---------------------------------------------------------------------------
# Supporting unit tests: InvalidTensionError (malformed Tension, before append)
# ---------------------------------------------------------------------------


def test_tension_payload_rejects_single_party() -> None:
    t = Tension(kind="k", parties=("a" * 64,), nature="n", resolution_hint="h")
    with pytest.raises(InvalidTensionError):
        tension_payload(t)


def test_tension_payload_rejects_duplicate_parties() -> None:
    p = "a" * 64
    t = Tension(kind="k", parties=(p, p), nature="n", resolution_hint="h")
    with pytest.raises(InvalidTensionError):
        tension_payload(t)


def test_tension_payload_rejects_empty_kind() -> None:
    t = Tension(kind="", parties=("a" * 64, "b" * 64), nature="n", resolution_hint="h")
    with pytest.raises(InvalidTensionError):
        tension_payload(t)


def test_tension_payload_rejects_non_json_detail() -> None:
    t = Tension(
        kind="k",
        parties=("a" * 64, "b" * 64),
        nature="n",
        resolution_hint="h",
        detail={"bad": lambda: 1},
    )
    with pytest.raises(InvalidTensionError):
        tension_payload(t)


# ---------------------------------------------------------------------------
# Supporting unit tests: resolution_payload
# ---------------------------------------------------------------------------


def test_resolution_payload_canonical_and_resolved_by_null_when_omitted() -> None:
    tref = "t" * 64
    body = {
        "v": 1,
        "kind": "resolution",
        "tension": tref,
        "resolution": "resolution text",
        "resolved_by": None,
    }
    payload = resolution_payload(tref, "resolution text", None)
    assert json.loads(payload.decode("utf-8")) == body
    # Canonical: sorted keys, compact separators (no space after ',' or ':').
    assert payload == json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# Supporting unit tests: TemporalContradictionCritic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_predicates", [(), ("@integrity:tension",), ("__invalidate__:x",)])
def test_temporal_contradiction_critic_rejects_bad_predicates(bad_predicates: Any) -> None:
    with pytest.raises(ValueError):
        TemporalContradictionCritic(predicates=bad_predicates)


def test_temporal_contradiction_critic_two_and_three_object_conflicts() -> None:
    store = AmplifierStore()
    s = store.write_cell(b"s")
    o1, o2, o3 = store.write_cell(b"o1"), store.write_cell(b"o2"), store.write_cell(b"o3")
    store.assert_fact(s, "status", o1)
    store.assert_fact(s, "status", o2)

    critic = TemporalContradictionCritic(predicates=("status",))
    detected = critic.detect(store)
    assert len(detected) == 1
    assert detected[0].parties == tuple(sorted((o1, o2)))

    store.assert_fact(s, "status", o3)
    detected2 = critic.detect(store)
    assert len(detected2) == 1
    assert detected2[0].parties == tuple(sorted((o1, o2, o3)))


def test_temporal_contradiction_critic_invalidated_side_stops_detection() -> None:
    store = AmplifierStore()
    s = store.write_cell(b"s")
    o1, o2 = store.write_cell(b"o1"), store.write_cell(b"o2")
    store.assert_fact(s, "status", o1)
    store.assert_fact(s, "status", o2)
    critic = TemporalContradictionCritic(predicates=("status",))
    assert len(critic.detect(store)) == 1

    store.invalidate_fact(s, "status", o1)
    assert critic.detect(store) == []

    store.assert_fact(s, "status", o1)  # re-assert
    assert len(critic.detect(store)) == 1


def test_temporal_contradiction_critic_output_ordering() -> None:
    store = AmplifierStore()
    s1, s2 = store.write_cell(b"s1"), store.write_cell(b"s2")
    o1, o2 = store.write_cell(b"o1"), store.write_cell(b"o2")
    store.assert_fact(s2, "alpha", o1)
    store.assert_fact(s2, "alpha", o2)
    store.assert_fact(s1, "beta", o1)
    store.assert_fact(s1, "beta", o2)

    critic = TemporalContradictionCritic(predicates=("beta", "alpha"))
    detected = critic.detect(store)
    keys = [(t.detail["predicate"], t.detail["subject"]) for t in detected]
    assert keys == sorted(keys)


def test_temporal_contradiction_critic_never_flags_predicates_not_given() -> None:
    store = AmplifierStore()
    s = store.write_cell(b"s")
    o1, o2 = store.write_cell(b"o1"), store.write_cell(b"o2")
    store.assert_fact(s, "other", o1)
    store.assert_fact(s, "other", o2)
    critic = TemporalContradictionCritic(predicates=("status",))
    assert critic.detect(store) == []


def test_temporal_contradiction_critic_never_regresses_on_its_own_tension_edges() -> None:
    """Regress test: after materialization, detect() must not flag the tension's
    own @integrity:tension fan-out (structurally impossible: predicates are
    validated to never start with '@')."""
    store = AmplifierStore()
    critic = TemporalContradictionCritic(predicates=("status",))
    monitor = IntegrityMonitor(store, critic)
    try:
        s, o1, o2 = _seed_contradiction(store)
        refs = monitor.run_once(force=True)
        assert len(refs) == 1

        detected = critic.detect(store)
        assert len(detected) == 1
        assert detected[0].parties == tuple(sorted((o1, o2)))
    finally:
        monitor.detach()


# ---------------------------------------------------------------------------
# Supporting unit tests: Critic protocol satisfiability
# ---------------------------------------------------------------------------


def test_critic_protocol_satisfiability() -> None:
    assert isinstance(TemporalContradictionCritic(("status",)), Critic)

    class _FakeLLMCritic:
        def detect(self, store: Any) -> list[Tension]:
            return []

    assert isinstance(_FakeLLMCritic(), Critic)


# ---------------------------------------------------------------------------
# Supporting unit tests: IntegrityMonitor
# ---------------------------------------------------------------------------


def test_monitor_threshold_must_be_positive() -> None:
    store = AmplifierStore()
    with pytest.raises(ValueError):
        IntegrityMonitor(store, TemporalContradictionCritic(("status",)), threshold=0)


def test_monitor_threshold_respected() -> None:
    store = AmplifierStore()
    monitor = IntegrityMonitor(store, TemporalContradictionCritic(("status",)), threshold=100)
    try:
        _seed_contradiction(store)
        assert monitor.run_once() == []  # well below threshold
    finally:
        monitor.detach()


def test_observer_ignores_integrity_edges_but_counts_tension_cell_write() -> None:
    """The tension cell CellWriteEvent self-echoes into pending (harmless, per
    design); the @integrity: RelationshipEvents fanning out from it do not."""
    store = AmplifierStore()
    critic = TemporalContradictionCritic(predicates=("status",))
    monitor = IntegrityMonitor(store, critic, threshold=1000)
    try:
        _seed_contradiction(store)
        monitor.run_once(force=True)
        assert monitor.pending == 1  # only the tension cell's own CellWriteEvent
    finally:
        monitor.detach()


def test_monitor_start_stop_detach_idempotent() -> None:
    store = AmplifierStore()
    monitor = IntegrityMonitor(store, TemporalContradictionCritic(("status",)), threshold=1)
    monitor.start(interval=0.01)
    monitor.start(interval=0.01)  # no-op, already running
    assert monitor.running is True
    monitor.stop()
    monitor.stop()  # idempotent
    assert monitor.running is False
    monitor.detach()
    monitor.detach()  # idempotent


class _RaisingCritic:
    def detect(self, store: Any) -> list[Tension]:
        raise RuntimeError("boom")


def test_monitor_background_loop_records_last_error_and_survives() -> None:
    store = AmplifierStore()
    monitor = IntegrityMonitor(store, _RaisingCritic(), threshold=1)
    monitor.start(interval=0.01)
    try:
        store.write_cell(b"x")  # bump pending past threshold
        deadline = time.time() + 3.0
        while time.time() < deadline and monitor.last_error is None:
            time.sleep(0.02)
        assert monitor.last_error is not None
        assert isinstance(monitor.last_error, RuntimeError)
        assert monitor.running is True  # a bad tick must not kill the loop
    finally:
        monitor.stop()
        monitor.detach()


def test_monitor_run_once_propagates_critic_exception_and_log_unchanged() -> None:
    store = AmplifierStore()
    monitor = IntegrityMonitor(store, _RaisingCritic())
    try:
        before = _count_events(store)
        with pytest.raises(RuntimeError):
            monitor.run_once(force=True)
        assert _count_events(store) == before
    finally:
        monitor.detach()


class _MalformedTensionCritic:
    def detect(self, store: Any) -> list[Tension]:
        return [Tension(kind="k", parties=("only-one",), nature="n", resolution_hint="h")]


def test_monitor_run_once_invalid_tension_is_loud_and_log_unchanged() -> None:
    store = AmplifierStore()
    monitor = IntegrityMonitor(store, _MalformedTensionCritic())
    try:
        before = _count_events(store)
        with pytest.raises(InvalidTensionError):
            monitor.run_once(force=True)
        assert _count_events(store) == before
    finally:
        monitor.detach()


def test_monitor_concurrent_run_once_produces_no_duplicates() -> None:
    store = AmplifierStore()
    critic = TemporalContradictionCritic(predicates=("status",))
    monitor = IntegrityMonitor(store, critic)
    try:
        _seed_contradiction(store)

        results: list[list[str]] = []
        results_lock = threading.Lock()

        def _run() -> None:
            r = monitor.run_once(force=True)
            with results_lock:
                results.append(r)

        threads = [threading.Thread(target=_run) for _ in range(5)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        total_appended = sum(len(r) for r in results)
        assert total_appended == 1  # exactly one thread materialized it
        assert len(store.query_tensions()) == 1
    finally:
        monitor.detach()


def test_store_factory_wires_indexer_absence() -> None:
    """IntegrityMonitor is constructed directly (like AutonomicIndexer); it is
    deliberately NOT exposed as a store convenience method (design doc)."""
    assert not hasattr(AmplifierStore, "integrity_monitor")


# ---------------------------------------------------------------------------
# Supporting unit tests: resolve_tension
# ---------------------------------------------------------------------------


def test_resolve_tension_not_a_tension_error_on_plain_cell() -> None:
    store = AmplifierStore()
    ref = store.write_cell(b"just a regular cell, not a tension envelope")
    with pytest.raises(NotATensionError):
        store.resolve_tension(ref, "whatever")


def test_resolve_tension_not_found_on_unknown_ref() -> None:
    store = AmplifierStore()
    with pytest.raises(NotFound):
        store.resolve_tension("f" * 64, "whatever")


def test_resolve_tension_type_error_on_event_ref() -> None:
    store = AmplifierStore()
    # A RelationshipEvent's identity is not a resolvable Cell ref via get_cell.
    a = store.write_cell(b"a")
    b = store.write_cell(b"b")
    store.relate(a, b, "likes")
    with pytest.raises((NotFound, TypeError)):
        store.resolve_tension("not-a-real-ref", "whatever")


def test_resolve_tension_duplicate_identical_resolution_dedupes() -> None:
    store = AmplifierStore()
    critic = TemporalContradictionCritic(predicates=("status",))
    monitor = IntegrityMonitor(store, critic)
    try:
        _seed_contradiction(store)
        [t] = monitor.run_once(force=True)
        r1 = store.resolve_tension(t, "picked O2", resolved_by="test")
        r2 = store.resolve_tension(t, "picked O2", resolved_by="test")
        assert r1 == r2
        records = store.query_tensions()
        assert records[0].resolution_refs == (r1,)
    finally:
        monitor.detach()


def test_resolve_tension_conflicting_resolutions_both_recorded() -> None:
    store = AmplifierStore()
    critic = TemporalContradictionCritic(predicates=("status",))
    monitor = IntegrityMonitor(store, critic)
    try:
        _seed_contradiction(store)
        [t] = monitor.run_once(force=True)
        r1 = store.resolve_tension(t, "picked O1", resolved_by="a")
        r2 = store.resolve_tension(t, "picked O2", resolved_by="b")
        assert r1 != r2
        records = store.query_tensions()
        assert records[0].resolved is True
        assert records[0].resolution_refs == (r1, r2)  # log order
    finally:
        monitor.detach()


# ---------------------------------------------------------------------------
# Supporting unit tests: query_tensions
# ---------------------------------------------------------------------------


def test_query_tensions_empty_store() -> None:
    store = AmplifierStore()
    assert store.query_tensions() == []
    assert store.query_tensions(unresolved_only=True) == []


def test_query_tensions_reflects_no_access_events_side_effects_beyond_decode() -> None:
    """D4: query_tensions decodes tension payloads via record_access=False."""
    store = AmplifierStore()
    critic = TemporalContradictionCritic(predicates=("status",))
    monitor = IntegrityMonitor(store, critic)
    try:
        _seed_contradiction(store)
        monitor.run_once(force=True)
        before = _count_events(store)
        store.query_tensions()
        assert _count_events(store) == before  # no AccessEvents from the fold
    finally:
        monitor.detach()
