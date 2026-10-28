"""Worked story: convergent integrity (Plan 4b, E4).

Writes are never rejected. An injected Critic detects tensions post-hoc by
folding the log; each detected tension is materialized as an ordinary,
content-addressed Cell with reserved @integrity: provenance edges;
resolutions are themselves events; and unresolved contradictions remain
queryable forever -- through the store's existing query surface (facts,
graph, QueryIR).

Run:  python examples/convergent_integrity.py   (exits 0 on success)
"""

from __future__ import annotations

from amplifier_data import AmplifierStore
from amplifier_data.integrity import (
    IN_TENSION,
    RESOLVES,
    IntegrityMonitor,
    InvalidTensionError,
    TemporalContradictionCritic,
    Tension,
)


def main() -> None:
    print("Convergent integrity (Plan 4b, E4)\n")

    store = AmplifierStore()
    critic = TemporalContradictionCritic(predicates=("status",))
    monitor = IntegrityMonitor(store, critic)

    try:
        # ------------------------------------------------------------------ #
        # 1. Seed a conflict. Both writes are accepted -- no gate.           #
        # ------------------------------------------------------------------ #
        s = store.write_cell(b"deploy #42")
        o1 = store.write_cell(b"succeeded")
        o2 = store.write_cell(b"failed")
        store.assert_fact(s, "status", o1)
        store.assert_fact(s, "status", o2)
        print("  wrote deploy #42 with two contradictory 'status' facts")
        print("  [\u2713] both writes accepted; nothing was rejected\n")

        # ------------------------------------------------------------------ #
        # 2. Detect: a post-hoc fold, never a gate.                         #
        # ------------------------------------------------------------------ #
        refs = monitor.run_once(force=True)
        assert len(refs) == 1
        t = refs[0]
        cell = store.get_cell(t, record_access=False)
        print(f"  tension ref: {t[:16]}...")
        print(f"  tension payload: {cell.payload.decode('utf-8')}")
        print("  [\u2713] the contradiction, both provenances, and a resolution hint,")
        print("      in one self-contained cell\n")

        # ------------------------------------------------------------------ #
        # 3. Query it, three ways.                                          #
        # ------------------------------------------------------------------ #
        records = store.query_tensions()
        assert len(records) == 1 and records[0].ref == t and records[0].resolved is False
        print(f"  query_tensions() -> 1 unresolved tension (parties={records[0].parties})")

        scanned = store.query([{"op": "scan", "lens": "facts", "predicate": IN_TENSION}])
        assert t in {row["object"] for row in scanned.output}
        print("  QueryIR scan for @integrity:in_tension finds it too")

        neighbors = store.query_graph(o1, max_hops=1).output
        assert t in neighbors and o2 in neighbors
        print("  query_graph(O1, max_hops=1) reaches the tension AND its adversary O2")
        print("  [\u2713] a contradiction found WITH a query -- reflexivity, for free\n")

        # ------------------------------------------------------------------ #
        # 4. Idempotent: re-detecting the same conflict appends nothing.    #
        # ------------------------------------------------------------------ #
        before = len(store.kernel.all_events())  # type: ignore[attr-defined]
        again = monitor.run_once(force=True)
        after = len(store.kernel.all_events())  # type: ignore[attr-defined]
        assert again == [] and after == before
        print("  run_once(force=True) again -> [] appended, same content-addressed ref")
        print("  [\u2713] idempotent -- the known set is folded from the log, not memory\n")

        # ------------------------------------------------------------------ #
        # 5. Converge: someone decides. History is preserved, not deleted.  #
        # ------------------------------------------------------------------ #
        store.invalidate_fact(s, "status", o1)
        resolution_ref = store.resolve_tension(
            t, "run 1 superseded by re-run", resolved_by="example"
        )
        assert store.query_tensions(unresolved_only=True) == []
        resolved_record = store.query_tensions()[0]
        assert resolved_record.resolved is True
        assert resolved_record.resolution_refs == (resolution_ref,)
        print(f"  resolved via: {resolution_ref[:16]}...")
        print("  converged -- history preserved, nothing deleted\n")

        # ------------------------------------------------------------------ #
        # 6. Reopen: no new verb needed -- invalidate the resolves fact.    #
        # ------------------------------------------------------------------ #
        store.invalidate_fact(resolution_ref, RESOLVES, t)
        reopened_record = store.query_tensions()[0]
        assert reopened_record.resolved is False
        print("  invalidate_fact(resolution_ref, '@integrity:resolves', t) reopens it")
        print("  [\u2713] resolution validity windows fall out of the existing temporal lens\n")

        # ------------------------------------------------------------------ #
        # 7. Loud: a bad critic, and a malformed tension, both fail loudly. #
        # ------------------------------------------------------------------ #
        class _BadCritic:
            def detect(self, store: object) -> list[Tension]:
                raise RuntimeError("critic exploded")

        bad_monitor = IntegrityMonitor(store, _BadCritic())
        try:
            before_bad = len(store.kernel.all_events())  # type: ignore[attr-defined]
            try:
                bad_monitor.run_once(force=True)
                raise AssertionError("expected RuntimeError to propagate")
            except RuntimeError:
                pass
            assert len(store.kernel.all_events()) == before_bad  # type: ignore[attr-defined]
            print("  a critic that raises propagates the exception unwrapped; log untouched")
        finally:
            bad_monitor.detach()

        class _OneSidedCritic:
            def detect(self, store: object) -> list[Tension]:
                return [
                    Tension(
                        kind="bogus",
                        parties=("only-one-party",),
                        nature="malformed",
                        resolution_hint="n/a",
                    )
                ]

        malformed_monitor = IntegrityMonitor(store, _OneSidedCritic())
        try:
            before_malformed = len(store.kernel.all_events())  # type: ignore[attr-defined]
            try:
                malformed_monitor.run_once(force=True)
                raise AssertionError("expected InvalidTensionError")
            except InvalidTensionError:
                pass
            assert len(store.kernel.all_events()) == before_malformed  # type: ignore[attr-defined]
            print(
                "  a critic emitting a one-party Tension raises InvalidTensionError; log untouched"
            )
        finally:
            malformed_monitor.detach()
        print("  [\u2713] honest stopping: no guessed default, no partial writes\n")

        print(
            "SUCCESS: writes never rejected -> post-hoc detection -> unresolved "
            "contradictions queryable forever -> resolutions are events, not "
            "silent winners."
        )
    finally:
        monitor.detach()


if __name__ == "__main__":
    main()
