"""Worked story: TemporalLens KG facts with validity windows (Plan 3f-temporal).

Asserts a knowledge-graph fact, invalidates it, and re-asserts it in a DURABLE
store, showing that validity is tracked by LOGICAL (SeqPos) time with NO
wall-clock — and that the current-validity view and full timeline survive a
process restart byte-identically (E1).

Run:  python examples/temporal_facts.py   (exits 0 on success)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from amplifier_data import AmplifierStore
from amplifier_data.lenses.temporal import Fact


def main() -> None:
    print("TemporalLens KG facts with validity windows\n")

    with tempfile.TemporaryDirectory() as tmp:
        log = str(Path(tmp) / "kg.log")
        fact = Fact("alice", "works_on", "acme")

        store = AmplifierStore(path=log)
        store.assert_fact("alice", "works_on", "acme")
        assert fact in store.query_facts().output
        print("  asserted   (alice, works_on, acme) -> currently valid")

        store.invalidate_fact("alice", "works_on", "acme")
        assert fact not in store.query_facts().output
        print("  invalidated                        -> no longer valid")

        store.assert_fact("alice", "works_on", "acme")
        assert fact in store.query_facts().output
        print("  re-asserted                        -> valid again")

        before_facts = store.query_facts().output
        before_timeline = [(e.op, e.predicate, e.object) for e in store.timeline("alice")]
        assert [e.op for e in store.timeline("alice")] == ["assert", "invalidate", "assert"]
        print("  timeline: assert -> invalidate -> assert (SeqPos order)")
        store.close()
        print("  closed store\n")

        # Reopen on the same path — validity + timeline must survive restart.
        reopened = AmplifierStore(path=log)
        after_facts = reopened.query_facts().output
        after_timeline = [(e.op, e.predicate, e.object) for e in reopened.timeline("alice")]
        reopened.close()

        assert after_facts == before_facts == [fact]
        assert after_timeline == before_timeline
        print("  [\u2713] validity + timeline identical across a durable restart (E1)\n")

    print(
        "SUCCESS: temporal facts — assert/invalidate/re-assert by logical time, "
        "with byte-identical regeneration across a durable restart."
    )


if __name__ == "__main__":
    main()
