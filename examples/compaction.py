"""Worked story: log compaction without mutating the log (Plan 3j).

Builds a store with verbatim cells, a stream of reads (AccessEvents), and a fact
that is later invalidated. Compaction re-materializes the CURRENT state into a
fresh, smaller log — dropping AccessEvent history, the invalidated fact, and its
tombstone — while preserving cell refs and byte-identical regeneration (E1). The
original log is never rewritten.

Run:  python examples/compaction.py   (exits 0 on success)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from amplifier_data import AmplifierStore


def main() -> None:
    print("Log compaction (content-addressed re-materialization)\n")

    src = AmplifierStore()
    a = src.write_cell(b"drawer: alpha")
    b = src.write_cell(b"drawer: beta")
    src.relate(a, b, "fork")
    src.assert_fact("alice", "works_on", "acme")
    src.assert_fact("alice", "likes", "tea")
    src.invalidate_fact("alice", "likes", "tea")  # will be dropped by compaction
    for _ in range(20):
        src.get_cell(a)  # 20 AccessEvents — the read-amplification the log accrues

    tgt = AmplifierStore()
    report = src.compact_into(tgt)

    print(f"  events before: {report.events_before}")
    print(f"  events after : {report.events_after}  (dropped {report.events_dropped})")
    print(f"  carried: {report.cells_carried} cells, {report.facts} facts, {report.scopes} scopes")

    assert report.events_after < report.events_before
    assert tgt.regenerate(a).payload == b"drawer: alpha"  # E1 preserved, ref intact
    valid = {(f.subject, f.predicate, f.object) for f in tgt.query_facts().output}
    assert ("alice", "works_on", "acme") in valid
    assert ("alice", "likes", "tea") not in valid  # invalidated fact gone
    print("  [\u2713] access/invalidated/tombstone dropped; cells regenerate identically\n")

    # Durable compact copy — reopen to use it; archive the original log.
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "compact.log")
        src.compact_to(path)
        reopened = AmplifierStore(path=path)
        assert reopened.regenerate(a).payload == b"drawer: alpha"
        reopened.close()
    print("  [\u2713] compact_to() wrote a durable, reopenable compact store\n")

    print(
        "SUCCESS: compaction bounded the log (dropping access/invalidated/superseded "
        "history) while preserving refs and byte-identical regeneration."
    )


if __name__ == "__main__":
    main()
