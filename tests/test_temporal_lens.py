"""Tests for TemporalLens validity windows (Plan 3f-temporal).

Validity is by LOGICAL (SeqPos) time — no wall-clock. A triple is valid iff its
latest event is an assertion; re-assertion after invalidation revalidates it.
"""

from __future__ import annotations

from pathlib import Path

from amplifier_data import AmplifierStore
from amplifier_data.kernel import InMemoryKernel
from amplifier_data.lenses.temporal import Fact


def test_assert_invalidate_reassert() -> None:
    store = AmplifierStore()
    f = Fact("alice", "works_on", "acme")

    store.assert_fact("alice", "works_on", "acme")
    assert f in store.query_facts().output

    store.invalidate_fact("alice", "works_on", "acme")
    assert f not in store.query_facts().output

    store.assert_fact("alice", "works_on", "acme")  # re-assert
    assert f in store.query_facts().output


def test_timeline_ordering() -> None:
    store = AmplifierStore()
    store.assert_fact("alice", "p", "x")
    store.invalidate_fact("alice", "p", "x")
    store.assert_fact("alice", "p", "x")

    tl = store.timeline("alice")
    assert [e.op for e in tl] == ["assert", "invalidate", "assert"]
    assert all(tl[i].seq_pos <= tl[i + 1].seq_pos for i in range(len(tl) - 1))


def test_filter_subject_and_predicate() -> None:
    store = AmplifierStore()
    store.assert_fact("alice", "works_on", "acme")
    store.assert_fact("bob", "works_on", "acme")
    store.assert_fact("alice", "likes", "tea")

    by_subject = {
        (f.subject, f.predicate, f.object) for f in store.query_facts(subject="alice").output
    }
    assert by_subject == {("alice", "works_on", "acme"), ("alice", "likes", "tea")}

    by_pred = {f.subject for f in store.query_facts(predicate="works_on").output}
    assert by_pred == {"alice", "bob"}


def test_reserved_scope_and_embedding_edges_excluded() -> None:
    """scoped_to / embedding_of edges are internal mechanism, not user facts."""
    store = AmplifierStore()
    c = store.write_cell(b"content")
    s = store.write_cell(b"scope")
    store.scope(c, s)  # SCOPED_TO edge
    store.add_embedding(c, [0.1, 0.2])  # EMBEDDING_OF edge
    store.assert_fact("alice", "p", "x")

    facts = {(f.subject, f.predicate, f.object) for f in store.query_facts().output}
    assert facts == {("alice", "p", "x")}


def test_determinism() -> None:
    store = AmplifierStore()
    store.assert_fact("a", "p", "b")
    assert store.query_facts().output == store.query_facts().output


def test_query_appends_no_access_events() -> None:
    store = AmplifierStore()
    store.assert_fact("a", "p", "b")
    kernel: InMemoryKernel = store.kernel  # type: ignore[assignment]
    before = len(kernel.all_events())
    store.query_facts()
    store.timeline("a")
    assert len(kernel.all_events()) == before


def test_durable_restart(tmp_path: Path) -> None:
    log = tmp_path / "kg.log"
    store = AmplifierStore(path=str(log))
    store.assert_fact("alice", "works_on", "acme")
    store.invalidate_fact("alice", "works_on", "acme")
    store.assert_fact("alice", "works_on", "acme")
    before = store.query_facts().output
    store.close()

    reopened = AmplifierStore(path=str(log))
    after = reopened.query_facts().output
    reopened.close()
    assert after == before == [Fact("alice", "works_on", "acme")]
