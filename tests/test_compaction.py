"""Tests for log compaction (Plan 3j)."""

from __future__ import annotations

from pathlib import Path

from amplifier_data import AmplifierStore
from amplifier_data.embedding import HashingEmbedder
from amplifier_data.kernel import InMemoryKernel
from amplifier_data.models import RelationshipEvent


def _facts(store: AmplifierStore) -> set[tuple[str, str, str]]:
    return {(f.subject, f.predicate, f.object) for f in store.query_facts().output}


def test_carried_cells_regenerate_byte_identical() -> None:
    src = AmplifierStore()
    a = src.write_cell(b"alpha")
    b = src.write_cell(b"beta")
    src.relate(a, b, "fork")
    src.scope(a, "S")
    src.assert_fact("x", "p", "y")
    src.get_cell(a)  # AccessEvent (record_access default)

    tgt = AmplifierStore()
    report = src.compact_into(tgt)

    # refs preserved + byte-identical regeneration (E1)
    assert tgt.regenerate(a).payload == b"alpha"
    assert tgt.regenerate(b).payload == b"beta"
    # current relationships (fork is a fact) + KG facts preserved
    assert (a, "fork", b) in _facts(tgt)
    assert ("x", "p", "y") in _facts(tgt)
    # scope preserved
    assert a in {r["cell_ref"] for r in tgt.query_relational(scope="S").output}
    assert report.cells_carried == 2


def test_access_events_dropped_and_log_shrinks() -> None:
    src = AmplifierStore()
    a = src.write_cell(b"alpha")
    for _ in range(5):
        src.get_cell(a)  # 5 AccessEvents
    tgt = AmplifierStore()
    report = src.compact_into(tgt)

    assert report.events_after < report.events_before
    assert report.events_dropped >= 5
    kernel: InMemoryKernel = tgt.kernel  # type: ignore[assignment]
    access = [e for _s, e in kernel.all_events() if type(e).__name__ == "AccessEvent"]
    assert access == []


def test_invalidated_fact_dropped_entirely() -> None:
    src = AmplifierStore()
    src.assert_fact("a", "likes", "tea")
    src.invalidate_fact("a", "likes", "tea")

    tgt = AmplifierStore()
    src.compact_into(tgt)

    assert tgt.query_facts().output == []
    kernel: InMemoryKernel = tgt.kernel  # type: ignore[assignment]
    types = [e.type for _s, e in kernel.all_events() if isinstance(e, RelationshipEvent)]
    assert all(not t.startswith("__invalidate__:") for t in types)  # no tombstone carried


def test_embeddings_preserved() -> None:
    src = AmplifierStore()
    emb = HashingEmbedder(dim=16)
    c = src.write_cell(b"doc")
    src.add_text_embedding(c, "hello world", emb)

    tgt = AmplifierStore()
    src.compact_into(tgt)
    assert tgt.query_vector(emb.embed("hello world"), 1)[0][0] == c


def test_compact_to_durable(tmp_path: Path) -> None:
    src = AmplifierStore()
    a = src.write_cell(b"durable cell")
    src.assert_fact("a", "p", "b")
    path = tmp_path / "compact.log"

    report = src.compact_to(str(path))
    assert report.cells_carried == 1

    reopened = AmplifierStore(path=str(path))
    assert reopened.regenerate(a).payload == b"durable cell"
    assert ("a", "p", "b") in _facts(reopened)
    reopened.close()
