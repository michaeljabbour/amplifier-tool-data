import hashlib

import pytest

from amplifier_data.kernel import InMemoryKernel, NotFound, Subscription
from amplifier_data.models import Cell, CellWriteEvent, Event, ToolResultEvent


def test_content_address_is_deterministic_sha256() -> None:
    k = InMemoryKernel()
    assert k.content_address(b"abc") == hashlib.sha256(b"abc").hexdigest()
    assert k.content_address(b"abc") == k.content_address(b"abc")
    assert k.content_address(b"abc") != k.content_address(b"abd")


def test_resolve_unknown_ref_raises_not_found() -> None:
    k = InMemoryKernel()
    with pytest.raises(NotFound):
        k.resolve("deadbeef")


def test_resolve_refolds_a_cell_from_the_log() -> None:
    k = InMemoryKernel()
    write = CellWriteEvent(payload=b"hello")
    # Seed the log directly (append_batch arrives in Task 8); index defining ref + by-hash.
    k._log.append(write)  # noqa: SLF001 - test seam, pre-append_batch
    k._cell_refs.add(write.cell_ref())  # noqa: SLF001
    result = k.resolve(write.cell_ref())
    assert result.payload == b"hello"  # type: ignore[union-attr]


def test_resolve_returns_a_tool_result_event_by_hash() -> None:
    k = InMemoryKernel()
    tool = ToolResultEvent(output=b"t")
    k._by_hash[tool.hash] = tool  # noqa: SLF001 - test seam, pre-append_batch
    assert k.resolve(tool.hash) is tool


def test_append_batch_assigns_contiguous_seqpos() -> None:
    k = InMemoryKernel()
    p1 = k.append_batch([CellWriteEvent(payload=b"a")])
    p2 = k.append_batch([CellWriteEvent(payload=b"b"), CellWriteEvent(payload=b"c")])
    assert p1 == [0]
    assert p2 == [1, 2]
    assert len(k.all_events()) == 3


def test_append_batch_indexes_defining_refs_and_tool_hashes() -> None:
    k = InMemoryKernel()
    write = CellWriteEvent(payload=b"a")
    tool = ToolResultEvent(output=b"t")
    k.append_batch([write, tool])
    # The written cell is now resolvable (re-folded), and the tool by its hash.
    assert k.resolve(write.cell_ref()).payload == b"a"  # type: ignore[union-attr]
    assert k.resolve(tool.hash) is tool


def test_append_batch_is_atomic_on_invalid_batch() -> None:
    k = InMemoryKernel()
    k.append_batch([CellWriteEvent(payload=b"ok")])
    with pytest.raises(ValueError):
        k.append_batch([CellWriteEvent(payload=b"x"), "not an event"])  # type: ignore[list-item]
    # The bad batch appended NOTHING (no torn provenance).
    assert len(k.all_events()) == 1


def test_append_batch_rejects_empty_batch() -> None:
    k = InMemoryKernel()
    with pytest.raises(ValueError):
        k.append_batch([])


def test_subscribe_notifies_post_commit_in_seqpos_order() -> None:
    k = InMemoryKernel()
    seen: list[tuple[int, Event]] = []
    k.subscribe(lambda pos, ev: seen.append((pos, ev)))
    a = CellWriteEvent(payload=b"a")
    b = CellWriteEvent(payload=b"b")
    k.append_batch([a, b])
    assert [pos for pos, _ in seen] == [0, 1]  # in SeqPos order
    assert [ev for _, ev in seen] == [a, b]


def test_observer_failure_does_not_roll_back_append() -> None:
    k = InMemoryKernel()

    def boom(pos: int, ev: Event) -> None:
        raise RuntimeError("observer kaboom")

    k.subscribe(boom)
    positions = k.append_batch([CellWriteEvent(payload=b"a")])
    assert positions == [0]  # append committed despite observer error
    assert len(k.all_events()) == 1  # the log grew (log is truth)
    assert len(k.observer_errors) == 1  # error recorded, NOT silently swallowed


def test_unsubscribe_detaches_the_observer() -> None:
    k = InMemoryKernel()
    seen: list[int] = []
    sub = k.subscribe(lambda pos, ev: seen.append(pos))
    k.append_batch([CellWriteEvent(payload=b"a")])
    sub.unsubscribe()
    k.append_batch([CellWriteEvent(payload=b"b")])
    assert seen == [0]  # only the first append was observed


def test_unsubscribe_is_idempotent() -> None:
    k = InMemoryKernel()
    seen: list[int] = []
    sub = k.subscribe(lambda pos, ev: seen.append(pos))
    sub.unsubscribe()
    sub.unsubscribe()  # must not raise
    k.append_batch([CellWriteEvent(payload=b"a")])
    assert seen == []  # observer was already detached; no notification


def test_drop_materialized_forces_resolve_to_refold() -> None:
    k = InMemoryKernel()
    write = CellWriteEvent(payload=b"hello")
    k.append_batch([write])
    ref = write.cell_ref()
    first = k.resolve(ref)  # materializes + caches
    assert ref in k._materialized  # noqa: SLF001 - cache populated
    k.drop_materialized(ref)
    assert ref not in k._materialized  # noqa: SLF001 - cache cleared
    second = k.resolve(ref)  # re-folds from the log
    assert second.payload == first.payload == b"hello"  # type: ignore[union-attr]


def test_drop_materialized_is_idempotent() -> None:
    k = InMemoryKernel()
    k.drop_materialized("never-materialized")  # must not raise


def test_append_amending_event_evicts_stale_materialized_cache() -> None:
    from amplifier_data.models import RelationshipEvent

    k = InMemoryKernel()
    a = CellWriteEvent(payload=b"cell A")
    b = CellWriteEvent(payload=b"cell B")
    k.append_batch([a, b])
    a_ref = a.cell_ref()

    first = k.resolve(a_ref)  # materializes A into the cache (no edge yet)
    assert isinstance(first, Cell)
    assert len(first.provenance) == 1

    edge = RelationshipEvent(from_ref=a_ref, to_ref=b.cell_ref(), type="links_to")
    k.append_batch([edge])  # amends A; stale cache MUST be evicted

    second = k.resolve(a_ref)  # must re-fold from the log, not return stale cache
    assert isinstance(second, Cell)
    assert edge in second.provenance  # edge visible WITHOUT an explicit drop_materialized
    assert len(second.provenance) == 2


def test_observer_can_unsubscribe_itself_without_skipping_others() -> None:
    k = InMemoryKernel()
    seen: list[str] = []
    sub_a: Subscription | None = None

    def observer_a(pos: int, ev: Event) -> None:
        seen.append("a")
        assert sub_a is not None
        sub_a.unsubscribe()  # mutate the observer list mid-notification

    def observer_b(pos: int, ev: Event) -> None:
        seen.append("b")

    sub_a = k.subscribe(observer_a)
    k.subscribe(observer_b)
    k.append_batch([CellWriteEvent(payload=b"x")])
    assert seen == ["a", "b"]  # b must NOT be skipped despite a unsubscribing itself
