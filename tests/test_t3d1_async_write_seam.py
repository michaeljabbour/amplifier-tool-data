"""T3D-1 killer gates — the async write seam and concurrent-writer safety.

The store is synchronous by design; T3D-1 adds an async seam that offloads writes
to a worker thread (``asyncio.to_thread``) while serializing them so concurrent
writers cannot corrupt the totally-ordered log. These gates prove:

* the async path lands writes that read back correctly,
* the existing sync API is unchanged (still works alongside the async path),
* N concurrent async writers all land with no lost / duplicated SeqPos,
* the kernel-level ``append_batch_async`` seam is concurrency-safe too.
"""

from __future__ import annotations

import asyncio

from amplifier_data import AmplifierStore
from amplifier_data.kernel import InMemoryKernel
from amplifier_data.models import CellWriteEvent


def _count_cell_writes(store: AmplifierStore) -> int:
    kernel: InMemoryKernel = store.kernel  # type: ignore[assignment]
    return sum(1 for _pos, ev in kernel.all_events() if isinstance(ev, CellWriteEvent))


def test_write_cell_async_round_trips() -> None:
    """A cell written through the async seam reads back byte-identically."""
    store = AmplifierStore()

    async def scenario() -> str:
        return await store.write_cell_async(b"async-payload")

    ref = asyncio.run(scenario())
    assert store.get_cell(ref).payload == b"async-payload"


def test_sync_api_unchanged_alongside_async() -> None:
    """The sync write_cell still works exactly as before (no behavioural drift)."""
    store = AmplifierStore()
    sync_ref = store.write_cell(b"sync")
    async_ref = asyncio.run(store.write_cell_async(b"async"))
    assert store.get_cell(sync_ref).payload == b"sync"
    assert store.get_cell(async_ref).payload == b"async"
    # Sync ref is the plain content address — async did not change identity rules.
    assert sync_ref == CellWriteEvent(payload=b"sync").cell_ref()


def test_concurrent_async_writers_all_land() -> None:
    """KILLER GATE: 50 concurrent async writers → 50 distinct cells, no loss.

    Uses an explicit InMemoryKernel so the thread-safety fix in
    ``InMemoryKernel.append_batch`` (RLock around SeqPos assignment + append) is
    the thing under test. Without it, two threads could compute the same ``start``
    and clobber a slot.
    """
    store = AmplifierStore(InMemoryKernel())
    n = 50

    async def scenario() -> list[str]:
        tasks = [store.write_cell_async(f"cell-{i}".encode()) for i in range(n)]
        return await asyncio.gather(*tasks)

    refs = asyncio.run(scenario())

    assert len(set(refs)) == n  # every write produced a distinct cell
    assert _count_cell_writes(store) == n  # nothing was lost or overwritten
    for i, ref in enumerate(refs):
        assert store.get_cell(ref).payload == f"cell-{i}".encode()


def test_kernel_append_batch_async_concurrency_preserves_total_order() -> None:
    """KILLER GATE: concurrent kernel-level async appends keep contiguous SeqPos.

    Each batch must occupy a contiguous, non-overlapping SeqPos run; the union of
    all assigned positions must be exactly ``range(total)`` with no gaps/dupes.
    """
    kernel = InMemoryKernel()
    batches = 40

    async def scenario() -> list[list[int]]:
        tasks = [
            kernel.append_batch_async([CellWriteEvent(payload=f"b{i}".encode())])
            for i in range(batches)
        ]
        return await asyncio.gather(*tasks)

    position_runs = asyncio.run(scenario())

    all_positions = sorted(p for run in position_runs for p in run)
    assert all_positions == list(range(batches))  # no lost, no duplicated SeqPos
    # Each batch of one event got exactly one contiguous position.
    assert all(len(run) == 1 for run in position_runs)
