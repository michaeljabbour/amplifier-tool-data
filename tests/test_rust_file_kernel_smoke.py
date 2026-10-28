"""Smoke test for the RustFileKernel PyO3 binding.

Phase 2 Task 6: RustFileKernel pyclass wrapping the Rust FileKernel.

Exercises the open/append/read/close lifecycle through the Python binding.

KEY DESIGN DECISION — value-equality durable-read identity
==========================================================
RustFileKernel does NOT store original Python objects in a parallel Vec (unlike
RustKernel, which keeps the originals for identity semantics).  Events are
persisted as encoded Rust bytes and later reconstructed via rust_event_to_py.

Consequence: all_events() returns Python events that compare == to the
originals (VALUE-EQUALITY), but are NOT the same Python objects (no identity).

    py_ev == ev   ← True   (same data)
    py_ev is ev   ← False  (different Python object — reconstructed from Rust)
"""

import os
import tempfile

import pytest

from amplifier_data import _rust  # type: ignore[import]
from amplifier_data.models import CellWriteEvent, ToolResultEvent

# RustFileKernel is only available after maturin develop has built this task;
# use getattr so a missing attribute gives a failing test rather than a
# collection-time ImportError that would break unrelated tests.
RustFileKernel = getattr(_rust, "RustFileKernel", None)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _require_rust_file_kernel() -> type:
    """Return RustFileKernel or fail the test with a helpful message."""
    if RustFileKernel is None:
        pytest.fail(
            "RustFileKernel not yet implemented in amplifier_data._rust; "
            "implement the #[pyclass] RustFileKernel in "
            "crates/amplifier-data-py/src/lib.rs and rebuild with `maturin develop`."
        )
    return RustFileKernel  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Lifecycle: open creates the file
# ---------------------------------------------------------------------------


def test_rust_file_kernel_open_creates_file() -> None:
    """open() creates a new log file on disk when the path does not exist."""
    fk_cls = _require_rust_file_kernel()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "events.log")
        assert not os.path.exists(path), "precondition: file must not exist yet"
        fk = fk_cls.open(path)
        assert os.path.exists(path), "open() must create the file"
        fk.close()


# ---------------------------------------------------------------------------
# Core round-trip: append + all_events (value-equality, not identity)
# ---------------------------------------------------------------------------


def test_rust_file_kernel_append_and_read_back_value_equal() -> None:
    """append() persists; all_events() reads back as a VALUE-EQUAL Python event."""
    fk_cls = _require_rust_file_kernel()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "events.log")
        ev = CellWriteEvent(payload=b"hello", interpreters=())
        fk = fk_cls.open(path)
        fk.append([ev])
        events = fk.all_events()
        fk.close()

        assert len(events) == 1
        seq_pos, py_ev = events[0]
        assert seq_pos == 0
        # VALUE-EQUALITY: same data, different Python object
        assert py_ev == ev, "read-back event must compare == to the original"


def test_rust_file_kernel_read_back_is_not_identity() -> None:
    """all_events() returns NEW Python objects, not the originals (value-equality design)."""
    fk_cls = _require_rust_file_kernel()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "events.log")
        ev = CellWriteEvent(payload=b"data", interpreters=())
        fk = fk_cls.open(path)
        fk.append([ev])
        events = fk.all_events()
        fk.close()

        _, py_ev = events[0]
        # NOT identity — rust_event_to_py reconstructs a fresh Python object
        assert py_ev is not ev, (
            "all_events() must return a new Python object, not the original "
            "(value-equality durable-read identity, see task spec)"
        )


def test_rust_file_kernel_multiple_events_in_one_batch() -> None:
    """append() accepts multiple events; all_events() reads them back in order."""
    fk_cls = _require_rust_file_kernel()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "events.log")
        ev1 = CellWriteEvent(payload=b"first", interpreters=())
        ev2 = ToolResultEvent(output=b"tool output")
        fk = fk_cls.open(path)
        fk.append([ev1, ev2])
        events = fk.all_events()
        fk.close()

        assert len(events) == 2
        pos0, py_ev1 = events[0]
        pos1, py_ev2 = events[1]
        assert pos0 == 0
        assert pos1 == 1
        assert py_ev1 == ev1
        assert py_ev2 == ev2


def test_rust_file_kernel_two_separate_appends_accumulate() -> None:
    """Two separate append() calls accumulate events in the log."""
    fk_cls = _require_rust_file_kernel()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "events.log")
        ev1 = CellWriteEvent(payload=b"alpha", interpreters=())
        ev2 = CellWriteEvent(payload=b"beta", interpreters=())
        fk = fk_cls.open(path)
        fk.append([ev1])
        fk.append([ev2])
        events = fk.all_events()
        fk.close()

        assert len(events) == 2
        assert events[0][1] == ev1
        assert events[1][1] == ev2


# ---------------------------------------------------------------------------
# Lifecycle: close
# ---------------------------------------------------------------------------


def test_rust_file_kernel_close_releases_file_handle() -> None:
    """close() flushes and releases the handle; the same path can be re-opened."""
    fk_cls = _require_rust_file_kernel()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "events.log")
        fk = fk_cls.open(path)
        fk.close()
        # Reopening after close must succeed (handle was released)
        fk2 = fk_cls.open(path)
        fk2.close()


# ---------------------------------------------------------------------------
# Existing RustKernel binding is untouched
# ---------------------------------------------------------------------------


def test_existing_rust_kernel_still_works() -> None:
    """RustKernel (identity semantics) must not be disturbed by this task."""
    from amplifier_data._rust import RustKernel  # type: ignore[import]

    k = RustKernel()
    ev = CellWriteEvent(payload=b"probe", interpreters=())
    k.append_batch([ev])
    all_evs = k.all_events()
    assert len(all_evs) == 1
    _, returned_ev = all_evs[0]
    # RustKernel preserves IDENTITY (is), not just equality
    assert returned_ev is ev
