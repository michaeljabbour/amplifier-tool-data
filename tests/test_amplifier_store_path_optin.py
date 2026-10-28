"""Tests for the additive AmplifierStore(path=...) durable opt-in.

Task T7 specification:
- AmplifierStore() with no path behaves exactly as before (in-memory)
- AmplifierStore(path=...) uses the durable RustFileKernel via DurableKernel
- close() flushes/closes cleanly; safe no-op on in-memory stores
- RUST_AVAILABLE-false guard is present and verified by code inspection
- New symbols exported from amplifier_data.__init__
"""

from __future__ import annotations

import inspect
import os
import tempfile

from amplifier_data.store import AmplifierStore

# ---------------------------------------------------------------------------
# Existing in-memory behaviour is unchanged
# ---------------------------------------------------------------------------


def test_no_path_store_write_and_read_unchanged() -> None:
    """AmplifierStore() without path= behaves exactly as before: write + read round-trip."""
    store = AmplifierStore()
    ref = store.write_cell(b"hello world")
    cell = store.get_cell(ref)
    assert cell.payload == b"hello world"


def test_no_path_store_uses_in_memory_kernel() -> None:
    """AmplifierStore() without path= does NOT use a DurableKernel."""
    from amplifier_data.kernel import DurableKernel

    store = AmplifierStore()
    assert not isinstance(store._kernel, DurableKernel)


def test_no_path_close_is_noop() -> None:
    """close() on an in-memory store does not raise (safe no-op)."""
    store = AmplifierStore()
    store.close()  # must not raise


# ---------------------------------------------------------------------------
# AmplifierStore(path=...) selects DurableKernel
# ---------------------------------------------------------------------------


def test_path_creates_durable_kernel() -> None:
    """AmplifierStore(path=...) sets _kernel to a DurableKernel instance."""
    from amplifier_data.kernel import DurableKernel

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "store.log")
        store = AmplifierStore(path=path)
        try:
            assert isinstance(store._kernel, DurableKernel)
        finally:
            store.close()


def test_path_write_and_read_back() -> None:
    """Write a cell to a durable store and read it back within the same session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "store.log")
        store = AmplifierStore(path=path)
        try:
            ref = store.write_cell(b"durable data")
            cell = store.get_cell(ref)
            assert cell.payload == b"durable data"
        finally:
            store.close()


def test_path_relate_and_query_relational() -> None:
    """Relational lens works with a durable store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "store.log")
        store = AmplifierStore(path=path)
        try:
            ref_a = store.write_cell(b"cell A")
            ref_b = store.write_cell(b"cell B")
            store.relate(ref_a, ref_b, "depends_on")

            result = store.query_relational()
            assert result.success
        finally:
            store.close()


# ---------------------------------------------------------------------------
# close() lifecycle
# ---------------------------------------------------------------------------


def test_path_close_is_callable() -> None:
    """close() on a durable store completes without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "store.log")
        store = AmplifierStore(path=path)
        store.close()  # must not raise


def test_path_close_is_idempotent() -> None:
    """close() can be called multiple times without raising."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "store.log")
        store = AmplifierStore(path=path)
        store.close()
        store.close()  # second call must not raise


# ---------------------------------------------------------------------------
# Persistence: data survives close + reopen
# ---------------------------------------------------------------------------


def test_path_data_persists_across_reopen() -> None:
    """Data written to a durable store survives close + reopen."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "store.log")

        # Session 1: write
        store1 = AmplifierStore(path=path)
        ref = store1.write_cell(b"persisted payload")
        store1.close()

        # Session 2: reopen and verify
        store2 = AmplifierStore(path=path)
        try:
            cell = store2.get_cell(ref)
            assert cell.payload == b"persisted payload"
        finally:
            store2.close()


# ---------------------------------------------------------------------------
# RUST_AVAILABLE=False guard — verified by code inspection (not runtime)
#
# The spec acknowledges this branch is unreachable in the built-extension dev
# environment. We verify the guard is PRESENT by inspecting the source.
# ---------------------------------------------------------------------------


def test_rust_unavailable_guard_is_present_in_init() -> None:
    """AmplifierStore.__init__ contains the RUST_AVAILABLE guard and RuntimeError."""
    source = inspect.getsource(AmplifierStore.__init__)
    assert "RUST_AVAILABLE" in source, (
        "AmplifierStore.__init__ must check RUST_AVAILABLE when path= is supplied"
    )
    assert "RuntimeError" in source, (
        "AmplifierStore.__init__ must raise RuntimeError when RUST_AVAILABLE is False"
    )


# ---------------------------------------------------------------------------
# New symbols exported from amplifier_data.__init__
# ---------------------------------------------------------------------------


def test_amplifier_store_importable_from_package() -> None:
    """AmplifierStore is importable directly from the amplifier_data package."""
    from amplifier_data import AmplifierStore as _AS  # noqa: PLC0415

    assert _AS is AmplifierStore


def test_durable_kernel_importable_from_package() -> None:
    """DurableKernel is importable from the amplifier_data package."""
    from amplifier_data import DurableKernel  # noqa: PLC0415
    from amplifier_data.kernel import DurableKernel as _DK

    assert DurableKernel is _DK
