"""RED phase: Lens Protocol tests (Plan 2, E2 — lens-swap deliverable).

Design Decision D1: lenses read all_events(), never a side-car;
every lens recomputes from the totally-ordered log on each project().
"""

from __future__ import annotations

from typing import Any

from amplifier_data.kernel import InMemoryKernel
from amplifier_data.lenses import Lens
from amplifier_data.models import CellWriteEvent

# ---------------------------------------------------------------------------
# Minimal concrete lens used in tests only
# ---------------------------------------------------------------------------


class CountLens:
    """A stateless lens that counts events in the log (pure fold).

    This exists only to test the Lens Protocol.  It holds no mutable state
    between project() calls — each call folds kernel.all_events() fresh.
    """

    @property
    def name(self) -> str:
        return "count"

    def project(self, kernel: Any) -> dict[str, int]:
        """Pure fold: count all events in the log."""
        events = kernel.all_events()
        return {"event_count": len(events)}

    def query(self, **typed_args: Any) -> int:
        """Return typed query result (no natural-language planner)."""
        return int(typed_args.get("count", 0))


# ---------------------------------------------------------------------------
# Exported symbols
# ---------------------------------------------------------------------------


def test_lenses_exports_lens() -> None:
    """amplifier_data.lenses exports the Lens symbol."""
    import amplifier_data.lenses as m

    assert hasattr(m, "Lens")


def test_lenses_exports_view() -> None:
    """amplifier_data.lenses exports the View type alias."""
    import amplifier_data.lenses as m

    assert hasattr(m, "View")


def test_lenses_exports_result() -> None:
    """amplifier_data.lenses exports the Result type alias."""
    import amplifier_data.lenses as m

    assert hasattr(m, "Result")


# ---------------------------------------------------------------------------
# Protocol conformance (runtime_checkable)
# ---------------------------------------------------------------------------


def test_concrete_lens_satisfies_lens_protocol() -> None:
    """A concrete implementation with name/project/query satisfies Lens."""
    lens = CountLens()
    assert isinstance(lens, Lens)


def test_object_missing_name_does_not_satisfy_lens() -> None:
    """An object without a name attribute does NOT satisfy Lens."""

    class NoName:
        def project(self, kernel: Any) -> Any:
            return {}

        def query(self, **kw: Any) -> Any:
            return None

    assert not isinstance(NoName(), Lens)


def test_object_missing_project_does_not_satisfy_lens() -> None:
    """An object without a project method does NOT satisfy Lens."""

    class NoProject:
        @property
        def name(self) -> str:
            return "no-project"

        def query(self, **kw: Any) -> Any:
            return None

    assert not isinstance(NoProject(), Lens)


def test_object_missing_query_does_not_satisfy_lens() -> None:
    """An object without a query method does NOT satisfy Lens."""

    class NoQuery:
        @property
        def name(self) -> str:
            return "no-query"

        def project(self, kernel: Any) -> Any:
            return {}

    assert not isinstance(NoQuery(), Lens)


# ---------------------------------------------------------------------------
# name property
# ---------------------------------------------------------------------------


def test_lens_name_is_a_string() -> None:
    """name returns a non-empty string."""
    lens = CountLens()
    assert isinstance(lens.name, str)
    assert len(lens.name) > 0


# ---------------------------------------------------------------------------
# project() — pure-fold contract (Design Decision D1)
# ---------------------------------------------------------------------------


def test_project_returns_view_for_empty_kernel() -> None:
    """project() can be called on an empty kernel without error."""
    kernel = InMemoryKernel()
    lens = CountLens()
    view = lens.project(kernel)
    assert view["event_count"] == 0


def test_project_folds_all_events() -> None:
    """project() folds every event in kernel.all_events()."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"a")])
    kernel.append_batch([CellWriteEvent(payload=b"b"), CellWriteEvent(payload=b"c")])

    lens = CountLens()
    view = lens.project(kernel)
    assert view["event_count"] == 3


def test_project_reflects_newly_appended_events() -> None:
    """project() recomputes from all_events() each call — new events appear."""
    kernel = InMemoryKernel()
    lens = CountLens()

    view_before = lens.project(kernel)
    assert view_before["event_count"] == 0

    kernel.append_batch([CellWriteEvent(payload=b"x")])
    view_after = lens.project(kernel)
    assert view_after["event_count"] == 1


def test_project_is_pure_fold_same_result_on_same_kernel() -> None:
    """project() called twice on the same static kernel returns equal views."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"a")])
    lens = CountLens()

    view1 = lens.project(kernel)
    view2 = lens.project(kernel)
    assert view1 == view2


def test_project_holds_no_mutable_state_between_calls() -> None:
    """Mutating the returned view does not affect the next project() call."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"a")])
    lens = CountLens()

    view1 = lens.project(kernel)
    view1["event_count"] = 9999  # mutate returned view

    view2 = lens.project(kernel)
    assert view2["event_count"] == 1  # lens recomputed, not poisoned


# ---------------------------------------------------------------------------
# Independence of lens instances
# ---------------------------------------------------------------------------


def test_two_lens_instances_are_independent() -> None:
    """Two Lens instances over the same kernel are independent."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"a")])

    lens_a = CountLens()
    lens_b = CountLens()

    view_a = lens_a.project(kernel)
    view_b = lens_b.project(kernel)
    assert view_a == view_b  # same kernel → same view

    # Mutating one view does not affect the other's next fold.
    view_a["event_count"] = 9999
    fresh_b = lens_b.project(kernel)
    assert fresh_b["event_count"] == 1  # b refolds correctly


# ---------------------------------------------------------------------------
# query() — typed surface (no natural-language planner)
# ---------------------------------------------------------------------------


def test_query_returns_typed_result() -> None:
    """query(**typed_args) returns a typed result."""
    lens = CountLens()
    result = lens.query(count=42)
    assert isinstance(result, int)
    assert result == 42


def test_query_default_result() -> None:
    """query() with no args returns the default result."""
    lens = CountLens()
    result = lens.query()
    assert result == 0
