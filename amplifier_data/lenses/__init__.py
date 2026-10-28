"""Lens Protocol — swappable pure-fold views over the event log (Plan 2, E2).

Design Decision D1
-------------------
Lenses read ``all_events()``, never a side-car.  Every lens recomputes from
the totally-ordered log on each ``project()`` invocation.  This is the
load-bearing constraint of Plan 2: lenses are the only read path, and each
``project()`` call is a fresh fold over the complete event history.

Overview
--------
A :class:`Lens` is a pure fold over a storage kernel's event log:

* :meth:`~Lens.project` consumes ``kernel.all_events()`` and returns a
  freshly computed, immutable-by-contract :data:`View`.  It must persist
  **no mutated state** between calls.  Two :class:`Lens` instances over the
  same kernel are independent and may be used concurrently.

* :meth:`~Lens.query` is a **typed** surface over that view.  There is no
  natural-language planner; callers supply typed keyword arguments and receive
  a typed :data:`Result`.

:data:`View` and :data:`Result` are exported as type aliases.  Each concrete
lens implementation should define its own lens-local aliases that narrow these
to the shapes it actually produces.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias, runtime_checkable

# ---------------------------------------------------------------------------
# Lens-local type aliases
# ---------------------------------------------------------------------------
# Concrete lens implementations narrow these to their own view / result shapes.
# The Protocol itself uses the base ``Any`` form so it imposes no structural
# constraint at the Protocol level — shape checking happens in the concrete
# implementation, where the lens-local View / Result types are defined.

View: TypeAlias = Any
"""Materialized view produced by :meth:`Lens.project`.

Each concrete lens defines a lens-local alias that narrows this to its actual
shape, e.g.::

    # Inside a concrete lens module:
    View = dict[str, int]   # lens-local alias
"""

Result: TypeAlias = Any
"""Query result produced by :meth:`Lens.query`.

Each concrete lens defines a lens-local alias that narrows this to its actual
shape, e.g.::

    # Inside a concrete lens module:
    Result = int   # lens-local alias
"""


# ---------------------------------------------------------------------------
# Lens Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Lens(Protocol):
    """Swappable, pure-fold view over a storage kernel's event log.

    Contract (Design Decision D1)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * :meth:`project` is a **pure fold**: it reads ``kernel.all_events()``
      and returns a freshly computed :data:`View`.  No mutated state is held
      between calls.  Two :class:`Lens` instances over the same kernel are
      independent and may be used concurrently.
    * :meth:`query` is a **typed** query surface — there is no natural-language
      planner; callers supply typed keyword arguments and receive a typed
      :data:`Result`.

    Example concrete implementation::

        class MyLens:
            View   = dict[str, list[str]]  # lens-local alias
            Result = list[str]             # lens-local alias

            @property
            def name(self) -> str:
                return "my-lens"

            def project(self, kernel: Any) -> dict[str, list[str]]:
                # Pure fold: build the view from all_events()
                index: dict[str, list[str]] = {}
                for _pos, event in kernel.all_events():
                    ...
                return index

            def query(self, **typed_args: Any) -> list[str]:
                # Typed surface — no NL planner
                return typed_args.get("items", [])
    """

    @property
    def name(self) -> str:
        """Human-readable identifier for this lens type."""
        ...

    def project(self, kernel: Any) -> View:
        """Fold ``kernel.all_events()`` into a freshly computed view.

        Parameters
        ----------
        kernel:
            A storage kernel that exposes ``all_events() -> list[tuple[SeqPos, Event]]``.

        Returns
        -------
        View
            A freshly computed materialized view.  The lens must not persist any
            mutated state derived from this computation between calls.

        Constraints
        -----------
        * Must only read from ``kernel.all_events()``; must not consult any
          side-car or cached state held on the lens instance.
        * Two concurrent calls on different :class:`Lens` instances backed by the
          same kernel must not interfere.
        """
        ...

    def query(self, **typed_args: Any) -> Result:
        """Typed query surface over the view.

        Parameters
        ----------
        **typed_args:
            Typed keyword arguments specific to this lens's query contract.

        Returns
        -------
        Result
            A typed query result.  The exact type is defined by the concrete lens.

        Constraints
        -----------
        There is **no** natural-language planner.  Callers are expected to supply
        well-typed arguments matching the lens's documented query contract.
        """
        ...


__all__ = ["Lens", "View", "Result"]
