# Architecture — Layered View

This document describes the *shape* of the system layer by layer. For the full rationale
behind each layer, see `SPEC.md`. For the principles that motivate the shape, see
`PRINCIPLES.md`.

The system is four layers plus one feedback loop. Read bottom-up.

---

## Layer 0 — The append-only event log (the floor)

The single source of truth. Every write is an appended, immutable event. Nothing above
this layer is authoritative; everything above is a *recomputable view* of it.

- **Property:** durability and total order live here and nowhere else.
- **Invariant:** the log is the floor. If the entire rest of the system were deleted, the
  log alone must be sufficient to regenerate every Cell. (This is the load-bearing
  assumption of the whole design — see `CONSTRAINTS.md`, "The hard kernel question.")

---

## Layer 1 — The storage kernel (mechanism only)

A tiny, stable engine that knows *nothing* about meaning. Its entire surface area:

| Operation         | Responsibility                                              |
| ----------------- | ----------------------------------------------------------- |
| `append(event)`   | Add an event to the log                                     |
| `content-address(cell)` | Produce a stable, content-derived identity for a cell |
| `resolve(ref)`    | Turn a reference into the cell/event it points to           |
| `emit(event)`     | Broadcast that something happened (e.g. an access event)    |

> **Note (as implemented):** notification is **folded into `append`** — there is **no standalone
> `emit` verb** (you cannot signal-without-recording; observers are notified post-commit, in `SeqPos`
> order, as a side-effect of `append`). The kernel is implemented in **Rust behind the `StorageKernel`
> Protocol**, with a pure-Python fallback. See the [README](../README.md) and `amplifier_data/`.

That is the whole kernel. It is deliberately incapable of holding schema, query
semantics, consistency policy, or index strategy. Those are all pushed to Layer 3.

> **Design rule:** if a proposed kernel feature could be implemented as a lens, it does
> NOT belong in the kernel. The litmus test: "Could two teams want different behavior
> here?" → if yes, it's policy → it's a lens, not the kernel.

---

## Layer 2 — Cells (the storage unit / the brick)

The materialized unit that lenses and agents operate on. A Cell is a **fold over the
event log** — a regenerable materialized view, not a primary record.

```
Cell {
  payload      : verbatim bytes              # never summarized
  provenance   : event-log ref               # how it came to be
  interpreters : [agent refs]                # lenses able to read it
  studs        : [typed connection points]   # stable interface to other cells
}
```

- Cells connect via **typed studs**, not foreign keys.
- Cells carry their own logic (validator/migrator/summarizer) — **data plane = compute
  plane.**
- Interpreters/agents are themselves stored *as Cells*.

---

## Layer 3 — Lenses (policy, swappable)

This is where the "database types" of the old world live now — demoted from competing
products to **mounted interpretation modules**:

```
relational · graph · vector · temporal · domain-specific
```

- Mount a relational lens → the same cells present as tables.
- Mount a graph lens → they present as a property graph.
- Multiple, even **contradictory**, lenses can be mounted over identical bytes
  simultaneously, because schema is policy and the kernel holds only mechanism.

Lenses are the natural module boundary. In Amplifier terms, each lens is a swappable
"brick" with a stable contract to the kernel below and the planner above.

---

## Layer 4 — The query path (intent, compiled)

```
intent ─▶ planner-agent ─▶ execution plan (a Cell) ─▶ dispatched over lenses ─▶ result Cell
```

- **Input:** natural-language or structured intent.
- **Compiler:** a planner-agent produces an execution plan.
- **Output:** a typed result Cell, with provenance attached.
- The **plan itself is a stored Cell** — reusable, reviewable, optimizable, queryable.

---

## The reflexive loop (why it's "self-building")

```
Cell ──read──▶ access event ──▶ indexer-agent ──▶ self-built index Cell ──┐
  ▲                                                                        │
  └──────────────── regenerates the store's own structure ◀───────────────┘
```

Every read emits an access event. An indexer-agent folds hot access patterns and
regenerates index Cells to serve them — recording *why* it did so as provenance. Those
index Cells are themselves read, emitting more access events. **The database builds the
database.**

---

## Meaning-on-spawn (cross-cutting)

Orthogonal to the layers: interpretation is *deferred*. Raw cells are cheap (bytes +
history). When meaning is needed, the relevant interpreter agent is **spawned bound to
the cell**, returns a distilled result, and evaporates. Interpretation cost is paid
per-query, on demand, and is itself logged.

---

## Integrity (cross-cutting)

Not a gate at write-time. A **convergence process**: generate → critique → converge.
Unresolved conflicts do not vanish or get silently won — they become first-class,
queryable conflict Cells appended back to the log.

---

## Mapping to the Amplifier ecosystem (for implementers)

This architecture deliberately mirrors `amplifier-core`'s kernel/module split. Likely
integration points (to be decided in `OPEN_QUESTIONS.md`):

| amplifier-data concept | Amplifier analog                                  |
| ---------------------- | ------------------------------------------------- |
| storage kernel         | `amplifier-core` kernel (mechanism, not policy)   |
| lens                   | a module type (swappable, `mount()`-style)        |
| interpreter spawn      | agents-as-context-sinks (`session.spawn`)         |
| event log              | the JSONL session event log, generalized          |
| planner-agent          | recipe-author emitting a runnable recipe          |
| verbatim payload rule  | the memory bundle's former backend's "store verbatim, never summarize" rule |
