# Specification — The Amplifier-Shaped Store

> Status: **Design draft.** This is the canonical, consolidated specification.
> This export includes the maintained specification; development archives are excluded.

A data store designed from first principles for the AI age — carrying a specific set of
properties (mechanism not policy, bricks & studs, event-first, reflexivity, context
sinks) down into the storage layer. The goal is **not** "SQL but smarter." It is a
different *category* of thing.

---

## The core inversion

Every database before this stores data and applies meaning as a **separate act**:

| Paradigm | What it fixes           | When meaning is applied |
| -------- | ----------------------- | ----------------------- |
| SQL      | schema                  | on write                |
| NoSQL    | nothing (bags of bytes) | on read                 |
| Graph    | relationships           | on write                |
| Vector   | similarity              | on embed                |

An Amplifier-shaped store makes one move none of them make: **it stores data together
with the means of interpreting and regenerating that data — and the means are
themselves first-class, queryable data.** Schema-on-write and schema-on-read both become
obsolete. The new mode is **meaning-on-spawn**.

That single property — the interpreter travels with the data and is itself storable — is
what makes it "meta," and what makes SQL/NoSQL/graph/vector reduce to *lenses over it*
rather than competitors to it.

---

## 1. The unit: the Cell (storage as a brick)

Not a row, not a document, not a node. A **Cell** is a self-contained brick with studs:

```
Cell {
  payload      : verbatim bytes              # stored as-is, never summarized
  provenance   : event-log ref               # the append-only history of how it came to be
  interpreters : [agent refs]                # lenses able to read it — loaded on demand
  studs        : [typed connection points]   # its stable interface to other cells
}
```

Three consequences fall straight out of bricks & studs:

- **Every cell is regenerable from its provenance log alone.** Lose the materialized
  payload? Replay the events. Storage becomes a *cache of a computation*, not the source
  of truth.
- **Cells compose through typed studs, not foreign keys.** A relationship is just a stud
  one cell exposes and another consumes.
- **The interpreter is part of the cell, not the application.** Logic lives in the data
  plane, not outside it. A cell can carry its own validator, its own migrator, its own
  summarizer.

---

## 2. The kernel: a storage engine that knows nothing about meaning

The center stays still. The kernel does exactly **four things** and refuses to know
anything else:

```
append(event) · content-address(cell) · resolve(ref) · emit(event)
```

> **Note (as implemented):** notification is **folded into `append`** — there is **no standalone
> `emit` verb** (you cannot signal-without-recording; observers are notified post-commit, in `SeqPos`
> order). The kernel is implemented in **Rust behind the `StorageKernel` Protocol**, with a pure-Python
> fallback. See the [README](../README.md) and `amplifier_data/` for the built surface.

That's the whole kernel. **Schema, indexing strategy, query semantics, consistency
model, and the very choice of "relational vs graph vs vector" — all of it is a module
mounted at the edge.** The radical payoff:

> SQL, NoSQL, graph, and vector stop being *databases* and become **lenses** — swappable
> interpretation modules over one substrate. Mount a relational lens and the same cells
> present as tables. Mount a graph lens and they present as a property graph. Two teams
> can hold contradictory schemas over identical bytes simultaneously, because schema is
> policy and the kernel only holds mechanism.

You don't migrate *between* database types anymore. You mount a different lens.

---

## 3. Reflexivity: schemas, queries, and indices are stored Cells

The "machines that build machines" property, dropped into storage. **Output type = input
type.** A query result is a Cell. A schema is a Cell. An index is a materialized-view
Cell with its own provenance. This unlocks things no existing DB can do:

- **You query your queries.** The history of *how the data was interrogated* is itself in
  the store: "show me every lens that ever read this cell, and what it concluded."
- **Schemas generate schemas.** A schema-Cell can be the output of an agent that observed
  the data — schema *discovered*, versioned, and queryable, not declared upfront.
- **Indices self-build.** The kernel emits an access event on every read. An
  indexer-agent folds those events, notices a hot access pattern, and *regenerates* an
  index Cell to serve it — then records that decision as provenance. The store tunes
  itself and can explain why.

State, in this model, is just a **fold over the event log** — a materialized view any
lens can recompute. The log is the floor; everything above it is a regenerable opinion.

---

## 4. Meaning-on-spawn (context sinks for data)

Raw Cells are cheap and dumb — bytes plus a history. **Rich meaning is expensive, so you
don't pay for it until you ask.** When you need a cell interpreted, the store *spawns its
interpreter agent* bound to that cell. The interpreter is a context sink: heavy semantic
understanding loads only on spawn, returns a distilled result, and evaporates.

This is the storage analog of agents-as-context-sinks. A petabyte of cells costs almost
nothing in "meaning overhead" at rest. Interpretation cost is incurred per-query,
per-region, on demand — and is itself logged.

---

## 5. Integrity by convergence, not by gate

SQL rejects a bad write at the constraint gate. This store does something closer to how
agents ship code: **generate → critique → converge.** On conflicting or ambiguous writes,
critic-agents flag the tension and the store converges toward consistent meaning over
time (eventually-consistent *semantics*, not just eventually-consistent bytes).

And — crucially, per honest-stopping discipline — when it cannot reconcile, it does
**not** silently pick a winner. The unresolved conflict becomes a first-class, queryable
Cell: "here is a contradiction, here are the two provenances, here is what would resolve
it." Integrity violations are *data you can query*, not exceptions that vanish.

---

## 6. The query language is intent, compiled by an agent

You don't write SQL. You express **intent**; a planner-agent compiles it into an
execution plan across whichever lenses apply, runs it, and returns a typed result Cell
with provenance attached. The plan itself is stored — reviewable, reusable, optimizable,
and (reflexively) queryable. The plan is an artifact that re-enters the system.

> **Note:** the "language" of amplifier-data is *agent-compiled intent*, not a grammar.
> There is no `.g4`/`.lark`/EBNF to write. The closest analog is a planner that emits a
> runnable, storable plan — see `OPEN_QUESTIONS.md` for whether a stable intermediate
> representation (IR) should nonetheless be specified.

---

## The whole shape

```
intent ─▶ planner-agent ─▶ execution plan (a Cell) ─┐
                                                     ▼
   ┌───────────── LENSES (policy, swappable) ───────────────┐
   │  relational · graph · vector · temporal · domain-specific │   ← mounted modules
   └───────────────────────┬─────────────────────────────────┘
                              ▼
   ┌───────────── STORAGE KERNEL (mechanism only) ───────────┐
   │   append · content-address · resolve · emit               │
   └───────────────────────┬─────────────────────────────────┘
                              ▼
        APPEND-ONLY EVENT LOG  ── the single source of truth
                              │
        Cells = folds over the log (regenerable materialized views)
                              │
   interpreters/agents stored AS cells ──▶ data plane = compute plane
                              ▲
        access events ──▶ indexer-agent ──▶ self-built index cells (loop closes)
```

The loop at the bottom is the point: access generates events → events drive an agent →
the agent regenerates the store's own structures → those structures are Cells → which
generate more access events. **The database builds the database.** Recursion at the
scaffolding layer — the store organizing itself.

---

## The one-line spec

> An append-only event substrate where the unit of storage is a self-interpreting,
> regenerable Cell; where SQL/NoSQL/graph/vector are swappable lenses rather than
> database types; where schemas, queries, and indices are themselves first-class
> queryable Cells; and where the store observes its own access patterns and regenerates
> its own structure — a database whose outputs are valid inputs to itself. Right for
> interpretation-heavy knowledge; wrong for anything that needs a microsecond or a single
> fixed answer.

See `CONSTRAINTS.md` for the full honest account of where this is the wrong tool.
