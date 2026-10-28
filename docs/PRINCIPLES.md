# Design Principles

These are the principles `amplifier-data` carries down *into the storage layer*. They are
borrowed deliberately from the Amplifier kernel philosophy — the bet is that the same
properties that make a good agent framework make a good AI-age data store.

Each principle below states the idea, then how it manifests concretely in the store.

---

## 1. Mechanism, not policy

The kernel provides capabilities; modules decide behavior.

- **In the store:** the kernel does only `append · content-address · resolve · emit`.
  Schema, indexing, query semantics, and consistency model are all *policy*, pushed out
  to lenses.
- **Litmus test:** "Could two teams want different behavior here?" → if yes, it's policy →
  it does not belong in the kernel.

## 2. Bricks & studs (composition)

Self-contained units with stable, typed connection points.

- **In the store:** the Cell is the brick. Its `studs` are typed connection points. Cells
  compose through studs, not foreign keys. Any cell is independently regenerable.

## 3. Event-first (the log is the floor)

If it happened, it's an appended event. State is a fold over events.

- **In the store:** the append-only event log is the single source of truth. Cells are
  regenerable materialized views. Lose a cell → replay the log.

## 4. Reflexivity (output type = input type)

The system's outputs are valid inputs to itself.

- **In the store:** schemas, queries, indices, and execution plans are *themselves Cells*
  — stored, versioned, and queryable. You can query your queries. Schemas can generate
  schemas. Indices self-build.

## 5. Context sinks (pay for meaning only on demand)

Heavy understanding loads only when asked, returns a distilled result, evaporates.

- **In the store:** meaning-on-spawn. Raw cells are cheap; interpreters spawn per-query,
  bound to a cell, and disappear. A petabyte of cells costs ~nothing in meaning overhead
  at rest.

## 6. Regeneration over editing

Prefer recomputing a clean artifact to mutating a stale one.

- **In the store:** storage is a *cache of a computation*. The materialized payload is
  disposable; the provenance log is precious. The indexer regenerates structures rather
  than patching them.

## 7. Integrity by convergence, not by gate

Quality emerges from generate → critique → converge, not from an up-front constraint that
rejects on write.

- **In the store:** conflicting writes are flagged by critic-agents and converge over
  time. Eventually-consistent *semantics*, not just bytes.

## 8. Honest stopping (no silent winners)

When something can't be reconciled with real evidence, surface it — never fabricate a
resolution.

- **In the store:** an unresolvable conflict becomes a first-class, queryable conflict
  Cell carrying both provenances and what would resolve it. Integrity violations are data,
  not vanished exceptions.

---

## The unifying bet

> A store built for AI agents should look more like an **agent framework with a log at the
> bottom** than like a traditional database with an AI feature bolted on.

Every principle above is a wager that *meaning is computed, deferred, and reflexive* — and
that designing for that from the floor up beats retrofitting it. `CONSTRAINTS.md` is the
honest accounting of where this bet does **not** pay off.
