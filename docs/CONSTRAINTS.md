# Honest Constraints — Where This Is the Wrong Tool

This design is correct only for a **specific regime**, and is actively *worse* than
Postgres for most jobs. This document exists so the brainstorm starts grounded, not
seduced. A simplicity zealot would rightly attack this design — so we pre-empt.

---

## Where it is WRONG to use

### 1. Latency-sensitive paths
An agent-in-the-query-path is **milliseconds-to-seconds**, not microseconds. Never put
this in a checkout transaction, an ad-auction, or any hot path. It is for
interpretation-heavy, **low-QPS**, high-ambiguity data — knowledge bases, research
corpora, evolving domain models — not OLTP.

### 2. Anywhere you need determinism
Intent-compiled queries and convergent integrity are **non-deterministic by
construction.** That is disqualifying for ledgers, billing, and anywhere a query must
mean exactly one thing forever.
- *Partial recovery:* pin a lens + plan version and you recover determinism — **at the
  cost of the magic.** (Is that pinned mode worth specifying as a first-class feature?
  See `OPEN_QUESTIONS.md`.)

### 3. Cost-sensitive / unbounded-growth scenarios
"Meaning-on-spawn" is cheap at rest, but:
- the **event log grows unbounded**, and
- **replay-to-materialize has real compute cost.**

You are trading storage durability guarantees for interpretation flexibility. Without
compaction/checkpointing, cost is a real liability.

---

## The hard kernel question (the load-bearing risk)

> The entire design rests on the event log genuinely being the floor and Cells being
> **truly regenerable** from it.

The moment an interpreter has **hidden, non-reproducible state**, the "regenerable brick"
guarantee breaks — and you have a normal database wearing a costume.

This is the single assumption most likely to sink the project. Any implementation must
treat *reproducibility of interpretation* as a first-class, tested invariant — not an
aspiration. If an interpreter calls a non-deterministic model, that non-determinism must
be captured in the log (e.g. seed, model version, full output recorded as an event), or
the floor is a lie.

---

## What a skeptic should keep hammering

Bring these to the brainstorm and do not let them be hand-waved:

1. **"Why not just Postgres + pgvector + a good schema?"** — For most workloads, that *is*
   the right answer. What is the *specific* workload where it isn't?
2. **"Show me the regeneration actually working."** — Delete a materialized cell, replay,
   prove bit-for-bit (or semantically-equivalent) recovery.
3. **"Where does the non-determinism get pinned?"** — Concretely, in which event field?
4. **"What's the compaction story?"** — An append-only log with no compaction is a
   disk-fill bug with extra steps.
5. **"What is the smallest thing that proves the core inversion?"** — meaning-on-spawn +
   reflexive index. If a tiny prototype can't show that, the big version won't either.

---

## The honest one-liner

> Right for interpretation-heavy knowledge; **wrong** for anything that needs a microsecond
> or a single fixed answer.

If the use case in front of you needs a microsecond or a single fixed answer, stop —
`amplifier-data` is the wrong tool, and that is a feature of this document, not a failure
of the design.
