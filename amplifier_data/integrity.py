"""Convergent integrity (Plan 4b, E4) -- mechanism, not policy.

The substrate owns NO conflict rules and NO model. Critic is an injected
protocol (the Plan 3i Embedder pattern); a detected tension is stored as an
ordinary content-addressed Cell with @integrity: provenance edges (the
causal.py / Plan 4a pattern); detection is a post-hoc fold driven by an
AutonomicIndexer-shaped monitor (the T3E pattern). Writes are never gated;
unresolved tensions are queryable data, forever.

Design choices
--------------
* **The critic is injected.** ``Critic.detect(store) -> list[Tension]``
  receives the store's full D4 read surface (``query_facts``,
  ``graph_neighbors``, ``kernel.all_events``, ...) rather than pre-extracted
  facts -- "conflicts are about facts" is itself policy. The substrate ships
  :class:`TemporalContradictionCritic` (deterministic, pure fold) as the
  test/example reference.
* **A tension is an ordinary Cell.** Payload = canonical JSON of a small
  versioned envelope carrying the parties (conflicting provenances, as
  refs), the nature of the contradiction, and what would resolve it --
  stored via ``CellWriteEvent`` + reserved ``@integrity:`` ``RelationshipEvent``s
  in ONE atomic batch. No new event type, zero Rust changes.
* **Detection is post-hoc, never a gate.** :class:`IntegrityMonitor` mirrors
  :class:`~amplifier_data.autonomic.AutonomicIndexer`: a read-only observer
  counts write pressure; ``run_once()`` folds the log through the critic and
  materializes only NEW tensions (known set folded from the log itself, not
  from memory, so idempotence holds across process restarts and fresh
  monitor instances).
* **Resolution is an explicit event, and only that.** The monitor never
  auto-resolves and never auto-reopens -- that is consumer policy.
  Re-opening needs no new verb: ``invalidate_fact(resolution_ref,
  "@integrity:resolves", tension_ref)`` tombstones the resolves fact and the
  temporal fold shows the tension unresolved again.
* **Never a silent winner.** Detection touches nothing but the tension cell
  it appends; both conflicting facts remain currently valid.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .lenses.temporal import INVALIDATE_PREFIX
from .models import CellWriteEvent, Event, Hash, RelationshipEvent, SeqPos

if TYPE_CHECKING:
    from .kernel import Subscription

TensionRef = Hash  # readability alias; a tension ref IS a cell ref
TENSION_VERSION = 1  # envelope version pinned into tension bytes
DEFAULT_THRESHOLD = 16  # writes-since-last-run that trigger run_once

# Reserved edge namespace (owned by this module; user code must not mint these).
_NAMESPACE = "@integrity:"
TENSION = "@integrity:tension"  # tension cell --> each party
IN_TENSION = "@integrity:in_tension"  # each party --> tension cell
CONFLICTS_WITH = "@integrity:conflicts_with"  # party_i --> party_j (all ordered pairs)
RESOLVES = "@integrity:resolves"  # resolution cell --> tension cell


@dataclass(frozen=True)
class Tension:
    """One detected conflict, as emitted by a :class:`Critic`. Pure data."""

    kind: str  # critic-defined, e.g. "temporal_contradiction"
    parties: tuple[Hash, ...]  # the conflicting provenances (>= 2, distinct)
    nature: str  # human-readable: what contradicts what
    resolution_hint: str  # what would resolve it
    detail: dict[str, Any] = field(default_factory=dict)  # critic context; JSON data only


@runtime_checkable
class Critic(Protocol):
    """Name the conflicts. Bring your own rules/model; amplifier-data ships none.

    ``detect()`` MUST be read-only over the store's D4 surfaces (``query_facts``,
    ``graph_neighbors``, ``kernel.all_events``, ...) and SHOULD be deterministic --
    determinism of the whole loop is exactly the determinism of the critic.
    """

    def detect(self, store: Any) -> list[Tension]: ...


class InvalidTensionError(ValueError):
    """Critic emitted a malformed Tension (< 2 parties, duplicate parties,
    empty kind, or detail that is not canonically JSON-serializable)."""


class NotATensionError(ValueError):
    """``resolve_tension`` target exists but is not a decodable tension cell."""


def _tension_errors(t: Tension) -> list[str]:
    """Structural validation errors for *t* (``[]`` == valid). Pure."""
    errors: list[str] = []
    if not t.kind:
        errors.append("kind must be a non-empty string")
    if len(t.parties) < 2:
        errors.append("parties must have at least 2 entries")
    elif len(set(t.parties)) != len(t.parties):
        errors.append("parties must be distinct")
    return errors


def tension_payload(t: Tension) -> bytes:
    """Canonical tension bytes (sorts parties; validates first).

    Raises :class:`InvalidTensionError` -- before any bytes are produced --
    when *t* has fewer than 2 parties, duplicate parties, an empty ``kind``,
    or a ``detail`` that is not canonically JSON-serializable.
    """
    errors = _tension_errors(t)
    if errors:
        raise InvalidTensionError("invalid tension: " + "; ".join(errors))
    body = {
        "v": TENSION_VERSION,
        "kind": t.kind,
        "parties": sorted(t.parties),
        "nature": t.nature,
        "resolution_hint": t.resolution_hint,
        "detail": t.detail,
    }
    try:
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except TypeError as exc:
        raise InvalidTensionError(
            f"tension detail is not canonically JSON-serializable: {exc}"
        ) from exc


def decode_tension(payload: bytes) -> Tension:
    """Inverse of :func:`tension_payload`.

    Raises :class:`NotATensionError` on non-JSON payloads, a non-dict
    envelope, missing/unknown ``"v"``, missing required fields, or a body
    that fails :func:`Tension <_tension_errors>` validation.
    """
    try:
        obj = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NotATensionError(f"tension payload is not valid JSON: {exc}") from exc

    if not isinstance(obj, dict):
        raise NotATensionError("tension payload must decode to a JSON object")
    if obj.get("v") != TENSION_VERSION:
        raise NotATensionError(
            f"unsupported tension version {obj.get('v')!r} (expected {TENSION_VERSION})"
        )

    required = ("kind", "parties", "nature", "resolution_hint", "detail")
    missing = [key for key in required if key not in obj]
    if missing:
        raise NotATensionError(f"tension payload missing required fields: {missing}")

    parties = obj["parties"]
    if not isinstance(parties, list) or not all(isinstance(p, str) for p in parties):
        raise NotATensionError("tension 'parties' must be a list of strings")

    t = Tension(
        kind=obj["kind"],
        parties=tuple(parties),
        nature=obj["nature"],
        resolution_hint=obj["resolution_hint"],
        detail=obj["detail"],
    )
    errors = _tension_errors(t)
    if errors:
        raise NotATensionError("invalid tension body: " + "; ".join(errors))
    return t


def tension_ref(t: Tension) -> TensionRef:
    """The content-addressed cell ref this tension WILL have when materialized.

    Pure -- this is the idempotence primitive (same conflict => same bytes
    => same ref). Raises :class:`InvalidTensionError` for a malformed *t*
    (via :func:`tension_payload`).
    """
    return CellWriteEvent(payload=tension_payload(t)).cell_ref()


def resolution_payload(tension: TensionRef, resolution: str, resolved_by: str | None) -> bytes:
    """Canonical resolution bytes.

    ``resolved_by`` is always present (``null`` when not given) so identical
    resolutions are byte-identical.
    """
    body = {
        "v": TENSION_VERSION,
        "kind": "resolution",
        "tension": tension,
        "resolution": resolution,
        "resolved_by": resolved_by,
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class TensionRecord:
    """One row of ``store.query_tensions()``."""

    ref: TensionRef
    kind: str
    parties: tuple[Hash, ...]  # sorted (as stored in the payload)
    resolved: bool  # >= 1 currently-valid @integrity:resolves fact
    resolution_refs: tuple[Hash, ...]  # currently-valid resolutions, log order


class TemporalContradictionCritic:
    """Deterministic, dependency-free reference Critic (pure fold, no network).

    ``predicates`` -- the explicitly-named single-valued ("functional")
    predicates to police -- which predicates are single-valued is domain
    knowledge, so the consumer names them. Raises :class:`ValueError` if
    empty or if any predicate starts with ``"@"`` (the reserved machinery
    namespace is never a domain claim; this also makes tension-about-tension
    regress structurally impossible here) or with the temporal lens's
    invalidation tombstone prefix.
    """

    def __init__(self, predicates: tuple[str, ...]) -> None:
        if not predicates:
            raise ValueError("predicates must be non-empty")
        for p in predicates:
            if p.startswith("@"):
                raise ValueError(
                    f"predicate {p!r} is reserved (starts with '@') and cannot be policed"
                )
            if p.startswith(INVALIDATE_PREFIX):
                raise ValueError(f"predicate {p!r} is reserved (starts with {INVALIDATE_PREFIX!r})")
        self._predicates = predicates

    def detect(self, store: Any) -> list[Tension]:
        tensions: list[Tension] = []
        for predicate in self._predicates:
            facts = store.query_facts(predicate=predicate).output
            by_subject: dict[str, list[str]] = {}
            for f in facts:
                by_subject.setdefault(f.subject, []).append(f.object)
            for subject, objects in by_subject.items():
                distinct = sorted(set(objects))
                if len(distinct) < 2:
                    continue
                tensions.append(
                    Tension(
                        kind="temporal_contradiction",
                        parties=tuple(distinct),
                        nature=(
                            f"functional predicate {predicate!r} has {len(distinct)} "
                            f"contradictory current objects for subject {subject}"
                        ),
                        resolution_hint=(
                            f"invalidate all but one (subject, {predicate!r}, object) "
                            "fact, then record a resolution via resolve_tension"
                        ),
                        detail={"subject": subject, "predicate": predicate},
                    )
                )
        tensions.sort(key=lambda t: (t.detail["predicate"], t.detail["subject"]))
        return tensions


class IntegrityMonitor:
    """Self-driving convergent-integrity loop (the AutonomicIndexer shape).

    Read-only observer counts write pressure; ``run_once()`` does the only
    writes, at top level, idempotently. Always :meth:`detach` (or
    :meth:`stop`) when done.
    """

    def __init__(self, store: Any, critic: Critic, *, threshold: int = DEFAULT_THRESHOLD) -> None:
        if threshold < 1:
            raise ValueError(f"threshold must be >= 1, got {threshold}")
        self._store = store
        self._critic = critic
        self._threshold = threshold

        # Counter state (touched by writer threads via the observer AND by
        # run_once); guarded by _count_lock.
        self._count_lock = threading.Lock()
        self._pending = 0
        self._total_seen = 0

        # Materialization state; _materialize_lock ensures at most one concurrent
        # detect+append run (e.g. background thread vs a manual run_once).
        self._materialize_lock = threading.Lock()
        self._materializations = 0

        # Background runtime.
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_error: Exception | None = None

        # Begin observing immediately (read-only).
        self._subscription: Subscription = store.kernel.subscribe(self._observe)

    # ------------------------------------------------------------------
    # Read-only observer -- counts write pressure, never writes to the log
    # ------------------------------------------------------------------

    def _observe(self, _seq_pos: SeqPos, event: Event) -> None:
        if isinstance(event, CellWriteEvent):
            counted = True
        elif isinstance(event, RelationshipEvent):
            # Its own @integrity: edges don't feed its own pressure; the
            # tension cell's CellWriteEvent self-echoes harmlessly (a
            # subsequent run_once re-detects nothing new -- content-address
            # dedup -- and appends nothing, so the loop is a fixed point).
            counted = not event.type.startswith(_NAMESPACE)
        else:
            counted = False
        if counted:
            with self._count_lock:
                self._pending += 1
                self._total_seen += 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> int:
        return self._threshold

    @property
    def pending(self) -> int:
        """Write events observed since the last materialization."""
        with self._count_lock:
            return self._pending

    @property
    def total_seen(self) -> int:
        """Write events observed over this monitor's lifetime."""
        with self._count_lock:
            return self._total_seen

    @property
    def materializations(self) -> int:
        """How many distinct tensions this monitor has appended."""
        return self._materializations

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_error(self) -> Exception | None:
        """The most recent exception raised inside the background loop, if any."""
        return self._last_error

    # ------------------------------------------------------------------
    # The materialization step (the only place that writes)
    # ------------------------------------------------------------------

    def run_once(self, *, force: bool = False) -> list[TensionRef]:
        """Detect + materialize new tensions if warranted; return their refs.

        Materializes when ``pending >= threshold`` (or ``force=True``). The
        critic's ``detect`` may raise -- it propagates unwrapped (loud).
        Every detected :class:`Tension` is validated (via :func:`tension_ref`)
        BEFORE any append this run; an :class:`InvalidTensionError` leaves
        the log untouched. The known set of already-materialized tensions is
        folded from the LOG (never from memory), so idempotence holds across
        process restarts and fresh monitor instances.
        """
        with self._count_lock:
            if not force and self._pending < self._threshold:
                return []
            self._pending = 0  # consume the accumulated pressure

        # The materialize lock serializes concurrent run_once callers (manual
        # vs background loop) so the in-lock known-set re-fold is race-free.
        with self._materialize_lock:
            detected = self._critic.detect(self._store)

            # Validate + compute refs for EVERY detected tension before any
            # append (InvalidTensionError must leave the log untouched).
            by_ref: dict[TensionRef, Tension] = {}
            for t in detected:
                by_ref[tension_ref(t)] = t

            known: set[TensionRef] = set()
            for _pos, ev in self._store.kernel.all_events():
                if isinstance(ev, RelationshipEvent) and ev.type == TENSION:
                    known.add(ev.from_ref)

            new_items = sorted(item for item in by_ref.items() if item[0] not in known)
            appended: list[TensionRef] = []
            for ref, t in new_items:
                self._materialize_one(ref, t)
                appended.append(ref)
            self._materializations += len(appended)
            return appended

    def _materialize_one(self, ref: TensionRef, t: Tension) -> None:
        """Append ONE atomic batch for a single new tension (the event table)."""
        cell_event = CellWriteEvent(payload=tension_payload(t))
        parties = sorted(t.parties)
        events: list[Event] = [cell_event]
        for p in parties:
            events.append(RelationshipEvent(from_ref=ref, to_ref=p, type=TENSION))
        for p in parties:
            events.append(RelationshipEvent(from_ref=p, to_ref=ref, type=IN_TENSION))
        for i in parties:
            for j in parties:
                if i != j:
                    events.append(RelationshipEvent(from_ref=i, to_ref=j, type=CONFLICTS_WITH))
        self._store.kernel.append_batch(events)

    # ------------------------------------------------------------------
    # Optional autonomic runtime -- a daemon thread that pumps run_once
    # ------------------------------------------------------------------

    def start(self, *, interval: float = 0.5) -> None:
        """Start a background thread that calls :meth:`run_once` every *interval* s.

        Idempotent: a no-op if already running.
        """
        if self.running:
            return
        self._stop_event.clear()
        thread = threading.Thread(
            target=self._loop,
            args=(interval,),
            name="amplifier-data-integrity-monitor",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def _loop(self, interval: float) -> None:
        while not self._stop_event.wait(interval):
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001 - a bad tick must not kill the loop
                self._last_error = exc

    def stop(self) -> None:
        """Stop the background thread (idempotent); blocks until it has joined."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def detach(self) -> None:
        """Stop the runtime and unsubscribe from the kernel (idempotent)."""
        self.stop()
        self._subscription.unsubscribe()
