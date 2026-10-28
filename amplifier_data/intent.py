"""Intent-compiled queries (Plan 4a) -- mechanism, not policy.

The substrate owns NO planner and NO model. IntentCompiler is an injected
protocol (the Plan 3i Embedder pattern); the emitted plan is stored as an
ordinary content-addressed Cell with @plan: provenance edges (the causal.py
pattern); execution replays the stored plan through the unchanged Plan 3g
executor. Intent->plan may be non-deterministic; plan->result is deterministic
given the log.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .models import Hash, content_address
from .query import IR, QueryResult, Row

PlanRef = Hash  # readability alias; a plan ref IS a cell ref
PLAN_VERSION = 1  # envelope version pinned into plan bytes

# Reserved edge namespace (owned by this module; user code must not mint these).
COMPILED_FROM = "@plan:compiled_from"  # plan cell --> intent cell
COMPILED_BY = "@plan:compiled_by"  # plan cell --> compiler_ref(compiler_id)
RESULT_OF = "@plan:result_of"  # result cell --> plan cell


def compiler_ref(compiler_id: str) -> Hash:
    """Stable synthetic ref for *compiler_id* (mirrors causal.hook_ref)."""
    return content_address(b"@plan:compiler:" + compiler_id.encode("utf-8"))


# The queryable surface handed to compilers AND used by validate_ir -- one
# source of truth. JSON-serializable, static.
CATALOG: dict[str, Any] = {
    "version": 1,
    "ops": ["scan", "filter", "traverse", "project", "order_by", "limit", "aggregate"],
    "scan_lenses": ["relational", "grouped", "facts"],
    "filter_cmps": ["eq", "ne", "contains", "gt", "lt"],
    "relational_fields": [
        "seq_pos",
        "cell_ref",
        "payload",
        "event_type",
        "interpreters",
        "model",
    ],
    "plan_predicates": [COMPILED_FROM, COMPILED_BY, RESULT_OF],
}


@runtime_checkable
class IntentCompiler(Protocol):
    """Turn intent into QueryIR. Bring your own planner; amplifier-data ships none."""

    def compile(self, intent: str, catalog: dict[str, Any]) -> IR: ...


class InvalidPlanError(ValueError):
    """Compiler emitted structurally invalid IR, or a cell is not a decodable plan."""


class IntentNotUnderstood(ValueError):
    """Raised by RuleBasedCompiler when no rule matches -- never a guessed default plan."""


# ---------------------------------------------------------------------------
# Canonical plan/result codec (mirrors models._canonical convention exactly)
# ---------------------------------------------------------------------------


def plan_payload(ir: IR) -> bytes:
    """Canonical plan bytes: ``{"ir": ir, "v": PLAN_VERSION}``, sorted keys, compact."""
    return json.dumps({"ir": ir, "v": PLAN_VERSION}, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def decode_plan(payload: bytes) -> IR:
    """Inverse of :func:`plan_payload`.

    Raises :class:`InvalidPlanError` on non-JSON payloads, a non-dict envelope,
    missing/unknown ``"v"``, or IR that fails :func:`validate_ir`.
    """
    try:
        obj = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidPlanError(f"plan payload is not valid JSON: {exc}") from exc

    if not isinstance(obj, dict):
        raise InvalidPlanError("plan payload must decode to a JSON object")
    if "v" not in obj or "ir" not in obj:
        raise InvalidPlanError("plan payload missing required 'v'/'ir' envelope fields")

    version = obj["v"]
    if version != PLAN_VERSION:
        raise InvalidPlanError(f"unsupported plan version {version!r} (expected {PLAN_VERSION})")

    ir = obj["ir"]
    errors = validate_ir(ir)
    if errors:
        raise InvalidPlanError("invalid plan IR: " + "; ".join(errors))
    return ir


def result_payload(rows: list[Row]) -> bytes:
    """Canonical bytes for a recorded PlanRun result: ``{"rows": rows, "v": PLAN_VERSION}``.

    Non-JSON row values are made JSON-safe via a ``default=`` hook: ``bytes``
    values become ``value.hex()``; tuples already serialize natively as JSON
    arrays. The result cell is a record/cache, not a round-trip contract -- the
    plan + log regenerate the true result (Principle 6: regeneration over editing).
    """

    def _default(value: object) -> object:
        if isinstance(value, bytes):
            return value.hex()
        raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")

    return json.dumps(
        {"rows": rows, "v": PLAN_VERSION},
        sort_keys=True,
        separators=(",", ":"),
        default=_default,
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# validate_ir -- structural validation against the catalog (pure)
# ---------------------------------------------------------------------------

_REQUIRED_OP_KEYS: dict[str, tuple[str, ...]] = {
    "filter": ("field", "cmp", "value"),
    "traverse": ("max_hops",),
    "project": ("fields",),
    "order_by": ("field",),
    "limit": ("n",),
    "aggregate": ("group_by",),
}


def validate_ir(ir: object, catalog: dict[str, Any] | None = None) -> list[str]:
    """Return a list of human-readable structural errors (``[]`` == valid). Pure."""
    cat = CATALOG if catalog is None else catalog
    errors: list[str] = []

    if not isinstance(ir, list) or not ir:
        return ["ir must be a non-empty list of op dicts"]

    ops = cat.get("ops", [])
    scan_lenses = cat.get("scan_lenses", [])
    filter_cmps = cat.get("filter_cmps", [])

    for i, op in enumerate(ir):
        if not isinstance(op, dict):
            errors.append(f"ir[{i}] must be a dict")
            continue
        name = op.get("op")
        if not isinstance(name, str):
            errors.append(f"ir[{i}] missing a string 'op' key")
            continue
        if i == 0 and name != "scan":
            errors.append("ir[0]['op'] must be 'scan'")
        if name not in ops:
            errors.append(f"ir[{i}]: unknown op {name!r}")
            continue

        if name == "scan":
            lens = op.get("lens", "relational")
            if lens not in scan_lenses:
                errors.append(f"ir[{i}]: unknown scan lens {lens!r}")
        else:
            for key in _REQUIRED_OP_KEYS.get(name, ()):
                if key not in op:
                    errors.append(f"ir[{i}]: {name!r} missing required key {key!r}")
            if name == "filter":
                cmp = op.get("cmp")
                if cmp is not None and cmp not in filter_cmps:
                    errors.append(f"ir[{i}]: unknown filter cmp {cmp!r}")

    try:
        plan_payload(ir)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        errors.append(f"ir is not canonically JSON-serializable: {exc}")

    return errors


# ---------------------------------------------------------------------------
# RuleBasedCompiler -- deterministic, dependency-free reference compiler
# ---------------------------------------------------------------------------


def _rule_show_plans(_m: re.Match[str]) -> IR:
    return [{"op": "scan", "lens": "facts", "predicate": COMPILED_FROM}]


def _rule_count_cells_by(m: re.Match[str]) -> IR:
    return [
        {"op": "scan", "lens": "relational"},
        {"op": "aggregate", "group_by": [m.group(1)], "agg": "count"},
    ]


def _rule_cells_containing(m: re.Match[str]) -> IR:
    return [
        {"op": "scan", "lens": "relational"},
        {"op": "filter", "field": "payload", "cmp": "contains", "value": m.group(1)},
        {"op": "project", "fields": ["cell_ref", "payload"]},
    ]


def _rule_facts_with_predicate(m: re.Match[str]) -> IR:
    return [{"op": "scan", "lens": "facts", "predicate": m.group(1)}]


def _rule_first_n_cells(m: re.Match[str]) -> IR:
    return [
        {"op": "scan", "lens": "relational"},
        {"op": "order_by", "field": "seq_pos"},
        {"op": "limit", "n": int(m.group(1))},
    ]


def _rule_neighbors_of_within_hops(m: re.Match[str]) -> IR:
    return [
        {"op": "scan", "lens": "relational"},
        {"op": "filter", "field": "cell_ref", "cmp": "eq", "value": m.group(1)},
        {"op": "traverse", "max_hops": int(m.group(2))},
    ]


# Fixed order; first match wins. Table is the contract; exact regex spellings
# are free (per the design doc's Residual Latitude note).
_RULES: list[tuple[re.Pattern[str], Any]] = [
    (re.compile(r"(?:show|list) plans", re.IGNORECASE), _rule_show_plans),
    (re.compile(r"count cells by (\w+)", re.IGNORECASE), _rule_count_cells_by),
    (
        re.compile(r"""cells containing ["']([^"']+)["']""", re.IGNORECASE),
        _rule_cells_containing,
    ),
    (re.compile(r"facts with predicate (\S+)", re.IGNORECASE), _rule_facts_with_predicate),
    (re.compile(r"first (\d+) cells", re.IGNORECASE), _rule_first_n_cells),
    (
        re.compile(r"neighbors of ([0-9a-f]{64}) within (\d+) hops?", re.IGNORECASE),
        _rule_neighbors_of_within_hops,
    ),
]


class RuleBasedCompiler:
    """Deterministic, dependency-free reference compiler (regex rules, no network)."""

    compiler_id: str = "rule-based-v1"

    def compile(self, intent: str, catalog: dict[str, Any]) -> IR:
        normalized = intent.strip()
        for pattern, builder in _RULES:
            match = pattern.fullmatch(normalized)
            if match is not None:
                return builder(match)
        raise IntentNotUnderstood(intent)


@dataclass(frozen=True)
class PlanRun:
    """Result of execute_plan: the QueryResult plus the optional recorded result ref."""

    result: QueryResult
    result_ref: Hash | None = None
