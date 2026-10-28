"""BM25Lens — pure-fold lexical (BM25) search over text cell payloads.

Design Decision D1 compliance
------------------------------
``project()`` folds ``kernel.all_events()`` on every call and persists **no**
mutated state between calls.  Two ``BM25Lens`` instances over the same kernel
are fully independent — mirrors ``VectorLens`` (``lenses/vector.py``).

Why this exists
----------------
Embeddings (``VectorLens``) miss exact-identifier recall: a query for
``ERR_4417`` or ``processRequest`` should rank the document containing that
literal identifier first, which semantic/cosine similarity does not
guarantee.  ``BM25Lens`` is a second, independent retrieval arm over the same
substrate; a consumer fuses both arms (e.g. by reciprocal rank fusion) without
any kernel change — "kernel stays boring, meaning lives in lenses."

``BM25Index`` is also exported standalone so a consumer that already holds
``(ref, text)`` pairs from its own single log-fold pass can build/score a BM25
index directly, without asking this lens to re-fold the log a second time.

Tokenizer
---------
See :func:`tokenize` for the exact, deterministic tokenization contract.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from amplifier_data.lenses._scope import fold_scope

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

#: Tiny English stopword list — intentionally NOT a general-purpose stopword
#: corpus. Kept minimal so identifier tokens (error codes, path components,
#: single-letter-adjacent acronyms) are never accidentally suppressed.
STOPWORDS: frozenset[str] = frozenset(
    {"the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for", "on"}
)

#: Maximal runs of [A-Za-z0-9_] in the ORIGINAL-case text. This is the same
#: character class as the lowercase-only "[a-z0-9_]" in the task description
#: applied case-insensitively — matching on original case first is what lets
#: us camel-split *before* lowercasing.
_RUN_RE = re.compile(r"[A-Za-z0-9_]+")

# Camel-case boundaries:
#   1) lower/digit -> Upper            e.g. "process|Request"
#   2) UPPER run -> Upper+lower        e.g. "HTTP|Server" (acronym boundary)
_CAMEL_BOUNDARY_1 = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_BOUNDARY_2 = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")


def _camel_split(segment: str) -> list[str]:
    """Split *segment* on camelCase boundaries. Pure text manipulation, no case change."""
    spaced = _CAMEL_BOUNDARY_1.sub(" ", segment)
    spaced = _CAMEL_BOUNDARY_2.sub(" ", spaced)
    return spaced.split()


def tokenize(text: str) -> list[str]:
    """Tokenize *text* into a deterministic list of lowercase tokens.

    Contract
    --------
    1. Find maximal runs of ``[A-Za-z0-9_]`` in the ORIGINAL-case text (path
       separators, punctuation and whitespace — e.g. ``. / \\ - :`` — are not
       in this class, so they already split runs; e.g. ``"src/app/main.py"``
       yields runs ``"src"``, ``"app"``, ``"main"``, ``"py"``. No separate
       pass is needed for path components: the run regex already produces
       them.
    2. Each run contributes its own lowercase form as a whole token (exact
       identifier match, e.g. ``"ERR_4417"`` -> ``"err_4417"``,
       ``"processRequest"`` -> ``"processrequest"``).
    3. Each run is ALSO split into sub-tokens: first on ``_`` (snake_case),
       then each piece is camel-split (boundaries computed on the
       ORIGINAL-case text, before lowercasing) and lowercased. Sub-tokens are
       only emitted in addition to the whole token when they actually differ
       from it (i.e. real splitting occurred) — otherwise a plain word like
       ``"cat"`` would double-count itself as both whole and sub-token,
       inflating its term frequency for no reason.
    4. Tokens of length 1 are dropped UNLESS they are all-digit (so ``"x"`` is
       dropped but ``"5"`` is kept).
    5. A tiny stopword list (:data:`STOPWORDS`) is removed after the length
       filter.

    Deterministic: same input text always yields the same output list, in
    the same order (whole-token first, then its sub-tokens, per run, in the
    order runs appear in the text).

    Examples
    --------
    >>> tokenize("ERR_4417")
    ['err_4417', 'err', '4417']
    >>> tokenize("processRequest")
    ['processrequest', 'process', 'request']
    >>> tokenize("src/app/main.py")
    ['src', 'app', 'main', 'py']
    >>> tokenize("the cat sat")
    ['cat', 'sat']
    """
    tokens: list[str] = []
    for run in _RUN_RE.findall(text):
        whole = run.lower()

        segments = [s for s in run.split("_") if s]
        sub_pieces: list[str] = []
        for seg in segments:
            sub_pieces.extend(_camel_split(seg))
        sub_lower = [p.lower() for p in sub_pieces]

        run_tokens = [whole]
        if len(sub_lower) > 1 or (len(sub_lower) == 1 and sub_lower[0] != whole):
            run_tokens.extend(sub_lower)

        for tok in run_tokens:
            if len(tok) == 1 and not tok.isdigit():
                continue
            if tok in STOPWORDS:
                continue
            tokens.append(tok)

    return tokens


# ---------------------------------------------------------------------------
# BM25Index — standalone, reusable by any (ref, text) source
# ---------------------------------------------------------------------------

_K1: float = 1.2
_B: float = 0.75


class BM25Index:
    """Incremental BM25 index over ``(ref, text)`` pairs.

    Not a lens itself — a plain, reusable data structure so a consumer that
    already holds ``(ref, text)`` pairs from its own single log-fold pass
    (e.g. ``amplifier-bundle-memory``'s ``_SearchFold``) can build and score a
    BM25 index directly, without a second ``all_events()`` pass.

    Postings are ``dict[token, dict[ref, tf]]`` plus per-doc token counts, so
    :meth:`score` only iterates postings for the query's own tokens —
    sub-linear in corpus size, not a full corpus scan.
    """

    def __init__(self) -> None:
        self._postings: dict[str, dict[str, int]] = {}
        self._doc_len: dict[str, int] = {}

    def add(self, ref: str, text: str) -> None:
        """Add one document. Idempotent: re-adding an existing *ref* is a no-op.

        Cells are content-addressed — a given ``ref`` can only ever map to one
        piece of text, so silently skipping a repeat ``add(ref, ...)`` call
        cannot lose information; it just avoids double-counting term
        frequencies for a document already indexed.
        """
        if ref in self._doc_len:
            return

        tokens = tokenize(text)
        self._doc_len[ref] = len(tokens)

        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        for tok, count in tf.items():
            self._postings.setdefault(tok, {})[ref] = count

    def __len__(self) -> int:
        """Number of indexed documents (N in the BM25 formula)."""
        return len(self._doc_len)

    def _avgdl(self) -> float:
        if not self._doc_len:
            return 0.0
        return sum(self._doc_len.values()) / len(self._doc_len)

    def score(
        self,
        query: str,
        *,
        candidates: Iterable[str] | None = None,
        k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Score all indexed documents against *query*, BM25-ranked.

        Parameters
        ----------
        query:
            Free-text query; tokenized with the same :func:`tokenize` used at
            index time.
        candidates:
            When given, restrict scoring to this subset of refs (scope
            filter-before-score: F4). IDF and average document length are
            computed over the WHOLE corpus regardless of ``candidates`` —
            only which documents may appear in the output is restricted.
        k:
            Max results. ``None`` -> all non-zero-scoring documents.

        Returns
        -------
        list[tuple[str, float]]
            ``(ref, score)`` sorted by descending score, ties broken by
            ascending ``ref`` (deterministic). Zero-score documents are
            excluded.
        """
        n = len(self._doc_len)
        if n == 0:
            return []

        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []

        candidate_set = set(candidates) if candidates is not None else None
        avgdl = self._avgdl()

        scores: dict[str, float] = {}
        for tok in query_tokens:
            postings = self._postings.get(tok)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for ref, tf in postings.items():
                if candidate_set is not None and ref not in candidate_set:
                    continue
                dl = self._doc_len.get(ref, 0)
                norm_len = dl / avgdl if avgdl else 0.0
                denom = tf + _K1 * (1.0 - _B + _B * norm_len)
                scores[ref] = scores.get(ref, 0.0) + idf * (tf * (_K1 + 1.0)) / denom

        results = [(ref, s) for ref, s in scores.items() if s > 0.0]
        results.sort(key=lambda pair: (-pair[1], pair[0]))
        return results if k is None else results[: max(k, 0)]


# ---------------------------------------------------------------------------
# Lens-local dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BM25QueryResult:
    """ToolResult-shaped result from :meth:`BM25Lens.query` (reuse, not reinvent)."""

    success: bool
    output: list[tuple[str, float]]
    error: str | None = None


@dataclass(frozen=True)
class BM25View:
    """Immutable wrapper marking a :class:`BM25Index` as a lens :data:`View`.

    ``BM25Index`` itself is mutable (incremental postings) by design — it is
    a plain reusable data structure, not a frozen dataclass like
    ``VectorIndex``. ``project()`` still honors D1 (pure fold, fresh every
    call): each call constructs and returns a brand-new ``BM25Index``, never
    a cached one. ``index`` here is exactly that fresh instance.
    """

    index: BM25Index = field(compare=False)


# ---------------------------------------------------------------------------
# BM25Lens
# ---------------------------------------------------------------------------


class BM25Lens:
    """Static pure fold of text cell payloads into a :class:`BM25Index`.

    For each ``CellWriteEvent`` in the log whose payload decodes as UTF-8
    text, the decoded text is indexed under its ``cell_ref()``. Payloads that
    do not decode as UTF-8 (e.g. packed f32-LE embedding vectors) are silently
    skipped — there is no generic cross-lens marker distinguishing "content"
    cells from "embedding" or "scope" cells (both are ordinary
    ``CellWriteEvent`` payloads; only their *edges* carry that meaning), so
    the decode step is the filter, and ``scope=`` on :meth:`query` narrows
    further to a caller-chosen membership set.

    Design Decision D1
    ------------------
    * ``project()`` is a pure fold: reads ``kernel.all_events()`` fresh each
      call and returns a freshly built :class:`BM25Index`. No mutable state
      is held between calls.
    """

    @property
    def name(self) -> str:
        return "bm25"

    def project(self, kernel: Any) -> BM25Index:
        """Fold ``kernel.all_events()`` into a fresh :class:`BM25Index`.

        Pure: appends no events and caches no state between calls.

        * Every ``CellWriteEvent`` whose payload decodes as UTF-8 text
          contributes ``cell_ref() -> text``.
        * A cell whose payload does not decode as UTF-8 is silently skipped
          (binary content — embeddings, packed vectors, etc.).
        * Documents are added to the index in ``sorted(cell_ref)`` order for
          a deterministic build sequence.
        """
        from amplifier_data.models import CellWriteEvent

        docs: dict[str, str] = {}
        for _seq_pos, event in kernel.all_events():
            if not isinstance(event, CellWriteEvent):
                continue
            ref = event.cell_ref()
            if ref in docs:
                continue
            try:
                text = event.payload.decode("utf-8")
            except UnicodeDecodeError:
                continue
            docs[ref] = text

        index = BM25Index()
        for ref in sorted(docs):
            index.add(ref, docs[ref])
        return index

    def query(self, **typed_args: Any) -> BM25QueryResult:
        """Top-k BM25 lexical search over the folded text cells.

        Keyword arguments
        -----------------
        kernel : Any (required) — the storage kernel; ``project()`` runs fresh
                 each call.
        text   : str (required) — the query text.
        k      : int              — max results; defaults to ``10``.
        scope  : Hash | None      — when given, only cells with a direct
                 ``SCOPED_TO`` edge to this scope ref are considered.
                 Filtering happens **before** BM25 scoring and top-k selection
                 (F4: filter-before-score) — IDF and average document length
                 stay corpus-wide, only which docs may be returned narrows.
                 An unknown scope (no member cells) yields an empty output
                 with ``success=True`` (F6).

        Returns a ToolResult-shaped ``BM25QueryResult`` whose ``output`` is
        ``[(cell_ref, score)]`` sorted by descending BM25 score, ties broken
        by ``cell_ref``.

        Raises
        ------
        TypeError — if ``kernel`` or ``text`` is absent.
        """
        if "kernel" not in typed_args:
            raise TypeError("BM25Lens.query() requires a 'kernel' keyword argument")
        if "text" not in typed_args:
            raise TypeError("BM25Lens.query() requires a 'text' keyword argument")

        kernel = typed_args["kernel"]
        query_text: str = typed_args["text"]
        k: int = typed_args.get("k", 10)
        scope: str | None = typed_args.get("scope")

        index = self.project(kernel)

        candidates: frozenset[str] | None = None
        if scope is not None:
            scope_index = fold_scope(kernel)
            candidates = scope_index.cells_in_scope(scope)

        output = index.score(query_text, candidates=candidates, k=k)
        return BM25QueryResult(success=True, output=output)
