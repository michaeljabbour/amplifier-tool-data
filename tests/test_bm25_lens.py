"""Unit tests for BM25Lens / BM25Index / tokenize -- T1.1 (memory bundle BM25 arm).

Tests target the lens and index directly (not via AmplifierStore), using
InMemoryKernel and, where the Rust extension is available, DurableKernel as
the backing store. Covers:

* ``tokenize()``            -- snake_case / camelCase / paths / error codes / digits
* ``BM25Index``             -- add/idempotency/score/candidates/tie-break
* ``BM25Lens.name``
* ``BM25Lens.project()``    -- pure fold, D1 compliance, text-only filtering
* ``BM25Lens.query()``      -- top-k BM25 search, scope filter-before-score
* performance sanity        -- 10k synthetic docs build + query timing (soft)
"""

from __future__ import annotations

import struct
import tempfile
import time

import pytest

from amplifier_data import RUST_AVAILABLE
from amplifier_data.kernel import InMemoryKernel
from amplifier_data.lenses._scope import SCOPED_TO
from amplifier_data.lenses.bm25 import (
    STOPWORDS,
    BM25Index,
    BM25Lens,
    BM25QueryResult,
    tokenize,
)
from amplifier_data.lenses.vector import EMBEDDING_OF
from amplifier_data.models import CellWriteEvent, RelationshipEvent

# ---------------------------------------------------------------------------
# tokenize()
# ---------------------------------------------------------------------------


def test_tokenize_snake_case() -> None:
    """snake_case splits into whole token plus its underscore-delimited parts."""
    assert tokenize("hello_world") == ["hello_world", "hello", "world"]


def test_tokenize_camel_case() -> None:
    """camelCase splits into whole token plus its camel-boundary parts."""
    assert tokenize("processRequest") == ["processrequest", "process", "request"]


def test_tokenize_acronym_camel_case() -> None:
    """Acronym-then-word camelCase (HTTPServer) splits at the acronym boundary."""
    assert tokenize("HTTPServer") == ["httpserver", "http", "server"]


def test_tokenize_path_components() -> None:
    """Path separators (`/`, `.`) are not word characters, so they already split runs."""
    assert tokenize("src/app/main.py") == ["src", "app", "main", "py"]


def test_tokenize_windows_path_components() -> None:
    """Backslash-separated paths split the same way as forward-slash paths."""
    assert tokenize(r"src\app\main.py") == ["src", "app", "main", "py"]


def test_tokenize_error_code_snake_plus_digits() -> None:
    """An error-code identifier yields the whole token plus its word/digit parts."""
    assert tokenize("ERR_4417") == ["err_4417", "err", "4417"]


def test_tokenize_digits_kept_single_char() -> None:
    """A lone digit is kept; a lone letter is dropped."""
    assert tokenize("x=5 and y=10") == ["5", "10"]


def test_tokenize_plain_word_not_double_counted() -> None:
    """A plain word with no real splitting is emitted exactly once."""
    assert tokenize("cat") == ["cat"]


def test_tokenize_stopwords_removed() -> None:
    """The tiny stopword list is removed from output; content words remain."""
    assert tokenize("the cat sat on a mat") == ["cat", "sat", "mat"]
    assert STOPWORDS == frozenset(
        {"the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for", "on"}
    )


def test_tokenize_empty_string_returns_empty_list() -> None:
    assert tokenize("") == []


def test_tokenize_deterministic() -> None:
    text = "processRequest failed with ERR_4417 in src/app/main.py"
    assert tokenize(text) == tokenize(text)


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------


def test_bm25index_len_tracks_documents() -> None:
    index = BM25Index()
    assert len(index) == 0
    index.add("r1", "hello world")
    assert len(index) == 1
    index.add("r2", "goodbye world")
    assert len(index) == 2


def test_bm25index_add_is_idempotent() -> None:
    """Re-adding an existing ref is a no-op -- cells are content-addressed."""
    index = BM25Index()
    index.add("r1", "hello world")
    index.add("r1", "completely different text that would change scoring")
    assert len(index) == 1

    result = index.score("completely different")
    # The second add() must not have taken effect: no match for words only
    # present in the (ignored) second text.
    assert result == []


def test_bm25index_empty_index_returns_empty_score() -> None:
    index = BM25Index()
    assert index.score("anything") == []


def test_bm25index_empty_query_returns_empty_score() -> None:
    index = BM25Index()
    index.add("r1", "hello world")
    assert index.score("") == []
    assert index.score("the a an") == []  # all-stopword query tokenizes to nothing


def test_bm25index_exact_identifier_ranks_first_among_distractors() -> None:
    """A query for a literal identifier ranks the doc containing it #1.

    This is the whole reason BM25Lens exists alongside VectorLens: semantic
    neighbours ("processing the request", "handling a request") must not
    outrank the document that actually contains the literal identifier.
    """
    index = BM25Index()
    index.add("exact", "the processRequest function raised ERR_4417 unexpectedly")
    index.add("distractor1", "handling a request took a long time to process today")
    index.add("distractor2", "processing requests is a routine task for this service")
    index.add("distractor3", "an error occurred somewhere in the pipeline")

    results = index.score("ERR_4417")
    assert results[0][0] == "exact"
    assert results[0][1] > 0.0
    # Distractors never mention ERR_4417 / err_4417 / err / 4417 -- zero score,
    # excluded entirely.
    assert {ref for ref, _ in results} == {"exact"}


def test_bm25index_candidates_restricts_output_but_not_idf() -> None:
    """`candidates` narrows which docs may be returned; IDF/avgdl stay corpus-wide."""
    index = BM25Index()
    index.add("a", "apple banana cherry")
    index.add("b", "apple date")
    index.add("c", "apple fig grape")

    full = dict(index.score("apple"))
    restricted = dict(index.score("apple", candidates=["b", "c"]))

    assert set(restricted) == {"b", "c"}
    assert set(full) == {"a", "b", "c"}
    # Scores for the overlapping docs are identical whether or not `candidates`
    # was supplied -- corpus-wide IDF/avgdl are unaffected by the candidate set.
    assert restricted["b"] == pytest.approx(full["b"])
    assert restricted["c"] == pytest.approx(full["c"])


def test_bm25index_candidates_empty_returns_empty() -> None:
    index = BM25Index()
    index.add("a", "apple banana")
    assert index.score("apple", candidates=[]) == []


def test_bm25index_deterministic_tie_break_by_ref() -> None:
    """Equal-scoring docs are ordered by ascending ref."""
    index = BM25Index()
    # Identical text -> identical score.
    index.add("zzz", "same content here")
    index.add("aaa", "same content here")
    index.add("mmm", "same content here")

    results = index.score("same content")
    refs = [ref for ref, _ in results]
    assert refs == ["aaa", "mmm", "zzz"]


def test_bm25index_k_limits_results() -> None:
    index = BM25Index()
    for i in range(5):
        index.add(f"r{i}", f"shared common token plus unique{i}")

    results = index.score("shared", k=2)
    assert len(results) == 2


def test_bm25index_k_none_returns_all_nonzero() -> None:
    index = BM25Index()
    for i in range(5):
        index.add(f"r{i}", "shared token")

    results = index.score("shared", k=None)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# BM25Lens.name
# ---------------------------------------------------------------------------


def test_bm25_lens_name() -> None:
    assert BM25Lens().name == "bm25"


# ---------------------------------------------------------------------------
# BM25Lens.project -- pure fold, D1 compliance
# ---------------------------------------------------------------------------


def test_project_empty_kernel_returns_empty_index() -> None:
    kernel = InMemoryKernel()
    index = BM25Lens().project(kernel)
    assert isinstance(index, BM25Index)
    assert len(index) == 0


def test_project_indexes_text_cell_write_events() -> None:
    kernel = InMemoryKernel()
    event = CellWriteEvent(payload=b"hello world")
    ref = event.cell_ref()
    kernel.append_batch([event])

    index = BM25Lens().project(kernel)
    assert len(index) == 1
    assert index.score("hello")[0][0] == ref


def test_project_skips_non_utf8_binary_payload() -> None:
    """A packed f32-LE embedding payload (binary, not valid UTF-8) is skipped."""
    kernel = InMemoryKernel()
    # 0xFF 0xFE is not valid UTF-8 continuation/lead byte sequence.
    binary_event = CellWriteEvent(payload=struct.pack("<3f", 1.0, 2.0, 3.0))
    kernel.append_batch([binary_event])

    text_event = CellWriteEvent(payload=b"hello world")
    text_ref = text_event.cell_ref()
    kernel.append_batch([text_event])

    index = BM25Lens().project(kernel)
    assert len(index) == 1
    assert index.score("hello")[0][0] == text_ref


def test_project_is_pure_two_instances_independent() -> None:
    """D1: two BM25Lens instances over the same kernel produce equal results."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"hello world")])

    lens_a = BM25Lens()
    lens_b = BM25Lens()
    assert lens_a.project(kernel).score("hello") == lens_b.project(kernel).score("hello")


def test_project_reflects_new_events_immediately() -> None:
    """D1: project() reads fresh each time -- adding a cell shows up next call."""
    kernel = InMemoryKernel()
    lens = BM25Lens()

    assert len(lens.project(kernel)) == 0

    kernel.append_batch([CellWriteEvent(payload=b"new content")])
    assert len(lens.project(kernel)) == 1


def test_project_duplicate_cell_ref_indexed_once() -> None:
    """Content-addressing means identical payloads share one cell_ref; indexed once."""
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"repeat me")])
    kernel.append_batch([CellWriteEvent(payload=b"repeat me")])

    index = BM25Lens().project(kernel)
    assert len(index) == 1


# ---------------------------------------------------------------------------
# BM25Lens.query
# ---------------------------------------------------------------------------


def test_query_empty_store_returns_empty_result() -> None:
    kernel = InMemoryKernel()
    result = BM25Lens().query(kernel=kernel, text="anything")
    assert isinstance(result, BM25QueryResult)
    assert result.success is True
    assert result.output == []
    assert result.error is None


def test_query_default_k_is_ten() -> None:
    kernel = InMemoryKernel()
    for i in range(15):
        kernel.append_batch([CellWriteEvent(payload=f"shared token doc {i}".encode())])

    result = BM25Lens().query(kernel=kernel, text="shared")
    assert len(result.output) == 10


def test_query_k_overrides_default() -> None:
    kernel = InMemoryKernel()
    for i in range(5):
        kernel.append_batch([CellWriteEvent(payload=f"shared token doc {i}".encode())])

    result = BM25Lens().query(kernel=kernel, text="shared", k=2)
    assert len(result.output) == 2


def test_query_missing_kernel_arg_raises_type_error() -> None:
    with pytest.raises(TypeError, match="kernel"):
        BM25Lens().query(text="hello")


def test_query_missing_text_arg_raises_type_error() -> None:
    kernel = InMemoryKernel()
    with pytest.raises(TypeError, match="text"):
        BM25Lens().query(kernel=kernel)


def test_query_scope_filters_before_score() -> None:
    """F4: scope restricts candidates before scoring; F6: unknown scope -> empty."""
    kernel = InMemoryKernel()

    in_scope_event = CellWriteEvent(payload=b"apple pie recipe")
    in_scope_ref = in_scope_event.cell_ref()
    out_scope_event = CellWriteEvent(payload=b"apple pie history")
    out_scope_ref = out_scope_event.cell_ref()
    scope_event = CellWriteEvent(payload=b"cooking scope")
    scope_ref = scope_event.cell_ref()

    kernel.append_batch([in_scope_event, out_scope_event, scope_event])
    kernel.append_batch(
        [RelationshipEvent(type=SCOPED_TO, from_ref=in_scope_ref, to_ref=scope_ref)]
    )

    result = BM25Lens().query(kernel=kernel, text="apple pie", scope=scope_ref)
    assert result.success is True
    assert [ref for ref, _ in result.output] == [in_scope_ref]
    assert out_scope_ref not in [ref for ref, _ in result.output]


def test_query_unknown_scope_returns_empty_not_error() -> None:
    kernel = InMemoryKernel()
    kernel.append_batch([CellWriteEvent(payload=b"apple pie recipe")])

    result = BM25Lens().query(kernel=kernel, text="apple", scope="deadbeef" * 8)
    assert result.success is True
    assert result.output == []


def test_query_embedding_cells_never_match_text_queries() -> None:
    """Embedding vector cells (linked via EMBEDDING_OF) are binary and skipped."""
    kernel = InMemoryKernel()
    target_event = CellWriteEvent(payload=b"target document about apples")
    target_ref = target_event.cell_ref()
    kernel.append_batch([target_event])

    packed = struct.pack("<3f", 0.1, 0.2, 0.3)
    emb_event = CellWriteEvent(payload=packed)
    emb_ref = emb_event.cell_ref()
    kernel.append_batch([emb_event])
    kernel.append_batch([RelationshipEvent(type=EMBEDDING_OF, from_ref=emb_ref, to_ref=target_ref)])

    result = BM25Lens().query(kernel=kernel, text="apples")
    assert [ref for ref, _ in result.output] == [target_ref]


# ---------------------------------------------------------------------------
# DurableKernel coverage (Rust extension required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUST_AVAILABLE, reason="requires the compiled Rust extension")
def test_bm25_lens_over_durable_kernel_with_scope() -> None:
    from amplifier_data.kernel import DurableKernel

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/log.bin"
        kernel = DurableKernel.open(path)

        doc_event = CellWriteEvent(payload=b"processRequest raised ERR_4417")
        doc_ref = doc_event.cell_ref()
        scope_event = CellWriteEvent(payload=b"scope-cell")
        scope_ref = scope_event.cell_ref()

        kernel.append_batch([doc_event, scope_event])
        kernel.append_batch([RelationshipEvent(type=SCOPED_TO, from_ref=doc_ref, to_ref=scope_ref)])

        result = BM25Lens().query(kernel=kernel, text="ERR_4417", scope=scope_ref)
        assert result.success is True
        assert [ref for ref, _ in result.output] == [doc_ref]

        # Fresh, independent lens over the same durable kernel: identical result (D1).
        result2 = BM25Lens().query(kernel=kernel, text="ERR_4417", scope=scope_ref)
        assert result.output == result2.output


# ---------------------------------------------------------------------------
# Performance sanity (soft) -- 10k synthetic docs
# ---------------------------------------------------------------------------


def test_perf_10k_docs_build_and_query_are_fast() -> None:
    """Soft perf gate: index 10k synthetic docs quickly and query sub-linearly.

    Thresholds are deliberately generous (this is a sanity check against
    accidental O(n^2) regressions, not a strict benchmark) so it stays stable
    across the range of machines this suite runs on.
    """
    index = BM25Index()
    vocab = [
        "apple",
        "banana",
        "cherry",
        "processRequest",
        "ERR_4417",
        "src/app/main.py",
        "database",
        "connection",
        "timeout",
        "retry",
    ]

    start_build = time.perf_counter()
    for i in range(10_000):
        text = f"{vocab[i % len(vocab)]} document number {i} extra filler words here"
        index.add(f"doc-{i:06d}", text)
    build_elapsed = time.perf_counter() - start_build

    assert len(index) == 10_000
    assert build_elapsed < 3.0, f"build took {build_elapsed:.3f}s, expected < 3.0s"

    start_query = time.perf_counter()
    results = index.score("ERR_4417", k=10)
    query_elapsed = time.perf_counter() - start_query

    assert len(results) == 10
    assert query_elapsed < 0.05, f"query took {query_elapsed * 1000:.1f}ms, expected < 50ms"
