"""Tests for companion server codec primitives and HTTP server harness."""

from __future__ import annotations

import tempfile
import threading
import urllib.error
import urllib.request

import pytest

from amplifier_data._codec import (
    decode_bytes,
    decode_cell,
    decode_event,
    decode_interpreters,
    decode_query_result,
    decode_vector_hits,
    encode_bytes,
    encode_cell,
    encode_event,
    encode_interpreters,
    encode_query_result,
    encode_vector_hits,
)
from amplifier_data.lenses.grouped import QueryResult as GroupedQueryResult
from amplifier_data.lenses.index import QueryResult as IndexQueryResult
from amplifier_data.lenses.relational import QueryResult as RelationalQueryResult
from amplifier_data.models import (
    AccessEvent,
    Cell,
    CellWriteEvent,
    Event,
    IndexCellEvent,
    InterpreterOutputEvent,
    RelationshipEvent,
    ToolResultEvent,
)

# Representative 64-char hex hashes for tests
_H_A = "a" * 64
_H_B = "b" * 64
_H_C = "c" * 64


def test_codec_bytes_roundtrip() -> None:
    raw = b"\x00\xff\x10raw"
    encoded = encode_bytes(raw)
    assert isinstance(encoded, str), "encode_bytes must return str"
    assert decode_bytes(encoded) == raw, "round-trip must be value-equal"


def test_codec_interpreters_roundtrip() -> None:
    hashes = ("a" * 64, "b" * 64)
    encoded = encode_interpreters(hashes)
    result = decode_interpreters(encoded)
    assert result == hashes, "round-trip must be value-equal"
    assert isinstance(result, tuple), "decode_interpreters must return a tuple"


@pytest.mark.parametrize(
    "ev",
    [
        CellWriteEvent(payload=b"\x01\x02\x03", interpreters=(_H_A, _H_B)),
        RelationshipEvent(from_ref=_H_A, to_ref=_H_B, type="references"),
        ToolResultEvent(output=b"\xde\xad\xbe\xef"),
        InterpreterOutputEvent(
            cell_ref=_H_A,
            model="gpt-4",
            input_hash=_H_B,
            tool_results_consumed=(_H_C,),
            output=b"\x00\x01\x02",
        ),
        AccessEvent(cell_ref=_H_A),
        IndexCellEvent(
            cell_ref=_H_A,
            access_summary=b"summary-bytes",
            output=b"index-data",
        ),
    ],
    ids=[
        "CellWriteEvent",
        "RelationshipEvent",
        "ToolResultEvent",
        "InterpreterOutputEvent",
        "AccessEvent",
        "IndexCellEvent",
    ],
)
def test_codec_event_roundtrip_all_six(ev) -> None:
    encoded = encode_event(ev)
    assert isinstance(encoded, dict), "encode_event must return dict"
    assert "_event" in encoded, "encoded dict must have '_event' tag"
    assert encoded["_event"] == type(ev).__name__, "_event tag must match class name"
    result = decode_event(encoded)
    assert result == ev, "round-trip must be value-equal (frozen dataclass equality)"


def test_codec_cell_roundtrip() -> None:
    c = Cell(
        payload=b"\x01\x02",
        provenance=(
            CellWriteEvent(payload=b"p", interpreters=(_H_A,)),
            RelationshipEvent(from_ref=_H_A, to_ref=_H_B, type="references"),
        ),
        interpreters=(_H_A,),
    )
    encoded = encode_cell(c)
    assert isinstance(encoded, dict), "encode_cell must return dict"
    assert "payload" in encoded
    assert "provenance" in encoded
    assert "interpreters" in encoded
    assert isinstance(encoded["provenance"], list)
    result = decode_cell(encoded)
    assert result == c, "Cell round-trip must be value-equal"


def test_codec_event_decode_unknown_tag_raises() -> None:
    """decode_event must raise ValueError for unrecognised event tags."""
    with pytest.raises(ValueError, match="Unknown event tag"):
        decode_event({"_event": "BogusEventType", "x": 1})


def test_codec_query_result_roundtrip() -> None:
    """QueryResult encode/decode round-trips for relational, grouped, and index types."""
    # Relational: output row contains a bytes payload field
    relational = RelationalQueryResult(
        success=True,
        output=[{"cell_ref": "a" * 64, "payload": b"\x00bytes"}],
        error=None,
    )
    assert decode_query_result(encode_query_result(relational)) == relational

    # Grouped: output is dict[str, list[dict]] with bytes payload entries
    grouped = GroupedQueryResult(
        success=True,
        output={"cell_write": [{"cell_ref": "a" * 64, "payload": b"\x00grp"}]},
        error=None,
    )
    assert decode_query_result(encode_query_result(grouped)) == grouped

    # Index: output is dict[Hash, int] — no bytes, but must round-trip too
    index = IndexQueryResult(
        success=True,
        output={"a" * 64: 3},
        error=None,
    )
    assert decode_query_result(encode_query_result(index)) == index


def test_codec_vector_hits_roundtrip() -> None:
    """encode_vector_hits / decode_vector_hits round-trip and decode to list[tuple]."""
    hits = [("a" * 64, 0.5)]
    encoded = encode_vector_hits(hits)
    decoded = decode_vector_hits(encoded)
    assert decoded == hits
    # Each element must be a tuple, not a list
    assert all(isinstance(item, tuple) for item in decoded)


# ── Server / harness fixture ─────────────────────────────────────────────────


@pytest.fixture
def running_server():
    """Start a ThreadingHTTPServer on port 0; yield (base_url, store, server).

    Skipped when the Rust extension is unavailable (AmplifierStore(path=...) requires it).
    Teardown: server.shutdown() then store.close().
    """
    from amplifier_data import RUST_AVAILABLE

    if not RUST_AVAILABLE:
        pytest.skip("Rust extension required for AmplifierStore(path=...)")

    import os

    from amplifier_data.server import make_server  # ImportError when server.py missing
    from amplifier_data.store import AmplifierStore

    with tempfile.TemporaryDirectory() as tmp_dir:
        store = AmplifierStore(path=os.path.join(tmp_dir, "test.store"))
        server = make_server(store, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        port = server.server_address[1]
        base_url = f"http://127.0.0.1:{port}"

        yield base_url, store, server

        server.shutdown()
        store.close()


def test_server_starts_and_404s_unknown_route(running_server) -> None:
    """Server boots on port 0 and returns HTTP 404 for unknown routes."""
    base_url, _store, _server = running_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(
            urllib.request.Request(
                f"{base_url}/does_not_exist",
                method="POST",
                data=b"",
            )
        )
    assert exc_info.value.code == 404


def test_gate1_roundtrip(running_server) -> None:
    """Killer Gate 1: write_cell → get_cell round-trip through base64+JSON+HTTP.

    Asserts:
    - payload is byte-identical after the full round-trip
    - interpreters are value-identical
    - provenance reconstructs as typed Event instances
    """
    from amplifier_data.client import RemoteStore

    base_url, _store, _server = running_server
    rs = RemoteStore(base_url)

    ref = rs.write_cell(b"\x00\xffpayload", interpreters=("a" * 64,))
    cell = rs.get_cell(ref, record_access=False)

    assert cell.payload == b"\x00\xffpayload", "payload must be byte-identical"
    assert cell.interpreters == ("a" * 64,), "interpreters must be value-identical"
    assert all(isinstance(e, Event) for e in cell.provenance), (
        "provenance must reconstruct as typed Event instances"
    )


def test_relate_and_scope(running_server) -> None:
    """relate, scope, and the add_to_scope alias all succeed over HTTP returning None."""
    from amplifier_data.client import RemoteStore

    base_url, _store, _server = running_server
    rs = RemoteStore(base_url)

    # Write two cells to get valid refs
    ref_a = rs.write_cell(b"cell a")
    ref_b = rs.write_cell(b"cell b")

    # relate returns None without error
    result = rs.relate(ref_a, ref_b, "references")
    assert result is None

    # scope returns None without error
    result = rs.scope(ref_a, ref_b)
    assert result is None

    # add_to_scope is the documented alias for scope
    assert RemoteStore.add_to_scope is RemoteStore.scope
    result = rs.add_to_scope(ref_a, ref_b)
    assert result is None


def test_write_interpretation(running_server) -> None:
    """write_interpretation: write a source cell, derive an interpretation, read it back."""
    from amplifier_data.client import RemoteStore

    base_url, _store, _server = running_server
    rs = RemoteStore(base_url)

    # Write a source cell first
    src = rs.write_cell(b"source payload")

    # Write the interpretation
    derived = rs.write_interpretation("gpt-4o", src, (b"tool bytes",), b"derived output")

    # Retrieve the derived cell and verify its payload
    cell = rs.get_cell(derived, record_access=False)
    assert cell.payload == b"derived output"


def test_add_embedding_and_query_vector(running_server) -> None:
    """add_embedding + query_vector: store an embedding remotely, retrieve as top-1 hit."""
    from amplifier_data.client import RemoteStore

    base_url, _store, _server = running_server
    rs = RemoteStore(base_url)

    # Write a content cell to get a valid ref
    c = rs.write_cell(b"embedding target")

    # Store an embedding for that cell
    rs.add_embedding(c, [1.0, 0.0])

    # Query for the top-1 nearest neighbor
    hits = rs.query_vector([1.0, 0.0], k=1)

    assert isinstance(hits, list), "query_vector must return a list"
    assert len(hits) == 1, "expected exactly one hit"
    assert hits[0][0] == c, "top hit ref must match the embedded cell"
    assert isinstance(hits[0], tuple), "each hit must be a tuple"
    assert isinstance(hits[0][1], float), "hit score must be a float"


def test_regenerate_and_drop(running_server) -> None:
    """drop_materialized evicts cache; regenerate returns a byte-identical Cell after eviction."""
    from amplifier_data.client import RemoteStore

    base_url, _store, _server = running_server
    rs = RemoteStore(base_url)

    # Write a cell with known payload
    c = rs.write_cell(b"payload-x")

    # drop_materialized returns None without error
    result = rs.drop_materialized(c)
    assert result is None

    # regenerate returns the byte-identical Cell with typed-Event provenance
    cell = rs.regenerate(c, record_access=False)
    assert cell.payload == b"payload-x"
    assert all(isinstance(e, Event) for e in cell.provenance), (
        "provenance must reconstruct as typed Event instances"
    )


def test_query_relational_grouped_index(running_server) -> None:
    """query_relational, query_grouped, and query_index all round-trip over HTTP.

    Verifies:
    - query_relational: success=True, list output, byte-identical payloads preserved
    - query_grouped: success=True, dict with "cell_write" key present
    - query_index: success=True, dict output (access count table)
    """
    from amplifier_data.client import RemoteStore

    base_url, _store, _server = running_server
    rs = RemoteStore(base_url)

    # Write two cells to populate the store
    payload_a = b"\x00\xffcell-a"
    payload_b = b"\x01\x02cell-b"
    rs.write_cell(payload_a)
    rs.write_cell(payload_b)

    # query_relational: success=True, list output, bytes preserved through codec
    r = rs.query_relational()
    assert r.success is True
    assert isinstance(r.output, list)
    payloads = {row["payload"] for row in r.output}
    assert payload_a in payloads, "payload_a must be byte-identical after round-trip"
    assert payload_b in payloads, "payload_b must be byte-identical after round-trip"

    # query_grouped: success=True, dict output with "cell_write" key
    g = rs.query_grouped()
    assert g.success is True
    assert isinstance(g.output, dict)
    assert "cell_write" in g.output, '"cell_write" key must be present in grouped output'

    # query_index: success=True, dict output (access count table)
    i = rs.query_index()
    assert i.success is True
    assert isinstance(i.output, dict), "index output must be a dict"

    # Shape assertions from spec: r.output is list, i.output is dict
    assert isinstance(r.output, list)
    assert isinstance(i.output, dict)


def test_query_relational_rejects_filter() -> None:
    """RemoteStore.query_relational raises TypeError when filter= callable is passed.

    Filter callables are not transportable over RemoteStore (they are not JSON-
    serializable). This guard is enforced client-side before any HTTP request is made.
    No running server is required for this test.
    """
    from amplifier_data.client import RemoteStore

    # No actual server needed — TypeError is raised before any HTTP call
    rs = RemoteStore("http://127.0.0.1:9999")

    with pytest.raises(TypeError, match="filter callables are not supported over RemoteStore"):
        rs.query_relational(filter=lambda row: True)


def test_gate4_surface_coverage(running_server) -> None:
    """Killer Gate 4: full surface end-to-end over RemoteStore.

    Exercises in order:
    - relate: record a directed relationship between two cells
    - add_embedding + query_vector top-k: store two orthogonal embeddings; verify
      unscoped top-2 returns both and top hit is the closest
    - scope + scoped query: associate only cell_in with scope_ref; verify scoped
      query_vector(scope=scope_ref) returns only cell_in, excludes cell_out, and
      an unknown scope returns []
    - get_cell(record_access=False): suppressed-access read returns byte-identical payload
    """
    from amplifier_data.client import RemoteStore

    base_url, _store, _server = running_server
    rs = RemoteStore(base_url)

    # --- Write cells for the test ---
    cell_in = rs.write_cell(b"in-scope cell")
    cell_out = rs.write_cell(b"out-of-scope cell")
    scope_ref = rs.write_cell(b"scope definition")

    # --- relate: record a directed relationship ---
    result = rs.relate(cell_in, cell_out, "references")
    assert result is None, "relate must return None"

    # --- add_embedding + query_vector top-k ---
    # Embed both cells with orthogonal 2-D vectors:
    #   cell_in  → [1.0, 0.0]  (points "right")
    #   cell_out → [0.0, 1.0]  (points "up")
    # A query for [1.0, 0.0] must rank cell_in highest.
    rs.add_embedding(cell_in, [1.0, 0.0])
    rs.add_embedding(cell_out, [0.0, 1.0])

    # Unscoped top-2: both embeddings returned, cell_in is the closest match
    hits_all = rs.query_vector([1.0, 0.0], k=2)
    assert isinstance(hits_all, list), "query_vector must return a list"
    assert len(hits_all) == 2, "unscoped top-2 must return both embeddings"
    assert hits_all[0][0] == cell_in, "cell_in must be the top hit for [1.0, 0.0]"
    assert isinstance(hits_all[0], tuple), "each hit must be a (ref, score) tuple"
    assert isinstance(hits_all[0][1], float), "hit score must be a float"

    # --- scope: associate ONLY cell_in with scope_ref ---
    # cell_out is intentionally left unscoped (no rs.scope(cell_out, scope_ref))
    rs.scope(cell_in, scope_ref)

    # --- scoped query: only in-scope hits returned ---
    hits_scoped = rs.query_vector([1.0, 0.0], k=2, scope=scope_ref)
    assert isinstance(hits_scoped, list), "scoped query_vector must return a list"
    assert len(hits_scoped) == 1, (
        "scoped query must return exactly 1 hit (only cell_in is in scope)"
    )
    assert hits_scoped[0][0] == cell_in, "scoped query must return cell_in"

    # out-of-scope: cell_out must NOT appear among the scoped results
    scoped_refs = {h[0] for h in hits_scoped}
    assert cell_out not in scoped_refs, (
        "cell_out must be excluded from scope_ref query (not scoped to scope_ref)"
    )

    # --- unknown scope returns [] ---
    unknown_scope = "0" * 64  # valid hash length but no cells scoped to it
    hits_unknown = rs.query_vector([1.0, 0.0], k=2, scope=unknown_scope)
    assert hits_unknown == [], f"unknown scope must yield empty list, got {hits_unknown!r}"

    # --- get_cell(record_access=False): suppressed-access read ---
    cell = rs.get_cell(cell_in, record_access=False)
    assert cell.payload == b"in-scope cell", (
        "get_cell(record_access=False) payload must be byte-identical"
    )


def test_gate2_concurrent_writers(running_server) -> None:
    """Killer Gate 2: 8 concurrent writer threads × N writes ⇒ log length == 8*N.

    Each thread issues N write_cell requests through RemoteStore.  The server's
    module-global ``_lock`` (in server.py) serialises every ``/write_cell`` body
    before it touches the underlying kernel, so the non-thread-safe
    InMemoryKernel / DurableKernel sees exactly 8*N CellWriteEvents in a
    consistent total order without any corruption.

    Payloads are distinct (``b"t<thread>-w<write>"``), so each call appends
    exactly one new CellWriteEvent to the log.

    Expected event count: 8 threads × 25 writes = 200 CellWriteEvents.
    A missing or incorrect lock would surface as a count mismatch (< 200)
    or as a kernel-level corruption error (exception from the server).
    """
    from amplifier_data.client import RemoteStore

    base_url, store, _server = running_server
    rs = RemoteStore(base_url)

    NUM_THREADS = 8
    N = 25  # writes per thread; total expected = NUM_THREADS * N = 200

    errors: list[Exception] = []

    def writer(thread_idx: int) -> None:
        """Write N cells with distinct payloads, collecting any exceptions."""
        try:
            for write_idx in range(N):
                # Distinct payload per (thread, write) pair ensures each call
                # content-addresses to a unique CellWriteEvent in the log.
                payload = f"t{thread_idx}-w{write_idx}".encode()
                rs.write_cell(payload)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,), daemon=True) for i in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Surface any per-thread exceptions as a clear test failure.
    if errors:
        raise AssertionError(f"{len(errors)} writer thread(s) raised exceptions: {errors!r}")

    # The server-side log must contain exactly NUM_THREADS * N = 200 events.
    # Reading store.kernel.all_events() queries the same kernel object the
    # server used — no HTTP round-trip needed for this assertion.
    expected = NUM_THREADS * N  # 8 * 25 = 200
    actual = len(store.kernel.all_events())
    assert actual == expected, (
        f"Expected {expected} events in the log "
        f"({NUM_THREADS} threads × {N} writes each), got {actual}.  "
        "A missing or incorrect global lock would cause a count mismatch "
        "or a kernel corruption error."
    )


# ── Gate 3 helper + test ───────────────────────────────────────────────────────


def _serve_on_path(path: str):
    """Start a ThreadingHTTPServer on port 0 backed by AmplifierStore(path=path).

    Returns (base_url, store, server).  The caller is responsible for
    ``server.shutdown()`` followed by ``store.close()`` when done.

    Skipping when Rust is unavailable is the caller's responsibility (the
    helper itself always attempts to import AmplifierStore so it will raise
    RuntimeError in that case, which the test turns into a pytest.skip).
    """
    from amplifier_data.server import make_server
    from amplifier_data.store import AmplifierStore

    store = AmplifierStore(path=path)
    server = make_server(store, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    return base_url, store, server


def test_gate3_restart_survival() -> None:
    """Killer Gate 3: cells written via RemoteStore survive server close + reopen on the same path.

    Session 1: open a server on a fixed path (in a temp dir), write two cells
    via RemoteStore, then flush by calling ``server.shutdown()`` followed by
    ``store.close()``.

    Session 2: open a NEW server on the SAME path; via a fresh RemoteStore
    ``regenerate`` each ref and confirm ``cell.payload`` is byte-identical to
    what was written in session 1.

    Durability proof: this test fails when session 2 is pointed at a *different*
    path (cells absent → RuntimeError / assertion error), confirming the gate
    exercises on-disk durability and not residual in-process state.
    """
    import os

    from amplifier_data import RUST_AVAILABLE

    if not RUST_AVAILABLE:
        pytest.skip("Rust extension required for AmplifierStore(path=...)")

    from amplifier_data.client import RemoteStore

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_path = os.path.join(tmp_dir, "gate3.store")

        # ── Session 1: write cells via RemoteStore ────────────────────────────
        base_url1, store1, server1 = _serve_on_path(store_path)
        rs1 = RemoteStore(base_url1)

        payload_a = b"\x00\xffgate3-cell-a"
        payload_b = b"\x01\x02gate3-cell-b"

        ref_a = rs1.write_cell(payload_a)
        ref_b = rs1.write_cell(payload_b)

        # Flush: shut down server *first*, then close store — order mirrors
        # the fixture teardown (server.shutdown() before store.close()).
        server1.shutdown()
        store1.close()

        # ── Session 2: NEW server on the SAME path ────────────────────────────
        base_url2, store2, server2 = _serve_on_path(store_path)
        rs2 = RemoteStore(base_url2)

        try:
            # Regenerate each ref byte-identically from the on-disk log.
            cell_a = rs2.regenerate(ref_a, record_access=False)
            assert cell_a.payload == payload_a, (
                "Gate 3: cell_a payload must be byte-identical after server restart"
            )

            cell_b = rs2.regenerate(ref_b, record_access=False)
            assert cell_b.payload == payload_b, (
                "Gate 3: cell_b payload must be byte-identical after server restart"
            )
        finally:
            server2.shutdown()
            store2.close()


# ── Example smoke test ─────────────────────────────────────────────────────────


def test_example_companion_server() -> None:
    """examples/companion_server.py must run to completion and exit with code 0.

    Skipped when the Rust extension is unavailable (the example returns 2 in
    that case, matching the repo convention for RUST_AVAILABLE=False examples).
    """
    import subprocess
    import sys
    from pathlib import Path

    from amplifier_data import RUST_AVAILABLE

    if not RUST_AVAILABLE:
        pytest.skip("Rust extension required for AmplifierStore(path=...)")

    example = Path(__file__).parent.parent / "examples" / "companion_server.py"
    proc = subprocess.run(
        [sys.executable, str(example)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"example exited with return code {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
