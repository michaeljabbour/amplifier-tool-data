"""Companion HTTP server for AmplifierStore.

SECURITY NOTE — deliberate design constraints (consumer policy, out of scope here):
  * Binds to localhost (127.0.0.1) by default; the ``--host`` flag lets callers
    choose a different address, but no transport-level security is enforced.
  * No authentication / no TLS: callers are assumed to share the same trust
    boundary.  For remote access, wrap in an authenticated reverse proxy
    (nginx, caddy, etc.).
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlsplit

from . import _codec

if TYPE_CHECKING:
    from .store import AmplifierStore

# Module-global lock: every route body acquires it (context manager), so
# concurrent HTTP requests are serialised and reads/writes are safe.
_lock: threading.Lock = threading.Lock()


class _Handler(BaseHTTPRequestHandler):
    """Request handler wired to an AmplifierStore via a class-level attribute.

    Sub-classed per server instance (see :func:`make_server`) so that each
    server carries its own store reference without any global state.
    """

    # Injected by make_server via subclassing.
    store: AmplifierStore  # type: ignore[assignment]
    api_key: str | None = None  # when set, every request needs Bearer <api_key>

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_json(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _authorized(self) -> bool:
        """Bearer-token check (constant-time). Auth is OFF when ``api_key`` is None."""
        if self.api_key is None:
            return True
        header = self.headers.get("Authorization", "")
        return hmac.compare_digest(header, f"Bearer {self.api_key}")

    def _handle_write_cell(self) -> None:
        """POST /write_cell — body: {payload:<b64>, interpreters:[hex...]} → {ref}."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        payload = _codec.decode_bytes(body["payload"])
        interpreters = _codec.decode_interpreters(body.get("interpreters", []))
        ref = self.store.write_cell(payload, interpreters)
        self._send_json(200, {"ref": ref})

    def _handle_relate(self) -> None:
        """POST /relate — body: {from_ref, to_ref, rel_type} → {}."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        self.store.relate(body["from_ref"], body["to_ref"], body["rel_type"])
        self._send_json(200, {})

    def _handle_scope(self) -> None:
        """POST /scope — body: {cell_ref, scope_ref} → {}."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        self.store.scope(body["cell_ref"], body["scope_ref"])
        self._send_json(200, {})

    def _handle_write_interpretation(self) -> None:
        """POST /write_interpretation.

        Body: {model, input_hash, tool_results:[<b64>...], output:<b64>} → {ref}.
        """
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        model = body["model"]
        input_hash = body["input_hash"]
        tool_results = tuple(_codec.decode_bytes(tr) for tr in body["tool_results"])
        output = _codec.decode_bytes(body["output"])
        ref = self.store.write_interpretation(model, input_hash, tool_results, output)
        self._send_json(200, {"ref": ref})

    def _handle_add_embedding(self) -> None:
        """POST /add_embedding — body: {target_ref, vector:[float...]} → {ref}."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        target_ref = body["target_ref"]
        vector = body["vector"]
        ref = self.store.add_embedding(target_ref, vector)
        self._send_json(200, {"ref": ref})

    def _handle_query_vector(self) -> None:
        """POST /query_vector — body: {vector:[float...], k:int, scope:hash|null} → {hits}."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        vector = body["vector"]
        k = body["k"]
        scope = body.get("scope")
        hits = self.store.query_vector(vector, k, scope)
        self._send_json(200, {"hits": _codec.encode_vector_hits(hits)})

    def _handle_get_cell(self, query: str) -> None:
        """GET /get_cell?ref=...&record_access=true|false → {cell: encode_cell(...)}."""
        params = parse_qs(query)
        ref = params["ref"][0]
        record_access_values = params.get("record_access")
        if record_access_values is None:
            record_access: bool | None = None
        else:
            record_access = record_access_values[0].lower() == "true"
        cell = self.store.get_cell(ref, record_access=record_access)
        self._send_json(200, {"cell": _codec.encode_cell(cell)})

    def _handle_regenerate(self, query: str) -> None:
        """GET /regenerate?ref=...&record_access=true|false → {cell: encode_cell(...)}."""
        params = parse_qs(query)
        ref = params["ref"][0]
        record_access_values = params.get("record_access")
        if record_access_values is None:
            record_access: bool | None = None
        else:
            record_access = record_access_values[0].lower() == "true"
        cell = self.store.regenerate(ref, record_access=record_access)
        self._send_json(200, {"cell": _codec.encode_cell(cell)})

    def _handle_drop_materialized(self) -> None:
        """POST /drop_materialized — body: {ref} → {}."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        self.store.drop_materialized(body["ref"])
        self._send_json(200, {})

    def _handle_query_relational(self) -> None:
        """POST /query_relational — body: JSON-serializable kwargs → {result: ...}.

        Accepted body keys: ``project``, ``scope``.  The ``filter`` callable is
        not transportable over HTTP and must never appear in the body.
        """
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        result = self.store.query_relational(**body)
        self._send_json(200, {"result": _codec.encode_query_result(result)})

    def _handle_query_grouped(self) -> None:
        """POST /query_grouped — body: JSON-serializable kwargs → {result: ...}.

        Accepted body keys: ``group``, ``scope``.
        """
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        result = self.store.query_grouped(**body)
        self._send_json(200, {"result": _codec.encode_query_result(result)})

    def _handle_query_index(self) -> None:
        """POST /query_index — body: JSON-serializable kwargs → {result: ...}.

        Accepted body keys: ``cell_ref``, ``scope``.
        """
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        result = self.store.query_index(**body)
        self._send_json(200, {"result": _codec.encode_query_result(result)})

    def _handle_query_graph(self) -> None:
        """POST /query_graph — body: {start, max_hops, rel_type:str|null} → {output:[refs]}."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        result = self.store.query_graph(body["start"], body["max_hops"], body.get("rel_type"))
        self._send_json(200, {"output": result.output})

    def _handle_graph_neighbors(self) -> None:
        """POST /graph_neighbors — body: {ref, rel_type:str|null} → {neighbors:[refs]}."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        neighbors = self.store.graph_neighbors(body["ref"], body.get("rel_type"))
        self._send_json(200, {"neighbors": neighbors})

    def _handle_assert_fact(self) -> None:
        """POST /assert_fact — body: {subject, predicate, object} → {}."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        self.store.assert_fact(body["subject"], body["predicate"], body["object"])
        self._send_json(200, {})

    def _handle_invalidate_fact(self) -> None:
        """POST /invalidate_fact — body: {subject, predicate, object} → {}."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        self.store.invalidate_fact(body["subject"], body["predicate"], body["object"])
        self._send_json(200, {})

    def _handle_query_facts(self) -> None:
        """POST /query_facts — body: {subject?, predicate?} → {facts:[...]}."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        result = self.store.query_facts(body.get("subject"), body.get("predicate"))
        facts = [
            {"subject": f.subject, "predicate": f.predicate, "object": f.object}
            for f in result.output
        ]
        self._send_json(200, {"facts": facts})

    def _handle_timeline(self) -> None:
        """POST /timeline — body: {subject} → {timeline:[{seq_pos,op,predicate,object}]}."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        entries = self.store.timeline(body["subject"])
        timeline = [
            {"seq_pos": e.seq_pos, "op": e.op, "predicate": e.predicate, "object": e.object}
            for e in entries
        ]
        self._send_json(200, {"timeline": timeline})

    def _dispatch(self) -> None:
        """Acquire the global lock and route the request.

        All routes — including error paths — release the lock because the
        ``with`` statement guarantees release on exit, normal or exceptional.
        """
        parsed = urlsplit(self.path)
        path = parsed.path

        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return

        with _lock:
            if self.command == "POST" and path == "/write_cell":
                self._handle_write_cell()
            elif self.command == "GET" and path == "/get_cell":
                self._handle_get_cell(parsed.query)
            elif self.command == "POST" and path == "/relate":
                self._handle_relate()
            elif self.command == "POST" and path == "/scope":
                self._handle_scope()
            elif self.command == "POST" and path == "/write_interpretation":
                self._handle_write_interpretation()
            elif self.command == "POST" and path == "/add_embedding":
                self._handle_add_embedding()
            elif self.command == "POST" and path == "/query_vector":
                self._handle_query_vector()
            elif self.command == "GET" and path == "/regenerate":
                self._handle_regenerate(parsed.query)
            elif self.command == "POST" and path == "/drop_materialized":
                self._handle_drop_materialized()
            elif self.command == "POST" and path == "/query_relational":
                self._handle_query_relational()
            elif self.command == "POST" and path == "/query_grouped":
                self._handle_query_grouped()
            elif self.command == "POST" and path == "/query_index":
                self._handle_query_index()
            elif self.command == "POST" and path == "/query_graph":
                self._handle_query_graph()
            elif self.command == "POST" and path == "/graph_neighbors":
                self._handle_graph_neighbors()
            elif self.command == "POST" and path == "/assert_fact":
                self._handle_assert_fact()
            elif self.command == "POST" and path == "/invalidate_fact":
                self._handle_invalidate_fact()
            elif self.command == "POST" and path == "/query_facts":
                self._handle_query_facts()
            elif self.command == "POST" and path == "/timeline":
                self._handle_timeline()
            else:
                self._send_json(
                    404,
                    {"error": "not found", "path": self.path},
                )

    # ------------------------------------------------------------------
    # HTTP verb dispatch — all delegate to _dispatch
    # ------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch()

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch()

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch()

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch()

    def do_PATCH(self) -> None:  # noqa: N802
        self._dispatch()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the default per-request access log (noise in tests)."""


# ── Public API ───────────────────────────────────────────────────────────────


def make_server(
    store: AmplifierStore, host: str, port: int, api_key: str | None = None
) -> ThreadingHTTPServer:
    """Build a :class:`~http.server.ThreadingHTTPServer` bound to *store*.

    Parameters
    ----------
    store:
        An already-opened :class:`~amplifier_data.store.AmplifierStore`.
        The server does **not** take ownership; callers must call
        ``store.close()`` after ``server.shutdown()``.
    host:
        Bind address.  Recommended value: ``"127.0.0.1"`` (loopback only).
    port:
        Port number.  Pass ``0`` to let the OS choose an ephemeral port;
        inspect ``server.server_address[1]`` after construction to discover
        the chosen port.

    Returns
    -------
    ThreadingHTTPServer
        A server that has bound the socket but has not yet started serving.
        Call ``serve_forever()`` (typically in a daemon thread) to begin
        accepting requests.
    """

    class _BoundHandler(_Handler):
        pass

    _BoundHandler.store = store  # type: ignore[assignment]
    _BoundHandler.api_key = api_key

    return ThreadingHTTPServer((host, port), _BoundHandler)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the companion server.

    After binding the socket, prints a single JSON line to *stdout*::

        {"port": <chosen>}

    so that callers can discover the ephemeral port (useful when ``--port 0``
    is passed).  Then calls ``serve_forever()`` and, on shutdown, closes the
    store.

    Parameters
    ----------
    argv:
        Argument list; defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    int
        Exit code (always ``0`` on clean shutdown).
    """
    parser = argparse.ArgumentParser(
        description="AmplifierStore companion HTTP server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--path", required=True, help="Path to the store directory")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port number (0 = OS-chosen ephemeral port)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("AMPLIFIER_DATA_API_KEY"),
        help="Bearer token required on every request (default: $AMPLIFIER_DATA_API_KEY)",
    )
    args = parser.parse_args(argv)

    from .store import AmplifierStore  # noqa: PLC0415

    store = AmplifierStore(path=args.path)
    server = make_server(store, args.host, args.port, api_key=args.api_key)

    chosen_port = server.server_address[1]
    print(json.dumps({"port": chosen_port}), flush=True)

    try:
        server.serve_forever()
    finally:
        store.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
