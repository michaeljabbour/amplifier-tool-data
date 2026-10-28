"""Remote HTTP client for AmplifierStore companion server.

Provides :class:`RemoteStore`, a thin client that speaks the same codec as the
server (base64 + JSON over HTTP) without any external dependencies.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from typing import Any

from . import _codec
from .lenses.graph import QueryResult as GraphQueryResult
from .lenses.temporal import Fact, TimelineEntry
from .lenses.temporal import QueryResult as TemporalQueryResult
from .models import Cell, Hash


class RemoteStore:
    """Client for the AmplifierStore companion HTTP server.

    Parameters
    ----------
    base_url:
        Base URL of the running companion server, e.g. ``"http://127.0.0.1:8080"``.
        Trailing slashes are stripped on construction.
    """

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Public write methods
    # ------------------------------------------------------------------

    def write_cell(self, payload: bytes, interpreters: tuple[Hash, ...] = ()) -> Hash:
        """Write a cell to the remote store; return its content-addressed ref.

        Parameters
        ----------
        payload:
            Verbatim cell bytes.
        interpreters:
            Tuple of 64-char hex hashes identifying interpreters that can read
            this cell.  Defaults to the empty tuple.

        Returns
        -------
        Hash
            The content-addressed cell reference (sha256 hexdigest).
        """
        body = {
            "payload": _codec.encode_bytes(payload),
            "interpreters": _codec.encode_interpreters(interpreters),
        }
        result = self._post("/write_cell", body)
        return result["ref"]

    def relate(self, from_ref: Hash, to_ref: Hash, rel_type: str) -> None:
        """Record a directed relationship between two cells.

        Parameters
        ----------
        from_ref:
            Source cell hash.
        to_ref:
            Target cell hash.
        rel_type:
            Relationship type string (e.g. ``"references"``).
        """
        self._post("/relate", {"from_ref": from_ref, "to_ref": to_ref, "rel_type": rel_type})

    def scope(self, cell_ref: Hash, scope_ref: Hash) -> None:
        """Associate *cell_ref* with the scope identified by *scope_ref*.

        Parameters
        ----------
        cell_ref:
            Hash of the content cell being scoped.
        scope_ref:
            Hash of the scope cell this content belongs to.
        """
        self._post("/scope", {"cell_ref": cell_ref, "scope_ref": scope_ref})

    def write_interpretation(
        self,
        model: str,
        input_hash: Hash,
        tool_results: tuple[bytes, ...],
        output: bytes,
    ) -> Hash:
        """Record an interpreter run via the remote store; return the derived cell ref.

        Parameters
        ----------
        model:
            Model identifier string (e.g. ``"gpt-4o"``).
        input_hash:
            Content-addressed hash of the input cell.
        tool_results:
            Tuple of raw bytes for each tool result consumed.
        output:
            Raw bytes of the interpretation output.

        Returns
        -------
        Hash
            The content-addressed ref of the derived interpretation cell.
        """
        body = {
            "model": model,
            "input_hash": input_hash,
            "tool_results": [_codec.encode_bytes(tr) for tr in tool_results],
            "output": _codec.encode_bytes(output),
        }
        result = self._post("/write_interpretation", body)
        return result["ref"]

    def add_embedding(self, target_ref: Hash, vector: Sequence[float]) -> Hash:
        """Store a vector embedding linked to *target_ref*; return the embedding cell ref.

        Parameters
        ----------
        target_ref:
            Hash of the cell this embedding represents.
        vector:
            Sequence of float values to embed.

        Returns
        -------
        Hash
            Content-addressed ref of the new embedding cell.
        """
        result = self._post("/add_embedding", {"target_ref": target_ref, "vector": list(vector)})
        return result["ref"]

    def query_vector(
        self,
        vector: Sequence[float],
        k: int,
        scope: Hash | None = None,
    ) -> list[tuple[Hash, float]]:
        """Top-k nearest-neighbor query over stored embeddings.

        Parameters
        ----------
        vector:
            Query vector as a sequence of floats.
        k:
            Maximum number of results to return.
        scope:
            Optional scope hash to restrict results to a single scope.

        Returns
        -------
        list[tuple[Hash, float]]
            ``[(target_ref, score)]`` sorted by descending cosine similarity.
        """
        body: dict = {"vector": list(vector), "k": k, "scope": scope}
        result = self._post("/query_vector", body)
        return _codec.decode_vector_hits(result["hits"])

    #: Compatibility alias for :meth:`scope`; the canonical name is ``scope``.
    add_to_scope = scope

    # ------------------------------------------------------------------
    # Public read methods
    # ------------------------------------------------------------------

    def get_cell(self, ref: Hash, *, record_access: bool | None = None) -> Cell:
        """Retrieve a cell from the remote store by its ref.

        Parameters
        ----------
        ref:
            Content-addressed hash of the cell to retrieve.
        record_access:
            Tri-state access recording flag forwarded to the server:

            * ``None`` (default) — server defers to its own store default.
            * ``True`` — server forces an ``AccessEvent`` to be appended.
            * ``False`` — server suppresses the ``AccessEvent``.

        Returns
        -------
        Cell
            Decoded cell with byte-identical payload, value-identical
            interpreters, and provenance reconstructed as typed Events.
        """
        params: dict[str, str] = {"ref": ref}
        if record_access is not None:
            params["record_access"] = "true" if record_access else "false"
        result = self._get("/get_cell", params)
        return _codec.decode_cell(result["cell"])

    def regenerate(self, ref: Hash, *, record_access: bool | None = None) -> Cell:
        """Force a fresh deterministic fold from the log by dropping cache then re-folding.

        Parameters
        ----------
        ref:
            Content-addressed hash of the cell to regenerate.
        record_access:
            Tri-state access recording flag forwarded to the server:

            * ``None`` (default) — server defers to its own store default.
            * ``True`` — server forces an ``AccessEvent`` to be appended.
            * ``False`` — server suppresses the ``AccessEvent``.

        Returns
        -------
        Cell
            Decoded cell with byte-identical payload, value-identical
            interpreters, and provenance reconstructed as typed Events.
        """
        params: dict[str, str] = {"ref": ref}
        if record_access is not None:
            params["record_access"] = "true" if record_access else "false"
        result = self._get("/regenerate", params)
        return _codec.decode_cell(result["cell"])

    def drop_materialized(self, ref: Hash) -> None:
        """Evict *ref* from the server's materialized cache; idempotent.

        Parameters
        ----------
        ref:
            Content-addressed hash of the cell to evict from cache.
        """
        self._post("/drop_materialized", {"ref": ref})

    def query_relational(self, **kwargs: Any) -> Any:
        """Run a relational query against the remote store.

        Only JSON-serializable kwargs cross the wire: ``project`` and ``scope``.
        The RelationalLens ``filter`` callable is **not transportable** over
        RemoteStore — pass it here and a :class:`TypeError` is raised immediately,
        before any HTTP request is made.  This is an out-of-scope boundary: filter
        predicates are Python callables and cannot be serialised to JSON.

        Parameters
        ----------
        project : list[str] | None
            Column names to include in each output row.  ``None`` → all columns.
        scope : Hash | None
            Optional scope hash to restrict results to a single scope.

        Returns
        -------
        RelationalQueryResult
            ToolResult-shaped: ``success``, ``output`` (list of row dicts),
            ``error``.

        Raises
        ------
        TypeError
            If ``filter`` is passed.  Filter callables are not transportable
            over RemoteStore.
        """
        if "filter" in kwargs:
            raise TypeError("filter callables are not supported over RemoteStore")
        result = self._post("/query_relational", kwargs)
        return _codec.decode_query_result(result["result"])

    def query_grouped(self, **kwargs: Any) -> Any:
        """Run a grouped query against the remote store.

        Only JSON-serializable kwargs cross the wire: ``group`` and ``scope``.

        Parameters
        ----------
        group : str | None
            Group key to restrict output to a single group (e.g. ``"cell_write"``).
            ``None`` → all groups returned.
        scope : Hash | None
            Optional scope hash to restrict results to a single scope.

        Returns
        -------
        GroupedQueryResult
            ToolResult-shaped: ``success``, ``output`` (dict of group lists),
            ``error``.
        """
        result = self._post("/query_grouped", kwargs)
        return _codec.decode_query_result(result["result"])

    def query_index(self, **kwargs: Any) -> Any:
        """Run an index query against the remote store.

        Only JSON-serializable kwargs cross the wire: ``cell_ref`` and ``scope``.

        Parameters
        ----------
        cell_ref : Hash | None
            Optional cell reference to restrict output to a single cell.
            ``None`` → all cells with at least one access are returned.
        scope : Hash | None
            Optional scope hash to restrict results to a single scope.

        Returns
        -------
        IndexQueryResult
            ToolResult-shaped: ``success``, ``output`` (dict mapping Hash to
            access count), ``error``.
        """
        result = self._post("/query_index", kwargs)
        return _codec.decode_query_result(result["result"])

    def query_graph(
        self, start: Hash, max_hops: int, rel_type: str | None = None
    ) -> GraphQueryResult:
        """Traverse RelationshipEvent edges from *start* up to *max_hops* via the server.

        Mirrors :meth:`amplifier_data.store.AmplifierStore.query_graph`. ``rel_type``
        optionally restricts the walk to one edge type. Returns a ToolResult-shaped
        result whose ``output`` is the sorted list of reachable refs.
        """
        body = {"start": start, "max_hops": max_hops, "rel_type": rel_type}
        result = self._post("/query_graph", body)
        return GraphQueryResult(success=True, output=result["output"])

    def graph_neighbors(self, ref: Hash, rel_type: str | None = None) -> list[Hash]:
        """Return the direct out-neighbors of *ref* via the server."""
        result = self._post("/graph_neighbors", {"ref": ref, "rel_type": rel_type})
        return result["neighbors"]

    def assert_fact(self, subject: Hash, predicate: str, object: Hash) -> None:
        """Assert the KG fact (subject, predicate, object) via the server."""
        self._post("/assert_fact", {"subject": subject, "predicate": predicate, "object": object})

    def invalidate_fact(self, subject: Hash, predicate: str, object: Hash) -> None:
        """Invalidate the KG fact (subject, predicate, object) via the server."""
        self._post(
            "/invalidate_fact",
            {"subject": subject, "predicate": predicate, "object": object},
        )

    def query_facts(
        self, subject: Hash | None = None, predicate: str | None = None
    ) -> TemporalQueryResult:
        """Return the currently-valid KG facts via the server (optionally filtered)."""
        result = self._post("/query_facts", {"subject": subject, "predicate": predicate})
        facts = [Fact(f["subject"], f["predicate"], f["object"]) for f in result["facts"]]
        return TemporalQueryResult(success=True, output=facts)

    def timeline(self, subject: Hash) -> list[TimelineEntry]:
        """Return *subject*'s SeqPos-ordered assert/invalidate history via the server."""
        result = self._post("/timeline", {"subject": subject})
        return [
            TimelineEntry(
                seq_pos=e["seq_pos"], op=e["op"], predicate=e["predicate"], object=e["object"]
            )
            for e in result["timeline"]
        ]

    # ------------------------------------------------------------------
    # Private HTTP helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, obj: dict) -> dict:
        """POST *obj* as JSON to *path*; return the decoded JSON response.

        Raises :class:`RuntimeError` on any non-2xx response.
        """
        url = self._base_url + path
        data = json.dumps(obj).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(url, data=data, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())  # type: ignore[no-any-return]
        except urllib.error.HTTPError as exc:
            body = exc.read()
            raise RuntimeError(f"HTTP {exc.code} from POST {path}: {body!r}") from exc

    def _get(self, path: str, params: dict[str, str]) -> dict:
        """GET *path* with query *params*; return the decoded JSON response.

        Raises :class:`RuntimeError` on any non-2xx response.
        """
        qs = urllib.parse.urlencode(params)
        url = self._base_url + path + "?" + qs
        headers = {}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(url, method="GET", headers=headers)
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())  # type: ignore[no-any-return]
        except urllib.error.HTTPError as exc:
            body = exc.read()
            raise RuntimeError(f"HTTP {exc.code} from GET {path}: {body!r}") from exc
