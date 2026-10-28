"""Bearer-token auth on the companion server (Plan 3k)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from amplifier_data.client import RemoteStore
from amplifier_data.server import make_server
from amplifier_data.store import AmplifierStore


def _serve(store: AmplifierStore, api_key: str | None = None) -> tuple[object, str]:
    server = make_server(store, "127.0.0.1", 0, api_key=api_key)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{port}"


def test_auth_required_when_key_set(tmp_path: Path) -> None:
    store = AmplifierStore(path=str(tmp_path / "a.log"))
    server, url = _serve(store, api_key="secret")
    try:
        # No token -> 401 (POST path)
        with pytest.raises(RuntimeError, match="401"):
            RemoteStore(url).write_cell(b"x")
        # Wrong token -> 401
        with pytest.raises(RuntimeError, match="401"):
            RemoteStore(url, api_key="wrong").write_cell(b"x")
        # Correct token -> works on both POST and GET paths
        rc = RemoteStore(url, api_key="secret")
        ref = rc.write_cell(b"hello")
        assert rc.get_cell(ref).payload == b"hello"
    finally:
        server.shutdown()  # type: ignore[attr-defined]
        store.close()


def test_auth_off_by_default(tmp_path: Path) -> None:
    store = AmplifierStore(path=str(tmp_path / "b.log"))
    server, url = _serve(store)  # no api_key -> auth disabled (localhost dev)
    try:
        rc = RemoteStore(url)
        ref = rc.write_cell(b"hi")
        assert rc.get_cell(ref).payload == b"hi"
    finally:
        server.shutdown()  # type: ignore[attr-defined]
        store.close()
