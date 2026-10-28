"""Companion server multi-process story — two-process write + cross-read + restart.

Demonstrates the companion HTTP server as the transport layer for cross-process
:class:`~amplifier_data.client.RemoteStore` access.  Two independent
``RemoteStore`` "client" instances connect to a shared subprocess server,
write distinct cells and a relationship, run a query, and survive a full
server close + reopen cycle with byte-identical payloads.

Run::

    python examples/companion_server.py

What this script proves
-----------------------
Session 1  Spawn a ``python -m amplifier_data.server`` subprocess on a temp
           path with ``--port 0``.  Read the ``{"port": N}`` discovery line
           printed by the server to stdout.  Two independent ``RemoteStore``
           clients (``rs_a`` and ``rs_b``) each write one distinct cell, then
           ``rs_a`` records a directed relationship between them.  A relational
           query confirms both cells are visible.  The server subprocess is then
           shut down cleanly so the on-disk log is fully flushed.

Session 2  Spawn a second ``python -m amplifier_data.server`` subprocess on the
           **same** path (simulating a process restart that inherits the on-disk
           log).  A fresh ``RemoteStore`` calls ``regenerate`` on both refs and
           asserts that each ``Cell.payload`` is byte-identical to what was
           written in Session 1.

The assertions mirror the ``test_gate3_restart_survival`` killer gate in
``tests/test_companion_server.py``.  The example exits with code 0 on success
and code 1 on any assertion failure, so it can be used as a smoke test.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile

from amplifier_data import RUST_AVAILABLE
from amplifier_data.client import RemoteStore


def _step(msg: str) -> None:
    """Print a single progress line to stdout."""
    print(f"  {msg}")


def _start_server(store_path: str) -> tuple[subprocess.Popen, str]:
    """Launch ``python -m amplifier_data.server`` as a subprocess.

    Blocks until the server prints its JSON discovery line to stdout, then
    returns ``(process, base_url)`` so the caller can connect immediately.

    Parameters
    ----------
    store_path:
        Absolute path passed to ``--path``.  The server opens (or creates) an
        :class:`~amplifier_data.store.AmplifierStore` at this location.

    Returns
    -------
    tuple[subprocess.Popen, str]
        The live subprocess handle and the ``http://127.0.0.1:<port>`` base URL.
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "amplifier_data.server", "--path", store_path, "--port", "0"],
        stdout=subprocess.PIPE,
        text=True,
    )
    # The server prints exactly one JSON line before entering serve_forever().
    # readline() blocks until that line arrives (or the process exits, returning "").
    discovery_line = proc.stdout.readline()  # type: ignore[union-attr]
    if not discovery_line:
        proc.kill()
        raise RuntimeError("Server subprocess exited before printing the discovery line.")
    port_info = json.loads(discovery_line)
    port = port_info["port"]
    base_url = f"http://127.0.0.1:{port}"
    return proc, base_url


def _stop_server(proc: subprocess.Popen) -> None:
    """Shut down a server subprocess cleanly.

    Sends SIGINT (→ ``KeyboardInterrupt`` → ``finally: store.close()`` in
    ``server.main()``) so the on-disk log is fully flushed before the process
    exits.  Falls back to ``kill()`` if the process does not exit within 10 s.
    """
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main() -> int:
    """Run the companion-server multi-process demonstration.

    Returns
    -------
    int
        0 on success, 1 on any assertion failure.

    Raises
    ------
    SystemExit(2)
        If the Rust extension is not available (``path=`` is not supported
        without it, so the server subprocess would also fail to start).
    """
    if not RUST_AVAILABLE:
        print(
            "SKIP: companion_server requires the compiled Rust extension.\n"
            "Build it with `maturin develop` and retry.",
            file=sys.stderr,
        )
        return 2

    # A syntactically valid 64-hex-char interpreter reference — mimics a real hash.
    interp_ref = "a" * 64

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "companion.store")

        print(f"\nCompanion server demo  (store: {store_path})\n")

        # ------------------------------------------------------------------ #
        # Session 1 — spawn server, two clients write cells, run query.      #
        # ------------------------------------------------------------------ #
        print("Session 1: start server subprocess, write cells via two RemoteStore clients")

        proc1: subprocess.Popen | None = None
        ref_alpha: str
        ref_beta: str
        try:
            proc1, base_url1 = _start_server(store_path)
            _step(f"server subprocess started  (pid={proc1.pid}, url={base_url1})")

            # Two independent clients — "process A" and "process B".
            rs_a = RemoteStore(base_url1)
            rs_b = RemoteStore(base_url1)

            # Client A writes cell_alpha.
            ref_alpha = rs_a.write_cell(b"alpha payload")
            _step(f"client A wrote cell_alpha  ({ref_alpha[:16]}...)")

            # Client B writes cell_beta with one interpreter reference.
            ref_beta = rs_b.write_cell(b"beta payload", interpreters=(interp_ref,))
            _step(f"client B wrote cell_beta   ({ref_beta[:16]}...)")

            # Client A records a directed relationship between the two cells.
            rs_a.relate(ref_alpha, ref_beta, "references")
            _step("client A wrote relationship  alpha -> beta")

            # Relational query (either client can query) — both cells must appear.
            result = rs_a.query_relational()
            assert result.success is True, "pre-restart relational query must succeed"
            payloads = {row["payload"] for row in result.output}
            assert b"alpha payload" in payloads, "alpha must appear in relational query"
            assert b"beta payload" in payloads, "beta must appear in relational query"
            _step("relational query succeeded — both cells visible  ✓")

        finally:
            if proc1 is not None:
                _stop_server(proc1)
                _step("server subprocess shut down (store flushed to disk)")

        # ------------------------------------------------------------------ #
        # Session 2 — restart server on the same path, regenerate cells.     #
        # ------------------------------------------------------------------ #
        print("\nSession 2: restart server subprocess on same path (simulated restart)")

        proc2: subprocess.Popen | None = None
        try:
            proc2, base_url2 = _start_server(store_path)
            _step(f"server subprocess restarted  (pid={proc2.pid}, url={base_url2})")

            rs2 = RemoteStore(base_url2)

            # E1 core — regenerate must yield byte-identical payloads.
            cell_alpha = rs2.regenerate(ref_alpha, record_access=False)
            assert cell_alpha.payload == b"alpha payload", (
                "restart: cell_alpha payload must be byte-identical"
            )
            assert cell_alpha.interpreters == (), "restart: cell_alpha must have no interpreters"
            _step(
                f"cell_alpha   payload={cell_alpha.payload!r}  "
                f"interpreters={cell_alpha.interpreters!r}  ✓"
            )

            cell_beta = rs2.regenerate(ref_beta, record_access=False)
            assert cell_beta.payload == b"beta payload", (
                "restart: cell_beta payload must be byte-identical"
            )
            assert cell_beta.interpreters == (interp_ref,), (
                "restart: cell_beta interpreter list must be value-equal"
            )
            _step(
                f"cell_beta    payload={cell_beta.payload!r}  "
                f"interpreters=({cell_beta.interpreters[0][:8]}...)  ✓"
            )

        finally:
            if proc2 is not None:
                _stop_server(proc2)
                _step("server subprocess shut down")

    print(
        "\nSUCCESS: companion-server multi-process write + cross-read + restart "
        "demonstrated — all cells survived server close + reopen byte-identically.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
