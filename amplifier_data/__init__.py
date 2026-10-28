"""amplifier-data — Plan 3a: Rust floor + Python fallback.

The storage kernel (content-addressing, append-only event log, regenerate fold)
is implemented in Rust via two crates:

  crates/amplifier-data-core/  — pure Rust (no PyO3); safe to cargo test
  crates/amplifier-data-py/   — PyO3 cdylib; compiled by maturin develop

When the compiled extension is present, ``AmplifierStore`` uses ``RustKernel``
by default.  If the extension is absent (source checkout without a maturin
build), it falls back to the pure-Python ``InMemoryKernel`` automatically.
All 173 pytest tests and examples/synthetic_workload.py pass in both modes.

Build the extension:
    maturin develop   (run from repo root with the active venv)
"""

try:
    from . import _rust  # type: ignore  # noqa: F401  (maturin-built extension)

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

from .autonomic import AutonomicIndexer  # noqa: E402
from .envelope import Transaction, WriteBatch  # noqa: E402
from .kernel import DurableKernel  # noqa: E402
from .store import AmplifierStore  # noqa: E402

__all__ = [
    "RUST_AVAILABLE",
    "AmplifierStore",
    "AutonomicIndexer",
    "DurableKernel",
    "Transaction",
    "WriteBatch",
]
