# Amplifier Data

Keep original evidence once. Build the views each capability needs.

Watch the [63-second film](https://michaeljabbour.github.io/amplifier-tool-data/#film): one true sentence copied into three tools until no original is left, and how Data keeps the exact bytes once while each tool builds its own view. The example in it is fictional.

An independent Python library with a Rust storage kernel. It retains exact bytes as content-addressed Cells in an append-only log, then lets each consumer derive scopes, relations and query views. Memory is one consumer; the storage library does not import or own Memory, your application or an AI model.

[Explore the site](https://michaeljabbour.github.io/amplifier-tool-data/) · [Reviewed release and artifacts](https://github.com/michaeljabbour/amplifier-tool-data/releases/tag/v0.2.0-reviewed) · [Specification](docs/SPEC.md)

## Try it

Python 3.11+, Git, Rust (`cargo` and `rustc`) and a native compiler are needed for source installation. Start in an empty directory:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install 'git+https://github.com/michaeljabbour/amplifier-tool-data@40450647367db2a2b5058bea1d81a6d248c24c94'
amplifier-data-doctor --self-test
```

The doctor creates and removes its own temporary fictional store. For durable operation, check `rust_extension: true`, `durable_available: true` and `self_test.durable_restart: true`. The source fallback is in-memory only. A [prebuilt macOS Apple silicon wheel](https://github.com/michaeljabbour/amplifier-tool-data/releases/download/v0.2.0-reviewed/amplifier_data-0.2.0-cp311-abi3-macosx_11_0_arm64.whl) is available; a source-built native wheel also passes acceptance on Ubuntu 24.04/Python 3.12. Other native platforms remain unverified by this release.

## What you can build

- Evidence tools that retain the exact source behind a summary or decision.
- Independent capabilities with their own scopes and schemas over one storage substrate.
- Replaceable relational, graph, vector or other lenses over a replayable log.

This fictional example uses a temporary store and no model:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from amplifier_data import AmplifierStore

with TemporaryDirectory() as folder:
    store = AmplifierStore(path=str(Path(folder) / "fictional.log"), record_access=False)
    evidence = store.write_cell(b"Fictional parser check remains unverified.")
    action = store.write_cell(b"Fictional next step: run the parser check.")
    store.relate(action, evidence, "derived_from")
    assert store.regenerate(evidence, record_access=False).payload == b"Fictional parser check remains unverified."
    store.close()
```

The library owns no model and has no runtime Python dependencies. Optional intent compilers, critics and embedders belong to callers. One writer owns a durable file; multiple processes use the companion server/RemoteStore seam. Invalidating a fact does not erase bytes from the log. Read the [constraints](docs/CONSTRAINTS.md) before choosing it for real data.

## What is verified

The reviewed public release passed 533 installed-wheel Python checks, 121 Rust unit checks and 3 native durable gates on macOS Apple silicon. Public native CI also passes the installed-wheel suite and Rust kernel suite on Ubuntu 24.04/Python 3.12. Its two-consumer exercise checks scope separation, complete Cell regeneration and durable close/reopen. These are functional checks on temporary fictional data, not evidence of better agent answers, outside-user usefulness or production acceptance.

The [public release guide](docs/PUBLIC-RELEASE.md) identifies the exact reviewed source and artifact hashes. The original private repository, Git history, working notes and operational stores are excluded. Later public documentation and site commits keep the native implementation unchanged.

## Develop

Install `maturin>=1.15,<2` as a build tool, then run `python3 scripts/build_release.py`. Test an installed native wheel outside the checkout with `python3 scripts/test_installed.py`; source-only imports cannot establish durable behavior. See [architecture](docs/ARCHITECTURE.md), [specification](docs/SPEC.md) and the runnable `examples/`.

The repository is `amplifier-tool-data`; package/import identities remain `amplifier-data` / `amplifier_data`.

## License

MIT. Original notices are retained, together with [third-party Rust dependency notices](THIRD_PARTY_NOTICES.md) and the supplied license texts in `amplifier_data/licenses/`.

## Repository migration

This repository was renamed from `amplifier-tool-data` to `amplifier-capability-data`. Existing installed method names, bundle aliases and native interfaces stay the same. See the [migration guide](https://michaeljabbour.github.io/migration/) and [repository map](https://michaeljabbour.github.io/migration/mapping.json) for source URL updates.
