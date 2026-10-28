# AI Storage

An independently installable append-only storage library for content-addressed, regenerable Cells and swappable views. Native package/import identity stays `amplifier-data` / `amplifier_data`; the proposed repository name is `amplifier-tool-data`.

This is a generated, allowlisted release snapshot. It contains no prior private repository history, operational store or development handoff notes. Publishing this snapshot requires explicit owner approval.

## Install and verify

Python 3.11+, Rust (`cargo` and `rustc`) and a native build toolchain are required when installing from source. The compiled Rust/PyO3 kernel supplies durable paths. A source checkout can use an in-memory Python fallback; it must not claim persistence without the native kernel.

```sh
python3 -m pip install .
amplifier-data-doctor --self-test
```

The library owns no model and has no runtime Python dependencies. Optional intent compilers, critics and embedders are supplied by callers. The doctor opens only an explicitly requested temporary test store, never an existing user store.

## Two consumers, one substrate

```python
from amplifier_data import AmplifierStore

store = AmplifierStore(path="fictional-example.log", record_access=False)
evidence_scope = store.write_cell(b"fictional-evidence-consumer")
artifact_scope = store.write_cell(b"fictional-artifact-consumer")
evidence = store.write_cell(b"Fictional check remains unverified.")
artifact = store.write_cell(b"Fictional next action: run the existing check.")
store.scope(evidence, evidence_scope)
store.scope(artifact, artifact_scope)
store.relate(artifact, evidence, "derived_from")
assert store.regenerate(evidence, record_access=False).payload == b"Fictional check remains unverified."
store.close()
```

A single writer owns a durable file. Multiple processes use the companion server/RemoteStore seam. A fact invalidation does not erase bytes from the append-only log. See [constraints](docs/CONSTRAINTS.md), [specification](docs/SPEC.md) and [architecture](docs/ARCHITECTURE.md).

## Build artifacts

Install `maturin>=1.15,<2` as a build tool, then run `python3 scripts/build_release.py`. It normalizes the source archive, builds a wheel from that archive, remaps private compiler paths, normalizes SBOM path identifiers and regenerates wheel RECORD hashes. `dist/` contains the artifacts. The fixed-platform artifact hash is checked separately before making a byte-reproducibility claim.

## Verification scope

The canonical baseline passed 533 Python tests, 121 Rust unit tests and 3 native durable gates. The release retains 532 functional checks and replaces one private historical cleanup-policy check with one public export-layout gate. Install the wheel and pytest in a clean venv, then run `python3 scripts/test_installed.py` to test that installed artifact; source-only imports are refused. Export acceptance separately verifies the installed wheel, source archive and two-consumer protocol. No external provider, production store or existing Memory data is opened by these tests.

## License and provenance

The original MIT notice is retained. [Third-party notices](THIRD_PARTY_NOTICES.md) inventory the Rust build dependency closure; supplied license texts are packaged under `amplifier_data/licenses/`. A per-file export manifest is included; the private source revision and source-to-export mapping are retained in the local release-control record, not in this snapshot.
