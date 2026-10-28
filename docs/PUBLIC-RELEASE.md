# Reviewed public release

The owner explicitly approved publication of the code-only export on 2026-10-28. The first reviewed public source is commit `40450647367db2a2b5058bea1d81a6d248c24c94`, tagged [v0.2.0-reviewed](https://github.com/michaeljabbour/amplifier-tool-data/releases/tag/v0.2.0-reviewed). It contains 171 allowlisted files and no original private repository history, working notes or operational store. Native implementation files remain unchanged in subsequent public documentation/site commits.

## Install a reviewed artifact

Python 3.11+ is required. For macOS Apple silicon, the reviewed ABI3 wheel supplies the compiled kernel without a local Rust build:

```sh
python3 -m pip install 'https://github.com/michaeljabbour/amplifier-tool-data/releases/download/v0.2.0-reviewed/amplifier_data-0.2.0-cp311-abi3-macosx_11_0_arm64.whl'
amplifier-data-doctor --self-test
```

Use a virtual environment. Source installation needs Rust and a native compiler. [Native CI](https://github.com/michaeljabbour/amplifier-tool-data/actions/runs/40744636502) also passes installed-wheel and Rust acceptance on Ubuntu 24.04/Python 3.12; Windows and other platform/toolchain combinations remain unverified. The reviewed [source distribution](https://github.com/michaeljabbour/amplifier-tool-data/releases/download/v0.2.0-reviewed/amplifier_data-0.2.0.tar.gz) is public too.

| Artifact | SHA256 |
|---|---|
| macOS Apple silicon wheel | `8fd75ec51bc2ce56321ea7b0b4836a1f71d1676772a84a5adb6e0349e842da10` |
| Source distribution | `84740ac43e53d978cb27b94c926c13eebbee01d176d7007f4cff05719b1dce60` |

Both include original MIT and third-party Rust dependency notices. They were built twice with matching bytes on the fixed macOS/Python toolchain; that does not establish reproducibility across platforms or toolchains.

The exact reviewed native wheel was installed from the public release URL without credentials into an empty environment. All 533 Python checks passed. The doctor exercised two fictional scoped consumers, complete Cell regeneration and durable close/reopen. No existing store, model provider or user data was used. These functional checks are separate from real-user benefit and production acceptance.

## Public documentation boundary

`EXPORT-MANIFEST.json` records the allowlisted implementation and retained notices. Its README and .gitignore hashes follow the current public documentation/site files. The reviewed release tag and release-artifact hashes above preserve the original frozen source boundary. Site assets, their MIT theme notice, release docs and CI are later public additions; they contain no private source history.
