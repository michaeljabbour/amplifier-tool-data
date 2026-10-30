# Working on the public storage export

This directory is a generated release snapshot. Make implementation changes in the separately maintained canonical source, then regenerate and verify the export. Never copy its private working notes, stores, credentials or Git history into this directory.

The StorageKernel protocol and AmplifierStore API are stable. Keep append-only event truth and byte-stable Cell regeneration; meaning belongs in lenses/interpreters. No model is owned by this library. Consumers depend on the substrate one way; the substrate does not import consumer implementations.

Run Python tests against an installed native wheel outside this checkout and run cargo test for the native kernel. Source-only fallback tests cannot establish durable restart. Do not publish or change repository visibility without explicit owner approval.

## Market research upkeep

Local-only market research lives in `.research/market/` (gitignored; never commit it).

At session start, check `.research/market/last_updated`:
- If the file is missing, seed it from this repository's README and capability source/provenance links, using cited primary sources and local retrieval dates.
- If it is older than 30 days, or before any positioning, launch or naming work, refresh it with cited primary sources (URL + retrieved date).
- Keep it local. Never commit it and never copy private research into public docs.
- Log every change in `.research/market/CHANGELOG.md`.

## Capability identity

Public capability identity: `amplifier-capability-data`. Preserve the existing portable method folder, native interfaces and legacy installation routes across the approved repository rename.
