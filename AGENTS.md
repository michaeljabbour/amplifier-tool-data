# Working on the public storage export

This directory is a generated release snapshot. Make implementation changes in the separately maintained canonical source, then regenerate and verify the export. Never copy its private working notes, stores, credentials or Git history into this directory.

The StorageKernel protocol and AmplifierStore API are stable. Keep append-only event truth and byte-stable Cell regeneration; meaning belongs in lenses/interpreters. No model is owned by this library. Consumers depend on the substrate one way; the substrate does not import consumer implementations.

Run Python tests against an installed native wheel outside this checkout and run cargo test for the native kernel. Source-only fallback tests cannot establish durable restart. Do not publish or change repository visibility without explicit owner approval.
