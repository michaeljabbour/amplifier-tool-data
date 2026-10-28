//! Core data types and content-addressing primitives.
//!
//! # Canonical JSON contract
//!
//! [`content_address`] hashes arbitrary bytes with SHA-256 and returns a
//! 64-character lowercase hex string.  The private `canonical` helper
//! serialises a key-value map as *canonical JSON*, whose output must byte-match
//! Python's
//! ```python
//! json.dumps(obj, sort_keys=True, separators=(",", ":"))
//! ```
//! with `bytes.hex()` (lowercase) for byte fields.  This contract is load-
//! bearing: `store.write_cell` in Python computes `cell_ref` from the same
//! canonical form; the kernel resolves cells by matching against refs it
//! computes itself.  Any divergence causes silent `NotFound`.

use sha2::{Digest, Sha256};

// ─── Public type aliases ─────────────────────────────────────────────────────

/// A 64-character lowercase hex-encoded SHA-256 hash.
pub type Hash = String;

/// Monotone sequence position used for total ordering of events.
pub type SeqPos = u64;

// ─── Public data types ───────────────────────────────────────────────────────

/// Every event in the append-only log.
///
/// Mirrors Python's `Event` hierarchy in `amplifier_data.models`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Event {
    /// Defines a cell: verbatim payload plus the interpreter refs that can read it.
    ///
    /// Mirrors `CellWriteEvent`.
    CellWrite {
        payload: Vec<u8>,
        interpreters: Vec<Hash>,
    },
    /// A directed edge.  Lands in `from_ref`'s provenance only (not `to_ref`).
    ///
    /// Mirrors `RelationshipEvent`.
    Relationship {
        from_ref: Hash,
        to_ref: Hash,
        type_: String,
    },
    /// A verbatim external tool result, referenced by hash from an interpreter output.
    ///
    /// Mirrors `ToolResultEvent`.
    ToolResult { output: Vec<u8> },
    /// Defines a derived cell from a recorded interpreter run.
    ///
    /// Mirrors `InterpreterOutputEvent`.
    InterpreterOutput {
        /// Identity of the derived cell this event defines.
        cell_ref: Hash,
        model: String,
        input_hash: Hash,
        tool_results_consumed: Vec<Hash>,
        output: Vec<u8>,
    },
    /// Free-floating log entry recording that a cell was read (provenance-neutral).
    ///
    /// Mirrors `AccessEvent`.
    Access { cell_ref: Hash },
    /// Defines an index cell from a static fold over access events.
    ///
    /// Mirrors `IndexCellEvent`.
    IndexCell {
        /// Identity of the index cell this event defines.
        cell_ref: Hash,
        /// Serialised access summary ('why I indexed this').
        access_summary: Vec<u8>,
        /// The serialised index bytes produced by the static fold.
        output: Vec<u8>,
    },
}

/// A regenerable materialized view: verbatim payload + ordered provenance.
///
/// Mirrors Python's `Cell` dataclass.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Cell {
    pub payload: Vec<u8>,
    pub provenance: Vec<Event>,
    pub interpreters: Vec<Hash>,
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Compute `sha256(payload)` and return the result as 64 lowercase hex chars.
///
/// Matches Python's `hashlib.sha256(payload).hexdigest()`.
pub fn content_address(payload: &[u8]) -> Hash {
    let digest = Sha256::digest(payload);
    // Emit each byte as exactly two lowercase hex digits.
    digest.iter().fold(String::with_capacity(64), |mut s, b| {
        s.push(nibble_to_hex(b >> 4));
        s.push(nibble_to_hex(b & 0x0f));
        s
    })
}

/// Stable identity of a `CellWrite` event — `kind:"cell_write"` canonical map.
///
/// Mirrors Python's `CellWriteEvent.cell_ref()`.
///
/// The canonical map is:
/// ```json
/// {"interpreters":[…],"kind":"cell_write","payload":"<hex>"}
/// ```
pub fn cell_write_ref(payload: &[u8], interpreters: &[Hash]) -> Hash {
    let interps: Vec<CanonicalVal> = interpreters
        .iter()
        .map(|h| CanonicalVal::Str(h.clone()))
        .collect();
    let json = canonical(&[
        ("interpreters", CanonicalVal::Array(interps)),
        ("kind", CanonicalVal::Str("cell_write".into())),
        ("payload", CanonicalVal::Bytes(payload.to_vec())),
    ]);
    content_address(json.as_bytes())
}

/// Identity of a derived cell from a recorded interpreter run.
///
/// Mirrors Python's `InterpreterOutputEvent.derive_ref()`.
///
/// **Circularity guard**: `cell_ref` is excluded from the canonical map —
/// the identity is derived from the *defining content only*.  This matches
/// the Python implementation which also excludes `cell_ref` from `derive_ref`.
///
/// The canonical map is:
/// ```json
/// {"input_hash":"…","kind":"interpreter_output","model":"…","output":"<hex>","tool_results_consumed":[…]}
/// ```
pub fn interpreter_derive_ref(
    model: &str,
    input_hash: &Hash,
    tool_results_consumed: &[Hash],
    output: &[u8],
) -> Hash {
    let consumed: Vec<CanonicalVal> = tool_results_consumed
        .iter()
        .map(|h| CanonicalVal::Str(h.clone()))
        .collect();
    let json = canonical(&[
        ("input_hash", CanonicalVal::Str(input_hash.clone())),
        ("kind", CanonicalVal::Str("interpreter_output".into())),
        ("model", CanonicalVal::Str(model.into())),
        ("output", CanonicalVal::Bytes(output.to_vec())),
        ("tool_results_consumed", CanonicalVal::Array(consumed)),
    ]);
    content_address(json.as_bytes())
}

/// Identity of an index cell derived from a static fold over access events.
///
/// Mirrors Python's `IndexCellEvent.derive_ref()`.
///
/// **Circularity guard**: `cell_ref` is excluded from the canonical map,
/// matching the Python implementation.
///
/// The canonical map is:
/// ```json
/// {"access_summary":"<hex>","kind":"index_cell","output":"<hex>"}
/// ```
pub fn index_derive_ref(access_summary: &[u8], output: &[u8]) -> Hash {
    let json = canonical(&[
        (
            "access_summary",
            CanonicalVal::Bytes(access_summary.to_vec()),
        ),
        ("kind", CanonicalVal::Str("index_cell".into())),
        ("output", CanonicalVal::Bytes(output.to_vec())),
    ]);
    content_address(json.as_bytes())
}

/// Content address of a `ToolResult` — `sha256(output)`.
///
/// Mirrors Python's `ToolResultEvent.hash` property.
pub fn tool_result_hash(output: &[u8]) -> Hash {
    content_address(output)
}

// ─── Convert a single hex nibble (0–15) to lowercase ASCII ───────────────────

/// Convert a single hex nibble (0–15) to its lowercase ASCII hex character.
#[inline]
fn nibble_to_hex(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        _ => (b'a' + (n - 10)) as char,
    }
}

// ─── Private canonical JSON ───────────────────────────────────────────────────

/// A typed value used when constructing canonical JSON.
///
/// Mirrors the Python `_canonical` helper:
///
/// | Variant   | JSON encoding                            |
/// |-----------|------------------------------------------|
/// | `Bytes`   | `"<lowercase hex>"`                      |
/// | `Array`   | `[…]` (elements rendered in order)       |
/// | `Str`     | `"…"` (minimal JSON string escaping)    |
#[derive(Debug, PartialEq, Clone)]
enum CanonicalVal {
    /// Raw bytes — rendered as a quoted lowercase hex string like `bytes.hex()`.
    Bytes(Vec<u8>),
    /// A JSON array; elements are rendered in the order given (no re-sorting).
    Array(Vec<CanonicalVal>),
    /// A UTF-8 string rendered as a quoted JSON string.
    Str(String),
}

/// Serialise `fields` as canonical JSON.
///
/// * Keys are sorted alphabetically (byte order, matching Python's `sort_keys`).
/// * Separators are `,` (items) and `:` (key-value) — no spaces anywhere.
/// * `CanonicalVal::Bytes` → lowercase hex string (like `bytes.hex()`).
///
/// The output byte-matches Python:
/// ```python
/// json.dumps({k: v}, sort_keys=True, separators=(",", ":"))
/// ```
/// Critical: store.write_cell computes cell_ref with this same form; the
/// kernel must produce identical bytes or resolve returns NotFound.
fn canonical(fields: &[(&str, CanonicalVal)]) -> String {
    // Sort by key (byte-wise, same as Python's default string sort for ASCII keys).
    let mut indices: Vec<usize> = (0..fields.len()).collect();
    indices.sort_unstable_by_key(|&i| fields[i].0);

    let mut out = String::new();
    out.push('{');
    for (pos, &i) in indices.iter().enumerate() {
        if pos > 0 {
            out.push(',');
        }
        let (key, val) = &fields[i];
        write_json_string(key, &mut out);
        out.push(':');
        write_val(val, &mut out);
    }
    out.push('}');
    out
}

/// Append a JSON-quoted string to `out` with minimal escaping.
///
/// Only the characters required by the JSON spec are escaped; everything else
/// is emitted as-is (valid for UTF-8 content that doesn't contain control chars).
fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Recursively render a [`CanonicalVal`] into `out`.
fn write_val(val: &CanonicalVal, out: &mut String) {
    match val {
        CanonicalVal::Bytes(b) => {
            out.push('"');
            for byte in b {
                out.push(nibble_to_hex(byte >> 4));
                out.push(nibble_to_hex(byte & 0x0f));
            }
            out.push('"');
        }
        CanonicalVal::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_val(item, out);
            }
            out.push(']');
        }
        CanonicalVal::Str(s) => {
            write_json_string(s, out);
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── content_address ──────────────────────────────────────────────────────

    /// Golden fixture: well-known SHA-256 of ASCII "abc".
    ///
    /// Verified with Python:
    /// ```python
    /// import hashlib
    /// hashlib.sha256(b"abc").hexdigest()
    /// # 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    /// ```
    #[test]
    fn content_address_golden_abc() {
        assert_eq!(
            content_address(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        );
    }

    /// Same input always yields the same hash.
    #[test]
    fn content_address_determinism_same_bytes_give_same_hash() {
        let a = content_address(b"reproducible-payload");
        let b = content_address(b"reproducible-payload");
        assert_eq!(a, b, "same bytes must yield identical hash");
    }

    /// Different inputs must yield different hashes (collision sanity-check).
    #[test]
    fn content_address_determinism_different_bytes_give_different_hash() {
        assert_ne!(
            content_address(b"hello"),
            content_address(b"world"),
            "distinct payloads must have distinct hashes",
        );
    }

    /// Output is 64 lowercase hex characters.
    #[test]
    fn content_address_output_format() {
        let h = content_address(b"format-check");
        assert_eq!(h.len(), 64, "SHA-256 hex is exactly 64 chars");
        assert!(
            h.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')),
            "hash must be lowercase hex, got: {h:?}",
        );
    }

    // ── canonical ────────────────────────────────────────────────────────────

    /// Golden fixture: byte-matches Python
    /// ```python
    /// import json
    /// json.dumps({"a": bytes([1]).hex(), "b": []},
    ///            sort_keys=True, separators=(",", ":"))
    /// # '{"a":"01","b":[]}'
    /// ```
    /// where `"01"` is `bytes([1]).hex()`.
    #[test]
    fn canonical_golden_fixture() {
        let fields = vec![
            ("a", CanonicalVal::Bytes(vec![0x01])),
            ("b", CanonicalVal::Array(vec![])),
        ];
        assert_eq!(canonical(&fields), r#"{"a":"01","b":[]}"#);
    }

    /// Keys are sorted alphabetically regardless of input order.
    #[test]
    fn canonical_sorts_keys() {
        // Deliberately supply in reverse order.
        let fields = vec![
            ("z", CanonicalVal::Array(vec![])),
            ("a", CanonicalVal::Bytes(vec![0x01])),
        ];
        assert_eq!(canonical(&fields), r#"{"a":"01","z":[]}"#);
    }

    /// Byte fields are encoded as lowercase hex (`"ff"`, not `"FF"`).
    #[test]
    fn canonical_bytes_are_lowercase_hex() {
        let fields = vec![("x", CanonicalVal::Bytes(vec![0xAB, 0xCD, 0xEF]))];
        assert_eq!(canonical(&fields), r#"{"x":"abcdef"}"#);
    }

    /// Output contains no whitespace.
    #[test]
    fn canonical_no_whitespace() {
        let fields = vec![
            ("k1", CanonicalVal::Bytes(vec![0x00])),
            ("k2", CanonicalVal::Array(vec![])),
        ];
        let out = canonical(&fields);
        assert!(
            out.bytes()
                .all(|b| !matches!(b, b' ' | b'\n' | b'\r' | b'\t')),
            "canonical output must contain no whitespace, got: {out:?}",
        );
    }

    /// Nested arrays are rendered correctly.
    #[test]
    fn canonical_nested_empty_arrays() {
        let fields = vec![(
            "a",
            CanonicalVal::Array(vec![
                CanonicalVal::Array(vec![]),
                CanonicalVal::Array(vec![]),
            ]),
        )];
        assert_eq!(canonical(&fields), r#"{"a":[[],[]]}"#);
    }

    // ── cell_write_ref — golden cross-checks against Python ──────────────────

    /// Golden: `CellWriteEvent(payload=b"hello").cell_ref()` in Python.
    ///
    /// Generated with:
    /// ```python
    /// from amplifier_data.models import CellWriteEvent
    /// CellWriteEvent(payload=b"hello").cell_ref()
    /// # 'c016bb610fba601a6fa1ef80f32ef34cc2a0e62ac5dd9c4775489e1edc6aa415'
    /// ```
    #[test]
    fn golden_cell_write_ref_empty_interpreters() {
        assert_eq!(
            cell_write_ref(b"hello", &[]),
            "c016bb610fba601a6fa1ef80f32ef34cc2a0e62ac5dd9c4775489e1edc6aa415",
        );
    }

    /// Golden: `CellWriteEvent(payload=b"hello", interpreters=("a"*64,)).cell_ref()`.
    ///
    /// Generated with:
    /// ```python
    /// from amplifier_data.models import CellWriteEvent
    /// CellWriteEvent(payload=b"hello", interpreters=("a"*64,)).cell_ref()
    /// # 'f0790992e77cfced1241ee13291fafca1022edc1a5922ab59827a0ab811ec359'
    /// ```
    #[test]
    fn golden_cell_write_ref_one_interpreter() {
        let interpreter_hash = "a".repeat(64);
        assert_eq!(
            cell_write_ref(b"hello", &[interpreter_hash]),
            "f0790992e77cfced1241ee13291fafca1022edc1a5922ab59827a0ab811ec359",
        );
    }

    // ── interpreter_derive_ref — golden cross-check ───────────────────────────

    /// Golden: `InterpreterOutputEvent.derive_ref(model, input_hash, consumed, output)`.
    ///
    /// Generated with:
    /// ```python
    /// from amplifier_data.models import InterpreterOutputEvent
    /// InterpreterOutputEvent.derive_ref(
    ///     "gpt-4o", "b"*64, ("c"*64,), b"result"
    /// )
    /// # 'e8dc02e66819e011740ea6b243b87c954ca09dff021cd180019f967ee4ceb561'
    /// ```
    ///
    /// Circularity guard: the `cell_ref` field is excluded from the canonical map,
    /// so the result does NOT depend on the cell's own identity.  This is verified
    /// implicitly by matching the derive_ref output: if cell_ref were included the
    /// hash would differ.
    #[test]
    fn golden_interpreter_derive_ref() {
        let model = "gpt-4o";
        let input_hash = "b".repeat(64);
        let consumed = vec!["c".repeat(64)];
        assert_eq!(
            interpreter_derive_ref(model, &input_hash, &consumed, b"result"),
            "e8dc02e66819e011740ea6b243b87c954ca09dff021cd180019f967ee4ceb561",
        );
    }

    // ── index_derive_ref — golden cross-check ─────────────────────────────────

    /// Golden: `IndexCellEvent.derive_ref(access_summary, output)`.
    ///
    /// Generated with:
    /// ```python
    /// from amplifier_data.models import IndexCellEvent
    /// IndexCellEvent.derive_ref(b"summary", b"index")
    /// # '1b3b703916d63984c74e3dde4835bcbe4383e622bea8665f40e905cf228b7f8e'
    /// ```
    #[test]
    fn golden_index_derive_ref() {
        assert_eq!(
            index_derive_ref(b"summary", b"index"),
            "1b3b703916d63984c74e3dde4835bcbe4383e622bea8665f40e905cf228b7f8e",
        );
    }

    // ── tool_result_hash — golden cross-check ────────────────────────────────

    /// Golden: `ToolResultEvent(output=b"tool output").hash`.
    ///
    /// Generated with:
    /// ```python
    /// from amplifier_data.models import ToolResultEvent
    /// ToolResultEvent(output=b"tool output").hash
    /// # 'e2a78c7b5a05c0b34cbcac65007d1e2a5765341d79b159797db7baf5fa1ecb13'
    /// ```
    #[test]
    fn golden_tool_result_hash() {
        assert_eq!(
            tool_result_hash(b"tool output"),
            "e2a78c7b5a05c0b34cbcac65007d1e2a5765341d79b159797db7baf5fa1ecb13",
        );
    }

    // ── Event and Cell structural smoke tests ─────────────────────────────────

    /// `Event::CellWrite` round-trips through clone/eq.
    #[test]
    fn event_cell_write_clone_eq() {
        let ev = Event::CellWrite {
            payload: b"hello".to_vec(),
            interpreters: vec!["a".repeat(64)],
        };
        assert_eq!(ev.clone(), ev);
    }

    /// `Cell` round-trips through clone/eq with non-trivial provenance.
    #[test]
    fn cell_clone_eq() {
        let ev = Event::ToolResult {
            output: b"out".to_vec(),
        };
        let cell = Cell {
            payload: b"payload".to_vec(),
            provenance: vec![ev],
            interpreters: vec![],
        };
        assert_eq!(cell.clone(), cell);
    }

    /// All `Event` variants are constructible (ensures the enum is complete).
    #[test]
    fn event_all_variants_constructible() {
        let fake_hash = "0".repeat(64);
        let _variants: Vec<Event> = vec![
            Event::CellWrite {
                payload: vec![],
                interpreters: vec![],
            },
            Event::Relationship {
                from_ref: fake_hash.clone(),
                to_ref: fake_hash.clone(),
                type_: "uses".into(),
            },
            Event::ToolResult { output: vec![] },
            Event::InterpreterOutput {
                cell_ref: fake_hash.clone(),
                model: "gpt-4o".into(),
                input_hash: fake_hash.clone(),
                tool_results_consumed: vec![],
                output: vec![],
            },
            Event::Access {
                cell_ref: fake_hash.clone(),
            },
            Event::IndexCell {
                cell_ref: fake_hash.clone(),
                access_summary: vec![],
                output: vec![],
            },
        ];
    }
}
