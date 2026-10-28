//! Binary codec for [`Event`]: `encode_event` / `decode_event`.
//!
//! # Wire format — length-sandwich record
//!
//! ```text
//! [body_len : u32 LE]          ← leading length
//! [body_bytes : body_len]
//!   body = [tag : u8] [fields…]
//! [body_len : u32 LE]          ← trailing length (sandwich for forward/backward scan)
//! ```
//!
//! All multi-byte integers are **little-endian**.
//!
//! # Field encoding
//!
//! | Rust type            | Wire                                                        |
//! |----------------------|-------------------------------------------------------------|
//! | `Vec<u8>`            | `[len : u32 LE][bytes : len]`                               |
//! | `String` / `Hash`    | `[len : u32 LE][utf-8 bytes : len]`                         |
//! | `Vec<String>`        | `[count : u32 LE]` followed by each string (length-prefixed) |
//!
//! # Variant tags
//!
//! | Tag  | Variant             |
//! |------|---------------------|
//! | 0x00 | `CellWrite`         |
//! | 0x01 | `Relationship`      |
//! | 0x02 | `ToolResult`        |
//! | 0x03 | `InterpreterOutput` |
//! | 0x04 | `Access`            |
//! | 0x05 | `IndexCell`         |

use crate::models::{Event, Hash};

// ─── Public error type ────────────────────────────────────────────────────────

/// Decode error returned by [`decode_event`].
#[derive(Debug, PartialEq, Eq)]
pub enum CodecError {
    /// The byte slice is too short to contain a valid record (or a required field is missing).
    TooShort,
    /// The leading and trailing length fields in the record disagree.
    LengthMismatch,
    /// The tag byte does not correspond to any known [`Event`] variant.
    UnknownTag(u8),
    /// A string field contained invalid UTF-8.
    InvalidUtf8,
    /// Bytes remain unconsumed after all expected fields have been decoded.
    TrailingBytes,
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Write a single framed event record to `writer`.
///
/// Encodes `event` as a length-sandwich record (see module-level wire-format
/// doc) and writes the full record to `writer` in one `write_all` call.
/// The caller is responsible for flushing or closing `writer` as needed.
///
/// Use [`read_log`](crate::replay::read_log) to walk a sequence of records
/// written by this function, with transparent torn-tail tolerance.
pub fn write_frame<W: std::io::Write>(writer: &mut W, event: &Event) -> std::io::Result<()> {
    writer.write_all(&encode_event(event))
}

/// Encode `event` as a length-sandwich binary record.
///
/// The returned bytes are self-framing and can be decoded with [`decode_event`].
pub fn encode_event(event: &Event) -> Vec<u8> {
    let mut body = Vec::new();
    match event {
        Event::CellWrite {
            payload,
            interpreters,
        } => {
            body.push(0x00);
            put_bytes(&mut body, payload);
            put_string_list(&mut body, interpreters);
        }
        Event::Relationship {
            from_ref,
            to_ref,
            type_,
        } => {
            body.push(0x01);
            put_str(&mut body, from_ref);
            put_str(&mut body, to_ref);
            put_str(&mut body, type_);
        }
        Event::ToolResult { output } => {
            body.push(0x02);
            put_bytes(&mut body, output);
        }
        Event::InterpreterOutput {
            cell_ref,
            model,
            input_hash,
            tool_results_consumed,
            output,
        } => {
            body.push(0x03);
            put_str(&mut body, cell_ref);
            put_str(&mut body, model);
            put_str(&mut body, input_hash);
            put_string_list(&mut body, tool_results_consumed);
            put_bytes(&mut body, output);
        }
        Event::Access { cell_ref } => {
            body.push(0x04);
            put_str(&mut body, cell_ref);
        }
        Event::IndexCell {
            cell_ref,
            access_summary,
            output,
        } => {
            body.push(0x05);
            put_str(&mut body, cell_ref);
            put_bytes(&mut body, access_summary);
            put_bytes(&mut body, output);
        }
    }
    sandwich(body)
}

/// Decode a length-sandwich binary record produced by [`encode_event`].
///
/// Returns `Err` on any malformed or truncated input.  Does not panic.
pub fn decode_event(bytes: &[u8]) -> Result<Event, CodecError> {
    // Minimum valid record: 4 (len_head) + 1 (tag) + 4 (len_tail) = 9 bytes.
    if bytes.len() < 9 {
        return Err(CodecError::TooShort);
    }

    let head_len = get_u32(&bytes[0..4]) as usize;

    // Total expected: 4 (head) + body + 4 (tail).
    // Use checked_add to avoid overflow on 32-bit platforms with adversarial input.
    let total = match 8usize.checked_add(head_len) {
        Some(t) => t,
        None => return Err(CodecError::TooShort),
    };

    match bytes.len().cmp(&total) {
        std::cmp::Ordering::Less => return Err(CodecError::TooShort),
        std::cmp::Ordering::Greater => return Err(CodecError::TrailingBytes),
        std::cmp::Ordering::Equal => {}
    }

    let body = &bytes[4..4 + head_len];
    let tail_len = get_u32(&bytes[4 + head_len..8 + head_len]) as usize;

    if head_len != tail_len {
        return Err(CodecError::LengthMismatch);
    }

    if body.is_empty() {
        return Err(CodecError::TooShort);
    }

    let tag = body[0];
    let mut cur = &body[1..];

    match tag {
        0x00 => {
            let payload = take_bytes(&mut cur)?.to_vec();
            let interpreters = take_string_list(&mut cur)?;
            exhaust(cur)?;
            Ok(Event::CellWrite {
                payload,
                interpreters,
            })
        }
        0x01 => {
            let from_ref = take_str(&mut cur)?;
            let to_ref = take_str(&mut cur)?;
            let type_ = take_str(&mut cur)?;
            exhaust(cur)?;
            Ok(Event::Relationship {
                from_ref,
                to_ref,
                type_,
            })
        }
        0x02 => {
            let output = take_bytes(&mut cur)?.to_vec();
            exhaust(cur)?;
            Ok(Event::ToolResult { output })
        }
        0x03 => {
            let cell_ref = take_str(&mut cur)?;
            let model = take_str(&mut cur)?;
            let input_hash = take_str(&mut cur)?;
            let tool_results_consumed = take_string_list(&mut cur)?;
            let output = take_bytes(&mut cur)?.to_vec();
            exhaust(cur)?;
            Ok(Event::InterpreterOutput {
                cell_ref,
                model,
                input_hash,
                tool_results_consumed,
                output,
            })
        }
        0x04 => {
            let cell_ref = take_str(&mut cur)?;
            exhaust(cur)?;
            Ok(Event::Access { cell_ref })
        }
        0x05 => {
            let cell_ref = take_str(&mut cur)?;
            let access_summary = take_bytes(&mut cur)?.to_vec();
            let output = take_bytes(&mut cur)?.to_vec();
            exhaust(cur)?;
            Ok(Event::IndexCell {
                cell_ref,
                access_summary,
                output,
            })
        }
        _ => Err(CodecError::UnknownTag(tag)),
    }
}

// ─── Encode helpers ───────────────────────────────────────────────────────────

/// Wrap `body` in the length sandwich and return the full record bytes.
fn sandwich(body: Vec<u8>) -> Vec<u8> {
    let len_bytes = (body.len() as u32).to_le_bytes();
    let mut rec = Vec::with_capacity(8 + body.len());
    rec.extend_from_slice(&len_bytes);
    rec.extend_from_slice(&body);
    rec.extend_from_slice(&len_bytes);
    rec
}

/// Append a length-prefixed byte slice to `buf`.
fn put_bytes(buf: &mut Vec<u8>, data: &[u8]) {
    buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
    buf.extend_from_slice(data);
}

/// Append a length-prefixed UTF-8 string to `buf`.
fn put_str(buf: &mut Vec<u8>, s: &str) {
    put_bytes(buf, s.as_bytes());
}

/// Append a count-prefixed list of length-prefixed strings to `buf`.
fn put_string_list(buf: &mut Vec<u8>, list: &[String]) {
    buf.extend_from_slice(&(list.len() as u32).to_le_bytes());
    for s in list {
        put_str(buf, s);
    }
}

// ─── Decode helpers ───────────────────────────────────────────────────────────

/// Read a `u32` from 4 bytes (little-endian, caller ensures `b.len() >= 4`).
#[inline]
fn get_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Consume a length-prefixed byte slice from `*cur`, advancing the cursor.
fn take_bytes<'a>(cur: &mut &'a [u8]) -> Result<&'a [u8], CodecError> {
    if cur.len() < 4 {
        return Err(CodecError::TooShort);
    }
    let len = get_u32(&cur[..4]) as usize;
    *cur = &cur[4..];
    if cur.len() < len {
        return Err(CodecError::TooShort);
    }
    let data = &cur[..len];
    *cur = &cur[len..];
    Ok(data)
}

/// Consume a length-prefixed UTF-8 string from `*cur`, advancing the cursor.
fn take_str(cur: &mut &[u8]) -> Result<String, CodecError> {
    let bytes = take_bytes(cur)?;
    std::str::from_utf8(bytes)
        .map(|s| s.to_owned())
        .map_err(|_| CodecError::InvalidUtf8)
}

/// Consume a count-prefixed list of length-prefixed strings from `*cur`.
fn take_string_list(cur: &mut &[u8]) -> Result<Vec<Hash>, CodecError> {
    if cur.len() < 4 {
        return Err(CodecError::TooShort);
    }
    let count = get_u32(&cur[..4]) as usize;
    *cur = &cur[4..];
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(take_str(cur)?);
    }
    Ok(out)
}

/// Return `Err(TrailingBytes)` if `cur` is non-empty.
fn exhaust(cur: &[u8]) -> Result<(), CodecError> {
    if cur.is_empty() {
        Ok(())
    } else {
        Err(CodecError::TrailingBytes)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Event;

    // ── Test helpers (mirror the codec spec to build expected bytes) ──────────

    /// Build the expected sandwich bytes: `[len][body][len]`.
    fn mk_sandwich(body: &[u8]) -> Vec<u8> {
        let len = (body.len() as u32).to_le_bytes();
        let mut v = Vec::new();
        v.extend_from_slice(&len);
        v.extend_from_slice(body);
        v.extend_from_slice(&len);
        v
    }

    /// Build length-prefixed bytes: `[len u32 LE][data]`.
    fn lp_bytes(data: &[u8]) -> Vec<u8> {
        let mut v = (data.len() as u32).to_le_bytes().to_vec();
        v.extend_from_slice(data);
        v
    }

    fn lp_str(s: &str) -> Vec<u8> {
        lp_bytes(s.as_bytes())
    }

    fn lp_list(items: &[&str]) -> Vec<u8> {
        let mut v = (items.len() as u32).to_le_bytes().to_vec();
        for s in items {
            v.extend(lp_str(s));
        }
        v
    }

    // ── Round-trip tests — all 6 Event variants ───────────────────────────────

    #[test]
    fn round_trip_cell_write_with_interpreters() {
        let ev = Event::CellWrite {
            payload: b"hello world".to_vec(),
            interpreters: vec!["a".repeat(64), "b".repeat(64)],
        };
        assert_eq!(decode_event(&encode_event(&ev)).unwrap(), ev);
    }

    #[test]
    fn round_trip_cell_write_empty() {
        let ev = Event::CellWrite {
            payload: vec![],
            interpreters: vec![],
        };
        assert_eq!(decode_event(&encode_event(&ev)).unwrap(), ev);
    }

    #[test]
    fn round_trip_relationship() {
        let ev = Event::Relationship {
            from_ref: "a".repeat(64),
            to_ref: "b".repeat(64),
            type_: "depends_on".to_string(),
        };
        assert_eq!(decode_event(&encode_event(&ev)).unwrap(), ev);
    }

    #[test]
    fn round_trip_tool_result() {
        let ev = Event::ToolResult {
            output: b"tool output bytes \x00\x01\x02\xff".to_vec(),
        };
        assert_eq!(decode_event(&encode_event(&ev)).unwrap(), ev);
    }

    #[test]
    fn round_trip_interpreter_output() {
        let ev = Event::InterpreterOutput {
            cell_ref: "c".repeat(64),
            model: "gpt-4o".to_string(),
            input_hash: "d".repeat(64),
            tool_results_consumed: vec!["e".repeat(64), "f".repeat(64)],
            output: b"generated text".to_vec(),
        };
        assert_eq!(decode_event(&encode_event(&ev)).unwrap(), ev);
    }

    #[test]
    fn round_trip_interpreter_output_no_tool_results() {
        let ev = Event::InterpreterOutput {
            cell_ref: "c".repeat(64),
            model: "claude-3".to_string(),
            input_hash: "d".repeat(64),
            tool_results_consumed: vec![],
            output: vec![],
        };
        assert_eq!(decode_event(&encode_event(&ev)).unwrap(), ev);
    }

    #[test]
    fn round_trip_access() {
        let ev = Event::Access {
            cell_ref: "0".repeat(64),
        };
        assert_eq!(decode_event(&encode_event(&ev)).unwrap(), ev);
    }

    #[test]
    fn round_trip_index_cell() {
        let ev = Event::IndexCell {
            cell_ref: "9".repeat(64),
            access_summary: b"access data here".to_vec(),
            output: b"index payload".to_vec(),
        };
        assert_eq!(decode_event(&encode_event(&ev)).unwrap(), ev);
    }

    // ── Golden-byte fixtures — byte-identical encoding for all 6 variants ─────
    //
    // Each test constructs expected bytes from the codec spec primitives
    // (mk_sandwich / lp_bytes / lp_str / lp_list) and asserts:
    //   1. encode_event(ev) == expected_bytes    (encode matches spec)
    //   2. decode_event(&expected_bytes) == ev   (decode round-trips)
    //
    // Two variants also include a truly literal const byte array to pin the
    // format independently of the helper functions.

    // ── Truly literal golden constants ────────────────────────────────────────

    /// Literal golden bytes for `ToolResult { output: b"hi" }`.
    ///
    /// ```text
    /// [07 00 00 00]  body_len_head = 7
    /// [02]           tag = ToolResult
    /// [02 00 00 00]  output len = 2
    /// [68 69]        b"hi"
    /// [07 00 00 00]  body_len_tail = 7  (sandwich)
    /// ```
    const GOLDEN_TOOL_RESULT_HI: &[u8] = &[
        0x07, 0x00, 0x00, 0x00, // body_len_head = 7
        0x02, // tag = ToolResult
        0x02, 0x00, 0x00, 0x00, // output len = 2
        0x68, 0x69, // b"hi"
        0x07, 0x00, 0x00, 0x00, // body_len_tail = 7
    ];

    /// Literal golden bytes for `Access { cell_ref: "x".repeat(64) }`.
    ///
    /// ```text
    /// [45 00 00 00]  body_len = 69  (1 tag + 4 len + 64 data)
    /// [04]           tag = Access
    /// [40 00 00 00]  cell_ref len = 64
    /// [78 × 64]      "x" * 64
    /// [45 00 00 00]  body_len_tail = 69  (sandwich)
    /// ```
    const GOLDEN_ACCESS_X64: &[u8] = &[
        0x45, 0x00, 0x00, 0x00, // body_len_head = 69
        0x04, // tag = Access
        0x40, 0x00, 0x00, 0x00, // cell_ref len = 64
        // 64 bytes of 0x78 ('x')
        0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78,
        0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78,
        0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78,
        0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78, 0x78,
        0x78, 0x78, 0x78, 0x78, 0x45, 0x00, 0x00, 0x00, // body_len_tail = 69
    ];

    // ── Golden tests ──────────────────────────────────────────────────────────

    #[test]
    fn golden_tool_result() {
        let ev = Event::ToolResult {
            output: b"hi".to_vec(),
        };
        // Verify against the literal constant.
        assert_eq!(encode_event(&ev), GOLDEN_TOOL_RESULT_HI);
        assert_eq!(decode_event(GOLDEN_TOOL_RESULT_HI).unwrap(), ev);
        // Cross-check: helper construction agrees.
        let mut body = vec![0x02u8];
        body.extend(lp_bytes(b"hi"));
        assert_eq!(encode_event(&ev), mk_sandwich(&body));
    }

    #[test]
    fn golden_access() {
        let cell_ref = "x".repeat(64);
        let ev = Event::Access {
            cell_ref: cell_ref.clone(),
        };
        // Verify against the literal constant.
        assert_eq!(encode_event(&ev), GOLDEN_ACCESS_X64);
        assert_eq!(decode_event(GOLDEN_ACCESS_X64).unwrap(), ev);
        // Cross-check: helper construction agrees.
        let mut body = vec![0x04u8];
        body.extend(lp_str(&cell_ref));
        assert_eq!(encode_event(&ev), mk_sandwich(&body));
    }

    #[test]
    fn golden_cell_write_empty_interpreters() {
        let ev = Event::CellWrite {
            payload: b"p".to_vec(),
            interpreters: vec![],
        };
        let mut body = vec![0x00u8];
        body.extend(lp_bytes(b"p"));
        body.extend(lp_list(&[]));
        let expected = mk_sandwich(&body);
        assert_eq!(encode_event(&ev), expected);
        assert_eq!(decode_event(&expected).unwrap(), ev);
    }

    #[test]
    fn golden_cell_write_with_one_interpreter() {
        let interp = "a".repeat(64);
        let ev = Event::CellWrite {
            payload: b"data".to_vec(),
            interpreters: vec![interp.clone()],
        };
        let mut body = vec![0x00u8];
        body.extend(lp_bytes(b"data"));
        body.extend(lp_list(&[interp.as_str()]));
        let expected = mk_sandwich(&body);
        assert_eq!(encode_event(&ev), expected);
        assert_eq!(decode_event(&expected).unwrap(), ev);
    }

    #[test]
    fn golden_relationship() {
        let from = "a".repeat(64);
        let to = "b".repeat(64);
        let ev = Event::Relationship {
            from_ref: from.clone(),
            to_ref: to.clone(),
            type_: "uses".to_string(),
        };
        let mut body = vec![0x01u8];
        body.extend(lp_str(&from));
        body.extend(lp_str(&to));
        body.extend(lp_str("uses"));
        let expected = mk_sandwich(&body);
        assert_eq!(encode_event(&ev), expected);
        assert_eq!(decode_event(&expected).unwrap(), ev);
    }

    #[test]
    fn golden_interpreter_output_no_consumed() {
        let cell_ref = "c".repeat(64);
        let input_hash = "i".repeat(64);
        let ev = Event::InterpreterOutput {
            cell_ref: cell_ref.clone(),
            model: "gpt".to_string(),
            input_hash: input_hash.clone(),
            tool_results_consumed: vec![],
            output: b"o".to_vec(),
        };
        let mut body = vec![0x03u8];
        body.extend(lp_str(&cell_ref));
        body.extend(lp_str("gpt"));
        body.extend(lp_str(&input_hash));
        body.extend(lp_list(&[]));
        body.extend(lp_bytes(b"o"));
        let expected = mk_sandwich(&body);
        assert_eq!(encode_event(&ev), expected);
        assert_eq!(decode_event(&expected).unwrap(), ev);
    }

    #[test]
    fn golden_interpreter_output_with_consumed() {
        let cell_ref = "c".repeat(64);
        let input_hash = "i".repeat(64);
        let tr1 = "e".repeat(64);
        let tr2 = "f".repeat(64);
        let ev = Event::InterpreterOutput {
            cell_ref: cell_ref.clone(),
            model: "gpt-4o".to_string(),
            input_hash: input_hash.clone(),
            tool_results_consumed: vec![tr1.clone(), tr2.clone()],
            output: b"result bytes".to_vec(),
        };
        let mut body = vec![0x03u8];
        body.extend(lp_str(&cell_ref));
        body.extend(lp_str("gpt-4o"));
        body.extend(lp_str(&input_hash));
        body.extend(lp_list(&[tr1.as_str(), tr2.as_str()]));
        body.extend(lp_bytes(b"result bytes"));
        let expected = mk_sandwich(&body);
        assert_eq!(encode_event(&ev), expected);
        assert_eq!(decode_event(&expected).unwrap(), ev);
    }

    #[test]
    fn golden_index_cell() {
        let cell_ref = "d".repeat(64);
        let ev = Event::IndexCell {
            cell_ref: cell_ref.clone(),
            access_summary: b"s".to_vec(),
            output: b"o".to_vec(),
        };
        let mut body = vec![0x05u8];
        body.extend(lp_str(&cell_ref));
        body.extend(lp_bytes(b"s"));
        body.extend(lp_bytes(b"o"));
        let expected = mk_sandwich(&body);
        assert_eq!(encode_event(&ev), expected);
        assert_eq!(decode_event(&expected).unwrap(), ev);
    }

    // ── Decode-failure tests — malformed / truncated input returns Err ─────────

    #[test]
    fn decode_failure_empty_slice() {
        assert_eq!(decode_event(&[]), Err(CodecError::TooShort));
    }

    #[test]
    fn decode_failure_three_bytes() {
        assert_eq!(decode_event(&[0x01, 0x00, 0x00]), Err(CodecError::TooShort));
    }

    #[test]
    fn decode_failure_eight_bytes_no_tag() {
        // body_len = 0 and tail matches — body is empty, no tag.
        let bytes = [0x00u8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(decode_event(&bytes), Err(CodecError::TooShort));
    }

    #[test]
    fn decode_failure_body_truncated() {
        // Header claims body_len = 100 but only 2 bytes of body are present.
        let bytes = [100u8, 0, 0, 0, 0x04, 0x00];
        assert_eq!(decode_event(&bytes), Err(CodecError::TooShort));
    }

    #[test]
    fn decode_failure_trailing_bytes_after_record() {
        // Valid ToolResult followed by one extra byte.
        let ev = Event::ToolResult {
            output: b"x".to_vec(),
        };
        let mut bytes = encode_event(&ev);
        bytes.push(0xAA);
        assert_eq!(decode_event(&bytes), Err(CodecError::TrailingBytes));
    }

    #[test]
    fn decode_failure_length_mismatch() {
        // Valid Access record with the last byte of the tail length corrupted.
        let ev = Event::Access {
            cell_ref: "z".repeat(64),
        };
        let mut bytes = encode_event(&ev);
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        assert_eq!(decode_event(&bytes), Err(CodecError::LengthMismatch));
    }

    #[test]
    fn decode_failure_unknown_tag_ff() {
        // Tag 0xFF is unrecognised.
        let body = [0xFFu8];
        assert_eq!(
            decode_event(&mk_sandwich(&body)),
            Err(CodecError::UnknownTag(0xFF))
        );
    }

    #[test]
    fn decode_failure_unknown_tag_06() {
        // Tag 0x06 is one past IndexCell (0x05).
        let body = [0x06u8];
        assert_eq!(
            decode_event(&mk_sandwich(&body)),
            Err(CodecError::UnknownTag(0x06))
        );
    }

    #[test]
    fn decode_failure_invalid_utf8_in_string_field() {
        // Access event whose cell_ref bytes are not valid UTF-8.
        let mut body = vec![0x04u8];
        body.extend(lp_bytes(&[0xFF])); // 0xFF is not valid UTF-8
        assert_eq!(
            decode_event(&mk_sandwich(&body)),
            Err(CodecError::InvalidUtf8)
        );
    }

    #[test]
    fn decode_failure_field_claims_more_bytes_than_body() {
        // Relationship event where the third string field claims 100 bytes but
        // the body ends after the count prefix.
        let from = "a".repeat(64);
        let to = "b".repeat(64);
        let mut body = vec![0x01u8];
        body.extend(lp_str(&from));
        body.extend(lp_str(&to));
        body.extend_from_slice(&100u32.to_le_bytes()); // claim 100 bytes for type_
                                                       // (no actual bytes follow)
        assert_eq!(decode_event(&mk_sandwich(&body)), Err(CodecError::TooShort));
    }

    // ── write_frame tests ──────────────────────────────────────────────────────

    /// write_frame writes exactly the same bytes as encode_event.
    #[test]
    fn write_frame_writes_encode_event_bytes() {
        let ev = Event::Access {
            cell_ref: "x".repeat(64),
        };
        let expected = encode_event(&ev);
        let mut buf = Vec::new();
        write_frame(&mut buf, &ev).expect("write_frame must succeed");
        assert_eq!(
            buf, expected,
            "write_frame must produce identical bytes to encode_event"
        );
    }

    /// write_frame can be called multiple times to append multiple records.
    #[test]
    fn write_frame_appends_multiple_records() {
        let ev1 = Event::ToolResult {
            output: b"first".to_vec(),
        };
        let ev2 = Event::ToolResult {
            output: b"second".to_vec(),
        };
        let mut buf = Vec::new();
        write_frame(&mut buf, &ev1).expect("write_frame ev1");
        write_frame(&mut buf, &ev2).expect("write_frame ev2");

        let expected_len = encode_event(&ev1).len() + encode_event(&ev2).len();
        assert_eq!(
            buf.len(),
            expected_len,
            "two write_frame calls must produce two records back-to-back"
        );
    }
}
