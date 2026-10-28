//! Replay / fold engine — the E1 regeneration guarantee.
//!
//! Pure functions, no inference.  Mirrors `amplifier_data.replay` in Python.

use crate::codec::decode_event;
use crate::models::{cell_write_ref, Cell, Event, Hash, SeqPos};

// ── On-disk log walker ───────────────────────────────────────────────────────

/// Walk a flat sequence of length-sandwich records from `reader`, returning all
/// fully-written [`Event`]s.
///
/// ## Torn-tail tolerance
///
/// Each record has the framing `[body_len : u32 LE][body : body_len][body_len : u32 LE]`
/// produced by [`crate::codec::write_frame`].  If `reader` reaches EOF in the
/// middle of a record (a *torn tail* — the result of a crash or an interrupted
/// write), the incomplete final record is **silently discarded** and all
/// preceding complete records are returned without error.
///
/// ## Return value
///
/// - `Ok(events)` — all complete records decoded in order; zero events if the
///   log is empty or every record is torn.
/// - `Err(io_error)` — only for genuine I/O errors, never for EOF / torn tails.
pub fn read_log<R: std::io::Read>(mut reader: R) -> std::io::Result<Vec<Event>> {
    let mut events = Vec::new();

    loop {
        // ── Step 1: read the 4-byte leading length ───────────────────────────
        let mut len_buf = [0u8; 4];
        match reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break, // clean end or torn tail
            Err(e) => return Err(e),
        }
        let body_len = u32::from_le_bytes(len_buf) as usize;

        // ── Step 2: read the body bytes ──────────────────────────────────────
        let mut body = vec![0u8; body_len];
        match reader.read_exact(&mut body) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break, // torn tail
            Err(e) => return Err(e),
        }

        // ── Step 3: read the 4-byte trailing length ──────────────────────────
        let mut tail_buf = [0u8; 4];
        match reader.read_exact(&mut tail_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break, // torn tail
            Err(e) => return Err(e),
        }

        // ── Step 4: reassemble and decode ────────────────────────────────────
        // Reassemble [len_head][body][len_tail] so decode_event can verify the
        // sandwich and parse the event tag + fields.
        let mut record = Vec::with_capacity(8 + body_len);
        record.extend_from_slice(&len_buf);
        record.extend_from_slice(&body);
        record.extend_from_slice(&tail_buf);

        match decode_event(&record) {
            Ok(event) => events.push(event),
            Err(_) => break, // malformed / corrupted record — stop at recoverable boundary
        }
    }

    Ok(events)
}

/// The cell_ref whose provenance this event belongs to, or `None`.
///
/// Mirrors Python's `target_ref`:
///
/// | Event variant      | Return value                        |
/// |--------------------|-------------------------------------|
/// | `CellWrite`        | `cell_write_ref(payload, interps)`  |
/// | `InterpreterOutput`| stored `cell_ref` field (no recompute)|
/// | `IndexCell`        | stored `cell_ref` field (no recompute)|
/// | `Relationship`     | `from_ref`                          |
/// | `ToolResult`       | `None`                              |
/// | `Access`           | `None`                              |
///
/// **Narrowing note (from design decisions):** `InterpreterOutput` and
/// `IndexCell` read the event's *stored* `cell_ref` field (set in Python via
/// `derive_ref`); the core does NOT recompute those.  The only canonical hash
/// the kernel hot-path recomputes is `CellWrite.cell_ref`.
pub fn target_ref(event: &Event) -> Option<Hash> {
    match event {
        Event::CellWrite {
            payload,
            interpreters,
        } => Some(cell_write_ref(payload, interpreters)),
        Event::InterpreterOutput { cell_ref, .. } => Some(cell_ref.clone()),
        Event::IndexCell { cell_ref, .. } => Some(cell_ref.clone()),
        Event::Relationship { from_ref, .. } => Some(from_ref.clone()),
        Event::ToolResult { .. } => None,
        Event::Access { .. } => None,
    }
}

/// True if `event` *defines* the cell identified by `cell_ref`.
///
/// Mirrors Python's `is_defining`.  Only three event variants can define a
/// cell; all others return `false`.
///
/// | Event variant      | Condition                                |
/// |--------------------|------------------------------------------|
/// | `CellWrite`        | `cell_write_ref(payload, interps) == cell_ref` |
/// | `InterpreterOutput`| stored `cell_ref` field `== cell_ref`    |
/// | `IndexCell`        | stored `cell_ref` field `== cell_ref`    |
/// | anything else      | `false`                                  |
pub fn is_defining(event: &Event, cell_ref: &str) -> bool {
    match event {
        Event::CellWrite {
            payload,
            interpreters,
        } => cell_write_ref(payload, interpreters) == cell_ref,
        Event::InterpreterOutput {
            cell_ref: ev_ref, ..
        } => ev_ref == cell_ref,
        Event::IndexCell {
            cell_ref: ev_ref, ..
        } => ev_ref == cell_ref,
        _ => false,
    }
}

/// No defining event exists for the requested `cell_ref`.
///
/// Mirrors Python's `MissingDefiningEvent` exception.  The binding maps this
/// to the Python exception class in Phase 2.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MissingDefiningEvent {
    /// The cell reference for which no defining event was found.
    pub cell_ref: Hash,
}

impl std::fmt::Display for MissingDefiningEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "no defining event found for cell_ref {:?}",
            self.cell_ref
        )
    }
}

impl std::error::Error for MissingDefiningEvent {}

/// Deterministic reduce over the ordered selection of events targeting `cell_ref`.
///
/// Mirrors Python's `regenerate()` in `amplifier_data.replay`.
///
/// ## Fold semantics (mirrors Python exactly)
///
/// 1. **Select** all events where `target_ref(event) == cell_ref`.
/// 2. **Sort** selected events by ascending [`SeqPos`].
/// 3. **Reject** if no selecting event satisfies `is_defining` → `Err(MissingDefiningEvent)`.
/// 4. **Reduce** in order — every event is appended to `provenance`:
///    - `CellWrite`          → last-write-wins on both `payload` *and* `interpreters`.
///    - `InterpreterOutput`  → last-write-wins on `payload` only (sets `payload = output`).
///    - `IndexCell`          → last-write-wins on `payload` only (sets `payload = output`).
///    - `Relationship`       → provenance only; `payload` and `interpreters` unchanged.
///
/// `Access` and `ToolResult` are never selected (`target_ref` returns `None`).
pub fn regenerate(
    events: &[(SeqPos, Event)],
    cell_ref: &str,
) -> Result<Cell, MissingDefiningEvent> {
    // Step 1 & 2: select events targeting this cell_ref, sort by SeqPos ascending.
    let mut selected: Vec<&(SeqPos, Event)> = events
        .iter()
        .filter(|(_, ev)| target_ref(ev).as_deref() == Some(cell_ref))
        .collect();
    selected.sort_by_key(|(pos, _)| *pos);

    // Step 3: require at least one defining event in the selection.
    if !selected.iter().any(|(_, ev)| is_defining(ev, cell_ref)) {
        return Err(MissingDefiningEvent {
            cell_ref: cell_ref.to_string(),
        });
    }

    // Step 4: fold — accumulate provenance, apply last-write-wins on payload/interpreters.
    let mut payload: Option<Vec<u8>> = None;
    let mut interpreters: Vec<Hash> = Vec::new();
    let mut provenance: Vec<Event> = Vec::new();

    for (_, ev) in &selected {
        provenance.push((*ev).clone());
        match ev {
            Event::CellWrite {
                payload: p,
                interpreters: interps,
            } => {
                payload = Some(p.clone());
                interpreters = interps.clone();
            }
            Event::InterpreterOutput { output, .. } => {
                payload = Some(output.clone());
                // interpreters left unchanged — InterpreterOutput does not own that field
            }
            Event::IndexCell { output, .. } => {
                payload = Some(output.clone());
            }
            // Relationship: appended to provenance above; payload/interpreters unchanged.
            _ => {}
        }
    }

    Ok(Cell {
        // SAFETY: guaranteed non-None because the is_defining check above confirmed at
        // least one CellWrite / InterpreterOutput / IndexCell is in the selection, and
        // all three arms set `payload`.
        payload: payload.expect("defining-event guarantee: payload must be set"),
        provenance,
        interpreters,
    })
}

// --- Tests -------------------------------------------------------------------

#[cfg(test)]
mod regenerate_tests {
    use super::*;
    use crate::models::{cell_write_ref, index_derive_ref, interpreter_derive_ref, SeqPos};

    // -- helpers --------------------------------------------------------------

    /// Build a `CellWrite` event and return `(cell_ref, event)`.
    fn make_cell_write(payload: &[u8], interpreters: &[Hash]) -> (Hash, Event) {
        let cell_ref = cell_write_ref(payload, interpreters);
        let event = Event::CellWrite {
            payload: payload.to_vec(),
            interpreters: interpreters.to_vec(),
        };
        (cell_ref, event)
    }

    // -- basic selection ------------------------------------------------------

    /// A single `CellWrite` produces a `Cell` with matching payload and interpreters.
    ///
    /// Mirrors: `test_regenerate_folds_a_written_cell` in Python.
    #[test]
    fn regenerate_basic_cell_write() {
        let interp = "a".repeat(64);
        let (cr, ev) = make_cell_write(b"verbatim", std::slice::from_ref(&interp));
        let events: Vec<(SeqPos, Event)> = vec![(0, ev.clone())];
        let cell = regenerate(&events, &cr).expect("should succeed");
        assert_eq!(cell.payload, b"verbatim".to_vec());
        assert_eq!(cell.interpreters, vec![interp]);
        assert_eq!(cell.provenance, vec![ev]);
    }

    /// A single `InterpreterOutput` defines a derived cell; `payload == output`.
    ///
    /// Mirrors: `test_regenerate_folds_a_derived_interpreter_cell` in Python.
    #[test]
    fn regenerate_interpreter_output_defines_cell() {
        let output = b"recorded output";
        let cr = interpreter_derive_ref("gpt-4o", &"b".repeat(64), &[], output);
        let ev = Event::InterpreterOutput {
            cell_ref: cr.clone(),
            model: "gpt-4o".into(),
            input_hash: "b".repeat(64),
            tool_results_consumed: vec![],
            output: output.to_vec(),
        };
        let cell = regenerate(&[(0, ev.clone())], &cr).expect("should succeed");
        assert_eq!(cell.payload, output.to_vec());
        assert_eq!(cell.interpreters, Vec::<Hash>::new());
        assert_eq!(cell.provenance, vec![ev]);
    }

    /// A single `IndexCell` defines an index cell; `payload == output`.
    #[test]
    fn regenerate_index_cell_defines_cell() {
        let cr = index_derive_ref(b"summary", b"index-output");
        let ev = Event::IndexCell {
            cell_ref: cr.clone(),
            access_summary: b"summary".to_vec(),
            output: b"index-output".to_vec(),
        };
        let cell = regenerate(&[(0, ev.clone())], &cr).expect("should succeed");
        assert_eq!(cell.payload, b"index-output".to_vec());
        assert_eq!(cell.provenance, vec![ev]);
    }

    // -- MissingDefiningEvent -------------------------------------------------

    /// Empty event slice → `Err(MissingDefiningEvent)`.
    #[test]
    fn regenerate_empty_events_err_missing_defining() {
        let result = regenerate(&[], "ref-that-does-not-exist");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().cell_ref, "ref-that-does-not-exist");
    }

    /// No event targets the requested `cell_ref` → `Err`.
    ///
    /// Mirrors: `test_regenerate_unknown_ref_raises_missing_defining_event` in Python.
    #[test]
    fn regenerate_no_matching_target_ref_err_missing_defining() {
        let (_, write) = make_cell_write(b"x", &[]);
        let events = vec![(0u64, write)];
        let result = regenerate(&events, "totally-different-ref");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().cell_ref, "totally-different-ref");
    }

    /// Only a `Relationship` event targets `cell_ref` — non-defining → `Err`.
    ///
    /// A relationship lands in provenance but never defines a cell.
    #[test]
    fn regenerate_only_relationship_matches_err_missing_defining() {
        let from = "e".repeat(64);
        let ev = Event::Relationship {
            from_ref: from.clone(),
            to_ref: "f".repeat(64),
            type_: "uses".into(),
        };
        // Only event targeting `from` is a Relationship — not defining
        let result = regenerate(&[(0, ev)], &from);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().cell_ref, from);
    }

    // -- last-write-wins ------------------------------------------------------

    /// Two duplicate `CellWrite` events (same content → same `cell_ref`):
    /// both accumulate in provenance; fold uses the later value (same in this case).
    #[test]
    fn regenerate_duplicate_cell_writes_both_in_provenance() {
        let interp = "a".repeat(64);
        let (cr, ev) = make_cell_write(b"hello", std::slice::from_ref(&interp));
        let events = vec![(0u64, ev.clone()), (5u64, ev.clone())];
        let cell = regenerate(&events, &cr).expect("should succeed");
        assert_eq!(cell.payload, b"hello".to_vec());
        assert_eq!(cell.interpreters, vec![interp]);
        assert_eq!(
            cell.provenance.len(),
            2,
            "both events must appear in provenance"
        );
        assert_eq!(cell.provenance[0], ev);
        assert_eq!(cell.provenance[1], ev);
    }

    /// Two `InterpreterOutput` events sharing the same stored `cell_ref`:
    /// later event's `output` wins (last-write-wins on payload).
    #[test]
    fn regenerate_two_interpreter_outputs_last_payload_wins() {
        // Synthetic stored ref — both events claim to define the same cell.
        let the_ref = "a".repeat(64);
        let ev1 = Event::InterpreterOutput {
            cell_ref: the_ref.clone(),
            model: "gpt-4o".into(),
            input_hash: "b".repeat(64),
            tool_results_consumed: vec![],
            output: b"first-output".to_vec(),
        };
        let ev2 = Event::InterpreterOutput {
            cell_ref: the_ref.clone(),
            model: "gpt-4o".into(),
            input_hash: "b".repeat(64),
            tool_results_consumed: vec![],
            output: b"second-output".to_vec(),
        };
        // ev2 has higher SeqPos → its payload wins.
        let events = vec![(0u64, ev1.clone()), (5u64, ev2.clone())];
        let cell = regenerate(&events, &the_ref).expect("should succeed");
        assert_eq!(
            cell.payload,
            b"second-output".to_vec(),
            "last-write-wins on payload"
        );
        assert_eq!(
            cell.provenance,
            vec![ev1, ev2],
            "both events in provenance, in order"
        );
    }

    /// `CellWrite` at SeqPos 0, `InterpreterOutput` with same `cell_ref` at SeqPos 5:
    /// later `InterpreterOutput.output` replaces payload; `interpreters` unchanged.
    #[test]
    fn regenerate_interpreter_output_after_cell_write_wins_payload_only() {
        let (cr, write) = make_cell_write(b"original", &[]);
        let interp = Event::InterpreterOutput {
            cell_ref: cr.clone(),
            model: "gpt-4o".into(),
            input_hash: "b".repeat(64),
            tool_results_consumed: vec![],
            output: b"derived".to_vec(),
        };
        let events = vec![(0u64, write.clone()), (5u64, interp.clone())];
        let cell = regenerate(&events, &cr).expect("should succeed");
        assert_eq!(
            cell.payload,
            b"derived".to_vec(),
            "InterpreterOutput wins on payload"
        );
        // interpreters come from the CellWrite; InterpreterOutput must NOT override them.
        assert_eq!(cell.interpreters, Vec::<Hash>::new());
        assert_eq!(cell.provenance, vec![write, interp]);
    }

    // -- provenance ordering --------------------------------------------------

    /// Events in provenance are in ascending `SeqPos` order regardless of input order.
    #[test]
    fn regenerate_provenance_is_ordered_by_seqpos() {
        let (cr, write) = make_cell_write(b"datum", &[]);
        let rel = Event::Relationship {
            from_ref: cr.clone(),
            to_ref: "f".repeat(64),
            type_: "ref".into(),
        };
        // Supply in REVERSE SeqPos order: relationship (SeqPos 10) before write (SeqPos 0).
        let events = vec![(10u64, rel.clone()), (0u64, write.clone())];
        let cell = regenerate(&events, &cr).expect("should succeed");
        assert_eq!(
            cell.provenance,
            vec![write, rel],
            "provenance must be ordered by SeqPos ascending"
        );
    }

    // -- Relationship semantics -----------------------------------------------

    /// `RelationshipEvent` amends the SOURCE ref's provenance only; does NOT change payload.
    ///
    /// Mirrors: `test_relationship_event_amends_from_ref_only` in Python.
    #[test]
    fn regenerate_relationship_amends_source_ref_only() {
        let (a_ref, a_write) = make_cell_write(b"cell A", &[]);
        let (b_ref, b_write) = make_cell_write(b"cell B", &[]);
        let edge = Event::Relationship {
            from_ref: a_ref.clone(),
            to_ref: b_ref.clone(),
            type_: "links_to".into(),
        };
        let events = vec![
            (0u64, a_write.clone()),
            (1u64, b_write.clone()),
            (2u64, edge.clone()),
        ];

        // Cell A: write + edge in provenance; payload unchanged.
        let cell_a = regenerate(&events, &a_ref).expect("cell A");
        assert_eq!(
            cell_a.payload,
            b"cell A".to_vec(),
            "edge must NOT change payload"
        );
        assert_eq!(
            cell_a.provenance,
            vec![a_write.clone(), edge.clone()],
            "edge must appear in A's provenance"
        );

        // Cell B: only b_write in provenance; edge NOT included.
        let cell_b = regenerate(&events, &b_ref).expect("cell B");
        assert_eq!(cell_b.payload, b"cell B".to_vec());
        assert_eq!(
            cell_b.provenance,
            vec![b_write],
            "edge must NOT appear in B's provenance (to_ref is never amended)"
        );
    }

    // -- Access / ToolResult are never selected -------------------------------

    /// `Access` has `target_ref == None` and must never appear in the selection.
    #[test]
    fn regenerate_access_event_is_never_selected() {
        let (cr, write) = make_cell_write(b"payload", &[]);
        // Access stores the same cell_ref, but target_ref(Access) == None → excluded.
        let access = Event::Access {
            cell_ref: cr.clone(),
        };
        let events = vec![(0u64, write.clone()), (5u64, access)];
        let cell = regenerate(&events, &cr).expect("should succeed");
        assert_eq!(
            cell.provenance,
            vec![write],
            "Access must not appear in provenance"
        );
    }

    /// `ToolResult` has `target_ref == None` and must never appear in the selection.
    #[test]
    fn regenerate_tool_result_is_never_selected() {
        let (cr, write) = make_cell_write(b"payload", &[]);
        let tool = Event::ToolResult {
            output: b"tool-out".to_vec(),
        };
        let events = vec![(0u64, write.clone()), (1u64, tool)];
        let cell = regenerate(&events, &cr).expect("should succeed");
        assert_eq!(
            cell.provenance,
            vec![write],
            "ToolResult must not appear in provenance"
        );
    }

    // -- MissingDefiningEvent struct ------------------------------------------

    /// The error carries the `cell_ref` that was requested.
    #[test]
    fn missing_defining_event_carries_cell_ref() {
        let wanted = "deadbeef".repeat(8); // 64 chars
        let err = MissingDefiningEvent {
            cell_ref: wanted.clone(),
        };
        assert_eq!(err.cell_ref, wanted);
    }

    /// `MissingDefiningEvent` implements `std::error::Error`.
    #[test]
    fn missing_defining_event_is_std_error() {
        let err = MissingDefiningEvent {
            cell_ref: "0".repeat(64),
        };
        // Coerce to &dyn std::error::Error to confirm trait bound.
        let _: &dyn std::error::Error = &err;
    }

    /// `Display` output includes the cell_ref for diagnostics.
    #[test]
    fn missing_defining_event_display_contains_cell_ref() {
        let cr = "abc123".to_string();
        let err = MissingDefiningEvent {
            cell_ref: cr.clone(),
        };
        let msg = format!("{err}");
        assert!(msg.contains(&cr), "Display must include the cell_ref");
    }
}

#[cfg(test)]
mod target {
    use super::*;
    use crate::models::cell_write_ref;

    // -- target_ref -----------------------------------------------------------

    /// `target_ref(CellWrite)` must equal `cell_write_ref(payload, interpreters)`.
    ///
    /// Mirrors: `target_ref(CellWriteEvent) == event.cell_ref()` in Python.
    #[test]
    fn target_ref_cell_write_returns_cell_write_ref() {
        let payload = b"hello";
        let interpreters = vec![];
        let ev = Event::CellWrite {
            payload: payload.to_vec(),
            interpreters: interpreters.clone(),
        };
        let expected = cell_write_ref(payload, &interpreters);
        assert_eq!(
            target_ref(&ev),
            Some(expected),
            "CellWrite target_ref must equal cell_write_ref(payload, interpreters)",
        );
    }

    /// `target_ref(CellWrite)` respects non-empty interpreter list.
    #[test]
    fn target_ref_cell_write_with_interpreters() {
        let payload = b"world";
        let interp = "a".repeat(64);
        let interpreters = vec![interp.clone()];
        let ev = Event::CellWrite {
            payload: payload.to_vec(),
            interpreters: interpreters.clone(),
        };
        let expected = cell_write_ref(payload, &interpreters);
        assert_eq!(target_ref(&ev), Some(expected));
    }

    /// `target_ref(InterpreterOutput)` returns the stored `cell_ref` field,
    /// NOT a recomputed hash.
    ///
    /// Mirrors: `target_ref(InterpreterOutputEvent) == event.cell_ref` in Python.
    #[test]
    fn target_ref_interpreter_output_returns_stored_cell_ref() {
        let stored = "b".repeat(64);
        let ev = Event::InterpreterOutput {
            cell_ref: stored.clone(),
            model: "gpt-4o".into(),
            input_hash: "c".repeat(64),
            tool_results_consumed: vec![],
            output: b"result".to_vec(),
        };
        assert_eq!(
            target_ref(&ev),
            Some(stored),
            "InterpreterOutput target_ref must return the stored cell_ref field",
        );
    }

    /// `target_ref(IndexCell)` returns the stored `cell_ref` field.
    ///
    /// Mirrors: `target_ref(IndexCellEvent) == event.cell_ref` in Python.
    #[test]
    fn target_ref_index_cell_returns_stored_cell_ref() {
        let stored = "d".repeat(64);
        let ev = Event::IndexCell {
            cell_ref: stored.clone(),
            access_summary: b"summary".to_vec(),
            output: b"index".to_vec(),
        };
        assert_eq!(
            target_ref(&ev),
            Some(stored),
            "IndexCell target_ref must return the stored cell_ref field",
        );
    }

    /// `target_ref(Relationship)` returns `from_ref` (edge lands in the
    /// source cell's provenance only).
    ///
    /// Mirrors: `target_ref(RelationshipEvent) == event.from_ref` in Python.
    #[test]
    fn target_ref_relationship_returns_from_ref() {
        let from = "e".repeat(64);
        let to = "f".repeat(64);
        let ev = Event::Relationship {
            from_ref: from.clone(),
            to_ref: to,
            type_: "uses".into(),
        };
        assert_eq!(
            target_ref(&ev),
            Some(from),
            "Relationship target_ref must return from_ref, not to_ref",
        );
    }

    /// `target_ref(ToolResult)` returns `None`.
    ///
    /// Mirrors: Python's `target_ref` falls through to `return None` for
    /// `ToolResultEvent`.
    #[test]
    fn target_ref_tool_result_is_none() {
        let ev = Event::ToolResult {
            output: b"raw".to_vec(),
        };
        assert_eq!(
            target_ref(&ev),
            None,
            "ToolResult has no provenance target — must return None",
        );
    }

    /// `target_ref(Access)` returns `None`.
    ///
    /// Mirrors: Python's `target_ref` falls through to `return None` for
    /// `AccessEvent`.
    #[test]
    fn target_ref_access_is_none() {
        let ev = Event::Access {
            cell_ref: "0".repeat(64),
        };
        assert_eq!(
            target_ref(&ev),
            None,
            "Access is provenance-neutral — must return None",
        );
    }

    // -- is_defining ----------------------------------------------------------

    /// `is_defining(CellWrite, cell_ref)` is `true` when the hash matches.
    ///
    /// Mirrors: `is_defining(CellWriteEvent, cell_ref)` == `True` in Python.
    #[test]
    fn is_defining_cell_write_match_is_true() {
        let payload = b"hello";
        let interpreters: Vec<Hash> = vec![];
        let the_ref = cell_write_ref(payload, &interpreters);
        let ev = Event::CellWrite {
            payload: payload.to_vec(),
            interpreters: interpreters.clone(),
        };
        assert!(
            is_defining(&ev, &the_ref),
            "CellWrite is_defining must be true when hash matches",
        );
    }

    /// `is_defining(CellWrite, other_ref)` is `false` when the hash differs.
    #[test]
    fn is_defining_cell_write_no_match_is_false() {
        let payload = b"hello";
        let interpreters: Vec<Hash> = vec![];
        let wrong_ref = "0".repeat(64);
        let ev = Event::CellWrite {
            payload: payload.to_vec(),
            interpreters,
        };
        assert!(
            !is_defining(&ev, &wrong_ref),
            "CellWrite is_defining must be false when hash differs",
        );
    }

    /// `is_defining(InterpreterOutput, cell_ref)` is `true` when stored
    /// `cell_ref` matches the argument.
    ///
    /// Mirrors Python: reads stored field, no recompute.
    #[test]
    fn is_defining_interpreter_output_match_is_true() {
        let stored = "b".repeat(64);
        let ev = Event::InterpreterOutput {
            cell_ref: stored.clone(),
            model: "gpt-4o".into(),
            input_hash: "c".repeat(64),
            tool_results_consumed: vec![],
            output: vec![],
        };
        assert!(
            is_defining(&ev, &stored),
            "InterpreterOutput is_defining must be true when stored cell_ref matches",
        );
    }

    /// `is_defining(InterpreterOutput, other_ref)` is `false`.
    #[test]
    fn is_defining_interpreter_output_no_match_is_false() {
        let stored = "b".repeat(64);
        let other = "c".repeat(64);
        let ev = Event::InterpreterOutput {
            cell_ref: stored,
            model: "gpt-4o".into(),
            input_hash: "d".repeat(64),
            tool_results_consumed: vec![],
            output: vec![],
        };
        assert!(
            !is_defining(&ev, &other),
            "InterpreterOutput is_defining must be false when stored cell_ref differs",
        );
    }

    /// `is_defining(IndexCell, cell_ref)` is `true` when stored `cell_ref`
    /// matches the argument.
    #[test]
    fn is_defining_index_cell_match_is_true() {
        let stored = "d".repeat(64);
        let ev = Event::IndexCell {
            cell_ref: stored.clone(),
            access_summary: b"s".to_vec(),
            output: b"o".to_vec(),
        };
        assert!(
            is_defining(&ev, &stored),
            "IndexCell is_defining must be true when stored cell_ref matches",
        );
    }

    /// `is_defining(IndexCell, other_ref)` is `false`.
    #[test]
    fn is_defining_index_cell_no_match_is_false() {
        let stored = "d".repeat(64);
        let other = "e".repeat(64);
        let ev = Event::IndexCell {
            cell_ref: stored,
            access_summary: vec![],
            output: vec![],
        };
        assert!(
            !is_defining(&ev, &other),
            "IndexCell is_defining must be false when stored cell_ref differs",
        );
    }

    /// `is_defining(Relationship, …)` is always `false`.
    ///
    /// Relationships land in provenance but do not define a cell.
    #[test]
    fn is_defining_relationship_is_always_false() {
        let from = "e".repeat(64);
        let ev = Event::Relationship {
            from_ref: from.clone(),
            to_ref: "f".repeat(64),
            type_: "uses".into(),
        };
        assert!(
            !is_defining(&ev, &from),
            "Relationship never defines a cell, even matching from_ref",
        );
    }

    /// `is_defining(ToolResult, …)` is always `false`.
    #[test]
    fn is_defining_tool_result_is_always_false() {
        let ev = Event::ToolResult {
            output: b"raw".to_vec(),
        };
        // Any ref string — should still be false.
        assert!(
            !is_defining(&ev, &"0".repeat(64)),
            "ToolResult never defines a cell",
        );
    }

    /// `is_defining(Access, …)` is always `false`.
    #[test]
    fn is_defining_access_is_always_false() {
        let cell_ref = "0".repeat(64);
        let ev = Event::Access {
            cell_ref: cell_ref.clone(),
        };
        assert!(
            !is_defining(&ev, &cell_ref),
            "Access is provenance-neutral and never defines a cell",
        );
    }
}

// ── read_log tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod read_log_tests {
    use super::read_log;
    use crate::codec::{encode_event, write_frame};
    use crate::models::Event;
    use std::io::Cursor;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn access_event() -> Event {
        Event::Access {
            cell_ref: "a".repeat(64),
        }
    }

    fn tool_result_event() -> Event {
        Event::ToolResult {
            output: b"test output bytes".to_vec(),
        }
    }

    fn relationship_event() -> Event {
        Event::Relationship {
            from_ref: "f".repeat(64),
            to_ref: "t".repeat(64),
            type_: "depends_on".to_string(),
        }
    }

    // ── basic correctness ────────────────────────────────────────────────────

    /// Empty reader returns Ok([]).
    #[test]
    fn read_log_empty_reader_returns_empty_vec() {
        let events = read_log(Cursor::new(vec![])).expect("empty log must not error");
        assert!(events.is_empty(), "empty log must yield zero events");
    }

    /// A single framed record round-trips through write_frame + read_log.
    #[test]
    fn read_log_single_event_round_trips() {
        let ev = access_event();
        let mut buf = Vec::new();
        write_frame(&mut buf, &ev).expect("write_frame must succeed");

        let events = read_log(Cursor::new(buf)).expect("read_log must succeed");
        assert_eq!(events, vec![ev], "single framed record must round-trip");
    }

    /// Multiple framed records all round-trip in insertion order.
    #[test]
    fn read_log_multiple_events_round_trip_in_order() {
        let ev1 = access_event();
        let ev2 = tool_result_event();
        let ev3 = relationship_event();

        let mut buf = Vec::new();
        write_frame(&mut buf, &ev1).expect("write_frame ev1");
        write_frame(&mut buf, &ev2).expect("write_frame ev2");
        write_frame(&mut buf, &ev3).expect("write_frame ev3");

        let events = read_log(Cursor::new(buf)).expect("read_log must succeed");
        assert_eq!(
            events,
            vec![ev1, ev2, ev3],
            "events must come back in insertion order"
        );
    }

    /// All six Event variants survive a write_frame + read_log round-trip.
    #[test]
    fn read_log_all_event_variants_round_trip() {
        let variants: Vec<Event> = vec![
            Event::CellWrite {
                payload: b"p".to_vec(),
                interpreters: vec!["a".repeat(64)],
            },
            Event::Relationship {
                from_ref: "b".repeat(64),
                to_ref: "c".repeat(64),
                type_: "uses".to_string(),
            },
            Event::ToolResult {
                output: b"raw".to_vec(),
            },
            Event::InterpreterOutput {
                cell_ref: "d".repeat(64),
                model: "gpt-4o".to_string(),
                input_hash: "e".repeat(64),
                tool_results_consumed: vec!["f".repeat(64)],
                output: b"derived".to_vec(),
            },
            Event::Access {
                cell_ref: "g".repeat(64),
            },
            Event::IndexCell {
                cell_ref: "h".repeat(64),
                access_summary: b"summary".to_vec(),
                output: b"index".to_vec(),
            },
        ];

        let mut buf = Vec::new();
        for ev in &variants {
            write_frame(&mut buf, ev).expect("write_frame");
        }

        let recovered = read_log(Cursor::new(buf)).expect("read_log must succeed");
        assert_eq!(recovered, variants, "all six variants must round-trip");
    }

    // ── torn-tail tolerance ──────────────────────────────────────────────────

    /// Core torn-tail test: a log whose final record is deliberately truncated
    /// (cut in the middle of the body) returns all preceding complete records
    /// without error.  The torn record is silently discarded.
    #[test]
    fn read_log_torn_tail_truncated_body_is_discarded() {
        let ev1 = access_event();
        let ev2 = tool_result_event(); // this one gets torn

        // Build a log where ev2 is written only halfway.
        let mut buf = Vec::new();
        write_frame(&mut buf, &ev1).expect("write_frame ev1");
        let ev2_bytes = encode_event(&ev2);
        let truncated = ev2_bytes.len() / 2; // keep first half only
        buf.extend_from_slice(&ev2_bytes[..truncated]);

        let events = read_log(Cursor::new(buf)).expect("torn tail must NOT return Err");
        assert_eq!(
            events,
            vec![ev1],
            "torn final record must be silently discarded"
        );
    }

    /// Torn tail: only 2 of the 4 leading-length bytes are present for the
    /// final record.  Preceding complete records are returned; the partial
    /// leading length is discarded without error.
    #[test]
    fn read_log_torn_tail_partial_leading_length_is_discarded() {
        let ev1 = access_event();

        let mut buf = Vec::new();
        write_frame(&mut buf, &ev1).expect("write_frame ev1");
        buf.extend_from_slice(&[0x05u8, 0x00]); // only 2 of 4 leading-length bytes

        let events = read_log(Cursor::new(buf)).expect("partial leading length must NOT error");
        assert_eq!(events, vec![ev1]);
    }

    /// Torn tail: leading length + body are complete but the 4-byte trailing
    /// length is truncated (only 2 bytes present).  Preceding complete records
    /// are returned; the incomplete record is discarded without error.
    #[test]
    fn read_log_torn_tail_partial_trailing_length_is_discarded() {
        let ev1 = access_event();
        let ev2 = tool_result_event();

        let mut buf = Vec::new();
        write_frame(&mut buf, &ev1).expect("write_frame ev1");
        // Write ev2 but shave the last 2 bytes off the trailing length.
        let ev2_bytes = encode_event(&ev2);
        buf.extend_from_slice(&ev2_bytes[..ev2_bytes.len() - 2]);

        let events = read_log(Cursor::new(buf)).expect("partial trailing length must NOT error");
        assert_eq!(events, vec![ev1]);
    }

    /// Torn tail at the start: the very first record is truncated.
    /// The result must be Ok([]) — zero events, no error.
    #[test]
    fn read_log_torn_tail_only_record_is_truncated_returns_empty() {
        let ev = access_event();
        let bytes = encode_event(&ev);
        // Write only the first 5 bytes (past the 4-byte length but missing most of body).
        let buf = bytes[..5].to_vec();

        let events = read_log(Cursor::new(buf)).expect("single torn record must NOT error");
        assert!(
            events.is_empty(),
            "no complete records → must return empty vec"
        );
    }

    /// Many complete records followed by a torn tail: all N complete records
    /// are recovered; the torn final record is dropped.
    #[test]
    fn read_log_many_complete_records_then_torn_tail() {
        let evs: Vec<Event> = (0u8..5)
            .map(|i| Event::ToolResult {
                output: vec![i; 10],
            })
            .collect();
        let torn = Event::Access {
            cell_ref: "z".repeat(64),
        };

        let mut buf = Vec::new();
        for ev in &evs {
            write_frame(&mut buf, ev).expect("write_frame");
        }
        // Append a torn version of the final record (strip the last 10 bytes).
        let torn_bytes = encode_event(&torn);
        let keep = torn_bytes.len().saturating_sub(10);
        buf.extend_from_slice(&torn_bytes[..keep]);

        let events = read_log(Cursor::new(buf)).expect("torn tail must NOT error");
        assert_eq!(
            events, evs,
            "all 5 complete records recovered; torn record dropped"
        );
    }
}
