//! Killer-gate integration tests: durability guarantees at the cargo layer.
//!
//! # What these tests prove
//!
//! Unit tests in `file_kernel.rs` verify individual operations in isolation.
//! These gates prove the **end-to-end durability story**: a complete write
//! session followed by a process restart (simulated by drop + reopen) produces
//! state that is byte-identical to the pre-restart state.
//!
//! # TDD note
//!
//! Each gate was written BEFORE its implementation was wired. Before
//! `FileKernel::open` called `read_log` to replay the on-disk log, gate E1
//! would fail immediately at the first assertion:
//!
//! ```text
//! assertion `left == right` failed: gate_e1 E1: event count must match after restart
//!   left: 7
//!  right: 0   ← empty kernel returned by un-replayed open
//! ```
//!
//! After `FileKernel::open` was wired to replay the log, the gate passes.
//!
//! # Naming convention
//!
//! Gates are named `gate_<id>_<property>` where `<id>` is the durability
//! requirement identifier (E1, E2, ...) and `<property>` describes what the
//! gate proves.

use amplifier_data_core::codec::encode_event;
use amplifier_data_core::file_kernel::FileKernel;
use amplifier_data_core::kernel::Resolved;
use amplifier_data_core::models::{
    cell_write_ref, index_derive_ref, interpreter_derive_ref, tool_result_hash, Event, SeqPos,
};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

/// Build a unique temporary file path for a gate test.
///
/// Uses `suffix` (unique per test function) plus the process ID to avoid
/// collisions when the test binary runs in parallel with other instances.
fn tmp_path(suffix: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ad_gate_{}_{}", suffix, std::process::id()))
}

/// RAII guard: remove the file when dropped, even if the test panics.
///
/// Keeps the test environment clean without requiring explicit cleanup in
/// the test body.
struct DeferRemove(PathBuf);

impl Drop for DeferRemove {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0); // best-effort; ignore missing-file error
    }
}

// ---------------------------------------------------------------------------
// Event sequence factory
// ---------------------------------------------------------------------------

/// Content-addressed refs produced by [`build_full_event_sequence`].
///
/// Carried alongside the event vector so each test can call `resolve()` on
/// the correct refs without recomputing them.
struct FullEventRefs {
    /// `CellWrite { payload: b"alpha payload", interpreters: [] }`
    cell_ref_a: String,
    /// `CellWrite { payload: b"beta payload", interpreters: [interp_hash] }`
    cell_ref_b: String,
    /// `tool_result_hash(b"tool result bytes")`
    tool_hash_d: String,
    /// `interpreter_derive_ref("gpt-4o", fake_input_hash, [tool_hash_d], b"derived")`
    derived_ref_e: String,
    /// `index_derive_ref(b"access-summary", b"index-output")`
    index_ref_g: String,
}

/// Construct a comprehensive event sequence that covers all six `Event` variants.
///
/// Returns the event vector alongside the pre-computed content-addressed refs
/// so callers can call `resolve()` after restart and verify the correct values.
///
/// Variant coverage:
///
/// | Pos | Variant            | Notes                                          |
/// |-----|--------------------|------------------------------------------------|
/// | 0   | `CellWrite`        | no interpreters; defines `cell_ref_a`          |
/// | 1   | `CellWrite`        | with one interpreter; defines `cell_ref_b`     |
/// | 2   | `Relationship`     | `from_ref = cell_ref_a`; amends its provenance |
/// | 3   | `ToolResult`       | content-addressed by `sha256(output)`          |
/// | 4   | `InterpreterOutput`| consumes `tool_hash_d`; defines `derived_ref_e`|
/// | 5   | `Access`           | provenance-neutral; `target_ref` is `None`     |
/// | 6   | `IndexCell`        | defines `index_ref_g`                          |
fn build_full_event_sequence() -> (Vec<Event>, FullEventRefs) {
    let interp_hash = "a".repeat(64);
    let fake_input_hash = "0".repeat(64);

    // --- Event A (pos 0): CellWrite, no interpreters ---
    let cell_ref_a = cell_write_ref(b"alpha payload", &[]);
    let ev_a = Event::CellWrite {
        payload: b"alpha payload".to_vec(),
        interpreters: vec![],
    };

    // --- Event B (pos 1): CellWrite, one interpreter ---
    let cell_ref_b = cell_write_ref(b"beta payload", std::slice::from_ref(&interp_hash));
    let ev_b = Event::CellWrite {
        payload: b"beta payload".to_vec(),
        interpreters: vec![interp_hash.clone()],
    };

    // --- Event C (pos 2): Relationship from_ref=cell_ref_a ---
    // Amends cell_ref_a's provenance; does not appear in cell_ref_b's provenance.
    let ev_c = Event::Relationship {
        from_ref: cell_ref_a.clone(),
        to_ref: cell_ref_b.clone(),
        type_: "references".to_string(),
    };

    // --- Event D (pos 3): ToolResult ---
    let ev_d = Event::ToolResult {
        output: b"tool result bytes".to_vec(),
    };
    let tool_hash_d = tool_result_hash(b"tool result bytes");

    // --- Event E (pos 4): InterpreterOutput, consumes tool_hash_d ---
    let derived_ref_e = interpreter_derive_ref(
        "gpt-4o",
        &fake_input_hash,
        std::slice::from_ref(&tool_hash_d),
        b"derived",
    );
    let ev_e = Event::InterpreterOutput {
        cell_ref: derived_ref_e.clone(),
        model: "gpt-4o".to_string(),
        input_hash: fake_input_hash.clone(),
        tool_results_consumed: vec![tool_hash_d.clone()],
        output: b"derived".to_vec(),
    };

    // --- Event F (pos 5): Access (provenance-neutral) ---
    let ev_f = Event::Access {
        cell_ref: cell_ref_a.clone(),
    };

    // --- Event G (pos 6): IndexCell ---
    let index_ref_g = index_derive_ref(b"access-summary", b"index-output");
    let ev_g = Event::IndexCell {
        cell_ref: index_ref_g.clone(),
        access_summary: b"access-summary".to_vec(),
        output: b"index-output".to_vec(),
    };

    let events = vec![ev_a, ev_b, ev_c, ev_d, ev_e, ev_f, ev_g];
    let refs = FullEventRefs {
        cell_ref_a,
        cell_ref_b,
        tool_hash_d,
        derived_ref_e,
        index_ref_g,
    };
    (events, refs)
}

// ---------------------------------------------------------------------------
// Gate E1: restart-survival with byte-identical regeneration
// ---------------------------------------------------------------------------

/// **E1 gate** — restart-survival with byte-identical regeneration.
///
/// Appends a comprehensive event sequence (all 6 Event variants) through
/// `FileKernel`, drops/closes it to simulate a process exit, then reopens
/// the same path into a fresh `FileKernel` to simulate a process restart.
///
/// Asserts that the regenerated state is byte-identical / value-equivalent to
/// the pre-restart state:
///
/// 1. **Event sequence identity** — `all_events()` returns the same
///    `(SeqPos, Event)` pairs after restart.
/// 2. **Log file byte-identity** — the raw on-disk bytes equal the exact
///    concatenation of `encode_event(ev)` for every event in the batch.
/// 3. **Resolve correctness** — every content-addressed ref yields the same
///    resolved `Cell` / `Event` value after restart, including correct
///    payloads, interpreter lists, and provenance structure.
///
/// # Why this gate would fail without durability wiring
///
/// If `FileKernel::open` did not call `read_log` to replay the log, the
/// reopened kernel would be empty. The first assertion in session 2 would
/// then fail:
///
/// ```text
/// gate_e1 E1: event count must match after restart
///   left: 7  right: 0
/// ```
#[test]
fn gate_e1_restart_survival_byte_identical_regeneration() {
    let path = tmp_path("gate_e1");
    let _cleanup = DeferRemove(path.clone());

    let (events, refs) = build_full_event_sequence();

    // -----------------------------------------------------------------------
    // Session 1: write all events in one atomic batch, then close.
    //
    // This simulates a process that writes events and then exits cleanly.
    // -----------------------------------------------------------------------
    {
        let mut k = FileKernel::open(&path).expect("gate_e1 session 1: open must succeed");

        // Append all 7 events in one batch.
        let positions = k
            .append(events.clone())
            .expect("gate_e1 session 1: append must succeed");

        // SeqPos assignments must be contiguous from 0.
        let expected_positions: Vec<SeqPos> = (0..events.len() as u64).collect();
        assert_eq!(
            positions, expected_positions,
            "gate_e1 session 1: SeqPos must be contiguous [0..N)"
        );

        // Sanity check: all resolvable refs work BEFORE close.
        k.resolve(&refs.cell_ref_a)
            .expect("gate_e1 session 1: cell_ref_a must be resolvable before close");
        k.resolve(&refs.cell_ref_b)
            .expect("gate_e1 session 1: cell_ref_b must be resolvable before close");
        k.resolve(&refs.derived_ref_e)
            .expect("gate_e1 session 1: derived_ref_e must be resolvable before close");
        k.resolve(&refs.index_ref_g)
            .expect("gate_e1 session 1: index_ref_g must be resolvable before close");
        k.resolve(&refs.tool_hash_d)
            .expect("gate_e1 session 1: tool_hash_d must be resolvable before close");

        // Close simulates process exit: flushes + syncs the file handle.
        k.close().expect("gate_e1 session 1: close must succeed");
    }
    // The `FileKernel` is now dropped; the OS file handle is released.

    // -----------------------------------------------------------------------
    // Byte-identity check: the raw log file must equal the concatenation of
    // encode_event(ev) for every event in the original batch.
    //
    // This verifies the wire format is deterministic and the file was not
    // corrupted during write or close.
    // -----------------------------------------------------------------------
    {
        let log_bytes = std::fs::read(&path).expect("gate_e1: log file must exist and be readable");

        let expected_log_bytes: Vec<u8> = events.iter().flat_map(encode_event).collect();

        assert_eq!(
            log_bytes, expected_log_bytes,
            "gate_e1 E1 (byte-identity): log file bytes must be the exact concatenation \
             of encode_event(ev) for each event in the original batch"
        );
    }

    // -----------------------------------------------------------------------
    // Session 2: reopen at the same path (simulates process restart).
    //
    // FileKernel::open must replay the on-disk log and reconstruct the full
    // in-memory state without any external assistance.
    // -----------------------------------------------------------------------
    {
        let mut k =
            FileKernel::open(&path).expect("gate_e1 session 2 (restart): open must succeed");

        // -------------------------------------------------------------------
        // E1 core: event sequence is value-identical after restart.
        //
        // Each (SeqPos, Event) pair must match the original exactly.
        // -------------------------------------------------------------------
        let recovered_events = k.all_events();

        assert_eq!(
            recovered_events.len(),
            events.len(),
            "gate_e1 E1: event count must match after restart"
        );

        for (i, (orig_ev, (recovered_pos, recovered_ev))) in
            events.iter().zip(recovered_events.iter()).enumerate()
        {
            assert_eq!(
                *recovered_pos, i as SeqPos,
                "gate_e1 E1: event[{i}] SeqPos must be identical after restart"
            );
            assert_eq!(
                orig_ev, recovered_ev,
                "gate_e1 E1: event[{i}] value must be value-identical after restart"
            );
        }

        // -------------------------------------------------------------------
        // E1 resolve: cell_ref_a — CellWrite + Relationship in provenance.
        //
        // Payload must be b"alpha payload", interpreters must be empty,
        // and provenance must include both the CellWrite and the Relationship.
        // -------------------------------------------------------------------
        match k
            .resolve(&refs.cell_ref_a)
            .expect("gate_e1 E1: cell_ref_a must be resolvable after restart")
        {
            Resolved::Cell(cell) => {
                assert_eq!(
                    cell.payload,
                    b"alpha payload".to_vec(),
                    "gate_e1 E1: cell_ref_a payload must be byte-identical after restart"
                );
                assert_eq!(
                    cell.interpreters,
                    Vec::<String>::new(),
                    "gate_e1 E1: cell_ref_a interpreters must be empty after restart"
                );
                // Provenance = CellWrite (pos 0) + Relationship (pos 2)
                assert_eq!(
                    cell.provenance.len(),
                    2,
                    "gate_e1 E1: cell_ref_a provenance must have 2 entries \
                     (CellWrite + Relationship) after restart"
                );
            }
            Resolved::Event(_) => {
                panic!("gate_e1 E1: cell_ref_a must resolve to Resolved::Cell, got Event")
            }
        }

        // -------------------------------------------------------------------
        // E1 resolve: cell_ref_b — CellWrite only; interpreter preserved.
        // -------------------------------------------------------------------
        match k
            .resolve(&refs.cell_ref_b)
            .expect("gate_e1 E1: cell_ref_b must be resolvable after restart")
        {
            Resolved::Cell(cell) => {
                assert_eq!(
                    cell.payload,
                    b"beta payload".to_vec(),
                    "gate_e1 E1: cell_ref_b payload must be byte-identical after restart"
                );
                assert_eq!(
                    cell.interpreters,
                    vec!["a".repeat(64)],
                    "gate_e1 E1: cell_ref_b interpreters must be byte-identical after restart"
                );
                // Provenance = CellWrite (pos 1) only
                assert_eq!(
                    cell.provenance.len(),
                    1,
                    "gate_e1 E1: cell_ref_b provenance must have 1 entry (CellWrite) after restart"
                );
            }
            Resolved::Event(_) => {
                panic!("gate_e1 E1: cell_ref_b must resolve to Resolved::Cell, got Event")
            }
        }

        // -------------------------------------------------------------------
        // E1 resolve: derived_ref_e — InterpreterOutput; payload = b"derived".
        // -------------------------------------------------------------------
        match k
            .resolve(&refs.derived_ref_e)
            .expect("gate_e1 E1: derived_ref_e must be resolvable after restart")
        {
            Resolved::Cell(cell) => {
                assert_eq!(
                    cell.payload,
                    b"derived".to_vec(),
                    "gate_e1 E1: derived_ref_e payload must be byte-identical after restart"
                );
            }
            Resolved::Event(_) => {
                panic!("gate_e1 E1: derived_ref_e must resolve to Resolved::Cell, got Event")
            }
        }

        // -------------------------------------------------------------------
        // E1 resolve: index_ref_g — IndexCell; payload = b"index-output".
        // -------------------------------------------------------------------
        match k
            .resolve(&refs.index_ref_g)
            .expect("gate_e1 E1: index_ref_g must be resolvable after restart")
        {
            Resolved::Cell(cell) => {
                assert_eq!(
                    cell.payload,
                    b"index-output".to_vec(),
                    "gate_e1 E1: index_ref_g payload must be byte-identical after restart"
                );
            }
            Resolved::Event(_) => {
                panic!("gate_e1 E1: index_ref_g must resolve to Resolved::Cell, got Event")
            }
        }

        // -------------------------------------------------------------------
        // E1 resolve: tool_hash_d — ToolResult event by content-address hash.
        // -------------------------------------------------------------------
        match k
            .resolve(&refs.tool_hash_d)
            .expect("gate_e1 E1: tool_hash_d must be resolvable after restart")
        {
            Resolved::Event(ev) => {
                assert_eq!(
                    ev,
                    Event::ToolResult {
                        output: b"tool result bytes".to_vec()
                    },
                    "gate_e1 E1: tool result event must be value-identical after restart"
                );
            }
            Resolved::Cell(_) => {
                panic!("gate_e1 E1: tool_hash_d must resolve to Resolved::Event, got Cell")
            }
        }

        k.close()
            .expect("gate_e1 session 2 (restart): close must succeed");
    }
}

// ---------------------------------------------------------------------------
// Gate E2: torn-tail recovery at the cargo layer + clean re-append
//
// Simulates a crash mid-append: after a clean write session a partial
// (torn) record is injected directly into the on-disk file.  Reopening via
// `FileKernel` must silently discard the torn tail and recover all complete
// records (T2), and any append made immediately after recovery must itself
// survive a further reopen (T3).
//
// TDD note: this gate was written BEFORE `FileKernel::open` was wired to
// truncate the torn tail.  Without the truncation, T3 fails because the
// new event lands after the torn bytes and is never seen on the next reopen:
//
//   gate_e2 T3: all 4 events (3 recovered + 1 post-recovery) must survive
//               further reopen
//     left: 4   right: 3   ← event after torn bytes is invisible
// ---------------------------------------------------------------------------

/// **E2 gate** — torn-tail recovery + clean re-append at the cargo layer.
///
/// # What this gate proves
///
/// | Label | Property |
/// |-------|----------|
/// | T2    | `FileKernel::open` after a truncated final record recovers all complete records without error; torn tail silently discarded |
/// | T3    | A `FileKernel::append` immediately after torn-tail recovery produces a durable record that survives a further reopen |
///
/// # Failure mode without truncation on open
///
/// Without truncating the torn tail during `open`, subsequent appends land
/// *after* the torn bytes.  On the next reopen `read_log` stops at the
/// corrupted record, so the newly-appended valid event is invisible:
///
/// ```text
/// gate_e2 T3: all 4 events (3 recovered + 1 post-recovery) must survive further reopen
///   left: 4  right: 3
/// ```
#[test]
fn gate_e2_cargo_torn_tail_recovery_and_clean_reappend() {
    use std::io::Write as _;

    let path = tmp_path("gate_e2_torn_tail");
    let _cleanup = DeferRemove(path.clone());

    // -----------------------------------------------------------------------
    // Build the initial event set: 3 complete events that will form the
    // "good" baseline on disk.
    // -----------------------------------------------------------------------
    let ev_0 = Event::CellWrite {
        payload: b"payload-zero".to_vec(),
        interpreters: vec![],
    };
    let ev_1 = Event::CellWrite {
        payload: b"payload-one".to_vec(),
        interpreters: vec![],
    };
    let ev_2 = Event::ToolResult {
        output: b"tool-output-two".to_vec(),
    };
    let cell_ref_0 = cell_write_ref(b"payload-zero", &[]);
    let cell_ref_1 = cell_write_ref(b"payload-one", &[]);

    // ev_post is the event written after torn-tail recovery (T3).
    let ev_post = Event::CellWrite {
        payload: b"post-recovery-event".to_vec(),
        interpreters: vec![],
    };
    let cell_ref_post = cell_write_ref(b"post-recovery-event", &[]);

    // -----------------------------------------------------------------------
    // Session 1: write 3 complete events, then close cleanly.
    //
    // After this session the file contains exactly:
    //   encode_event(ev_0) || encode_event(ev_1) || encode_event(ev_2)
    // -----------------------------------------------------------------------
    {
        let mut k = FileKernel::open(&path).expect("gate_e2 session 1: open must succeed");
        k.append(vec![ev_0.clone(), ev_1.clone(), ev_2.clone()])
            .expect("gate_e2 session 1: append must succeed");
        k.close().expect("gate_e2 session 1: close must succeed");
    }

    // -----------------------------------------------------------------------
    // Inject a torn tail: append a partial record to the on-disk file.
    //
    // We take a well-formed encoded event and write only the first half of
    // its bytes — exactly what a crash mid-append would leave behind.
    // The partial bytes include the leading length and part of the body, but
    // the record is incomplete (trailing length is missing).
    // -----------------------------------------------------------------------
    let ev_torn = Event::CellWrite {
        payload: b"this-write-was-interrupted-mid-append".to_vec(),
        interpreters: vec![],
    };
    let torn_full_bytes = encode_event(&ev_torn);
    // Write only the first half of the encoded bytes — definitely incomplete.
    let torn_partial_len = torn_full_bytes.len() / 2;
    // Sanity: the partial write must include at least the 4-byte leading length.
    assert!(
        torn_partial_len >= 4,
        "gate_e2: partial must be ≥ 4 bytes to include the leading length field"
    );
    let torn_bytes = &torn_full_bytes[..torn_partial_len];
    {
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("gate_e2: open file for torn-write injection");
        f.write_all(torn_bytes)
            .expect("gate_e2: writing torn bytes must succeed");
        // Intentionally do NOT call sync_data — mirrors a crash that left
        // the bytes unfsynced.  The file is flushed to OS on drop (good
        // enough for test purposes; we just need the bytes present on disk).
    }

    // Verify the torn bytes are actually present on disk before proceeding.
    {
        let disk_len = std::fs::metadata(&path)
            .expect("gate_e2: file must exist after torn-write injection")
            .len();
        let clean_len: u64 = [&ev_0, &ev_1, &ev_2]
            .iter()
            .map(|ev| encode_event(ev).len() as u64)
            .sum();
        assert_eq!(
            disk_len,
            clean_len + torn_partial_len as u64,
            "gate_e2: on-disk file must contain the 3 clean records plus the torn partial bytes"
        );
    }

    // -----------------------------------------------------------------------
    // Session 2: reopen after torn tail.
    //
    // FileKernel::open must:
    //  (a) succeed — no error returned
    //  (b) recover all 3 complete events (T2)
    //  (c) truncate the torn bytes so the subsequent append lands at the
    //      correct offset and is recoverable on the next reopen (prerequisite
    //      for T3)
    // -----------------------------------------------------------------------
    {
        let mut k = FileKernel::open(&path)
            .expect("gate_e2 session 2 (torn-tail reopen): open must succeed — no error");

        // -----------------------------------------------------------------
        // T2: all complete records recovered; torn tail silently discarded.
        // -----------------------------------------------------------------
        let recovered = k.all_events();
        assert_eq!(
            recovered.len(),
            3,
            "gate_e2 T2: exactly 3 complete events must be recovered after torn-tail reopen"
        );

        assert_eq!(recovered[0].0, 0u64, "gate_e2 T2: ev_0 must have SeqPos 0");
        assert_eq!(recovered[1].0, 1u64, "gate_e2 T2: ev_1 must have SeqPos 1");
        assert_eq!(recovered[2].0, 2u64, "gate_e2 T2: ev_2 must have SeqPos 2");

        assert_eq!(
            recovered[0].1, ev_0,
            "gate_e2 T2: ev_0 must be value-identical after torn-tail recovery"
        );
        assert_eq!(
            recovered[1].1, ev_1,
            "gate_e2 T2: ev_1 must be value-identical after torn-tail recovery"
        );
        assert_eq!(
            recovered[2].1, ev_2,
            "gate_e2 T2: ev_2 must be value-identical after torn-tail recovery"
        );

        // Resolvable refs after recovery.
        k.resolve(&cell_ref_0)
            .expect("gate_e2 T2: cell_ref_0 must be resolvable after torn-tail recovery");
        k.resolve(&cell_ref_1)
            .expect("gate_e2 T2: cell_ref_1 must be resolvable after torn-tail recovery");

        // -----------------------------------------------------------------
        // T3 (write): append one new event immediately after recovery.
        //
        // This write must land at the correct offset (after the 3 clean
        // records, NOT after the torn bytes).  Correct placement is only
        // guaranteed if open() truncated the torn tail.
        // -----------------------------------------------------------------
        let positions = k
            .append(vec![ev_post.clone()])
            .expect("gate_e2 T3: append after torn-tail recovery must succeed");
        assert_eq!(
            positions,
            vec![3u64],
            "gate_e2 T3: post-recovery event must be assigned SeqPos 3"
        );

        // New event immediately visible in memory.
        k.resolve(&cell_ref_post)
            .expect("gate_e2 T3: cell_ref_post must be resolvable immediately after append");

        k.close().expect("gate_e2 session 2: close must succeed");
    }

    // -----------------------------------------------------------------------
    // T3 durability: reopen a third time and verify the post-recovery append
    // survives.
    //
    // Critical assertion: if the torn tail was NOT truncated before appending
    // in session 2, ev_post would have been written after the torn bytes.
    // read_log would stop at the torn record and ev_post would be invisible.
    // With correct truncation the log is clean and all 4 records are present.
    //
    // Byte-identity: the file must equal exactly the concatenation of the 4
    // valid encoded events — no torn bytes, no padding.
    // -----------------------------------------------------------------------
    {
        let mut k =
            FileKernel::open(&path).expect("gate_e2 T3 (further reopen): open must succeed");

        let all = k.all_events();
        assert_eq!(
            all.len(),
            4,
            "gate_e2 T3: all 4 events (3 recovered + 1 post-recovery) must survive further reopen"
        );

        assert_eq!(all[0].0, 0u64, "gate_e2 T3: ev_0 SeqPos must still be 0");
        assert_eq!(all[1].0, 1u64, "gate_e2 T3: ev_1 SeqPos must still be 1");
        assert_eq!(all[2].0, 2u64, "gate_e2 T3: ev_2 SeqPos must still be 2");
        assert_eq!(all[3].0, 3u64, "gate_e2 T3: ev_post SeqPos must be 3");

        assert_eq!(
            all[0].1, ev_0,
            "gate_e2 T3: ev_0 must be value-identical after further reopen"
        );
        assert_eq!(
            all[1].1, ev_1,
            "gate_e2 T3: ev_1 must be value-identical after further reopen"
        );
        assert_eq!(
            all[2].1, ev_2,
            "gate_e2 T3: ev_2 must be value-identical after further reopen"
        );
        assert_eq!(
            all[3].1, ev_post,
            "gate_e2 T3: ev_post must be value-identical after further reopen"
        );

        // Byte-identity: the log must be exactly the 4 clean encoded records.
        let expected_bytes: Vec<u8> = [&ev_0, &ev_1, &ev_2, &ev_post]
            .iter()
            .flat_map(|ev| encode_event(ev))
            .collect();
        let actual_bytes = std::fs::read(&path)
            .expect("gate_e2 T3: log file must be readable after further reopen");
        assert_eq!(
            actual_bytes, expected_bytes,
            "gate_e2 T3 (byte-identity): log file must equal the exact concatenation \
             of the 4 valid encoded events — no torn bytes, no padding"
        );

        // All refs resolvable after further reopen.
        k.resolve(&cell_ref_0)
            .expect("gate_e2 T3: cell_ref_0 must be resolvable after further reopen");
        k.resolve(&cell_ref_1)
            .expect("gate_e2 T3: cell_ref_1 must be resolvable after further reopen");
        k.resolve(&cell_ref_post)
            .expect("gate_e2 T3: cell_ref_post must be resolvable after further reopen");

        k.close()
            .expect("gate_e2 T3 (further reopen): close must succeed");
    }
}

// ---------------------------------------------------------------------------
// Gate E1-multi: restart-survival across multiple separate sessions
//
// A more realistic scenario: events arrive in three separate sessions
// (simulating three distinct process runs).  The final restart must recover
// all events from all three sessions.
// ---------------------------------------------------------------------------

/// **E1 multi-session gate** — events written across three separate sessions
/// must all survive the final restart.
///
/// This gate catches regressions where only the most-recent session's events
/// are replayed (e.g., the log was truncated or overwritten on each open).
#[test]
fn gate_e1_multi_session_restart_survival() {
    let path = tmp_path("gate_e1_multi");
    let _cleanup = DeferRemove(path.clone());

    // --- Three separate write sessions ---

    // Session A: write a CellWrite event
    let cell_ref_a = cell_write_ref(b"session-a", &[]);
    let ev_a = Event::CellWrite {
        payload: b"session-a".to_vec(),
        interpreters: vec![],
    };
    {
        let mut k = FileKernel::open(&path).expect("gate_e1_multi session A: open");
        k.append(vec![ev_a.clone()]).expect("session A: append");
        k.close().expect("session A: close");
    }

    // Session B: write a ToolResult event
    let ev_b = Event::ToolResult {
        output: b"session-b-tool".to_vec(),
    };
    let tool_hash_b = tool_result_hash(b"session-b-tool");
    {
        let mut k = FileKernel::open(&path).expect("gate_e1_multi session B: open");
        k.append(vec![ev_b.clone()]).expect("session B: append");
        k.close().expect("session B: close");
    }

    // Session C: write an Access event
    let ev_c = Event::Access {
        cell_ref: cell_ref_a.clone(),
    };
    {
        let mut k = FileKernel::open(&path).expect("gate_e1_multi session C: open");
        k.append(vec![ev_c.clone()]).expect("session C: append");
        k.close().expect("session C: close");
    }

    // --- Final restart: all three events must be present ---
    {
        let mut k =
            FileKernel::open(&path).expect("gate_e1_multi final restart: open must succeed");

        let recovered = k.all_events();

        assert_eq!(
            recovered.len(),
            3,
            "gate_e1_multi: all three events from three sessions must survive restart"
        );

        // SeqPos must be contiguous [0, 1, 2].
        assert_eq!(recovered[0].0, 0u64, "gate_e1_multi: ev_a SeqPos must be 0");
        assert_eq!(recovered[1].0, 1u64, "gate_e1_multi: ev_b SeqPos must be 1");
        assert_eq!(recovered[2].0, 2u64, "gate_e1_multi: ev_c SeqPos must be 2");

        // Event values must match.
        assert_eq!(
            recovered[0].1, ev_a,
            "gate_e1_multi: ev_a must be value-identical after restart"
        );
        assert_eq!(
            recovered[1].1, ev_b,
            "gate_e1_multi: ev_b must be value-identical after restart"
        );
        assert_eq!(
            recovered[2].1, ev_c,
            "gate_e1_multi: ev_c must be value-identical after restart"
        );

        // cell_ref_a and tool_hash_b must both be resolvable.
        k.resolve(&cell_ref_a)
            .expect("gate_e1_multi: cell_ref_a must be resolvable after final restart");
        k.resolve(&tool_hash_b)
            .expect("gate_e1_multi: tool_hash_b must be resolvable after final restart");

        // Byte-identity of the log: must equal enc(ev_a) || enc(ev_b) || enc(ev_c).
        let expected_bytes: Vec<u8> = [&ev_a, &ev_b, &ev_c]
            .iter()
            .flat_map(|ev| encode_event(ev))
            .collect();
        let actual_bytes = std::fs::read(&path).expect("gate_e1_multi: log file must be readable");
        assert_eq!(
            actual_bytes, expected_bytes,
            "gate_e1_multi: log file bytes must be byte-identical to the \
             concatenation of all three session's encoded events"
        );

        k.close()
            .expect("gate_e1_multi final restart: close must succeed");
    }
}
