//! In-memory storage kernel — content addressing, event log, resolve.
//!
//! Mirrors `amplifier_data.kernel.InMemoryKernel` in Python.

use std::collections::{HashMap, HashSet};

use crate::models::{tool_result_hash, Cell, Event, Hash, SeqPos};
use crate::replay::{is_defining, regenerate, target_ref};

// --- BatchError --------------------------------------------------------------

/// Error returned by [`InMemoryKernel::append_batch`] when the batch is invalid.
///
/// In the Rust core, the only validation failure is an empty batch — type-safety
/// ensures every item is an [`Event`].  The PyO3 binding adds a type-check layer
/// for the Python boundary (Phase 2).
///
/// Mirrors Python's `ValueError` raised by `append_batch([])`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum BatchError {
    /// The batch contained no events.
    EmptyBatch,
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchError::EmptyBatch => write!(f, "batch must contain at least one event"),
        }
    }
}

impl std::error::Error for BatchError {}

// --- NotFound ----------------------------------------------------------------

/// Requested ref was not found in the kernel's log or hash index.
///
/// Mirrors Python's `NotFound` exception.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct NotFound {
    pub ref_: Hash,
}

impl std::fmt::Display for NotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ref not found: {:?}", self.ref_)
    }
}

impl std::error::Error for NotFound {}

// --- Resolved ----------------------------------------------------------------

/// Successful result returned by [`InMemoryKernel::resolve`].
///
/// Mirrors the `Event | Cell` union returned by Python's `resolve()`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Resolved {
    /// A materialized (or re-folded) cell.
    Cell(Cell),
    /// A content-addressed event (currently only `ToolResult`).
    Event(Event),
}

// --- InMemoryKernel ----------------------------------------------------------

/// In-memory, append-only storage kernel.
///
/// Fields are private but accessible to the `#[cfg(test)]` block below,
/// matching Python's test seam (`k._log`, `k._cell_refs`, `k._by_hash`).
pub struct InMemoryKernel {
    log: Vec<Event>,
    cell_refs: HashSet<Hash>,
    by_hash: HashMap<Hash, Event>,
    materialized: HashMap<Hash, Cell>,
}

impl InMemoryKernel {
    pub fn new() -> Self {
        InMemoryKernel {
            log: Vec::new(),
            cell_refs: HashSet::new(),
            by_hash: HashMap::new(),
            materialized: HashMap::new(),
        }
    }

    /// Delegate `sha256(payload) -> 64-char lowercase hex` to `models::content_address`.
    ///
    /// Mirrors `InMemoryKernel.content_address` in Python.
    pub fn content_address(&self, payload: &[u8]) -> Hash {
        crate::models::content_address(payload)
    }

    /// Read accessor for the fold engine: returns `(SeqPos, Event)` pairs in
    /// insertion order.
    ///
    /// Mirrors `InMemoryKernel.all_events()` in Python:
    /// ```python
    /// return list(enumerate(self._log))
    /// ```
    pub fn all_events(&self) -> Vec<(SeqPos, Event)> {
        self.log
            .iter()
            .enumerate()
            .map(|(i, ev)| (i as SeqPos, ev.clone()))
            .collect()
    }

    /// Resolve a ref to a [`Cell`] or [`Event`].
    ///
    /// Resolution order (mirrors Python exactly):
    /// 1. Return cached `materialized` cell if present.
    /// 2. Re-fold from the log if `ref_` is in `cell_refs`; cache the result.
    /// 3. Return the content-addressed event from `by_hash` if present.
    /// 4. Return `Err(NotFound)`.
    pub fn resolve(&mut self, ref_: &str) -> Result<Resolved, NotFound> {
        // 1. Cached materialized cell — return immediately without re-folding.
        if let Some(cell) = self.materialized.get(ref_) {
            return Ok(Resolved::Cell(cell.clone()));
        }
        // 2. Known defining ref — fold from the log and cache the result.
        if self.cell_refs.contains(ref_) {
            let all = self.all_events();
            let cell = regenerate(&all, ref_)
                .expect("cell_refs invariant: defining event must be present in log");
            self.materialized.insert(ref_.to_string(), cell.clone());
            return Ok(Resolved::Cell(cell));
        }
        // 3. Content-addressed event (e.g. ToolResult) stored by its hash.
        if let Some(ev) = self.by_hash.get(ref_) {
            return Ok(Resolved::Event(ev.clone()));
        }
        // 4. Unknown ref.
        Err(NotFound {
            ref_: ref_.to_string(),
        })
    }

    /// Append a validated, atomic batch of events to the log.
    ///
    /// # Behaviour
    ///
    /// 1. **Validate** — the batch must be non-empty; returns `Err(BatchError::EmptyBatch)`
    ///    otherwise.  In the Rust core, the type `Vec<Event>` guarantees every item is
    ///    an `Event`; the PyO3 binding adds a Python-side type check (Phase 2).
    /// 2. **Assign positions** — each event gets the next available `SeqPos`, starting
    ///    from `log.len()` before the append.
    /// 3. **Update indexes** (all or nothing — mutation happens after validation):
    ///    - For every event that has a [`target_ref`](crate::replay::target_ref):
    ///      - If it is also a *defining* event, insert its `target_ref` into `cell_refs`.
    ///      - Evict any stale entry in `materialized` for that `target_ref` so the next
    ///        `resolve()` call re-folds from the updated log.
    ///    - For `Event::ToolResult`, index the event in `by_hash` under `sha256(output)`.
    /// 4. **Extend the log** with the batch.
    /// 5. **Return** the assigned positions.
    ///
    /// Observer notification is the PyO3 binding's responsibility (Task 11), not the
    /// core kernel's — this method does **not** call any observers.
    ///
    /// Mirrors `InMemoryKernel.append_batch()` in Python.
    pub fn append_batch(&mut self, events: Vec<Event>) -> Result<Vec<SeqPos>, BatchError> {
        // Step 1: Validate.
        if events.is_empty() {
            return Err(BatchError::EmptyBatch);
        }

        // Step 2: Compute the positions that will be assigned.
        let start = self.log.len() as SeqPos;
        let positions: Vec<SeqPos> = (0..events.len() as u64).map(|i| start + i).collect();

        // Step 3: Update indexes — runs AFTER validation so a bad batch commits nothing.
        for ev in &events {
            // For events that target a cell_ref: update cell_refs and evict materialized.
            if let Some(tr) = target_ref(ev) {
                // Defining events establish the cell identity — add to cell_refs.
                if is_defining(ev, &tr) {
                    self.cell_refs.insert(tr.clone());
                }
                // Any event targeting a ref makes the cached view stale.
                self.materialized.remove(&tr);
            }
            // ToolResult is indexed by its content-address hash.
            if let Event::ToolResult { output } = ev {
                let h = tool_result_hash(output);
                self.by_hash.insert(h, ev.clone());
            }
        }

        // Step 4: Extend the log.
        self.log.extend(events);

        // Step 5: Return assigned positions.
        Ok(positions)
    }

    /// Evict a cached materialized cell so the next `resolve()` re-folds.
    ///
    /// Idempotent — silently succeeds if `ref_` was never materialized.
    ///
    /// Mirrors `InMemoryKernel.drop_materialized()` in Python.
    pub fn drop_materialized(&mut self, ref_: &str) {
        self.materialized.remove(ref_);
    }
}

impl Default for InMemoryKernel {
    fn default() -> Self {
        Self::new()
    }
}

// --- Tests -------------------------------------------------------------------

#[cfg(test)]
mod append_batch_tests {
    use super::*;
    use crate::models::{cell_write_ref, tool_result_hash};

    fn make_cell_write(payload: &[u8], interpreters: &[Hash]) -> (Hash, Event) {
        let cell_ref = cell_write_ref(payload, interpreters);
        let event = Event::CellWrite {
            payload: payload.to_vec(),
            interpreters: interpreters.to_vec(),
        };
        (cell_ref, event)
    }

    // -- empty batch ----------------------------------------------------------

    /// An empty batch must return Err without mutating the kernel.
    ///
    /// Mirrors: `test_append_batch_rejects_empty_batch` in Python.
    #[test]
    fn append_batch_rejects_empty_batch() {
        let mut k = InMemoryKernel::new();
        let result = k.append_batch(vec![]);
        assert!(result.is_err(), "empty batch must return Err(BatchError)");
    }

    // -- SeqPos assignment ----------------------------------------------------

    /// First batch assigns positions starting at 0 (log is empty before).
    ///
    /// Mirrors: `test_append_batch_assigns_contiguous_seqpos` in Python.
    #[test]
    fn append_batch_first_batch_starts_at_zero() {
        let mut k = InMemoryKernel::new();
        let (_, ev) = make_cell_write(b"a", &[]);
        let positions = k.append_batch(vec![ev]).expect("valid batch must succeed");
        assert_eq!(positions, vec![0u64], "first event must get position 0");
    }

    /// Second batch starts immediately after the last committed position.
    ///
    /// Mirrors: `test_append_batch_assigns_contiguous_seqpos` in Python.
    #[test]
    fn append_batch_second_batch_continues_from_log_len() {
        let mut k = InMemoryKernel::new();
        let (_, ev1) = make_cell_write(b"a", &[]);
        let (_, ev2) = make_cell_write(b"b", &[]);
        let (_, ev3) = make_cell_write(b"c", &[]);
        let p1 = k.append_batch(vec![ev1]).expect("batch 1");
        let p2 = k.append_batch(vec![ev2, ev3]).expect("batch 2");
        assert_eq!(p1, vec![0u64]);
        assert_eq!(p2, vec![1u64, 2u64]);
    }

    // -- log extension --------------------------------------------------------

    /// The log grows by exactly the number of events in the batch.
    #[test]
    fn append_batch_extends_log_by_batch_length() {
        let mut k = InMemoryKernel::new();
        let (_, ev1) = make_cell_write(b"a", &[]);
        let (_, ev2) = make_cell_write(b"b", &[]);
        k.append_batch(vec![ev1, ev2]).expect("valid batch");
        assert_eq!(
            k.all_events().len(),
            2,
            "log must contain both events after batch"
        );
    }

    // -- cell_refs indexing ---------------------------------------------------

    /// CellWrite adds its cell_ref to cell_refs so resolve() can fold it.
    ///
    /// Mirrors: `test_append_batch_indexes_defining_refs_and_tool_hashes` in Python.
    #[test]
    fn append_batch_indexes_defining_ref_in_cell_refs() {
        let mut k = InMemoryKernel::new();
        let (cr, ev) = make_cell_write(b"hello", &[]);
        k.append_batch(vec![ev]).expect("valid batch");
        assert!(
            k.cell_refs.contains(&cr),
            "append_batch must add the defining cell_ref to cell_refs"
        );
    }

    /// After append_batch, a CellWrite cell is resolvable via resolve().
    #[test]
    fn append_batch_makes_cell_write_resolvable() {
        let mut k = InMemoryKernel::new();
        let (cr, ev) = make_cell_write(b"hello", &[]);
        k.append_batch(vec![ev]).expect("valid batch");
        let result = k
            .resolve(&cr)
            .expect("cell must be resolvable after append_batch");
        match result {
            Resolved::Cell(c) => assert_eq!(c.payload, b"hello".to_vec()),
            Resolved::Event(_) => panic!("expected Resolved::Cell, got Event"),
        }
    }

    // -- ToolResult indexing --------------------------------------------------

    /// ToolResult is indexed by its content-address hash so resolve(hash) returns it.
    ///
    /// Mirrors: `test_append_batch_indexes_defining_refs_and_tool_hashes` in Python.
    #[test]
    fn append_batch_indexes_tool_result_by_hash() {
        let mut k = InMemoryKernel::new();
        let ev = Event::ToolResult {
            output: b"tool output".to_vec(),
        };
        let h = tool_result_hash(b"tool output");
        k.append_batch(vec![ev.clone()]).expect("valid batch");
        assert!(
            k.by_hash.contains_key(&h),
            "append_batch must index ToolResult in by_hash under its output hash"
        );
        let result = k
            .resolve(&h)
            .expect("ToolResult must be resolvable by hash");
        match result {
            Resolved::Event(e) => assert_eq!(e, ev),
            Resolved::Cell(_) => panic!("expected Resolved::Event, got Cell"),
        }
    }

    // -- materialized eviction ------------------------------------------------

    /// An amending Relationship event evicts the stale materialized cache for its
    /// target_ref so the next resolve() re-folds and includes the edge in provenance.
    ///
    /// Mirrors: `test_append_amending_event_evicts_stale_materialized_cache` in Python.
    #[test]
    fn append_batch_amending_event_evicts_stale_materialized_cache() {
        let mut k = InMemoryKernel::new();
        let (a_ref, a_write) = make_cell_write(b"cell A", &[]);
        let (b_ref, b_write) = make_cell_write(b"cell B", &[]);
        k.append_batch(vec![a_write.clone(), b_write.clone()])
            .expect("initial batch");

        // Materialize A — puts a stale (edge-less) entry in the cache.
        let first = k.resolve(&a_ref).expect("first resolve of A");
        match &first {
            Resolved::Cell(c) => {
                assert_eq!(c.provenance.len(), 1, "before edge: only one event");
            }
            _ => panic!("expected Cell"),
        }
        assert!(
            k.materialized.contains_key(&a_ref),
            "A must be cached after first resolve"
        );

        // Append an amending Relationship (from_ref = a_ref).
        let edge = Event::Relationship {
            from_ref: a_ref.clone(),
            to_ref: b_ref.clone(),
            type_: "links_to".into(),
        };
        k.append_batch(vec![edge.clone()]).expect("amending batch");

        // Stale entry must have been evicted by append_batch.
        assert!(
            !k.materialized.contains_key(&a_ref),
            "append_batch must evict stale materialized entry when target_ref is amended"
        );

        // Next resolve must re-fold and show the edge without an explicit drop_materialized.
        let second = k.resolve(&a_ref).expect("second resolve of A");
        match second {
            Resolved::Cell(c) => {
                assert!(
                    c.provenance.contains(&edge),
                    "edge must appear in A's provenance after amending append (no explicit drop needed)"
                );
                assert_eq!(c.provenance.len(), 2, "provenance must be write + edge");
            }
            _ => panic!("expected Cell"),
        }
    }

    // -- atomicity ------------------------------------------------------------

    /// Empty batch is rejected before mutation: the log is unchanged.
    ///
    /// Mirrors the all-or-nothing invariant from `test_append_batch_is_atomic_on_invalid_batch`
    /// in Python. In Rust, type safety eliminates non-Event items; the only rejection
    /// point is the empty-batch check, which must also commit nothing.
    #[test]
    fn append_batch_atomicity_empty_batch_commits_nothing() {
        let mut k = InMemoryKernel::new();
        let (_, ev) = make_cell_write(b"ok", &[]);
        k.append_batch(vec![ev]).expect("initial valid batch");
        // Empty batch must fail.
        let result = k.append_batch(vec![]);
        assert!(result.is_err(), "empty batch must return Err");
        // Log must be unchanged: only the first event persists.
        assert_eq!(
            k.all_events().len(),
            1,
            "failed batch must commit nothing (atomicity)"
        );
    }

    // -- mixed batch ----------------------------------------------------------

    /// A batch with both a CellWrite and a ToolResult is handled correctly.
    #[test]
    fn append_batch_mixed_batch_indexes_both() {
        let mut k = InMemoryKernel::new();
        let (cr, write_ev) = make_cell_write(b"a", &[]);
        let tool_ev = Event::ToolResult {
            output: b"t".to_vec(),
        };
        let tool_hash = tool_result_hash(b"t");
        let positions = k
            .append_batch(vec![write_ev, tool_ev])
            .expect("mixed batch");
        assert_eq!(positions, vec![0u64, 1u64], "positions must be [0, 1]");
        // Both must be resolvable.
        assert!(k.resolve(&cr).is_ok(), "cell write must be resolvable");
        assert!(
            k.resolve(&tool_hash).is_ok(),
            "tool result must be resolvable by hash"
        );
    }
}

#[cfg(test)]
mod resolve {
    use super::*;
    use crate::models::{cell_write_ref, tool_result_hash};

    // -- helpers --------------------------------------------------------------

    fn make_cell_write(payload: &[u8], interpreters: &[Hash]) -> (Hash, Event) {
        let cell_ref = cell_write_ref(payload, interpreters);
        let event = Event::CellWrite {
            payload: payload.to_vec(),
            interpreters: interpreters.to_vec(),
        };
        (cell_ref, event)
    }

    // -- content_address ------------------------------------------------------

    /// `content_address` must delegate to `models::content_address`.
    #[test]
    fn content_address_delegates_to_models() {
        let k = InMemoryKernel::new();
        let expected = crate::models::content_address(b"abc");
        assert_eq!(
            k.content_address(b"abc"),
            expected,
            "content_address must match models::content_address"
        );
    }

    /// Same input always produces the same hash; different inputs produce different hashes.
    #[test]
    fn content_address_is_deterministic() {
        let k = InMemoryKernel::new();
        assert_eq!(k.content_address(b"x"), k.content_address(b"x"));
        assert_ne!(k.content_address(b"x"), k.content_address(b"y"));
    }

    // -- all_events -----------------------------------------------------------

    /// Empty log returns an empty vec.
    #[test]
    fn all_events_empty_log() {
        let k = InMemoryKernel::new();
        assert!(k.all_events().is_empty());
    }

    /// Events are returned in insertion order with ascending SeqPos.
    ///
    /// Mirrors Python: `list(enumerate(self._log))`.
    #[test]
    fn all_events_insertion_order_with_seqpos() {
        let mut k = InMemoryKernel::new();
        let (_, ev1) = make_cell_write(b"a", &[]);
        let (_, ev2) = make_cell_write(b"b", &[]);
        // Seed directly — test seam, mirrors Python's pre-append_batch seam.
        k.log.push(ev1.clone());
        k.log.push(ev2.clone());
        let events = k.all_events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], (0u64, ev1));
        assert_eq!(events[1], (1u64, ev2));
    }

    // -- resolve: NotFound ----------------------------------------------------

    /// Unknown ref returns `Err(NotFound)` carrying the requested ref.
    ///
    /// Mirrors: `test_resolve_unknown_ref_raises_not_found` in Python.
    #[test]
    fn resolve_unknown_ref_returns_not_found() {
        let mut k = InMemoryKernel::new();
        let result = k.resolve("deadbeef");
        assert!(result.is_err(), "unknown ref must return Err(NotFound)");
        assert_eq!(result.unwrap_err().ref_, "deadbeef");
    }

    // -- resolve: cached materialized -----------------------------------------

    /// When a Cell is already in `materialized`, resolve() returns it without re-folding.
    #[test]
    fn resolve_returns_cached_cell_if_materialized() {
        let mut k = InMemoryKernel::new();
        let (cr, ev) = make_cell_write(b"hello", &[]);
        let cell = Cell {
            payload: b"hello".to_vec(),
            provenance: vec![ev],
            interpreters: vec![],
        };
        // Seed materialized cache directly.
        k.materialized.insert(cr.clone(), cell.clone());

        let result = k.resolve(&cr).expect("materialized ref must succeed");
        match result {
            Resolved::Cell(c) => assert_eq!(c.payload, b"hello".to_vec()),
            Resolved::Event(_) => panic!("expected Resolved::Cell, got Event"),
        }
    }

    /// Materialized cache takes strict priority: if cached, no re-fold occurs.
    #[test]
    fn resolve_prefers_materialized_over_cell_refs_fold() {
        let mut k = InMemoryKernel::new();
        let (cr, ev) = make_cell_write(b"live", &[]);
        k.log.push(ev.clone());
        k.cell_refs.insert(cr.clone());
        // Inject stale cache — resolve() must return THIS without re-folding.
        let stale = Cell {
            payload: b"stale".to_vec(),
            provenance: vec![],
            interpreters: vec![],
        };
        k.materialized.insert(cr.clone(), stale);

        let result = k.resolve(&cr).expect("cached ref must succeed");
        match result {
            Resolved::Cell(c) => assert_eq!(
                c.payload,
                b"stale".to_vec(),
                "materialized cache must take strict priority over re-folding"
            ),
            Resolved::Event(_) => panic!("expected Resolved::Cell"),
        }
    }

    // -- resolve: fold + cache ------------------------------------------------

    /// resolve() re-folds from the log when ref is in cell_refs but not materialized.
    ///
    /// Mirrors: `test_resolve_refolds_a_cell_from_the_log` in Python.
    #[test]
    fn resolve_folds_and_caches_for_known_cell_ref() {
        let mut k = InMemoryKernel::new();
        let (cr, ev) = make_cell_write(b"hello", &[]);
        // Seed directly — test seam, mirrors Python's pre-append_batch seam.
        k.log.push(ev.clone());
        k.cell_refs.insert(cr.clone());

        let result = k.resolve(&cr).expect("known cell_ref must succeed");
        match result {
            Resolved::Cell(c) => assert_eq!(c.payload, b"hello".to_vec()),
            Resolved::Event(_) => panic!("expected Resolved::Cell, got Event"),
        }
        // Verify it was cached for subsequent calls.
        assert!(
            k.materialized.contains_key(&cr),
            "resolve() must cache the re-folded cell in materialized"
        );
    }

    // -- resolve: ToolResult by hash ------------------------------------------

    /// resolve() returns the ToolResult event by its content-address hash.
    ///
    /// Mirrors: `test_resolve_returns_a_tool_result_event_by_hash` in Python.
    #[test]
    fn resolve_returns_tool_result_event_by_hash() {
        let mut k = InMemoryKernel::new();
        let tool_ev = Event::ToolResult {
            output: b"tool output".to_vec(),
        };
        let h = tool_result_hash(b"tool output");
        // Seed by_hash directly — test seam.
        k.by_hash.insert(h.clone(), tool_ev.clone());

        let result = k.resolve(&h).expect("known tool hash must succeed");
        match result {
            Resolved::Event(ev) => assert_eq!(ev, tool_ev),
            Resolved::Cell(_) => panic!("expected Resolved::Event, got Cell"),
        }
    }

    // -- NotFound -------------------------------------------------------------

    /// `NotFound` carries the ref that was requested.
    #[test]
    fn not_found_carries_ref() {
        let nf = NotFound {
            ref_: "some-ref".to_string(),
        };
        assert_eq!(nf.ref_, "some-ref");
    }

    /// `NotFound` implements `std::error::Error`.
    #[test]
    fn not_found_is_std_error() {
        let nf = NotFound {
            ref_: "r".to_string(),
        };
        let _: &dyn std::error::Error = &nf;
    }

    /// `NotFound` Display includes the ref.
    #[test]
    fn not_found_display_contains_ref() {
        let nf = NotFound {
            ref_: "abc123".to_string(),
        };
        let msg = format!("{nf}");
        assert!(
            msg.contains("abc123"),
            "Display must include ref_, got: {msg:?}"
        );
    }

    // -- drop_materialized ----------------------------------------------------

    /// drop_materialized evicts the cache entry.
    #[test]
    fn drop_materialized_removes_from_cache() {
        let mut k = InMemoryKernel::new();
        let (cr, ev) = make_cell_write(b"x", &[]);
        let cell = Cell {
            payload: b"x".to_vec(),
            provenance: vec![ev],
            interpreters: vec![],
        };
        k.materialized.insert(cr.clone(), cell);
        k.drop_materialized(&cr);
        assert!(!k.materialized.contains_key(&cr));
    }

    /// drop_materialized is idempotent — calling on unknown ref must not panic.
    #[test]
    fn drop_materialized_is_idempotent() {
        let mut k = InMemoryKernel::new();
        k.drop_materialized("never-materialized"); // must not panic
    }
}
