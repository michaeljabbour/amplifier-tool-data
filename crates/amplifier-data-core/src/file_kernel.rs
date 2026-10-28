//! Durable `FileKernel`: composes [`InMemoryKernel`] with an append-only log file.
//!
//! # Overview
//!
//! [`FileKernel`] wraps an [`InMemoryKernel`] to add persistence: events are encoded
//! with the binary codec, appended to a log file, fsynced, and then indexed into the
//! composed in-memory kernel.  On re-open, the log is replayed via
//! [`crate::replay::read_log`] to reconstruct the full in-memory state.
//!
//! Torn-tail tolerance means that only fully-written, fsynced events survive a crash.
//!
//! # Composition pattern
//!
//! [`FileKernel`] **composes** [`InMemoryKernel`] — it holds an instance as a field
//! and delegates all read/query operations via forwarding methods.  There is no Rust
//! trait extraction or inheritance; [`InMemoryKernel`] is untouched.
//!
//! # Atomic-write helper
//!
//! [`write_atomic`] provides file-level atomic replacement (temp-file + rename) for
//! use-cases that require it (e.g. compaction, checkpoints).  It has no external
//! dependency beyond `std::fs`.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::codec::encode_event;
use crate::kernel::{BatchError, InMemoryKernel, NotFound, Resolved};
use crate::models::{Event, Hash, SeqPos};
use crate::replay::read_log;

// --- FileKernelError ----------------------------------------------------------

/// Error type for [`FileKernel`] operations.
#[derive(Debug)]
pub enum FileKernelError {
    /// A batch-level validation error (e.g. empty batch).
    Batch(BatchError),
    /// An I/O error during file operations.
    Io(std::io::Error),
}

impl From<BatchError> for FileKernelError {
    fn from(e: BatchError) -> Self {
        FileKernelError::Batch(e)
    }
}

impl From<std::io::Error> for FileKernelError {
    fn from(e: std::io::Error) -> Self {
        FileKernelError::Io(e)
    }
}

impl std::fmt::Display for FileKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileKernelError::Batch(e) => write!(f, "batch error: {e}"),
            FileKernelError::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for FileKernelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FileKernelError::Batch(e) => Some(e),
            FileKernelError::Io(e) => Some(e),
        }
    }
}

// --- FileKernel ---------------------------------------------------------------

/// Durable kernel: an [`InMemoryKernel`] composed with an append-only log file.
///
/// Provides the same query operations as [`InMemoryKernel`] (via delegation) plus
/// durable [`append`](FileKernel::append), [`open`](FileKernel::open), and
/// [`close`](FileKernel::close).
pub struct FileKernel {
    /// The composed in-memory state rebuilt from (or mirrored by) the on-disk log.
    inner: InMemoryKernel,
    /// Open file handle for appending; writes always go to the end.
    file: std::fs::File,
    /// The log file path — retained for diagnostics.
    #[allow(dead_code)]
    path: PathBuf,
}

impl FileKernel {
    // --- Lifecycle ---

    /// Open or create the log file at `path`, rebuilding in-memory state from any
    /// existing events.
    ///
    /// If `path` does not exist it is created empty.  If it exists, all complete
    /// records are replayed via [`read_log`]; torn or incomplete records at the end
    /// are silently discarded (torn-tail tolerance).
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Replay any existing events (gracefully handles a missing file)
        let events = match std::fs::File::open(&path) {
            Ok(f) => read_log(f)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e),
        };

        // Open the log for appending (creates the file if it doesn't exist yet)
        let file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)?;

        // Truncate any torn tail left by a prior crash mid-append.
        //
        // If the file ends with an incomplete (partially-written) record,
        // the bytes of that record are still present on disk.  Without
        // truncation, subsequent appends land *after* those bytes, making
        // the new records invisible on the next reopen because `read_log`
        // stops when it encounters the torn record.
        //
        // Truncating to `valid_bytes` removes the torn tail and ensures
        // the next append is placed immediately after the last fully-written
        // record.  On a clean file `valid_bytes` equals the actual file size
        // and `set_len` is a no-op.
        let valid_bytes: u64 = events.iter().map(|e| encode_event(e).len() as u64).sum();
        file.set_len(valid_bytes)?;

        // Rebuild in-memory state from the replayed events
        let mut inner = InMemoryKernel::new();
        if !events.is_empty() {
            inner.append_batch(events).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("log replay failed: {e}"),
                )
            })?;
        }

        Ok(FileKernel { inner, file, path })
    }

    /// Flush and close the file handle.
    ///
    /// Calls `sync_all` to ensure both data and metadata are flushed before
    /// releasing the OS file handle.  Consumes `self`.
    pub fn close(self) -> std::io::Result<()> {
        let mut f = self.file;
        f.flush()?;
        f.sync_all()?;
        drop(f);
        Ok(())
    }

    // --- Durable append -------------------------------------------------------

    /// Encode+frame all events, persist to the log, fsync, then index into the
    /// composed [`InMemoryKernel`].
    ///
    /// # Atomicity
    ///
    /// One `sync_data` call per batch — either all events in the batch are durable
    /// (sync succeeded) or none are (crash before sync leaves a torn tail that is
    /// discarded on the next [`open`](FileKernel::open)).
    ///
    /// # Errors
    ///
    /// - [`FileKernelError::Batch`]`(`[`BatchError::EmptyBatch`]`)` — empty batch.
    /// - [`FileKernelError::Io`] — any file-system error during write or sync.
    pub fn append(&mut self, events: Vec<Event>) -> Result<Vec<SeqPos>, FileKernelError> {
        if events.is_empty() {
            return Err(BatchError::EmptyBatch.into());
        }

        // Encode all events into a single buffer for one write + one fsync
        let mut buf = Vec::new();
        for ev in &events {
            buf.extend_from_slice(&encode_event(ev));
        }

        // Persist to the log file
        self.file.write_all(&buf)?;

        // fsync — guarantees durability before we update the in-memory index
        self.file.sync_data()?;

        // Index/apply into the composed in-memory kernel.
        // SAFETY: we checked non-empty above, so append_batch cannot return EmptyBatch.
        Ok(self
            .inner
            .append_batch(events)
            .expect("non-empty batch invariant violated"))
    }

    // --- Delegation to InMemoryKernel ----------------------------------------

    /// Delegates to [`InMemoryKernel::content_address`].
    pub fn content_address(&self, payload: &[u8]) -> Hash {
        self.inner.content_address(payload)
    }

    /// Delegates to [`InMemoryKernel::all_events`].
    pub fn all_events(&self) -> Vec<(SeqPos, Event)> {
        self.inner.all_events()
    }

    /// Delegates to [`InMemoryKernel::resolve`].
    pub fn resolve(&mut self, ref_: &str) -> Result<Resolved, NotFound> {
        self.inner.resolve(ref_)
    }

    /// Delegates to [`InMemoryKernel::drop_materialized`].
    pub fn drop_materialized(&mut self, ref_: &str) {
        self.inner.drop_materialized(ref_)
    }
}

// --- Atomic write helper ------------------------------------------------------

/// Write `content` to `path` atomically using a sibling temp file followed by an
/// OS-level rename.
///
/// The rename is atomic on POSIX filesystems when both paths share the same mount
/// point (guaranteed for a sibling temp file in the same directory).  On failure,
/// the temp file is removed as a best-effort clean-up.
///
/// No external crate dependency — implemented entirely with `std::fs`.
pub fn write_atomic(path: &Path, content: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path.file_name().and_then(|n| n.to_str()).unwrap_or("data");
    let tmp_name = format!(".{stem}.tmp");
    let tmp_path = dir.join(&tmp_name);

    {
        let mut tmp = std::fs::File::create(&tmp_path)?;
        tmp.write_all(content)?;
        tmp.sync_data()?;
        // tmp is closed here (drop)
    }

    match std::fs::rename(&tmp_path, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp_path); // best-effort clean-up
            Err(e)
        }
    }
}

// --- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{cell_write_ref, Event};

    // -- Helpers --

    /// Create a unique temp-dir path for a test.  Uses process ID to avoid collisions
    /// across parallel test runs while keeping names stable within a single process.
    fn tmp_path(suffix: &str) -> PathBuf {
        std::env::temp_dir().join(format!("ad_test_{}_{}", suffix, std::process::id()))
    }

    /// Build a `CellWrite` event and return `(cell_ref, event)`.
    fn make_cell_write(payload: &[u8]) -> (Hash, Event) {
        let cell_ref = cell_write_ref(payload, &[]);
        let event = Event::CellWrite {
            payload: payload.to_vec(),
            interpreters: vec![],
        };
        (cell_ref, event)
    }

    /// RAII guard: remove the file when dropped.
    struct DeferRemove(PathBuf);

    impl Drop for DeferRemove {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    // -- open: fresh path ------------------------------------------------------

    /// `FileKernel::open` on a fresh path creates the file and returns an empty kernel.
    #[test]
    fn open_fresh_path_works() {
        let path = tmp_path("open_fresh");
        let _cleanup = DeferRemove(path.clone());

        let kernel = FileKernel::open(&path).expect("open must succeed on a fresh path");
        assert!(path.exists(), "open must create the file");
        assert!(
            kernel.all_events().is_empty(),
            "fresh kernel must have no events"
        );
        kernel.close().expect("close must succeed");
    }

    // -- append: persists and reflects in delegated reads ---------------------

    /// `append` persists events; delegated reads reflect them immediately.
    #[test]
    fn append_persists_and_reflects_in_delegate_reads() {
        let path = tmp_path("append_persists");
        let _cleanup = DeferRemove(path.clone());

        let (cr, ev) = make_cell_write(b"hello");

        let mut kernel = FileKernel::open(&path).expect("open");
        let positions = kernel
            .append(vec![ev.clone()])
            .expect("append must succeed");
        assert_eq!(positions, vec![0u64], "first event must get SeqPos 0");

        // Delegated all_events reflects the appended event
        let all = kernel.all_events();
        assert_eq!(all.len(), 1, "all_events must include the appended event");
        assert_eq!(all[0], (0u64, ev.clone()), "event content must match");

        // Delegated resolve works
        let resolved = kernel
            .resolve(&cr)
            .expect("resolve must succeed after append");
        match resolved {
            Resolved::Cell(c) => assert_eq!(c.payload, b"hello".to_vec()),
            Resolved::Event(_) => panic!("expected Cell, got Event"),
        }

        kernel.close().expect("close");
    }

    // -- reopen: replays log and reconstructs state ---------------------------

    /// Reopening a path replays the log and reconstructs equivalent in-memory state.
    #[test]
    fn reopen_replays_log_and_reconstructs_state() {
        let path = tmp_path("reopen_replay");
        let _cleanup = DeferRemove(path.clone());

        let (cr, ev) = make_cell_write(b"durable");

        // Session 1: write and close
        {
            let mut k = FileKernel::open(&path).expect("open first session");
            k.append(vec![ev.clone()]).expect("append");
            k.close().expect("close first session");
        }

        // Session 2: reopen and verify state is fully reconstructed
        {
            let mut k = FileKernel::open(&path).expect("open second session");

            let all = k.all_events();
            assert_eq!(all.len(), 1, "replayed log must contain 1 event");
            assert_eq!(all[0].1, ev, "replayed event must match original");

            let resolved = k.resolve(&cr).expect("resolve must succeed after reopen");
            match resolved {
                Resolved::Cell(c) => assert_eq!(c.payload, b"durable".to_vec()),
                Resolved::Event(_) => panic!("expected Cell after reopen"),
            }

            k.close().expect("close second session");
        }
    }

    // -- close: releases the file handle --------------------------------------

    /// After `close`, the file handle is released; the file can be reopened.
    #[test]
    fn close_releases_file() {
        let path = tmp_path("close_releases");
        let _cleanup = DeferRemove(path.clone());

        let k = FileKernel::open(&path).expect("open");
        k.close().expect("close must succeed");

        // Reopening must succeed — the previous handle must have been released
        let k2 = FileKernel::open(&path).expect("reopen after close must succeed");
        k2.close().expect("second close");
    }

    // -- composition: InMemoryKernel is untouched -----------------------------

    /// `InMemoryKernel` is still usable independently; `FileKernel`'s composition
    /// is non-intrusive.
    #[test]
    fn composition_preserved_in_memory_kernel_untouched() {
        use crate::kernel::InMemoryKernel;

        let (cr, ev) = make_cell_write(b"standalone");
        let mut mem = InMemoryKernel::new();
        mem.append_batch(vec![ev])
            .expect("InMemoryKernel must still work independently");
        assert_eq!(mem.all_events().len(), 1);
        mem.resolve(&cr)
            .expect("resolve must work on InMemoryKernel");
    }

    // -- multiple appends accumulate and survive reopen -----------------------

    #[test]
    fn multiple_appends_accumulate_across_reopen() {
        let path = tmp_path("multi_append");
        let _cleanup = DeferRemove(path.clone());

        let (cr1, ev1) = make_cell_write(b"first");
        let (cr2, ev2) = make_cell_write(b"second");

        // Session 1: two separate batches
        {
            let mut k = FileKernel::open(&path).expect("open");
            k.append(vec![ev1.clone()]).expect("append ev1");
            k.append(vec![ev2.clone()]).expect("append ev2");
            k.close().expect("close");
        }

        // Session 2: both events must be recovered
        {
            let mut k = FileKernel::open(&path).expect("reopen");
            let all = k.all_events();
            assert_eq!(all.len(), 2, "both events must survive reopen");
            k.resolve(&cr1)
                .expect("cr1 must be resolvable after reopen");
            k.resolve(&cr2)
                .expect("cr2 must be resolvable after reopen");
            k.close().expect("close");
        }
    }

    // -- empty batch is rejected -----------------------------------------------

    #[test]
    fn append_empty_batch_is_rejected() {
        let path = tmp_path("empty_batch");
        let _cleanup = DeferRemove(path.clone());

        let mut k = FileKernel::open(&path).expect("open");
        let result = k.append(vec![]);
        assert!(result.is_err(), "empty batch must return Err");
        match result.unwrap_err() {
            FileKernelError::Batch(BatchError::EmptyBatch) => {}
            other => panic!("expected Batch(EmptyBatch), got: {other}"),
        }
        k.close().expect("close");
    }

    // -- write_atomic helper ---------------------------------------------------

    #[test]
    fn write_atomic_creates_file_with_content() {
        let path = tmp_path("atomic_write");
        let _cleanup = DeferRemove(path.clone());

        write_atomic(&path, b"atomic content").expect("write_atomic must succeed");
        let content = std::fs::read(&path).expect("file must exist after write_atomic");
        assert_eq!(content, b"atomic content");
    }

    #[test]
    fn write_atomic_overwrites_existing_file() {
        let path = tmp_path("atomic_overwrite");
        let _cleanup = DeferRemove(path.clone());

        std::fs::write(&path, b"old content").expect("write old content");
        write_atomic(&path, b"new content").expect("write_atomic must succeed");
        let content = std::fs::read(&path).expect("file must exist");
        assert_eq!(
            content, b"new content",
            "write_atomic must replace old content"
        );
    }

    // -- delegation: content_address ------------------------------------------

    #[test]
    fn delegate_content_address_matches_models() {
        let path = tmp_path("content_addr");
        let _cleanup = DeferRemove(path.clone());

        let k = FileKernel::open(&path).expect("open");
        let expected = crate::models::content_address(b"abc");
        assert_eq!(
            k.content_address(b"abc"),
            expected,
            "delegated content_address must match models::content_address"
        );
        k.close().expect("close");
    }

    // -- delegation: drop_materialized ----------------------------------------

    #[test]
    fn delegate_drop_materialized_is_idempotent() {
        let path = tmp_path("drop_mat");
        let _cleanup = DeferRemove(path.clone());

        let mut k = FileKernel::open(&path).expect("open");
        k.drop_materialized("never-materialized"); // must not panic
        k.close().expect("close");
    }
}
