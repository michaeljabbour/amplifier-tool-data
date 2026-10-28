// amplifier-data-py: PyO3 binding crate (cdylib).
//
// Phase 2, Task 9: Python<->Rust marshaling + exception mapping.
// Phase 2, Task 10: RustKernel #[pyclass] -- five StorageKernel verbs + all_events.
// Phase 2, Task 11: RustKernel observers -- subscribe/Subscription + observer_errors.
//
// Two temporary helper functions are exported for the task-9 test suite:
//   `_roundtrip_event(ev)` -- Python Event -> Rust Event -> Python Event (for marshaling tests)
//   `_fold(events, cell_ref)` -- Python Events -> Rust regenerate() -> Python Cell
//
// Marshaling is ASYMMETRIC in the real system: Python->Rust extracts fields;
// Rust->Python only ever constructs a Cell (events are handed back as stored
// originals via the parallel object vec -- Task 10).  The Rust->Python Event
// direction implemented here is for the round-trip test helper only.
//
// `extension-module` is NOT listed in Cargo.toml features; maturin injects
// it via pyproject.toml `features = ["pyo3/extension-module"]` at build time.
// This keeps `cargo test` / `cargo build` at the workspace root working.
//
// PyO3 note: `useless_conversion` fires on `?` inside `#[pyfunction]` /
// `#[pymethods]` functions because `From<PyErr> for PyErr` is an identity
// conversion.  This is a known false-positive in PyO3 bindings; suppress it
// for the whole file rather than sprinkling per-function allows.
#![allow(clippy::useless_conversion)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple, PyType};

use amplifier_data_core::file_kernel::FileKernel;
use amplifier_data_core::kernel::{InMemoryKernel, Resolved};
use amplifier_data_core::models::{tool_result_hash, Event};
use amplifier_data_core::replay::{target_ref, MissingDefiningEvent};

// --- Python Event -> Rust Event ------------------------------------------------

/// Convert a Python `Event` subclass instance to a Rust [`Event`] enum.
///
/// Discriminates by `isinstance` check (in base->leaf order: `CellWriteEvent`,
/// `RelationshipEvent`, `ToolResultEvent`, `InterpreterOutputEvent`,
/// `AccessEvent`, `IndexCellEvent`).  Rejects any other type with a Python
/// `ValueError`, mirroring `InMemoryKernel.append_batch`'s validation message.
fn py_event_to_rust(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Event> {
    let models = py.import_bound("amplifier_data.models")?;

    // Closure: return true if `obj` is an instance of the named class.
    let is = |class_name: &str| -> PyResult<bool> {
        let cls = models.getattr(class_name)?;
        obj.is_instance(&cls)
    };

    if is("CellWriteEvent")? {
        let payload: Vec<u8> = obj.getattr("payload")?.extract()?;
        let interpreters: Vec<String> = obj.getattr("interpreters")?.extract()?;
        Ok(Event::CellWrite {
            payload,
            interpreters,
        })
    } else if is("RelationshipEvent")? {
        let from_ref: String = obj.getattr("from_ref")?.extract()?;
        let to_ref: String = obj.getattr("to_ref")?.extract()?;
        // Python field is named "type"; "type" is a keyword in Rust -> stored as type_
        let type_: String = obj.getattr("type")?.extract()?;
        Ok(Event::Relationship {
            from_ref,
            to_ref,
            type_,
        })
    } else if is("ToolResultEvent")? {
        let output: Vec<u8> = obj.getattr("output")?.extract()?;
        Ok(Event::ToolResult { output })
    } else if is("InterpreterOutputEvent")? {
        let cell_ref: String = obj.getattr("cell_ref")?.extract()?;
        let model: String = obj.getattr("model")?.extract()?;
        let input_hash: String = obj.getattr("input_hash")?.extract()?;
        let tool_results_consumed: Vec<String> = obj.getattr("tool_results_consumed")?.extract()?;
        let output: Vec<u8> = obj.getattr("output")?.extract()?;
        Ok(Event::InterpreterOutput {
            cell_ref,
            model,
            input_hash,
            tool_results_consumed,
            output,
        })
    } else if is("AccessEvent")? {
        let cell_ref: String = obj.getattr("cell_ref")?.extract()?;
        Ok(Event::Access { cell_ref })
    } else if is("IndexCellEvent")? {
        let cell_ref: String = obj.getattr("cell_ref")?.extract()?;
        let access_summary: Vec<u8> = obj.getattr("access_summary")?.extract()?;
        let output: Vec<u8> = obj.getattr("output")?.extract()?;
        Ok(Event::IndexCell {
            cell_ref,
            access_summary,
            output,
        })
    } else {
        let type_name: String = obj
            .get_type()
            .qualname()
            .and_then(|s| s.extract::<String>())
            .unwrap_or_else(|_| "unknown".to_owned());
        Err(PyValueError::new_err(format!(
            "append_batch requires every item to be an Event, got: {type_name}"
        )))
    }
}

// --- Rust Event -> Python Event (for _roundtrip_event test helper only) -------
//
// In the real kernel (Task 10), events are NEVER reconstructed from Rust
// variants -- the binding stores the original Python objects in a parallel
// `Vec<Py<PyAny>>` and returns those directly.  This function exists only so
// that `_roundtrip_event` can verify that the Python->Rust conversion is
// lossless.

fn rust_event_to_py<'py>(py: Python<'py>, event: &Event) -> PyResult<Bound<'py, PyAny>> {
    let models = py.import_bound("amplifier_data.models")?;

    match event {
        Event::CellWrite {
            payload,
            interpreters,
        } => {
            let cls = models.getattr("CellWriteEvent")?;
            let kw = PyDict::new_bound(py);
            kw.set_item("payload", PyBytes::new_bound(py, payload))?;
            kw.set_item(
                "interpreters",
                PyTuple::new_bound(py, interpreters.iter().map(|s| s.as_str())),
            )?;
            cls.call((), Some(&kw))
        }
        Event::Relationship {
            from_ref,
            to_ref,
            type_,
        } => {
            let cls = models.getattr("RelationshipEvent")?;
            let kw = PyDict::new_bound(py);
            kw.set_item("from_ref", from_ref.as_str())?;
            kw.set_item("to_ref", to_ref.as_str())?;
            // Python keyword: use string "type", not the Rust keyword
            kw.set_item("type", type_.as_str())?;
            cls.call((), Some(&kw))
        }
        Event::ToolResult { output } => {
            let cls = models.getattr("ToolResultEvent")?;
            let kw = PyDict::new_bound(py);
            kw.set_item("output", PyBytes::new_bound(py, output))?;
            cls.call((), Some(&kw))
        }
        Event::InterpreterOutput {
            cell_ref,
            model,
            input_hash,
            tool_results_consumed,
            output,
        } => {
            let cls = models.getattr("InterpreterOutputEvent")?;
            let kw = PyDict::new_bound(py);
            kw.set_item("cell_ref", cell_ref.as_str())?;
            kw.set_item("model", model.as_str())?;
            kw.set_item("input_hash", input_hash.as_str())?;
            kw.set_item(
                "tool_results_consumed",
                PyTuple::new_bound(py, tool_results_consumed.iter().map(|s| s.as_str())),
            )?;
            kw.set_item("output", PyBytes::new_bound(py, output))?;
            cls.call((), Some(&kw))
        }
        Event::Access { cell_ref } => {
            let cls = models.getattr("AccessEvent")?;
            let kw = PyDict::new_bound(py);
            kw.set_item("cell_ref", cell_ref.as_str())?;
            cls.call((), Some(&kw))
        }
        Event::IndexCell {
            cell_ref,
            access_summary,
            output,
        } => {
            let cls = models.getattr("IndexCellEvent")?;
            let kw = PyDict::new_bound(py);
            kw.set_item("cell_ref", cell_ref.as_str())?;
            kw.set_item("access_summary", PyBytes::new_bound(py, access_summary))?;
            kw.set_item("output", PyBytes::new_bound(py, output))?;
            cls.call((), Some(&kw))
        }
    }
}

// --- Exception mapping helpers ------------------------------------------------

/// Map Rust [`MissingDefiningEvent`] -> Python `amplifier_data.replay.MissingDefiningEvent`.
///
/// The Python exception class is looked up at call time (no static registration
/// required).  If the import or downcast fails, the inner `PyErr` from that
/// failure is returned as-is rather than silently swallowed.
fn make_missing_defining_event(py: Python<'_>, e: &MissingDefiningEvent) -> PyErr {
    let make = || -> PyResult<PyErr> {
        let replay_mod = py.import_bound("amplifier_data.replay")?;
        let exc_cls = replay_mod.getattr("MissingDefiningEvent")?;
        // Downcast Bound<'_, PyAny> -> Bound<'_, PyType> (the class IS a type object).
        let exc_type: Bound<'_, PyType> = exc_cls.downcast_into::<PyType>().map_err(PyErr::from)?;
        // PyErr::from_type_bound calls exc_type(cell_ref) in Python and wraps the result.
        // Use an owned String (not &str) to satisfy the 'static + Send + Sync bounds.
        Ok(PyErr::from_type_bound(exc_type, (e.cell_ref.clone(),)))
    };
    make().unwrap_or_else(|err| err)
}

/// Map Rust `NotFound { ref_ }` -> Python `amplifier_data.kernel.NotFound`.
///
/// Not exercised by task-9 tests (those come in Task 10 via `resolve()`),
/// but the mapping is implemented here as part of the exception-mapping spec.
#[allow(dead_code)]
fn make_not_found(py: Python<'_>, ref_: &str) -> PyErr {
    let make = || -> PyResult<PyErr> {
        let kernel_mod = py.import_bound("amplifier_data.kernel")?;
        let exc_cls = kernel_mod.getattr("NotFound")?;
        let exc_type: Bound<'_, PyType> = exc_cls.downcast_into::<PyType>().map_err(PyErr::from)?;
        // Use owned String to satisfy 'static + Send + Sync bounds on PyErrArguments.
        Ok(PyErr::from_type_bound(exc_type, (ref_.to_owned(),)))
    };
    make().unwrap_or_else(|err| err)
}

// --- Exported helpers (temporary, for task-9 marshaling test coverage) --------

/// Round-trip a single Python Event via the Rust representation and back.
///
/// Python Event -> Rust `Event` (field extraction) -> Python Event (reconstruction).
/// The returned object compares `==` to the input.  This is a **test helper
/// only**; the production kernel never reconstructs Python events from Rust
/// (it returns the stored originals -- Task 10).
#[pyfunction]
fn _roundtrip_event(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let rust_event = py_event_to_rust(py, obj)?;
    let py_event = rust_event_to_py(py, &rust_event)?;
    Ok(py_event.unbind())
}

/// Fold a list of Python Events through the Rust `regenerate()` engine.
///
/// Returns a Python `Cell` whose `provenance` tuple contains the **exact same
/// Python objects** passed in (not reconstructions), preserving identity
/// semantics (`is` checks pass).  `payload` and `interpreters` come from the
/// Rust fold result.
///
/// Raises `amplifier_data.replay.MissingDefiningEvent` (with `.cell_ref`)
/// when the fold has no defining event for the given `cell_ref`, mirroring
/// the Python `regenerate()` contract.
#[pyfunction]
fn _fold(py: Python<'_>, events: Vec<Bound<'_, PyAny>>, cell_ref: String) -> PyResult<PyObject> {
    use amplifier_data_core::replay::regenerate;

    // 1. Convert each Python event to a Rust event (SeqPos = input index).
    let rust_with_seq: Vec<(u64, Event)> = events
        .iter()
        .enumerate()
        .map(|(i, ev)| py_event_to_rust(py, ev).map(|e| (i as u64, e)))
        .collect::<PyResult<_>>()?;

    // 2. Run the Rust regenerate() fold.
    let rust_cell =
        regenerate(&rust_with_seq, &cell_ref).map_err(|e| make_missing_defining_event(py, &e))?;

    // 3. Compute the provenance Python objects.
    //    Select the original Python events whose Rust `target_ref == cell_ref`,
    //    in input (SeqPos) order -- exactly what regenerate() selects internally.
    let prov_py: Vec<Bound<'_, PyAny>> = events
        .iter()
        .zip(rust_with_seq.iter())
        .filter_map(|(py_ev, (_, rust_ev))| {
            if target_ref(rust_ev).as_deref() == Some(cell_ref.as_str()) {
                Some(py_ev.clone())
            } else {
                None
            }
        })
        .collect();

    // 4. Build Python Cell(payload=..., provenance=(...), interpreters=(...)).
    let models = py.import_bound("amplifier_data.models")?;
    let cell_cls = models.getattr("Cell")?;

    let provenance_tuple = PyTuple::new_bound(py, prov_py);
    let interpreters_tuple =
        PyTuple::new_bound(py, rust_cell.interpreters.iter().map(|s| s.as_str()));

    let kw = PyDict::new_bound(py);
    kw.set_item("payload", PyBytes::new_bound(py, &rust_cell.payload))?;
    kw.set_item("provenance", provenance_tuple)?;
    kw.set_item("interpreters", interpreters_tuple)?;

    let result = cell_cls.call((), Some(&kw))?;
    Ok(result.unbind())
}

// --- RustKernelSubscription (Task 11) ----------------------------------------
//
// Handle returned by `RustKernel::subscribe()`.
//
// Holds a shared reference to the observer slot-list (`SharedObservers`) and
// the index of the registered observer.  Calling `unsubscribe()` sets the slot
// to `None`, detaching the observer from future notifications.  The operation
// is idempotent: setting `None` a second time is a no-op.

/// Shared observer slot-list.
///
/// Each element is `Some(callable)` while the observer is active, or `None`
/// after `unsubscribe()`.  Wrapped in `Arc<Mutex<...>>` so both the
/// `RustKernelSubscription` and the `RustKernel` can access it safely across
/// PyO3 call boundaries (the GIL is held by PyO3 when either side runs, but
/// the Mutex guards the slot mutation itself).
type SharedObservers = Arc<Mutex<Vec<Option<Py<PyAny>>>>>;

#[pyclass]
struct RustKernelSubscription {
    observers: SharedObservers,
    index: usize,
}

#[pymethods]
impl RustKernelSubscription {
    /// Detach the observer from the kernel's notification list.
    ///
    /// Idempotent: safe to call multiple times without error.
    fn unsubscribe(&self) {
        if let Ok(mut list) = self.observers.lock() {
            if let Some(slot) = list.get_mut(self.index) {
                *slot = None;
            }
        }
    }
}

// --- RustKernel ---------------------------------------------------------------
//
// Phase 2, Task 10: #[pyclass] wrapping `InMemoryKernel` with a parallel
// `py_events` Vec so that `resolve()` returns the ORIGINAL Python event objects
// (identity semantics).
//
// Phase 2, Task 11: Observer list + observer_errors added to the binding layer.
// Observers are Python callables; they live here (not in the pure Rust core)
// to avoid coupling the core to the Python runtime.  After each successful
// `core.append_batch`, committed (pos, py_event) pairs are dispatched to all
// live observers.  Exceptions are recorded in `observer_errors` and never
// propagated (same contract as the Python `InMemoryKernel`).
//
// Fields:
//   core                  -- Rust InMemoryKernel (storage, indexing, fold)
//   py_events             -- parallel Vec<Py<PyAny>>; py_events[seqpos] is the
//                            original Python event object appended at that SeqPos
//   tool_hash_to_seqpos   -- maps content-address hash -> SeqPos for ToolResult
//                            events so resolve(tool_hash) returns the exact object
//   observers             -- shared slot-list of active Python observer callables
//   observer_errors_inner -- list of (SeqPos, Event, Exception) Python 3-tuples

#[pyclass]
struct RustKernel {
    core: InMemoryKernel,
    py_events: Vec<Py<PyAny>>,
    tool_hash_to_seqpos: HashMap<String, u64>,
    observers: SharedObservers,
    observer_errors_inner: Vec<Py<PyAny>>,
}

#[pymethods]
impl RustKernel {
    #[new]
    fn new() -> Self {
        RustKernel {
            core: InMemoryKernel::new(),
            py_events: Vec::new(),
            tool_hash_to_seqpos: HashMap::new(),
            observers: Arc::new(Mutex::new(Vec::new())),
            observer_errors_inner: Vec::new(),
        }
    }

    /// Append a batch of Python Event objects to the log.
    ///
    /// Converts each Python event to a Rust `Event` (type validation), calls
    /// `InMemoryKernel::append_batch`, stores the original Python objects in
    /// `py_events` at the assigned SeqPos slots, and indexes any ToolResult
    /// events by their content-address hash for identity-preserving resolution.
    ///
    /// After committing, fires all active observers for each committed
    /// (SeqPos, event) pair in SeqPos order.  Observer exceptions are recorded
    /// in `observer_errors` and never propagated (same contract as the Python
    /// `InMemoryKernel`).
    ///
    /// Returns the list of assigned SeqPos values (same length as `events`).
    fn append_batch(
        &mut self,
        py: Python<'_>,
        events: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Vec<u64>> {
        if events.is_empty() {
            return Err(PyValueError::new_err(
                "append_batch requires a non-empty list of events",
            ));
        }

        // Convert all Python events to Rust events first (type validation).
        let rust_events: Vec<Event> = events
            .iter()
            .map(|e| py_event_to_rust(py, e))
            .collect::<PyResult<_>>()?;

        // The start SeqPos = current log length (py_events and the core log
        // are always in sync).
        let start_pos = self.py_events.len() as u64;

        // Commit to the Rust core kernel.
        let positions = self
            .core
            .append_batch(rust_events.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Store original Python objects and update the ToolResult hash index.
        for (i, (rust_ev, py_ev)) in rust_events.iter().zip(events.iter()).enumerate() {
            let seqpos = start_pos + i as u64;
            self.py_events.push(py_ev.clone().unbind());
            if let Event::ToolResult { output } = rust_ev {
                let h = tool_result_hash(output);
                self.tool_hash_to_seqpos.insert(h, seqpos);
            }
        }

        // Post-commit: fire observers in SeqPos order.
        //
        // The observer list is snapshotted *per position* (matching the Python
        // `list(self._observers)` idiom) so that a re-entrant observer that
        // calls `unsubscribe()` during notification does not skip subsequent
        // observers for the same event.  The snapshot clones the `Py<PyAny>`
        // handles out of the `Mutex` and then immediately releases the lock,
        // so observers can safely call `unsubscribe()` without deadlocking.
        for (pos, py_ev_bound) in positions.iter().zip(events.iter()) {
            // Snapshot: clone live observer handles, release the Mutex before calling.
            let snapshot: Vec<Py<PyAny>> = {
                let list = self.observers.lock().unwrap();
                list.iter()
                    .filter_map(|slot| slot.as_ref().map(|o| o.clone_ref(py)))
                    .collect()
            };

            let py_ev_owned: Py<PyAny> = py_ev_bound.clone().unbind();

            for observer in &snapshot {
                match observer.call_bound(py, (*pos, py_ev_owned.clone_ref(py)), None) {
                    Ok(_) => {}
                    Err(exc) => {
                        // Record the error as a (pos, event, exception) tuple.
                        // Never propagate: observer failures must not roll back the commit.
                        let exc_obj: Py<PyAny> = exc.into_py(py);
                        let err_tuple: Py<PyAny> =
                            (*pos, py_ev_owned.clone_ref(py), exc_obj).into_py(py);
                        self.observer_errors_inner.push(err_tuple);
                    }
                }
            }
        }

        Ok(positions)
    }

    /// Return `sha256(payload)` as a 64-character lowercase hex string.
    fn content_address(&self, payload: &[u8]) -> String {
        self.core.content_address(payload)
    }

    /// Resolve a ref to the matching Python `Cell` or Python Event object.
    ///
    /// Resolution order (mirrors Python `InMemoryKernel.resolve`):
    ///   1. Return cached materialized cell if present.
    ///   2. Re-fold from the log if `ref_` is in `cell_refs`; cache result.
    ///   3. Return the content-addressed event (ToolResult) from `by_hash`.
    ///   4. Raise `amplifier_data.kernel.NotFound`.
    ///
    /// For a Cell result, the returned Python `Cell`'s `provenance` tuple
    /// contains the **exact same Python objects** from `py_events` (identity
    /// semantics -- no reconstruction from Rust fields).
    ///
    /// For a ToolResult result, the returned Python object is the **exact same
    /// Python object** passed to `append_batch` (identity semantics).
    fn resolve(&mut self, py: Python<'_>, ref_: &str) -> PyResult<PyObject> {
        match self.core.resolve(ref_) {
            Ok(Resolved::Cell(cell_data)) => {
                // Compute provenance by filtering (SeqPos, Event) pairs for
                // those whose target_ref equals this cell ref, in SeqPos order.
                // all_events() returns events in insertion (= SeqPos ascending) order.
                let all = self.core.all_events();
                let prov_seqpos: Vec<u64> = all
                    .iter()
                    .filter(|(_, ev)| target_ref(ev).as_deref() == Some(ref_))
                    .map(|(pos, _)| *pos)
                    .collect();

                // Build the Python Cell with original Python provenance objects.
                let models = py.import_bound("amplifier_data.models")?;
                let cell_cls = models.getattr("Cell")?;

                let prov_items: Vec<Py<PyAny>> = prov_seqpos
                    .iter()
                    .map(|&pos| self.py_events[pos as usize].clone_ref(py))
                    .collect();
                let provenance_tuple = PyTuple::new_bound(py, prov_items);
                let interpreters_tuple =
                    PyTuple::new_bound(py, cell_data.interpreters.iter().map(|s| s.as_str()));

                let kw = PyDict::new_bound(py);
                kw.set_item("payload", PyBytes::new_bound(py, &cell_data.payload))?;
                kw.set_item("provenance", provenance_tuple)?;
                kw.set_item("interpreters", interpreters_tuple)?;

                Ok(cell_cls.call((), Some(&kw))?.unbind())
            }
            Ok(Resolved::Event(_)) => {
                // ToolResult -- return the original Python object via the hash index.
                let seqpos = self
                    .tool_hash_to_seqpos
                    .get(ref_)
                    .copied()
                    .ok_or_else(|| make_not_found(py, ref_))?;
                Ok(self.py_events[seqpos as usize].clone_ref(py))
            }
            Err(nf) => Err(make_not_found(py, &nf.ref_)),
        }
    }

    /// Subscribe a Python observer callable and return a `RustKernelSubscription`.
    ///
    /// The observer is called as `observer(seq_pos: int, event: Event)` for
    /// each event committed by `append_batch`, in SeqPos order, synchronously
    /// after the commit.  Observer exceptions are captured in `observer_errors`
    /// and never propagated.
    ///
    /// Returns a `RustKernelSubscription` whose `unsubscribe()` method detaches
    /// the observer.  `unsubscribe()` is idempotent.
    fn subscribe(&mut self, py: Python<'_>, observer: PyObject) -> PyResult<PyObject> {
        let mut list = self.observers.lock().unwrap();
        let index = list.len();
        list.push(Some(observer));
        drop(list); // release the Mutex before constructing the Subscription

        let sub = RustKernelSubscription {
            observers: Arc::clone(&self.observers),
            index,
        };
        Ok(Py::new(py, sub)?.into_py(py))
    }

    /// Evict `ref_` from the materialized cache; idempotent.
    fn drop_materialized(&mut self, ref_: &str) {
        self.core.drop_materialized(ref_);
    }

    /// Return `[(SeqPos, original_python_event)]` for every event in the log.
    ///
    /// Each Python event object is the **exact same** object that was passed
    /// to `append_batch` (identity semantics -- no reconstruction from Rust).
    fn all_events(&self, py: Python<'_>) -> Vec<(u64, PyObject)> {
        self.py_events
            .iter()
            .enumerate()
            .map(|(i, ev)| (i as u64, ev.clone_ref(py)))
            .collect()
    }

    /// List of `(SeqPos, Event, Exception)` 3-tuples for observer failures.
    ///
    /// Mirrors `InMemoryKernel.observer_errors` for parity.  Errors are
    /// appended by `append_batch` when an observer raises; the commit is
    /// never rolled back.
    #[getter]
    fn observer_errors(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.observer_errors_inner
            .iter()
            .map(|e| e.clone_ref(py))
            .collect()
    }
}

// --- RustFileKernel -----------------------------------------------------------
//
// Phase 2, Task 6: #[pyclass] wrapping `FileKernel` (Rust) with durable
// append-only persistence.
//
// KEY DESIGN DECISION: value-equality durable-read identity.
//
// Unlike `RustKernel` (which stores the original Python objects in a parallel
// `py_events` Vec for identity semantics), `RustFileKernel` does NOT retain
// Python objects.  Events are encoded to bytes, fsynced to disk, and replayed
// from Rust `Event` variants when read back.  The Python objects returned by
// `all_events()` are therefore freshly constructed via `rust_event_to_py` —
// they compare `==` (VALUE-EQUAL) to the originals but are not the same
// Python objects (`is` would be False).
//
// This is intentional and load-bearing: events can outlive the Python objects
// that wrote them (the file persists across process restarts), so identity
// cannot be preserved.
//
// Fields:
//   inner  -- `Option<FileKernel>` so that `close()` can consume the handle
//             and subsequent calls receive a clear "already closed" error.

/// Durable file-backed kernel exposed to Python.
///
/// Use `RustFileKernel.open(path)` to create or re-open a kernel, then call
/// `append()`, `all_events()`, and finally `close()`.
///
/// Read identity: **value-equality** (==), not object identity (is).
#[pyclass]
struct RustFileKernel {
    inner: Option<FileKernel>,
}

#[pymethods]
impl RustFileKernel {
    /// Open or create the durable log file at `path`.
    ///
    /// If `path` does not exist it is created empty.  If it exists, all
    /// complete records are replayed to reconstruct the in-memory state;
    /// any torn tail left by a prior crash is discarded (torn-tail tolerance).
    #[classmethod]
    fn open(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let fk = FileKernel::open(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(RustFileKernel { inner: Some(fk) })
    }

    /// Encode+frame all events, persist to the log (fsync), and index them.
    ///
    /// Converts each Python `Event` subclass to a Rust `Event` (type
    /// validation), calls `FileKernel::append`, and returns the list of
    /// assigned SeqPos values.
    ///
    /// Raises `ValueError` on empty batch or if the kernel is already closed.
    fn append(&mut self, py: Python<'_>, events: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<u64>> {
        let fk = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("RustFileKernel is already closed"))?;

        if events.is_empty() {
            return Err(PyValueError::new_err(
                "append requires a non-empty list of events",
            ));
        }

        // Convert Python events to Rust events (type-validates each item).
        let rust_events: Vec<Event> = events
            .iter()
            .map(|e| py_event_to_rust(py, e))
            .collect::<PyResult<_>>()?;

        // Persist durably (encode → write → fsync → index).
        let positions = fk
            .append(rust_events)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(positions)
    }

    /// Return `[(SeqPos, python_event)]` for every event in the log.
    ///
    /// Events are **reconstructed** from the Rust log via `rust_event_to_py`.
    /// The returned Python objects are VALUE-EQUAL (`==`) to the originals
    /// but are NOT the same Python objects (identity `is` would be False).
    ///
    /// This is the load-bearing value-equality durable-read identity decision:
    /// events outlive the Python process, so object identity cannot be
    /// preserved across restarts.
    ///
    /// Raises `ValueError` if the kernel is already closed.
    fn all_events(&self, py: Python<'_>) -> PyResult<Vec<(u64, PyObject)>> {
        let fk = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("RustFileKernel is already closed"))?;

        fk.all_events()
            .into_iter()
            .map(|(pos, ev)| {
                let py_ev = rust_event_to_py(py, &ev)?;
                Ok((pos, py_ev.unbind()))
            })
            .collect()
    }

    /// Flush and release the file handle.
    ///
    /// Calls `sync_all` on the underlying file before dropping it.  After
    /// `close()`, all method calls raise `ValueError("already closed")`.
    /// Calling `close()` on an already-closed kernel is a no-op (idempotent).
    fn close(&mut self) -> PyResult<()> {
        if let Some(fk) = self.inner.take() {
            fk.close()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        Ok(())
    }
}

// --- Module registration ------------------------------------------------------

/// Sentinel: returns `True` to prove the extension loaded correctly.
#[pyfunction]
fn __built() -> bool {
    true
}

/// The extension module exported as `amplifier_data._rust`.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(__built, m)?)?;
    m.add_function(wrap_pyfunction!(_roundtrip_event, m)?)?;
    m.add_function(wrap_pyfunction!(_fold, m)?)?;
    m.add_class::<RustKernel>()?;
    m.add_class::<RustKernelSubscription>()?;
    m.add_class::<RustFileKernel>()?;
    Ok(())
}
