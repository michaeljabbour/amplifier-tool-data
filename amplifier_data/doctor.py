"""Read-only installation report, with an explicit temporary-store self-test."""
from __future__ import annotations
import argparse
from importlib import metadata
import json
import shutil
import tempfile
from pathlib import Path
from . import AmplifierStore, RUST_AVAILABLE


def self_test(durable=False):
    """Fictional two-consumer protocol exercise, not a production integration claim."""
    with tempfile.TemporaryDirectory(prefix='amplifier-data-doctor-') as tmp:
        path=str(Path(tmp)/'shared.log') if durable else None
        s=AmplifierStore(path=path,record_access=False)
        memory_scope=s.write_cell(b'fictional-memory-consumer')
        handoff_scope=s.write_cell(b'fictional-handoff-consumer')
        evidence=s.write_cell(b'Fictional source evidence: parser check remains unverified.')
        s.scope(evidence,memory_scope)
        handoff=s.write_cell(b'Fictional handoff: run the existing parser check before editing.')
        s.scope(handoff,handoff_scope)
        s.relate(handoff,evidence,'derived_from')
        s.add_embedding(evidence,(1.0,0.0))
        before=s.get_cell(handoff,record_access=False)
        s.kernel.drop_materialized(handoff)
        after=s.regenerate(handoff,record_access=False)
        assert before==after, 'Regeneration must reproduce the complete Cell'
        memory_members={r['cell_ref'] for r in s.query_relational(scope=memory_scope).output}
        handoff_members={r['cell_ref'] for r in s.query_relational(scope=handoff_scope).output}
        assert evidence in memory_members and handoff not in memory_members
        assert handoff in handoff_members and evidence not in handoff_members
        assert s.query_vector((1.0,0.0),k=1,scope=memory_scope)[0][0]==evidence
        refs={evidence:b'Fictional source evidence: parser check remains unverified.',handoff:b'Fictional handoff: run the existing parser check before editing.'}
        s.close()
        if durable:
            s=AmplifierStore(path=path,record_access=False)
            for ref,payload in refs.items():assert s.get_cell(ref,record_access=False).payload==payload
            s.close()
        return {'fictional_consumers':2,'shared_kernel':True,'scoped_views':True,'cell_regeneration':True,'durable_restart':durable,
                'limits':'Protocol exercise through AmplifierStore; not live Memory or Project Handoff host acceptance.'}


def report(run_test=False):
    try: version=metadata.version('amplifier-data')
    except metadata.PackageNotFoundError:version='source-checkout; distribution not installed'
    result={'format':'amplifier-data-doctor/1','distribution':version,'rust_extension':RUST_AVAILABLE,
            'in_memory_available':True,'durable_available':RUST_AVAILABLE,
            'build_tools':{'cargo':bool(shutil.which('cargo')),'rustc':bool(shutil.which('rustc'))},
            'owned_model':False,'network_calls':False,
            'limits':'Source fallback is in-memory only. Durable paths require the compiled Rust kernel. No user store is opened.'}
    if run_test:result['self_test']=self_test(durable=RUST_AVAILABLE)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true');a=p.parse_args(argv)
    print(json.dumps(report(a.self_test),indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
