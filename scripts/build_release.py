#!/usr/bin/env python3
"""Build a source-complete, normalized release in dist/. Requires maturin and Rust.

No provider calls or user store access. Wheel is built from the source distribution.
"""
from pathlib import Path
import base64
import csv
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import uuid
import zipfile

ROOT=Path(__file__).resolve().parents[1]
EPOCH=315532800


def run(argv,cwd,env):
    subprocess.run(argv,cwd=cwd,env=env,check=True)


def normalize_sdist(path):
    with tarfile.open(path,'r:gz') as src:
        entries=[(info,src.extractfile(info).read() if info.isfile() else None) for info in src.getmembers()]
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w',format=tarfile.PAX_FORMAT) as out:
        for info,data in sorted(entries,key=lambda x:x[0].name):
            if not (info.isfile() or info.isdir()):raise ValueError('Source release must not contain links or special files')
            info.uid=info.gid=0;info.uname=info.gname='';info.mtime=EPOCH;info.pax_headers={}
            info.mode=0o755 if info.isdir() else 0o644
            out.addfile(info,io.BytesIO(data) if data is not None else None)
    with path.open('wb') as f:
        with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=EPOCH) as gz:gz.write(buffer.getvalue())


def normalize_wheel(path,source):
    # The compiler remaps source locations. Maturin's SBOM independently embeds
    # local path-based component IDs; normalize the IDs and matching references.
    mappings=[(str(source),'/source'),(str(ROOT),'/source'),(str(Path.home()),'/build-home')]
    from urllib.parse import quote
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        if isinstance(x,str):
            for old,new in mappings:
                x=x.replace(old,new).replace(quote(old,safe=''),quote(new,safe=''))
            return x
        return x
    with zipfile.ZipFile(path) as z:
        data={info.filename:z.read(info) for info in z.infolist() if not info.is_dir()}
    for name,raw in list(data.items()):
        if name.endswith('.cyclonedx.json'):
            doc=clean(json.loads(raw));doc.pop('serialNumber',None)
            if 'timestamp' in doc.get('metadata',{}):doc['metadata']['timestamp']='1980-01-01T00:00:00Z'
            canonical=json.dumps(doc,sort_keys=True,separators=(',',':'))
            doc['serialNumber']='urn:uuid:'+str(uuid.uuid5(uuid.NAMESPACE_URL,canonical))
            data[name]=(json.dumps(doc,sort_keys=True,separators=(',',':'))+'\n').encode()
    # Abort rather than mask a compiled private source path.
    for name,raw in data.items():
        if str(Path.home()).encode() in raw or str(source).encode() in raw:
            raise ValueError('Unremapped build path remains in '+name)
    record=next(n for n in data if n.endswith('.dist-info/RECORD'))
    rows=[]
    for name in sorted(n for n in data if n!=record):
        raw=data[name];value=base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b'=').decode()
        rows.append([name,'sha256='+value,str(len(raw))])
    rows.append([record,'','']);buffer=io.StringIO();csv.writer(buffer,lineterminator='\n').writerows(rows);data[record]=buffer.getvalue().encode()
    tmp=path.with_suffix('.normalized.whl')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as out:
        for name in sorted(data):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(0o755 if name.endswith(('.so','.pyd')) else 0o644)<<16
            out.writestr(info,data[name])
    tmp.replace(path)


def main():
    dest=ROOT/'dist';dest.mkdir(exist_ok=True)
    env=dict(os.environ,SOURCE_DATE_EPOCH=str(EPOCH))
    run([sys.executable,'-m','maturin','sdist','--out',str(dest)],ROOT,env)
    source_dist=next(dest.glob('amplifier_data-*.tar.gz'));normalize_sdist(source_dist)
    with tempfile.TemporaryDirectory(prefix='data-release-build-') as temp:
        work=Path(temp)
        with tarfile.open(source_dist,'r:gz') as archive:
            for member in archive.getmembers():
                if not (work/member.name).resolve().is_relative_to(work.resolve()):raise ValueError('Archive path escape')
            archive.extractall(work,filter='data')
        source=next(work.iterdir())
        flags=env.get('RUSTFLAGS','')
        flags+=' --remap-path-prefix='+str(Path.home())+'=/build-home'
        flags+=' --remap-path-prefix='+str(ROOT)+'=/source'
        flags+=' --remap-path-prefix='+str(source)+'=/source'
        build_env=dict(env,RUSTFLAGS=flags)
        run([sys.executable,'-m','maturin','build','--release','--strip','--out',str(dest)],source,build_env)
        wheel=next(dest.glob('amplifier_data-*.whl'));normalize_wheel(wheel,source)
    print(json.dumps({'sdist':source_dist.name,'sdist_sha256':hashlib.sha256(source_dist.read_bytes()).hexdigest(),'wheel':wheel.name,'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'wheel_built_from_sdist':True}))

if __name__=='__main__':main()
