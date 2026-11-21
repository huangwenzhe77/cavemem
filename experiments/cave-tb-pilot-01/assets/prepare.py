"""Fetch pinned official task files, content-verified image layers and offline verifier dependencies."""
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path
from urllib.request import Request, urlopen

BASE = Path(__file__).resolve().parent
REV = '2fd12b88aafdd04a52c298e3940bcb189f9766d6'
NAMES = ('regex-log','log-summary-date-ranges','regex-chess','polyglot-c-py',
         'cancel-async-tasks','constraints-scheduling')
CACHE = Path(os.environ['TEMP'])/'cave-tb-verified-blobs'
CACHE.mkdir(exist_ok=True)
LOCKS = {}
MUTEX = threading.Lock()

def fetch(url, headers=None):
    for i in range(4):
        try:
            return urlopen(Request(url,headers={'User-Agent':'CAVE-TB-pilot',**(headers or {})}),timeout=100)
        except OSError:
            if i == 3: raise
            time.sleep(1+i)

def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def raw(path):
    with fetch(f'https://raw.githubusercontent.com/harbor-framework/terminal-bench-2/{REV}/{path}') as f:
        return f.read()

def download_blob(repo, header, descriptor):
    h=descriptor['digest'].split(':')[1]
    dest=CACHE/h
    with MUTEX:
        lock=LOCKS.setdefault(h,threading.Lock())
    with lock:
        if not dest.exists() or sha(dest)!=h:
            with fetch(f'https://registry-1.docker.io/v2/{repo}/blobs/sha256:{h}',header) as f, dest.with_suffix('.part').open('wb') as o:
                shutil.copyfileobj(f,o)
            if sha(dest.with_suffix('.part'))!=h: raise ValueError('layer digest mismatch')
            dest.with_suffix('.part').replace(dest)
    return dest

def linux(p):
    p=Path(p).resolve()
    return '/mnt/'+p.drive[0].lower()+p.as_posix()[2:]

def wsl(*args):
    p=subprocess.run(['wsl','-d','Ubuntu','--',*args],capture_output=True)
    if p.returncode: raise RuntimeError(p.stderr[-1800:])
    return p.stdout.decode().strip()

def prepare(name,tree):
    dest=BASE/name; dest.mkdir(exist_ok=True)
    files={}
    for entry in tree:
        p=entry['path']
        if not p.startswith(name+'/') or entry['type']!='blob':continue
        rel=p[len(name)+1:]
        if not (rel in ('instruction.md','task.toml','environment/Dockerfile') or rel.startswith('tests/')):continue
        if entry.get('mode')=='120000':raise ValueError('symlink in verifier')
        data=raw(p)
        if hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()!=entry['sha']:
            raise ValueError('official task file drift')
        file=dest/rel;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(data)
        files[rel]=hashlib.sha256(data).hexdigest()
    image=tomllib.loads((dest/'task.toml').read_text())['environment']['docker_image']
    repo,tag=image.rsplit(':',1)
    if (dest/'ready.json').exists():
        tag=json.loads((dest/'ready.json').read_text())['digest']
    with fetch(f'https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull') as f:token=json.load(f)['token']
    header={'Authorization':'Bearer '+token,'Accept':'application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json'}
    with fetch(f'https://registry-1.docker.io/v2/{repo}/manifests/{tag}',header) as f:
        data=f.read(); manifest_hash=f.headers['docker-content-digest']
    assert 'sha256:'+hashlib.sha256(data).hexdigest()==manifest_hash
    manifest=json.loads(data); (dest/'image.json').write_bytes(data)
    config=json.loads(download_blob(repo,header,manifest['config']).read_bytes())
    assert config['architecture']=='amd64' and config['os']=='linux'
    layers=[download_blob(repo,header,b) for b in manifest['layers']]
    home=wsl('sh','-c','printf %s "$HOME"')
    root=wsl('mktemp','-d',home+'/cave-tb-image.XXXXXXXX')
    for layer in layers:
        wsl('python3',linux(BASE/'layer.py'),root,linux(layer))
    wsl('mkdir','-p',root+'/runtime',root+'/tests',root+'/logs/verifier')
    assert wsl('test','-d',root+'/app')==''
    env=config['config'].get('Env',[])
    values={'task':name,'rootfs':root,'revision':REV,'image':image,'digest':manifest_hash,'files':files,
            'env':env,'grader_dependencies':{'python':'3.13','pytest':'8.4.1','pytest-json-ctrf':'0.3.5'}}
    if name=='regex-chess':values['grader_dependencies']['chess']='1.11.2'
    (dest/'ready.tmp').write_text(json.dumps(values,indent=2));(dest/'ready.tmp').replace(dest/'ready.json')
    print('READY',name,flush=True)

def reusable(n):
    f=BASE/n/'ready.json'
    if not f.exists():return False
    m=json.loads(f.read_text())
    try:wsl('test','-d',m['rootfs']+'/app')
    except RuntimeError:return False
    return all(sha(BASE/n/p)==h for p,h in m['files'].items() if (BASE/n/p).exists()) and all(
        (BASE/n/p).is_file() for p in m['files'])

if __name__=='__main__':
    with fetch(f'https://api.github.com/repos/harbor-framework/terminal-bench-2/git/trees/{REV}?recursive=1') as f:tree=json.load(f)['tree']
    with concurrent.futures.ThreadPoolExecutor(4) as exe:
        jobs={exe.submit(prepare,n,tree):n for n in NAMES if not reusable(n)}
        for future in concurrent.futures.as_completed(jobs):
            try:future.result()
            except Exception as exc:print('FAILED',jobs[future],repr(exc),flush=True)
