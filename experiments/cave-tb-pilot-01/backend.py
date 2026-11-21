"""Windows client for verified, network-isolated Terminal-Bench CPU tasks."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path, PurePosixPath

HERE=Path(__file__).resolve().parent
ASSETS=HERE/'assets'
ALLOW=('regex-log','log-summary-date-ranges','regex-chess','polyglot-c-py','cancel-async-tasks','constraints-scheduling')
REV='2fd12b88aafdd04a52c298e3940bcb189f9766d6'

def tasks():
    return [n for n in ALLOW if (ASSETS/n/'ready.json').exists()]

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def _meta(n):
    if n not in ALLOW:raise ValueError('unregistered task')
    m=json.loads((ASSETS/n/'ready.json').read_text())
    if m['task']!=n or m['revision']!=REV or 'sha256:'+sha(ASSETS/n/'image.json')!=m['digest']:
        raise ValueError('changed task image/registration')
    return m

def _linux(p):
    p=Path(p).resolve()
    return '/mnt/'+p.drive[0].lower()+p.as_posix()[2:]

def _wsl(args,input=None,timeout=120,check=True):
    p=subprocess.run(['wsl','-d','Ubuntu','--',*args],input=input,capture_output=True,timeout=timeout)
    if check and p.returncode:raise RuntimeError(f'WSL {p.returncode}: '+_err(p.stderr)[-1200:])
    return p

def _err(data):
    return data.rsplit(b'\n\x00',1)[-1].decode(errors='replace')

def _app(meta, app):
    path=PurePosixPath(app)
    if path.parent!=PurePosixPath(meta['rootfs']).parent or not path.name.startswith('cave-tb-app.') or '..' in path.parts:
        raise ValueError('unregistered episode path')
    return str(path)

def instruction(task):
    m=_meta(task);p=ASSETS/task/'instruction.md'
    if sha(p)!=m['files']['instruction.md']:raise ValueError('instruction changed')
    return p.read_text(encoding='utf-8')

def _copy(m,src):
    app=_wsl(['mktemp','-d',str(PurePosixPath(m['rootfs']).parent/'cave-tb-app.XXXXXXXX')]).stdout.decode().strip()
    _wsl(['cp','-a','--',src+'/.',app+'/'])
    return app

def new(task):
    m=_meta(task)
    return _copy(m,m['rootfs']+'/app')

def clone(task,app):
    m=_meta(task)
    return _copy(m,_app(m,app))

def _isolated(task,app,mode,rest=(),**kwargs):
    m=_meta(task)
    return _wsl(['python3',_linux(ASSETS/'isolated.py'),mode,m['rootfs'],_app(m,app),
                 json.dumps(m['env'],separators=(',',':')),*rest],**kwargs)

def state(task,app):
    return _isolated(task,app,'hash').stdout.decode().strip()

def execute(task,app,script):
    start=time.perf_counter()
    p=_isolated(task,app,'actor',input=script.encode(),timeout=60,check=False)
    err=_err(p.stderr)
    if err.startswith('bwrap:'):raise RuntimeError(err)
    return {'code':p.returncode,'stdout':p.stdout[-8192:].decode(errors='replace'),'stderr':err[-4096:],
            'seconds':time.perf_counter()-start}

def grade(task,app,dest):
    m=_meta(task);dest=Path(dest); dest.mkdir(parents=True,exist_ok=True)
    before=state(task,app)
    receipt=dest/'result.json'
    if receipt.exists():
        prior=json.loads(receipt.read_text())
        if prior['state']!=before or prior['task']!=task:raise ValueError('conflicting grade receipt')
        return prior
    for path,h in m['files'].items():
        if path.startswith('tests/') and sha(ASSETS/task/path)!=h:raise ValueError('changed verifier file')
    stage=clone(task,app)
    runtime=json.loads((ASSETS/'runtime.json').read_text())['wsl_runtime']
    logs=dest/'verifier';logs.mkdir(exist_ok=True)
    if (logs/'ctrf.json').exists():raise RuntimeError('incomplete previous verifier receipt')
    start=time.perf_counter()
    p=_isolated(task,stage,'grade',[runtime,_linux(ASSETS/task/'tests'),_linux(logs),task],timeout=900,check=False)
    sec=time.perf_counter()-start
    (logs/'pytest.stdout').write_bytes(p.stdout);(logs/'pytest.stderr').write_text(_err(p.stderr),encoding='utf-8')
    if p.returncode not in (0,1) or not (logs/'ctrf.json').exists():
        raise RuntimeError(f'verifier unavailable ({task}, rc={p.returncode}): '+p.stdout[-800:].decode(errors='replace'))
    summary=json.loads((logs/'ctrf.json').read_text())['results']['summary']
    if not summary['tests'] or summary['passed']+summary['failed']!=summary['tests']:
        raise RuntimeError('incomplete test report')
    result={'task':task,'reward':int(p.returncode==0),'state':before,'seconds':sec,'summary':summary,
            'image':m['digest'],'revision':REV,'test_sha256':m['files']['tests/test_outputs.py'],
            'test_sh_sha256':m['files']['tests/test.sh'],'ctrf_sha256':sha(logs/'ctrf.json')}
    if (summary['failed']==0)!=bool(result['reward']):raise RuntimeError('grade mismatch')
    receipt.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
