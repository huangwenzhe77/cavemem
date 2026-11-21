"""Linux-side executor, verifier and filesystem snapshot helper."""
import hashlib
import json
import os
import resource
import stat
import sys
from pathlib import Path

mode, image, app = sys.argv[1:4]
if mode=='hash':
    h=hashlib.sha256()
    root=Path(app)
    for parent,dirs,files in os.walk(root,followlinks=False):
        dirs.sort();files.sort()
        for name in dirs+files:
            p=Path(parent)/name
            st=p.lstat()
            rec=[p.relative_to(root).as_posix(),st.st_mode]
            if stat.S_ISLNK(st.st_mode):rec.append(os.readlink(p))
            elif stat.S_ISREG(st.st_mode):
                with p.open('rb') as f:rec.append(hashlib.file_digest(f,'sha256').hexdigest())
            elif not stat.S_ISDIR(st.st_mode):raise RuntimeError('non-replayable filesystem entry')
            h.update(json.dumps(rec).encode()+b'\n')
    print(h.hexdigest());sys.exit()
opts=['bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL','--ro-bind',image,'/',
      '--dev','/dev','--proc','/proc','--tmpfs','/tmp','--clearenv','--setenv','HOME','/tmp',
      '--bind',app,'/app','--chdir','/app']
env=json.loads(sys.argv[4])
for kv in env:
    k,_,v=kv.partition('=')
    if k in {'PATH','LANG','LC_ALL','PYTHON_VERSION','PYTHONDONTWRITEBYTECODE'}:opts+=['--setenv',k,v]
resource.setrlimit(resource.RLIMIT_NOFILE,(256,256))
resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,64*1024**2))
resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
resource.setrlimit(resource.RLIMIT_NPROC,(96,96))
if mode=='actor':
    resource.setrlimit(resource.RLIMIT_CPU,(45,45))
    opts+=['/bin/bash','--norc','--noprofile','-s']
elif mode=='grade':
    runtime,tests,logs,task=sys.argv[5:]
    extra=':/usr/local/lib/python3.13/site-packages' if task=='regex-chess' else ''
    opts+=['--ro-bind',runtime,'/runtime','--ro-bind',tests,'/tests','--bind',logs,'/logs/verifier',
           '--setenv','PATH','/runtime/python/bin:/usr/local/bin:/usr/bin:/bin',
           '--setenv','PYTHONPATH','/runtime/wheels'+extra,'--setenv','PYTHONDONTWRITEBYTECODE','1']
    pytest=['/runtime/python/bin/python3.13','-m','pytest','--ctrf','/logs/verifier/ctrf.json',
            '/tests/test_outputs.py','-rA']
    if task=='cancel-async-tasks':
        # Official test.sh stages its helper in the grading copy only.
        opts+=['/bin/sh','-ec','cp --remove-destination /tests/test.py /app/test.py; exec "$@"','grade',*pytest]
    else:
        opts+=pytest
else:raise ValueError(mode)
if mode == 'actor':
    # Bound wall time as well as CPU for commands waiting on input or a network.
    # Killing the namespace supervisor also terminates the private child processes.
    os.execvp('timeout', ['timeout', '--kill-after=2s', '45s', *opts])
os.execvp(opts[0],opts)
