"""Unpack one official OCI layer in a confined namespace and apply overlay whiteouts."""
import os
import sys
import shutil
import tarfile
import subprocess
from pathlib import Path

if len(sys.argv)==3:
    os.execvp('bwrap',['bwrap','--unshare-all','--die-with-parent','--ro-bind','/usr','/usr',
              '--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64','--symlink','usr/bin','/bin',
              '--bind',sys.argv[1],'/image','--ro-bind',sys.argv[2],'/layer.tgz',
              '--ro-bind',__file__,'/apply.py','--chdir','/image','/usr/bin/python3','/apply.py'])
with tarfile.open('/layer.tgz') as tf:
    for m in tf:
        p=Path(m.name)
        if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe OCI member')
        if p.name.startswith('.wh.'):
            root=Path('/image').resolve()
            parent=(root/p.parent).resolve()
            parent.relative_to(root)
            targets=list(parent.iterdir()) if p.name=='.wh..wh..opq' and parent.is_dir() else [parent/p.name[4:]]
            for target in targets:
                if target.is_symlink() or target.is_file():target.unlink()
                elif target.is_dir():shutil.rmtree(target)
subprocess.run(['tar','--no-same-owner','--exclude=dev/*','--exclude=.wh.*','-xf','/layer.tgz','-C','/image'],check=True)
