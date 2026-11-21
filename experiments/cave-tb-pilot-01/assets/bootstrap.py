"""Prepare the offline verifier runtime in WSL."""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from prepare import BASE, fetch, linux, sha, wsl


VERSION = "3.13.18"
RELEASE = "20261110"
PYTHON_SHA = "f2d2199eb25880e07863b391cf7124e7ffa1338b0d65600dfcec117c4223a7fb"
PINS = ["pytest==8.4.1", "pytest-json-ctrf==0.3.5", "colorama==0.4.6", "iniconfig==2.3.1",
        "packaging==26.3", "pluggy==1.7.0", "pygments==2.21.0"]


def main():
    state = BASE / "runtime.json"
    if state.exists():
        runtime = json.loads(state.read_text())["wsl_runtime"]
        try:
            wsl("test", "-x", runtime + "/python/bin/python3.13")
        except RuntimeError:
            pass
        else:
            return
    with tempfile.TemporaryDirectory(prefix="cave-tb-runtime-") as name:
        files = Path(name)
        archive = files / "python.tgz"
        url = ("https://github.com/astral-sh/python-build-standalone/releases/download/"
               f"{RELEASE}/cpython-{VERSION}%2B{RELEASE}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz")
        with fetch(url) as response, archive.open("wb") as dest:
            shutil.copyfileobj(response, dest)
        if sha(archive) != PYTHON_SHA:
            raise ValueError("CPython archive digest mismatch")
        subprocess.run([sys.executable, "-m", "pip", "download", "--index-url", "https://pypi.org/simple",
                        "--platform", "manylinux_2_28_x86_64", "--python-version", "3.13",
                        "--only-binary=:all:", "--dest", str(files), *PINS], check=True)
        home = wsl("sh", "-c", 'printf %s "$HOME"')
        runtime = wsl("mktemp", "-d", home + "/cave-tb-runtime.XXXXXXXX")
        wsl("tar", "--no-same-owner", "-xzf", linux(archive), "-C", runtime)
        wsl("mkdir", "-p", runtime + "/wheels")
        for wheel in sorted(files.glob("*.whl")):
            wsl("python3", "-m", "zipfile", "-e", linux(wheel), runtime + "/wheels")
        wsl("test", "-x", runtime + "/python/bin/python3.13")
        state.write_text(json.dumps({"wsl_runtime": runtime}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
