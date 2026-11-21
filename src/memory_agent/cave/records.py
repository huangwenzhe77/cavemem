import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path


def encode(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False)


def digest(obj):
    return hashlib.sha256(encode(obj).encode()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(encode(obj) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(spec, state, executable=None):
    request_id = uuid.uuid4().hex
    payload = {"request_id": request_id, "family": spec["family"],
               "input": spec["input"], "submitted": state["submitted"]}
    verifier = Path(__file__).with_name("verifier.py") if executable is None else executable
    try:
        proc = subprocess.run([sys.executable, "-I", str(verifier)], input=encode(payload),
                              capture_output=True, text=True, timeout=20, check=True)
        response = json.loads(proc.stdout)
        reward = response["reward"]
        if response["request_id"] != request_id or type(reward) is not int or reward not in (0, 1):
            raise ValueError("invalid verifier response or stale request ID")
        return {"request_id": request_id, "status": "pass" if reward else "fail", "reward": reward}
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
        return {"request_id": request_id, "status": "unavailable", "reward": None,
                "error_type": type(error).__name__}
