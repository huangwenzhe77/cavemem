"""Build SSH commands from the experiment operator's local configuration."""

import os
import shlex


def model_command(script):
    repo = os.environ["CAVE_TB_REMOTE"]
    python = os.environ.get("CAVE_TB_PYTHON", "python3")
    weights = os.environ["CAVE_QWEN_WEIGHTS"]
    command = f"cd {shlex.quote(repo)} && " + shlex.join(
        ["env", f"CAVE_QWEN_WEIGHTS={weights}", python, "-u", f"experiments/{script}"]
    )
    return ssh(command)


def teacher_command(script):
    repo = os.environ["CAVE_TB_REMOTE"]
    python = os.environ.get("CAVE_TB_PYTHON", "python3")
    env_file = os.environ["CAVE_TB_TEACHER_ENV"]
    command = f"cd {shlex.quote(repo)} && . {shlex.quote(env_file)} && " + shlex.join(
        ["env", "PYTHONPATH=src", python, "-u", f"experiments/{script}"]
    )
    return ssh(command)


def ssh(command):
    return ["ssh", "-T", "-o", "BatchMode=yes", os.environ["CAVE_TB_HOST"], command]
