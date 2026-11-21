# Terminal-Bench pilot

There are six registered CPU tasks from the pinned Terminal-Bench 2.0 revision. The pilot uses the official task images and tests, a WSL Ubuntu Bubblewrap sandbox, a fixed five-turn actor budget, and one paired checkpoint per eligible target. Actor commands see a writable private `/app`. Task tests are staged in a disposable grading copy after each branch has been sealed.

The five-turn pilot and the follow-up artifact-completion condition have separate registrations. They are development protocols with overlapping tasks. The number of independent training tasks in the pilot is below the learned controller's minimum.

## Windows sandbox with a remote GPU

Clone this repository on Windows and on the GPU server. Use Windows Python 3.12+, WSL Ubuntu with Bubblewrap, and an SSH alias that works without prompts. Install the actor dependencies on the GPU server and place the model weights there. A remote shell file should export the teacher API variables without exposing them on the command line.

```powershell
$env:CAVE_TB_HOST = "my-gpu-host"
$env:CAVE_TB_REMOTE = "/path/to/server/checkout"
$env:CAVE_TB_PYTHON = "/path/to/venv/bin/python"
$env:CAVE_TB_TEACHER_ENV = "/path/to/teacher.env"
$env:CAVE_QWEN_WEIGHTS = "/path/to/Qwen2.5-7B"
$p = "experiments/cave-tb-pilot-01"
python "$p/assets/prepare.py"
python "$p/assets/bootstrap.py"
python "$p/experiment.py" natural regex-log cancel-async-tasks
python "$p/experiment.py" source regex-log cancel-async-tasks
python "$p/experiment.py" natural polyglot-c-py regex-chess constraints-scheduling log-summary-date-ranges
python "$p/experiment.py" target
python "$p/audit.py"
```

Source replay and target scoring consume frozen natural trajectories and write local receipts. The preparation step downloads the official benchmark inputs and resolves the registered images. Pilot logs, source traces, model weights, verifier files, and local paths stay in ignored runtime files.

For an existing Harbor deployment, use `python -m memory_agent.cave.transfer_suite harbor --help` and supply a matching task cache, Enroot images, and actor endpoint. The capture path uses `configs/cave_harbor.yaml`.
