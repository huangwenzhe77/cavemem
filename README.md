# CAVE-Mem

CAVE-Mem evaluates when an agent should use a memory from a past failure. It saves the actor's unaided action before retrieval. The controller learns when a source-verified reminder improves downstream outcomes after accounting for the risk of turning a success into a failure and the extra inference cost.

- **Source memory.** A teacher proposes corrections from historical failures. Matched source replays measure both the direct alternative and the text reminder before admission.
- **Target data.** An eligible target checkpoint records the original action. Restore it to compare silence with one reminder, or randomize a single arm and log its propensity. Correct original actions remain eligible.
- **Policy.** A frozen Qwen2.5-7B actor solves the task. A small controller uses outcome and cost data to predict reminder gain, learn a risk- and cost-constrained policy, and abstain when evidence is insufficient. A separate diagnostic cohort supports optional read-only inspection.

## Installation

Python 3.12 or later. Install a CUDA-compatible PyTorch build for local GPU inference.

```sh
python -m pip install -e ".[actor]"
python -m unittest discover -s tests
```

For the Harbor integration, install `python -m pip install -e ./external/harbor`. The replay code uses the vendored command parser, so work from a source checkout. For TextWorld experiments, install ALFWorld separately and provision its official game files and base environment configuration.

## Run

```sh
python -m memory_agent.cave --help
python -m memory_agent.cave.alfworld --help
python -m memory_agent.cave.alfworld pipeline \
  --out outputs/alfworld-study \
  --data /path/to/alfworld-data \
  --config /path/to/alfworld/configs/base_config.yaml \
  --weights /path/to/Qwen2.5-7B
```

Set `OPENAI_BASE_URL` and `OPENAI_API_KEY` in the process environment for teacher calls. Each new output directory registers its task splits and binds replay and model inputs. See [method and implementation map](docs/method.md) for the stages and [Terminal-Bench pilot](docs/terminal-bench.md) for the isolated task runner.

## Experiment status

The sorting environment supports development runs and policy comparisons. ALFWorld exposes the end-to-end controller pipeline. Terminal-Bench has an independent official-image pilot and a Harbor capture path. The paused Terminal-Bench pilot has yet to produce a validated held-out intervention result.

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for licensing and third-party attribution.
