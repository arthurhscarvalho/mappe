# mqa-project

## Environment

This project uses `uv` for Python dependency management.

```bash
uv sync
uv run python scripts/main_experiment.py
uv run python scripts/ablation_study.py
```

To run with AMD GPU, run: `uv sync --no-default-groups --extra rocm`.
