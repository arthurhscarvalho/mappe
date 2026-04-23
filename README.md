# mqa-project

## Environment

This project uses `uv` for Python dependency management.

```bash
uv sync
source .venv/bin/activate
python scripts/main_experiment.py
python scripts/ablation_study.py
```

To run with AMD GPU, run: `uv sync --no-default-groups --extra rocm`.

If the local packages (mesh-projector-pylib, pointpca3-pylib) were changed, pass `--reinstall` to `uv sync`:
`uv sync --reinstall`
