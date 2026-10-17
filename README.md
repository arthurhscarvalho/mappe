# MAPPE

## Purpose

MAPPE (Mesh Assessment via Point Cloud and Projection Estimators) is a full-reference mesh quality assessment framework. It combines image quality features from rendered views with PointPCA3 features from sampled point clouds. This repository is the entry point for the MAPPE experiments and regression-based evaluation.

The Python distribution and import are both named `mappe`.

## Related projects

| Project | Role |
| --- | --- |
| [mesh-projector-pylib](https://github.com/arthurhscarvalho/mesh-projector-pylib) | Python interface for rendering views and sampling colored point clouds. |
| [mesh-projector-rs](https://github.com/arthurhscarvalho/mesh-projector-rs) | Rust mesh processing, CPU rendering, and GPU rendering backend. |
| [pointpca3-pylib](https://github.com/arthurhscarvalho/pointpca3-pylib) | Python interface for PointPCA3 feature extraction. |
| [pointpca3-rs](https://github.com/arthurhscarvalho/pointpca3-rs) | Rust PointPCA3 backend with Voxelization-based Spatial Sampling (VoSS). |
| [pointpca2-pylib](https://github.com/arthurhscarvalho/pointpca2-pylib) | Python interface for the alternative PointPCA2-RS backend. |
| [pointpca2-rs](https://github.com/arthurhscarvalho/pointpca2-rs) | Rust PointPCA2 implementation, also used by PointPCA3. |

## Requirements

- Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).
- A current stable Rust toolchain, Cargo, and a native linker/C compiler for the local Python extensions.
- Git and network access for dependencies, including the PointPCA2 Rust dependency.
- Mesh datasets, including OBJ files and any associated MTL files and texture images.

The package metadata permits Python `>=3.10,<3.13`, but the current mesh-projector dependency uses `enum.StrEnum`, which requires Python 3.11 or later. A GPU is optional for the default CPU mesh renderer and CPU PyTorch execution. GPU execution requires compatible hardware and drivers.

## Installation

Clone the repositories into the following sibling directory layout:

```bash
git clone https://github.com/arthurhscarvalho/mappe.git mappe
git clone https://github.com/arthurhscarvalho/mesh-projector-pylib.git
git clone https://github.com/arthurhscarvalho/mesh-projector-rs.git
git clone https://github.com/arthurhscarvalho/pointpca3-pylib.git
git clone https://github.com/arthurhscarvalho/pointpca3-rs.git
```

```text
workspace/
├── mappe/
├── mesh-projector-pylib/
├── mesh-projector-rs/
├── pointpca3-pylib/
└── pointpca3-rs/
```

From `mappe/`, install the default environment:

```bash
uv sync --python 3.12
```

The configured local sources build `mesh-projector-pylib` and `pointpca3-pylib`; their Cargo manifests require the sibling Rust repositories. `pointpca2` is resolved from PyPI. Cargo fetches the PointPCA2 Rust backend used by PointPCA3, so a local PointPCA2 checkout is not required.

For the configured AMD ROCm environment, use:

```bash
uv sync --python 3.12 --no-default-groups --extra rocm
```

This selects the ROCm 6.3 package index configured in [pyproject.toml](pyproject.toml). Use a compatible ROCm installation and GPU. Run subsequent `uv run` commands with `--no-default-groups --extra rocm` as well, for example `uv run --no-default-groups --extra rocm python scripts/main_experiment.py`.

After changing a local Python binding or its Rust backend, rebuild the local dependencies:

```bash
uv sync --reinstall
```

For a ROCm environment, retain `--no-default-groups --extra rocm` on this command.

## Usage

Run commands from the `mappe/` directory.

### Dataset preparation

The [data](data) directory contains metadata CSV files, rather than the mesh datasets themselves. Obtain the corresponding meshes separately and preserve their material and texture paths. Metadata used by the main experiment requires `ID`, `SCORE`, `REF_OBJ_LOCATION`, and `SIGNAL_OBJ_LOCATION` columns. Mesh locations are joined to the dataset base path.

Before running experiments:

- In [scripts/main_experiment.py](scripts/main_experiment.py), change `BASE_PATH` from its author-specific path to your dataset directory, then select the dataset CSV, configuration, and output CSV in the `run_experiment(...)` call.
- Review the dataset-specific rendering, lighting, sampling density, neighborhood size, and view angles in [scripts/dataset_configs.py](scripts/dataset_configs.py).
- In [scripts/ablation_study.py](scripts/ablation_study.py), set `DATASET_BASE_PATH` and, if needed, `DATASET_CSV` and `OUTPUT_DIR`.

### Feature extraction and evaluation

```bash
uv run python scripts/main_experiment.py
uv run python scripts/ablation_study.py
```

The main experiment currently selects TexturedMeshQuality and writes `experiments/TMQ_MQAProject.csv`. The ablation study currently selects TSMD and writes CSV files under `experiments/ablation_study_tsmd/`. Feature tables contain `QUALITY_INDEX_IQA_*`, `QUALITY_INDEX_PCQA_*`, subjective scores, configuration, and timing information. The extraction pipeline returns features; regression is a separate evaluation step.

For group-based regression evaluation of an extracted table:

```bash
uv run python scripts/group_k_fold_test.py \
  --feature_csvs experiments/TMQ_MQAProject.csv \
  --output_xlsx experiments/TMQ_group_k_fold.xlsx

uv run python scripts/leave_one_group_out_test.py \
  --feature_csvs experiments/TMQ_MQAProject.csv \
  --output_xlsx experiments/TMQ_leave_one_group_out.xlsx \
  --output_plots_dir experiments/TMQ_feature_importance
```

Group k-fold evaluation requires at least five distinct reference groups in the `REF` column. Both scripts write evaluation workbooks; the leave-one-group-out script also writes feature-importance plots.

### PointPCA backend comparison

The pipeline defaults to `pointpca3`; `process_mqa(...)` also accepts `pointpca_backend="pointpca2_rs"`. The contribution-ablation script runs the two backends using the same TSMD configuration:

```bash
uv run python scripts/pointpca3_contribution_ablation.py \
  --backend both \
  --dataset_csv data/TSMD.csv \
  --dataset_base_path /path/to/meshes \
  --output_dir experiments/pointpca3_contribution_ablation_tsmd
```

This produces `TSMD_MAPPE_PointPCA3.csv` and `TSMD_MAPPE_PointPCA2RS.csv`, and, when automatic evaluation runs, `pointpca3_contribution_ablation_tsmd_k_fold.xlsx`. Use `--help` to inspect options such as `--limit`, `--max_workers`, and `--kfold`.

## Publications

1. Arthur Henrique Silva Carvalho and Pedro Garcia Freitas. “MAPPE: Mesh Assessment via Point Cloud and Projection Estimators.” *Signal Processing: Image Communication*, vol. 150, article 117722, 2027. [DOI: 10.1016/j.image.2026.117722](https://doi.org/10.1016/j.image.2026.117722). Available online 5 October 2026.

```bibtex
@article{carvalho2027mappe,
  author  = {Carvalho, Arthur Henrique Silva and Freitas, Pedro Garcia},
  title   = {{MAPPE}: Mesh Assessment via Point Cloud and Projection Estimators},
  journal = {Signal Processing: Image Communication},
  volume  = {150},
  pages   = {117722},
  year    = {2027},
  doi     = {10.1016/j.image.2026.117722}
}
```

## License

[MIT License](LICENSE).
