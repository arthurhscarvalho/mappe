# MAPPE experiment scripts

This directory contains dataset-level feature extraction, parameter searches, ablation studies, and regression evaluations for MAPPE. For extracting features from a single mesh pair, see the [project README](../README.md#usage).

## Setup and data

Run all commands below from the repository root, rather than from `scripts/`:

```bash
uv sync --python 3.12
```

For the configured ROCm environment, use `uv sync --python 3.12 --no-default-groups --extra rocm`, and retain `--no-default-groups --extra rocm` on subsequent `uv run` commands.

The [data directory](../data) contains metadata CSVs, not the mesh datasets. Obtain the meshes separately and retain their associated MTL files and texture images. Feature extraction expects these CSV columns:

| Column | Meaning |
| --- | --- |
| `ID` | Identifier for the reference/distorted pair. |
| `SCORE` | Subjective quality score used for regression evaluation. |
| `REF_OBJ_LOCATION` | Reference OBJ path, joined to the dataset base directory. |
| `SIGNAL_OBJ_LOCATION` | Distorted OBJ path, joined to the dataset base directory. |

Extraction produces tables with `REF`, `SIGNAL`, `SCORE`, `QUALITY_INDEX_IQA_*`, `QUALITY_INDEX_PCQA_*`, configuration, and timing information. Regression scripts use columns beginning with `QUALITY_INDEX` as predictors, `SCORE` as the target, and `REF` to group distortions of the same reference mesh.

Feature extraction and regression evaluation are separate stages: extract feature CSVs first, then supply them to an evaluation script. Generated results can be stored under `experiments/`, which is ignored by Git. DISTS may download pretrained model weights on its first use.

## Script overview

| Script | Purpose |
| --- | --- |
| [dataset_configs.py](dataset_configs.py) | Rendering, lighting, sampling, neighborhood, and view configurations for TSMD, SJTU-TMQA, TexturedMeshQuality, and Yana TAP. Imported by the experiments; not a command-line entry point. |
| [main_experiment.py](main_experiment.py) | Extract MAPPE features and timing information for a selected dataset. |
| [ablation_study.py](ablation_study.py) | Measure the effect of view count, lighting, sampling density, and neighborhood size on TSMD features. |
| [pointpca3_contribution_ablation.py](pointpca3_contribution_ablation.py) | Compare PointPCA3 and PointPCA2-RS within MAPPE using the same TSMD configuration, including backend runtime and memory measurements. |
| [grid_search_optimization.py](grid_search_optimization.py) | Search feature-extraction settings and optionally regressor hyperparameters using grouped cross-validation. |
| [group_k_fold_test.py](group_k_fold_test.py) | Compare regressors with five-fold cross-validation grouped by reference mesh. |
| [leave_one_group_out_test.py](leave_one_group_out_test.py) | Evaluate a Random Forest while holding out one reference mesh at a time, and export feature-importance plots. |

## Dataset feature extraction

Configure [main_experiment.py](main_experiment.py) before running it:

- Set `BASE_PATH` to the mesh dataset directory.
- Select the metadata CSV, configuration class, and output CSV in the `run_experiment(...)` call. Update the configuration import when selecting another dataset.
- Adjust `DEFAULT_MAX_WORKERS` if needed.

```bash
uv run python scripts/main_experiment.py
```

The current entry point uses `data/TexturedMeshQuality.csv` with `TexturedMeshQualityConfig` and writes `experiments/TMQ_MQAProject.csv`. It performs a one-row warmup before collecting features. The script uses constants and its entry-point call for configuration; it does not accept experiment arguments on the command line.

## TSMD parameter ablation

[ablation_study.py](ablation_study.py) starts from `TSMDConfig` and varies one setting at a time:

| Setting | Values |
| --- | --- |
| View count | 1, 3, 6, 12, 20. |
| Lighting | TSMD baseline, flat ambient, and headlight with ambient point-cloud sampling. |
| Point-cloud density factor | 0.3, 0.5, 1.0. |
| Neighborhood size | 121, 289, 361. |

Set `DATASET_BASE_PATH`, `DATASET_CSV`, `OUTPUT_DIR`, and `MAX_WORKERS` in the script as needed. The default CSV is `data/TSMD.csv`.

```bash
uv run python scripts/ablation_study.py
```

The script reuses shared computations across configurations for each mesh pair. It writes one feature CSV per configuration under `experiments/ablation_study_tsmd/`, with names such as `views_6.csv`, `light_baseline_tsmd.csv`, `density_0_3.csv`, and `search_size_361.csv`. Evaluate these tables with the regression scripts below.

## PointPCA3 contribution ablation

[pointpca3_contribution_ablation.py](pointpca3_contribution_ablation.py) compares `pointpca3` and `pointpca2_rs` using `TSMDConfig`. With `--backend both`, each backend runs in a separate child process. The tables include image and point-cloud features, stage runtimes, and baseline, peak, and incremental process RSS during backend computation.

```bash
uv run python scripts/pointpca3_contribution_ablation.py \
  --backend both \
  --dataset_csv data/TSMD.csv \
  --dataset_base_path /path/to/meshes \
  --output_dir experiments/pointpca3_contribution_ablation_tsmd \
  --max_workers 16
```

The output directory contains:

- `TSMD_MAPPE_PointPCA3.csv`.
- `TSMD_MAPPE_PointPCA2RS.csv`.
- `pointpca3_contribution_ablation_tsmd_k_fold.xlsx` when regression evaluation runs.

By default, five-fold evaluation runs after processing both backends without a row limit. It requires matching pair IDs, six IQA features, 40 PCQA features, and at least five distinct `REF` groups in both tables. Use `--kfold never` to skip evaluation or `--kfold always` to request it explicitly.

For a small extraction check, add `--limit 1 --kfold never`. The script appends results and skips IDs already present in each output CSV. Use a new output directory or `--overwrite` when changing the experiment settings. It performs a warmup for a new output; `--no_warmup` disables it.

This script currently imports the Unix `resource` module and therefore requires a Unix-like system.

## Parameter search

[grid_search_optimization.py](grid_search_optimization.py) searches neighborhood size, point-cloud density, resolution, and lighting settings. It evaluates each configuration with `GroupKFold`, keeping all distortions of a reference mesh in the same fold. Supported regressors are `RandomForest`, `SVR`, `XGBoost`, and `ExtraTrees`.

For example, run a restricted search rather than the full default parameter grid:

```bash
uv run python scripts/grid_search_optimization.py \
  --dataset_csv data/TSMD.csv \
  --dataset_base_path /path/to/meshes \
  --output_dir experiments/grid_search_tsmd \
  --model ExtraTrees \
  --n_folds 5 \
  --search_sizes 121 289 361 \
  --pc_densities 0.3 0.5 \
  --resolutions 512 \
  --light_types PointLights \
  --light_intensities 0.6
```

This example writes `grid_search_results_ExtraTrees.csv` and `best_config_ExtraTrees.json` in the selected output directory. The configuration builder starts from `TexturedMeshQualityConfig` and applies the searched settings; review it when adapting the search to another dataset.

Use `--save_features` to retain feature CSVs, `--independent_pcqa_lights` to search point-cloud lighting independently, and `--optimize_model_params` to include regressor hyperparameters. `--model_param_search` selects `coarse`, `fine`, or `grid` hyperparameter search. The number of distinct reference groups must be at least `--n_folds` after any sampling or failed-row exclusions.

## Regression evaluation

Both evaluation scripts accept one or more feature tables through `--feature_csvs` and write one workbook sheet per input table. They report Spearman and Pearson correlations. These scripts evaluate existing feature CSVs; they do not load meshes or extract new features.

### Five-fold grouped evaluation

[group_k_fold_test.py](group_k_fold_test.py) compares the regressors provided by LazyPredict using five folds grouped by `REF`. Each input table needs at least five distinct reference groups.

```bash
mkdir -p experiments
uv run python scripts/group_k_fold_test.py \
  --feature_csvs experiments/TMQ_MQAProject.csv \
  --output_xlsx experiments/TMQ_group_k_fold.xlsx
```

The workbook retains results for each model and fold, including the `FOLD` column.

### Leave-one-reference-out evaluation

[leave_one_group_out_test.py](leave_one_group_out_test.py) trains a Random Forest on all but one reference group and evaluates on the held-out group. It repeats this for each reference and exports a feature-importance plot for each split.

```bash
mkdir -p experiments
uv run python scripts/leave_one_group_out_test.py \
  --feature_csvs experiments/TMQ_MQAProject.csv \
  --output_xlsx experiments/TMQ_leave_one_group_out.xlsx \
  --output_plots_dir experiments/TMQ_feature_importance \
  --n_estimators 100 \
  --random_state 42
```

The workbook contains one result row per held-out reference, identified by `GROUP`. Optional arguments include `--max_depth` for the Random Forest.

For the command-line scripts, append `--help` to inspect their arguments. `main_experiment.py` and `ablation_study.py` instead require editing their configuration constants and entry points.
