"""
Grid Search Optimization for MQA Project Parameters

This script performs a grid search over the following parameters:
- SEARCH_SIZE: odd number squared (40 <= SEARCH_SIZE <= 1100)
- PC_DENSITY: 0.1 to 2.0
- resolution: 512, 1024, 2048
- LIGHT_CONFIG_IQA: mp.PointLights or mp.Headlight with varying lighting parameters
- LIGHT_CONFIG_PCQA: similar to LIGHT_CONFIG_IQA

The search evaluates model performance using GroupKFold cross-validation.
"""

import argparse
import json
import os
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from absl import app
import xgboost as xgb
from absl.flags import argparse_flags
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVR
from tqdm import tqdm

# Add parent directory to path to import mqa_project
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

warnings.filterwarnings("ignore")


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset_csv",
        type=str,
        required=True,
        help="Path to the dataset CSV file (e.g., data/SJTU-TMQA.csv)",
    )
    parser.add_argument(
        "--dataset_base_path",
        type=str,
        required=True,
        help="Base path to the mesh dataset files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="grid_search_results",
        help="Directory to store grid search results",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["RandomForest", "SVR", "XGBoost", "ExtraTrees"],
        default="RandomForest",
        help="Model to optimize for",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds for GroupKFold cross-validation",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for faster testing)",
    )
    parser.add_argument(
        "--search_sizes",
        type=int,
        nargs="+",
        default=[49, 81, 121, 361, 529, 729, 1089],
        help="SEARCH_SIZE values to test (odd number squared)",
    )
    parser.add_argument(
        "--pc_densities",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        help="PC_DENSITY values to test",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048],
        help="Resolution values to test",
    )
    parser.add_argument(
        "--light_intensities",
        type=float,
        nargs="+",
        default=[0.0, 0.3, 0.6, 1.0],
        help="Light intensity values (same for R, G, B channels)",
    )
    parser.add_argument(
        "--light_types",
        type=str,
        nargs="+",
        default=["PointLights", "Headlight"],
        help="Light types to test",
    )
    parser.add_argument(
        "--independent_pcqa_lights",
        action="store_true",
        help="If set, PCQA light config will be searched independently from IQA",
    )
    parser.add_argument(
        "--save_features",
        action="store_true",
        help="If set, save extracted features for each parameter combination",
    )
    parser.add_argument(
        "--optimize_model_params",
        action="store_true",
        help="If set, also optimize model hyperparameters (slower but more thorough)",
    )
    parser.add_argument(
        "--model_param_search",
        type=str,
        choices=["grid", "coarse", "fine"],
        default="coarse",
        help="Model hyperparameter search granularity (grid=exhaustive, coarse=fast, fine=thorough)",
    )
    return parser.parse_args(argv[1:])


def srocc(x, y):
    """Spearman Rank Order Correlation Coefficient"""
    score = spearmanr(x, y)[0]
    if np.isnan(score):
        gaussian_noise = np.random.normal(y.mean(), 0.05, y.shape)
        score = spearmanr(x, y + gaussian_noise, nan_policy="omit")[0]
    return score


def plcc(x, y):
    """Pearson Linear Correlation Coefficient"""
    score = pearsonr(x, y)[0]
    if np.isnan(score):
        gaussian_noise = np.random.normal(y.mean(), 0.05, y.shape)
        score = pearsonr(x, y + gaussian_noise, nan_policy="omit")[0]
    return score


def get_model_param_grid(model_name, search_type="coarse"):
    """
    Get hyperparameter grid for a given model.

    Args:
        model_name: Name of the model
        search_type: 'grid' (exhaustive), 'coarse' (fast), or 'fine' (thorough)

    Returns:
        List of parameter dictionaries to try
    """
    if model_name == "RandomForest":
        if search_type == "grid":
            return [
                {
                    "n_estimators": n,
                    "max_depth": d,
                    "min_samples_split": s,
                    "max_features": f,
                }
                for n in [50, 100, 200, 300]
                for d in [None, 10, 20, 30]
                for s in [2, 5, 10]
                for f in ["sqrt", "log2", None]
            ]
        elif search_type == "fine":
            return [
                {
                    "n_estimators": n,
                    "max_depth": d,
                    "min_samples_split": s,
                    "max_features": f,
                }
                for n in [100, 200, 300]
                for d in [None, 20, 30]
                for s in [2, 5]
                for f in ["sqrt", None]
            ]
        else:  # coarse
            return [
                {
                    "n_estimators": 100,
                    "max_depth": None,
                    "min_samples_split": 2,
                    "max_features": "sqrt",
                },
                {
                    "n_estimators": 200,
                    "max_depth": None,
                    "min_samples_split": 2,
                    "max_features": "sqrt",
                },
                {
                    "n_estimators": 100,
                    "max_depth": 20,
                    "min_samples_split": 5,
                    "max_features": "sqrt",
                },
            ]

    elif model_name == "SVR":
        if search_type == "grid":
            return [
                {"kernel": k, "C": c, "gamma": g, "epsilon": e}
                for k in ["rbf", "poly", "sigmoid"]
                for c in [0.1, 1, 10, 100]
                for g in ["scale", "auto", 0.001, 0.01, 0.1]
                for e in [0.01, 0.1, 0.2]
            ]
        elif search_type == "fine":
            return [
                {"kernel": k, "C": c, "gamma": g, "epsilon": 0.1}
                for k in ["rbf", "poly"]
                for c in [1, 10, 100]
                for g in ["scale", 0.01, 0.1]
            ]
        else:  # coarse
            return [
                {"kernel": "rbf", "C": 1.0, "gamma": "scale", "epsilon": 0.1},
                {"kernel": "rbf", "C": 10.0, "gamma": "scale", "epsilon": 0.1},
                {"kernel": "rbf", "C": 100.0, "gamma": "auto", "epsilon": 0.1},
            ]

    elif model_name == "XGBoost":
        if search_type == "grid":
            return [
                {
                    "n_estimators": n,
                    "max_depth": d,
                    "learning_rate": lr,
                    "subsample": s,
                    "colsample_bytree": c,
                }
                for n in [50, 100, 200, 300]
                for d in [3, 5, 7, 9]
                for lr in [0.01, 0.05, 0.1, 0.3]
                for s in [0.6, 0.8, 1.0]
                for c in [0.6, 0.8, 1.0]
            ]
        elif search_type == "fine":
            return [
                {
                    "n_estimators": n,
                    "max_depth": d,
                    "learning_rate": lr,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                }
                for n in [100, 200, 300]
                for d in [3, 5, 7]
                for lr in [0.05, 0.1, 0.2]
            ]
        else:  # coarse
            return [
                {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                },
                {
                    "n_estimators": 200,
                    "max_depth": 5,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                },
                {
                    "n_estimators": 100,
                    "max_depth": 7,
                    "learning_rate": 0.05,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                },
            ]

    elif model_name == "ExtraTrees":
        if search_type == "grid":
            return [
                {
                    "n_estimators": n,
                    "max_depth": d,
                    "min_samples_split": s,
                    "max_features": f,
                }
                for n in [50, 100, 200, 300]
                for d in [None, 10, 20, 30]
                for s in [2, 5, 10]
                for f in ["sqrt", "log2", None]
            ]
        elif search_type == "fine":
            return [
                {
                    "n_estimators": n,
                    "max_depth": d,
                    "min_samples_split": s,
                    "max_features": f,
                }
                for n in [100, 200, 300]
                for d in [None, 20, 30]
                for s in [2, 5]
                for f in ["sqrt", None]
            ]
        else:  # coarse
            return [
                {
                    "n_estimators": 100,
                    "max_depth": None,
                    "min_samples_split": 2,
                    "max_features": "sqrt",
                },
                {
                    "n_estimators": 200,
                    "max_depth": None,
                    "min_samples_split": 2,
                    "max_features": "sqrt",
                },
                {
                    "n_estimators": 100,
                    "max_depth": 20,
                    "min_samples_split": 5,
                    "max_features": "sqrt",
                },
            ]

    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_model(model_name, model_params=None, random_state=42):
    """
    Get a model instance by name with optional hyperparameters.

    Args:
        model_name: Name of the model
        model_params: Dictionary of model hyperparameters (if None, uses defaults)
        random_state: Random seed for reproducibility

    Returns:
        Instantiated model
    """
    if model_params is None:
        model_params = {}

    if model_name == "RandomForest":
        default_params = {
            "n_estimators": 100,
            "random_state": random_state,
            "n_jobs": -1,
        }
        default_params.update(model_params)
        return RandomForestRegressor(**default_params)

    elif model_name == "SVR":
        default_params = {"kernel": "rbf"}
        default_params.update(model_params)
        return SVR(**default_params)

    elif model_name == "XGBoost":
        default_params = {
            "n_estimators": 100,
            "random_state": random_state,
            "n_jobs": -1,
        }
        default_params.update(model_params)
        return xgb.XGBRegressor(**default_params)

    elif model_name == "ExtraTrees":
        default_params = {
            "n_estimators": 100,
            "random_state": random_state,
            "n_jobs": -1,
        }
        default_params.update(model_params)
        return ExtraTreesRegressor(**default_params)

    else:
        raise ValueError(f"Unknown model: {model_name}")


def evaluate_model_cv(features_df, model_name, model_params=None, n_folds=5):
    """
    Evaluate a model using GroupKFold cross-validation.

    Args:
        features_df: DataFrame with extracted features
        model_name: Name of the model to use
        model_params: Dictionary of model hyperparameters (optional)
        n_folds: Number of folds for cross-validation

    Returns:
        Tuple of (mean_srocc, mean_plcc, std_srocc, std_plcc)
    """
    X_columns = [col for col in features_df.columns if col.startswith("QUALITY_INDEX")]
    y_column = "SCORE"

    if len(X_columns) == 0:
        raise ValueError("No QUALITY_INDEX columns found in features")

    gkf = GroupKFold(n_splits=n_folds)
    srocc_scores = []
    plcc_scores = []

    for train_idx, test_idx in gkf.split(features_df, groups=features_df["REF"]):
        train_data = features_df.iloc[train_idx]
        test_data = features_df.iloc[test_idx]

        X_train = train_data[X_columns].values
        y_train = train_data[y_column].values
        X_test = test_data[X_columns].values
        y_test = test_data[y_column].values

        # Train model with specified hyperparameters
        model = get_model(model_name, model_params=model_params)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics
        srocc_scores.append(srocc(y_test, y_pred))
        plcc_scores.append(plcc(y_test, y_pred))

    return (
        np.mean(srocc_scores),
        np.mean(plcc_scores),
        np.std(srocc_scores),
        np.std(plcc_scores),
    )


def generate_light_config(light_type, ambient, diffuse, specular, positions=None):
    """Generate a light configuration dictionary"""
    config = {
        "type": light_type,
        "global_ambient": (ambient, ambient, ambient),
        "global_diffuse": (diffuse, diffuse, diffuse),
        "global_specular": (specular, specular, specular),
    }
    if positions is not None:
        config["positions"] = positions
    return config


def extract_features_with_params(
    dataset_csv, dataset_base_path, params, max_samples=None
):
    """
    Extract features using specified parameters by modifying mqa_project module.
    Returns a DataFrame with the extracted features.
    """
    import importlib
    import mqa_project
    import mesh_projector as mp

    # Reload the module to ensure we get fresh parameters
    importlib.reload(mqa_project)

    # Temporarily modify the parameters
    original_params = {
        "SEARCH_SIZE": mqa_project.SEARCH_SIZE,
        "PC_DENSITY": mqa_project.PC_DENSITY,
        "RENDER_CONFIG": mqa_project.RENDER_CONFIG,
        "LIGHT_CONFIG_IQA": mqa_project.LIGHT_CONFIG_IQA,
        "LIGHT_CONFIG_PCQA": mqa_project.LIGHT_CONFIG_PCQA,
    }

    try:
        # Set new parameters
        mqa_project.SEARCH_SIZE = params["search_size"]
        mqa_project.PC_DENSITY = params["pc_density"]

        # Update render config
        mqa_project.RENDER_CONFIG = mp.RenderConfig(
            resolution=params["resolution"],
            ignore_opacity=True,
        )

        # Update light configs
        light_iqa_cfg = params["light_config_iqa"]
        light_pcqa_cfg = params["light_config_pcqa"]

        if light_iqa_cfg["type"] == "PointLights":
            mqa_project.LIGHT_CONFIG_IQA = mp.PointLights(
                positions=light_iqa_cfg.get(
                    "positions", [(-1.0, 1.0, 1.0), (1.0, 1.0, -1.0)]
                ),
                global_ambient=light_iqa_cfg["global_ambient"],
                global_diffuse=light_iqa_cfg["global_diffuse"],
                global_specular=light_iqa_cfg["global_specular"],
            )
        else:  # Headlight
            mqa_project.LIGHT_CONFIG_IQA = mp.Headlight(
                global_ambient=light_iqa_cfg["global_ambient"],
                global_diffuse=light_iqa_cfg["global_diffuse"],
                global_specular=light_iqa_cfg["global_specular"],
            )

        if light_pcqa_cfg["type"] == "PointLights":
            mqa_project.LIGHT_CONFIG_PCQA = mp.PointLights(
                positions=light_pcqa_cfg.get(
                    "positions", [(-1.0, 1.0, 1.0), (1.0, 1.0, -1.0)]
                ),
                global_ambient=light_pcqa_cfg["global_ambient"],
                global_diffuse=light_pcqa_cfg["global_diffuse"],
                global_specular=light_pcqa_cfg["global_specular"],
            )
        else:  # Headlight
            mqa_project.LIGHT_CONFIG_PCQA = mp.Headlight(
                global_ambient=light_pcqa_cfg["global_ambient"],
                global_diffuse=light_pcqa_cfg["global_diffuse"],
                global_specular=light_pcqa_cfg["global_specular"],
            )

        # Load dataset
        df = pd.read_csv(dataset_csv)
        if max_samples is not None and max_samples < len(df):
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        # Extract features
        result_rows = []
        for idx, row in tqdm(
            df.iterrows(), desc="Extracting features", total=len(df), leave=False
        ):
            try:
                result_row = mqa_project.compute_features(row, dataset_base_path)
                result_rows.append(result_row)
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

        result_df = pd.DataFrame(result_rows)
        return result_df

    finally:
        # Restore original parameters
        mqa_project.SEARCH_SIZE = original_params["SEARCH_SIZE"]
        mqa_project.PC_DENSITY = original_params["PC_DENSITY"]
        mqa_project.RENDER_CONFIG = original_params["RENDER_CONFIG"]
        mqa_project.LIGHT_CONFIG_IQA = original_params["LIGHT_CONFIG_IQA"]
        mqa_project.LIGHT_CONFIG_PCQA = original_params["LIGHT_CONFIG_PCQA"]


def generate_parameter_grid(args):
    """Generate all parameter combinations for grid search"""
    # Generate light configurations
    light_configs = []
    for light_type in args.light_types:
        for ambient in args.light_intensities:
            for diffuse in args.light_intensities:
                for specular in args.light_intensities:
                    positions = (
                        [(-1.0, 1.0, 1.0), (1.0, 1.0, -1.0)]
                        if light_type == "PointLights"
                        else None
                    )
                    light_configs.append(
                        generate_light_config(
                            light_type, ambient, diffuse, specular, positions
                        )
                    )

    # Generate all combinations
    param_combinations = []

    if args.independent_pcqa_lights:
        # PCQA lights are independent from IQA lights
        for search_size in args.search_sizes:
            for pc_density in args.pc_densities:
                for resolution in args.resolutions:
                    for light_iqa in light_configs:
                        for light_pcqa in light_configs:
                            param_combinations.append(
                                {
                                    "search_size": search_size,
                                    "pc_density": pc_density,
                                    "resolution": resolution,
                                    "light_config_iqa": deepcopy(light_iqa),
                                    "light_config_pcqa": deepcopy(light_pcqa),
                                }
                            )
    else:
        # PCQA lights same as IQA lights (more sensible default)
        for search_size in args.search_sizes:
            for pc_density in args.pc_densities:
                for resolution in args.resolutions:
                    for light_config in light_configs:
                        param_combinations.append(
                            {
                                "search_size": search_size,
                                "pc_density": pc_density,
                                "resolution": resolution,
                                "light_config_iqa": deepcopy(light_config),
                                "light_config_pcqa": deepcopy(light_config),
                            }
                        )

    return param_combinations


def model_params_to_string(model_params):
    """Convert model parameters dict to a compact string"""
    if not model_params:
        return "default"

    parts = []
    for key, value in sorted(model_params.items()):
        if key in ["random_state", "n_jobs"]:
            continue  # Skip these as they don't affect performance
        if isinstance(value, float):
            parts.append(f"{key[:3]}={value:.2f}")
        elif value is None:
            parts.append(f"{key[:3]}=None")
        else:
            parts.append(f"{key[:3]}={value}")
    return "_".join(parts) if parts else "default"


def params_to_string(params):
    """Convert parameter dict to a readable string"""
    light_iqa = params["light_config_iqa"]
    light_pcqa = params["light_config_pcqa"]

    base_str = (
        f"SS={params['search_size']}_"
        f"PCD={params['pc_density']:.1f}_"
        f"RES={params['resolution']}_"
        f"IQA_{light_iqa['type'][:3]}_A{light_iqa['global_ambient'][0]:.1f}_"
        f"D{light_iqa['global_diffuse'][0]:.1f}_S{light_iqa['global_specular'][0]:.1f}_"
        f"PCQA_{light_pcqa['type'][:3]}_A{light_pcqa['global_ambient'][0]:.1f}_"
        f"D{light_pcqa['global_diffuse'][0]:.1f}_S{light_pcqa['global_specular'][0]:.1f}"
    )

    # Add model params if present
    if "model_params" in params and params["model_params"]:
        model_str = model_params_to_string(params["model_params"])
        base_str += f"_MODEL_{model_str}"

    return base_str


def run_grid_search(args):
    """Main grid search function"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate parameter grid for MQA params
    print("Generating MQA parameter grid...")
    mqa_param_grid = generate_parameter_grid(args)
    print(f"MQA parameter combinations: {len(mqa_param_grid)}")

    # Generate model parameter grid if requested
    if args.optimize_model_params:
        model_param_grid = get_model_param_grid(args.model, args.model_param_search)
        print(f"Model parameter combinations: {len(model_param_grid)}")
        print(
            f"Total combinations (MQA × Model): {len(mqa_param_grid) * len(model_param_grid)}"
        )
    else:
        model_param_grid = [None]  # Use default parameters
        print("Using default model parameters (no model optimization)")

    # Results storage
    results = []
    results_file = os.path.join(
        args.output_dir, f"grid_search_results_{args.model}.csv"
    )

    # Load existing results if resuming
    if os.path.exists(results_file):
        try:
            existing_results = pd.read_csv(results_file)
            results = existing_results.to_dict("records")
            print(f"Loaded {len(results)} existing results. Resuming...")
        except Exception as e:
            print(f"Could not load existing results: {e}")
            results = []

    # Cache for extracted features to avoid re-extracting for different model params
    features_cache = {}

    # Run grid search
    total_combinations = len(mqa_param_grid) * len(model_param_grid)
    pbar = tqdm(total=total_combinations, desc=f"Grid Search ({args.model})")

    for mqa_params in mqa_param_grid:
        # Extract features once for each MQA parameter combination
        mqa_param_str = params_to_string(mqa_params)

        # Check if features are already extracted
        if mqa_param_str not in features_cache:
            try:
                features_df = extract_features_with_params(
                    args.dataset_csv,
                    args.dataset_base_path,
                    mqa_params,
                    max_samples=args.max_samples,
                )

                # Save features if requested
                if args.save_features:
                    features_file = os.path.join(
                        args.output_dir, f"features_{mqa_param_str}.csv"
                    )
                    features_df.to_csv(features_file, index=False)

                features_cache[mqa_param_str] = features_df
            except Exception as e:
                print(f"\nError extracting features for {mqa_param_str}: {e}")
                # Skip all model params for this MQA param combination
                pbar.update(len(model_param_grid))
                continue

        features_df = features_cache[mqa_param_str]

        # Try different model parameters
        for model_params in model_param_grid:
            # Combine params for unique identification
            full_params = deepcopy(mqa_params)
            full_params["model_params"] = model_params
            param_str = params_to_string(full_params)

            # Check if already computed
            if any(r["param_string"] == param_str for r in results):
                pbar.update(1)
                continue

            try:
                start_time = time.time()

                # Evaluate model with these hyperparameters
                mean_srocc, mean_plcc, std_srocc, std_plcc = evaluate_model_cv(
                    features_df,
                    args.model,
                    model_params=model_params,
                    n_folds=args.n_folds,
                )

                elapsed_time = time.time() - start_time

                # Store results
                result = {
                    "param_string": param_str,
                    "search_size": mqa_params["search_size"],
                    "pc_density": mqa_params["pc_density"],
                    "resolution": mqa_params["resolution"],
                    "light_iqa_type": mqa_params["light_config_iqa"]["type"],
                    "light_iqa_ambient": mqa_params["light_config_iqa"][
                        "global_ambient"
                    ][0],
                    "light_iqa_diffuse": mqa_params["light_config_iqa"][
                        "global_diffuse"
                    ][0],
                    "light_iqa_specular": mqa_params["light_config_iqa"][
                        "global_specular"
                    ][0],
                    "light_pcqa_type": mqa_params["light_config_pcqa"]["type"],
                    "light_pcqa_ambient": mqa_params["light_config_pcqa"][
                        "global_ambient"
                    ][0],
                    "light_pcqa_diffuse": mqa_params["light_config_pcqa"][
                        "global_diffuse"
                    ][0],
                    "light_pcqa_specular": mqa_params["light_config_pcqa"][
                        "global_specular"
                    ][0],
                    "model_params": str(model_params) if model_params else "default",
                    "mean_srocc": mean_srocc,
                    "mean_plcc": mean_plcc,
                    "std_srocc": std_srocc,
                    "std_plcc": std_plcc,
                    "elapsed_time": elapsed_time,
                    "model": args.model,
                    "n_folds": args.n_folds,
                }
                results.append(result)

                # Save intermediate results
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values(by="mean_srocc", ascending=False)
                results_df.to_csv(results_file, index=False)

                # Print current best
                pbar.set_postfix(
                    {"Best SROCC": f"{results_df.iloc[0]['mean_srocc']:.4f}"}
                )

            except Exception as e:
                print(f"\nError with params {param_str}: {e}")

            pbar.update(1)

    pbar.close()

    # Final results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="mean_srocc", ascending=False)
    results_df.to_csv(results_file, index=False)

    print("\n" + "=" * 80)
    print("Grid Search Complete!")
    print("=" * 80)
    print("\nTop 5 parameter combinations (by SROCC):")
    print(
        results_df[["param_string", "mean_srocc", "mean_plcc"]]
        .head(10)
        .to_string(index=False)
    )
    print(f"\nFull results saved to: {results_file}")

    return results_df


def main(args):
    print(f"Starting grid search for {args.model} model")
    print(f"Dataset: {args.dataset_csv}")
    print(f"Output directory: {args.output_dir}")

    results_df = run_grid_search(args)

    # Save best parameters as JSON for easy use
    best_params = results_df.iloc[0]
    best_config = {
        "search_size": int(best_params["search_size"]),
        "pc_density": float(best_params["pc_density"]),
        "resolution": int(best_params["resolution"]),
        "light_config_iqa": {
            "type": best_params["light_iqa_type"],
            "ambient": float(best_params["light_iqa_ambient"]),
            "diffuse": float(best_params["light_iqa_diffuse"]),
            "specular": float(best_params["light_iqa_specular"]),
        },
        "light_config_pcqa": {
            "type": best_params["light_pcqa_type"],
            "ambient": float(best_params["light_pcqa_ambient"]),
            "diffuse": float(best_params["light_pcqa_diffuse"]),
            "specular": float(best_params["light_pcqa_specular"]),
        },
        "model_params": best_params.get("model_params", "default"),
        "performance": {
            "mean_srocc": float(best_params["mean_srocc"]),
            "mean_plcc": float(best_params["mean_plcc"]),
            "std_srocc": float(best_params["std_srocc"]),
            "std_plcc": float(best_params["std_plcc"]),
        },
    }

    best_config_file = os.path.join(args.output_dir, f"best_config_{args.model}.json")
    with open(best_config_file, "w") as f:
        json.dump(best_config, f, indent=2)

    print(f"\nBest configuration saved to: {best_config_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Model Params: {best_params.get('model_params', 'default')}")
    print("\nMQA Parameters:")
    print(f"  Search Size: {best_params['search_size']}")
    print(f"  PC Density: {best_params['pc_density']}")
    print(f"  Resolution: {best_params['resolution']}")
    print("\nPerformance:")
    print(
        f"  Mean SROCC: {best_params['mean_srocc']:.4f} ± {best_params['std_srocc']:.4f}"
    )
    print(
        f"  Mean PLCC: {best_params['mean_plcc']:.4f} ± {best_params['std_plcc']:.4f}"
    )
    print("=" * 80)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
