import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from absl import app
from absl.flags import argparse_flags
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--feature_csvs",
        type=str,
        nargs="+",
        required=True,
        help="List of paths to the files with the extracted features.",
    )
    parser.add_argument(
        "--output_xlsx", type=str, required=True, help="Output file with the results."
    )
    parser.add_argument(
        "--output_plots_dir",
        type=str,
        default="./feature_importance_plots_gkf",
        help="Directory to save feature importance plots.",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of trees in the random forest.",
    )
    parser.add_argument(
        "--max_depth", type=int, default=None, help="Maximum depth of the trees."
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random state for reproducibility."
    )
    return parser.parse_args(argv[1:])


def srocc(x, y):
    score = spearmanr(x, y)[0]
    if np.isnan(score):
        gaussian_noise = np.random.normal(y.mean(), 0.05, y.shape)
        score = spearmanr(x, y + gaussian_noise, nan_policy="omit")[0]
    return score


def plcc(x, y):
    score = pearsonr(x, y)[0]
    if np.isnan(score):
        gaussian_noise = np.random.normal(y.mean(), 0.05, y.shape)
        score = spearmanr(x, y + gaussian_noise, nan_policy="omit")[0]
    return score


def custom_metric(x, y):
    s = srocc(x, y)
    p = plcc(x, y)
    return s, p


def plot_feature_importance(
    feature_names, importances, fold_number, dataset_name, output_dir
):
    """
    Create and save a feature importance plot for a specific fold.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Sort features by importance
    sorted_indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importances = importances[sorted_indices]

    # Create the plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(sorted_features)), sorted_importances)
    plt.xlabel("Features")
    plt.ylabel("Feature Importance")
    plt.title(f"Feature Importance - {dataset_name} - Fold {fold_number}")
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha="right")

    # Color bars by importance (darker = more important)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_features)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.tight_layout()

    # Save the plot
    filename = f"{dataset_name}_fold_{fold_number}_feature_importance.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    return filepath


def evaluate_random_forest(
    file_path,
    n_estimators=100,
    max_depth=None,
    random_state=42,
    output_plots_dir="./feature_importance_plots",
):
    """
    Evaluates Random Forest regressor using GroupKFold cross-validation with 5 splits.
    The data is split into 5 folds while ensuring that samples from the same group
    (determined by the "REF" column) stay together in the same fold. Feature importance
    plots are generated and saved for each fold.
    """
    data = pd.read_csv(file_path)
    X_columns = [col for col in data.columns if col.startswith("QUALITY_INDEX")]
    y_column = ["SCORE"]

    # Get dataset name for file naming
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]

    gkf = GroupKFold(n_splits=5)
    model_results = []
    iterator = gkf.split(data, groups=data["REF"])
    total = sum(1 for _ in gkf.split(data, groups=data["REF"]))

    for fold_idx, (train_inds, test_inds) in enumerate(
        tqdm(iterator, desc="Processing Random Forest GKF", total=total)
    ):
        # Get the current fold number
        fold_number = fold_idx + 1

        train, test = data.iloc[train_inds], data.iloc[test_inds]
        X_train, X_test = train[X_columns], test[X_columns]
        y_train, y_test = train[y_column].squeeze(), test[y_column].squeeze()

        # Train Random Forest
        rf = RandomForestRegressor(
            random_state=42,
            n_estimators=300,
            n_jobs=-1,
            max_depth=30,
            max_features=8,
            max_samples=0.25,
            min_samples_leaf=1,
            min_samples_split=2,
        )
        rf.fit(X_train, y_train)

        # Make predictions
        y_pred = rf.predict(X_test)

        # Calculate metrics
        spearman_score = srocc(y_test, y_pred)
        pearson_score = plcc(y_test, y_pred)

        # Create feature importance plot
        plot_feature_importance(
            X_columns,
            rf.feature_importances_,
            fold_number,
            dataset_name,
            output_plots_dir,
        )

        # Store results
        result = {
            "Model": "RandomForestRegressor",
            "Spearman": spearman_score,
            "Pearson": pearson_score,
            "FOLD": fold_number,
        }
        model_results.append(result)

    # Convert results to DataFrame
    df = pd.DataFrame(model_results)
    return df


def process_directory_with_correlation(
    paths,
    filename,
    n_estimators=100,
    max_depth=None,
    random_state=42,
    output_plots_dir="./feature_importance_plots_gkf",
):
    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        for path in paths:
            df = evaluate_random_forest(
                path,
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                output_plots_dir=output_plots_dir,
            )
            df = df.sort_values(by=["Model"], ascending=True)
            df.reset_index(drop=True, inplace=True)
            workbook = writer.book
            sheet_name = os.path.basename(path)
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:31]
            worksheet = workbook.add_worksheet(sheet_name)
            writer.sheets[sheet_name] = worksheet
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def main(args):
    process_directory_with_correlation(
        args.feature_csvs,
        args.output_xlsx,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        output_plots_dir=args.output_plots_dir,
    )


if __name__ == "__main__":
    matplotlib.use("Agg")  # Use non-interactive backend to avoid tkinter issues
    app.run(main, flags_parser=parse_args)
