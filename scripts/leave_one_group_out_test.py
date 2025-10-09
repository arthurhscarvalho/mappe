import argparse
import os

import numpy as np
import pandas as pd
from absl import app
from absl.flags import argparse_flags
from lazypredict.Supervised import LazyRegressor
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import LeaveOneGroupOut


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


def evaluate_regressors(file_path):
    """
    Evaluates multiple regressors using LeaveOneGroupOut cross-validation.
    For each unique group (determined by the "REF" column), the group is left out
    as the test set, and the remaining data is used for training. The resulting
    correlations for each model are stored along with the test group in a column "GROUP_OUT".
    """
    data = pd.read_csv(file_path)
    X_columns = [col for col in data.columns if col.startswith("QUALITY_INDEX")]
    y_column = ["SCORE"]

    logo = LeaveOneGroupOut()
    model_results = []
    iterator = logo.split(data, groups=data["REF"])
    total = sum(1 for _ in logo.split(data, groups=data["REF"]))

    for train_inds, test_inds in tqdm(iterator, desc="Processing LOGO", total=total):
        # Get the current test group (assumed to be a single unique value)
        group_value = data["REF"].iloc[test_inds].unique()[0]

        train, test = data.iloc[train_inds], data.iloc[test_inds]
        X_train, X_test = train[X_columns], test[X_columns]
        y_train, y_test = train[y_column].squeeze(), test[y_column].squeeze()
        reg = LazyRegressor(
            verbose=0,
            ignore_warnings=True,
            custom_metric=custom_metric,
            random_state=42,
        )
        models, _ = reg.fit(X_train, X_test, y_train, y_test)
        # Reset the index so that model names become a column called "Model"
        models = models.reset_index().rename(columns={"index": "Model"})
        # Split the custom_metric tuple into two separate numeric columns
        models[["Spearman", "Pearson"]] = models["custom_metric"].apply(pd.Series)
        models = models.drop("custom_metric", axis=1)
        # Add the current test group as a column
        models["GROUP_OUT"] = group_value
        model_results.append(models)

    # Concatenate the results from each fold without averaging
    df = pd.concat(model_results).reset_index(drop=True)
    return df


def process_directory_with_correlation(paths, filename):
    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        for path in paths:
            df = evaluate_regressors(path)
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
    )


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
