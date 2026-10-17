import os
from pathlib import Path

import mappe
import pandas as pd
import torch
from dataset_configs import (
    TexturedMeshQualityConfig,
)
from mappe import MQAConfig
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAX_WORKERS = 10


def compute_features(
    row: pd.Series,
    dataset_base_path: str,
    mqa_config: MQAConfig,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict:
    ref_obj_location = os.path.normpath(
        os.path.join(dataset_base_path, row["REF_OBJ_LOCATION"])
    )
    dst_obj_location = os.path.normpath(
        os.path.join(dataset_base_path, row["SIGNAL_OBJ_LOCATION"])
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mqa_features = mappe.process_mqa(
        ref_obj_location,
        dst_obj_location,
        mqa_config,
        device,
        max_workers,
    )
    result_row = {
        "ID": row["ID"],
        "SCORE": row["SCORE"],
        "REF": ref_obj_location,
        "SIGNAL": dst_obj_location,
        "MAX_WORKERS": max_workers,
        "RENDER_CONFIG": mqa_config.render_config.model_dump(),
        "IQA_PROJECTION_CONFIG": mqa_config.iqa_projection_config.model_dump(),
        "IQA_LIGHT_CONFIG": mqa_config.iqa_projection_config.light_config.model_dump(),
        "IQA_VIEW_ANGLES": mqa_config.view_angles,
        "PCQA_SEARCH_SIZE": mqa_config.search_size,
        "PCQA_PROJECTION_CONFIG": mqa_config.pcqa_projection_config.model_dump(),
        "PCQA_PC_DENSITY": mqa_config.pcqa_projection_config.density_factor,
        "PCQA_LIGHT_CONFIG": mqa_config.pcqa_projection_config.light_config.model_dump(),
    }
    result_row.update(mqa_features)
    return result_row


def run_experiment(
    path_dataset_csv: str,
    mqa_config: MQAConfig,
    dataset_path: str = "",
    output_csv: str = "",
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> None:
    if not output_csv:
        raise ValueError("Invalid output_csv (Empty str or None)")
    folder_path = os.path.dirname(output_csv)
    os.makedirs(folder_path, exist_ok=True)
    df = pd.read_csv(path_dataset_csv)
    result_rows = []
    # Compute for the first row (warmup)
    first_row = df.iloc[0]
    compute_features(first_row, dataset_path, mqa_config, max_workers)
    # Now run the pipeline
    for idx, row in tqdm(df.iterrows(), "Processing rows", len(df)):
        try:
            result_row = compute_features(row, dataset_path, mqa_config, max_workers)
        except Exception as e:
            print(f"Exception on row {idx}:")
            print(e)
            continue
        result_rows.append(result_row)
    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(output_csv)


if __name__ == "__main__":
    BASE_PATH = "/home/arthurc/meshes/"

    # run_experiment(
    #     path_dataset_csv="data/SJTU-TMQA.csv",
    #     mqa_config=SJTUConfig(),
    #     dataset_path=BASE_PATH,
    #     output_csv="experiments/SJTU_MQAProject.csv",
    # )

    # run_experiment(
    #     path_dataset_csv="data/TSMD.csv",
    #     mqa_config=TSMDConfig(),
    #     dataset_path=BASE_PATH,
    #     output_csv="experiments/TSMD_MQAProject.csv",
    # )

    run_experiment(
        path_dataset_csv=str(PROJECT_ROOT / "data" / "TexturedMeshQuality.csv"),
        mqa_config=TexturedMeshQualityConfig(),
        dataset_path=BASE_PATH,
        output_csv=str(PROJECT_ROOT / "experiments" / "TMQ_MQAProject.csv"),
    )

    # run_experiment(
    #     path_dataset_csv="data/Yana_TAP_2021_ACR-HR.csv",
    #     mqa_config=YanaTAPConfig(),
    #     dataset_path=BASE_PATH,
    #     output_csv="experiments/Yana_TAP_2021_ACR-HR_MQAProject.csv",
    # )

    # run_experiment(
    #     path_dataset_csv="data/Yana_TAP_2021_DSIS.csv",
    #     mqa_config=YanaTAPConfig(),
    #     dataset_path=BASE_PATH,
    #     output_csv="experiments/Yana_TAP_2021_DSIS_MQAProject.csv",
    # )

    # run_experiment(
    #     path_dataset_csv="data/Yana_TAP_2021_SAMVIQ.csv",
    #     mqa_config=YanaTAPConfig(),
    #     dataset_path=BASE_PATH,
    #     output_csv="experiments/Yana_TAP_2021_SAMVIQ_MQAProject.csv",
    # )
