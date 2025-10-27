import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, wait

import mesh_projector as mp
import numpy as np
import pandas as pd
import torch
from piq import DISTS
from pointpca3 import compute_pointpca3
from tqdm import tqdm

import config

MAX_WORKERS = 10
# Ignore tensorflow warnings
warnings.filterwarnings("ignore")
print(f"CUDA available: {torch.cuda.is_available()}")


def init_dists(device: torch.device) -> DISTS:
    model = DISTS(reduction="none").to(device)
    return model


def process_io(
    ref_obj_location: str,
    dst_obj_location: str,
    render_config: mp.RenderConfig,
    device: torch.device,
) -> tuple[mp.Mesh, mp.Mesh, DISTS]:
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                mp.read_mesh,
                obj_path=ref_obj_location,
                render_config=render_config,
            ),
            executor.submit(
                mp.read_mesh,
                obj_path=dst_obj_location,
                render_config=render_config,
            ),
            executor.submit(
                init_dists,
                device,
            ),
        ]
        wait(futures)
        ref_mesh = futures[0].result()
        dst_mesh = futures[1].result()
        dists = futures[2].result()
    return ref_mesh, dst_mesh, dists


def prepare_single_tensor(array: np.ndarray, device: torch.device):
    tensor = torch.tensor(array, dtype=torch.float32, device=device)
    tensor = tensor.permute(0, 3, 1, 2).div(255.0)
    return tensor


def process_iqa(
    ref_mesh: mp.Mesh,
    dst_mesh: mp.Mesh,
    config: config.DatasetConfig,
    dists: DISTS,
    device: torch.device,
) -> dict:
    ref_projs, _, _, _ = mp.project_to_images(
        ref_mesh, config.view_angles, config.light_config_iqa
    )
    dst_projs, _, _, _ = mp.project_to_images(
        dst_mesh, config.view_angles, config.light_config_iqa
    )
    with torch.no_grad():
        ref_projs = prepare_single_tensor(ref_projs, device)
        dst_projs = prepare_single_tensor(dst_projs, device)
        iqa_indices = [i.item() for i in dists(ref_projs, dst_projs)]
    indices_data = {}
    for i, m in enumerate(iqa_indices):
        indices_data[f"QUALITY_INDEX_IQA_{i}"] = m
    return indices_data


def process_pcqa(
    ref_mesh: mp.Mesh, dst_mesh: mp.Mesh, config: config.DatasetConfig
) -> dict:
    ref_pc = mp.project_to_point_cloud(
        ref_mesh, config.light_config_pcqa, config.pc_density
    )
    dst_pc = mp.project_to_point_cloud(
        dst_mesh, config.light_config_pcqa, config.pc_density
    )
    pcqa_indices = compute_pointpca3(*ref_pc, *dst_pc, config.search_size, MAX_WORKERS)
    indices_data = {
        "PCQA_REF_PC_NUM_POINTS": ref_pc[0].shape[0],
        "PCQA_DST_PC_NUM_POINTS": dst_pc[0].shape[0],
    }
    for i, m in enumerate(pcqa_indices):
        indices_data[f"QUALITY_INDEX_PCQA_{i}"] = m
    return indices_data


def compute_features(
    row: pd.Series, dataset_base_path: str, config: config.DatasetConfig
) -> dict:
    ref_obj_location = row["REF_OBJ_LOCATION"]
    dst_obj_location = row["SIGNAL_OBJ_LOCATION"]
    ref_obj_location = os.path.join(dataset_base_path, ref_obj_location)
    dst_obj_location = os.path.join(dataset_base_path, dst_obj_location)
    ref_obj_location = os.path.normpath(ref_obj_location)
    dst_obj_location = os.path.normpath(dst_obj_location)
    signal_id = row["ID"]
    score = row["SCORE"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with ThreadPoolExecutor() as executor:
        t0 = time.time()
        # Read meshes
        ref_mesh, dst_mesh, dists = process_io(
            ref_obj_location, dst_obj_location, config.render_config, device
        )
        t1 = time.time()
        # Compute metrics
        device = "cuda" if torch.cuda.is_available() else "cpu"
        futures = [
            executor.submit(process_iqa, ref_mesh, dst_mesh, config, dists, device),
            executor.submit(process_pcqa, ref_mesh, dst_mesh, config),
        ]
        wait(futures)
        for future in futures:
            if future.exception():
                raise future.exception()
        future_results = [future.result() for future in futures]
        iqa_indices, pcqa_indices = future_results
        t2 = time.time()
    result_row = {
        "ID": signal_id,
        "SCORE": score,
        "REF": ref_obj_location,
        "SIGNAL": dst_obj_location,
        "TIME_DELTA_IO": t1 - t0,
        "TIME_DELTA_INFERENCE": t2 - t1,
        "TIME_DELTA_END_TO_END": t2 - t0,
        "MAX_WORKERS": MAX_WORKERS,
        "RENDER_CONFIG": config.render_config.model_dump(),
        "IQA_LIGHT_CONFIG": config.light_config_iqa.model_dump(),
        "IQA_VIEW_ANGLES": config.view_angles,
        "PCQA_SEARCH_SIZE": config.search_size,
        "PCQA_PC_DENSITY": config.pc_density,
        "PCQA_LIGHT_CONFIG": config.light_config_pcqa.model_dump(),
    }
    result_row.update(iqa_indices)
    result_row.update(pcqa_indices)
    return result_row


def run_experiment(
    path_dataset_csv: str,
    config: config.DatasetConfig,
    dataset_path: str = "",
    output_csv: str = "",
):
    if not output_csv:
        raise ValueError("Invalid output_csv (Empty str or None)")
    folder_path = os.path.dirname(output_csv)
    os.makedirs(folder_path, exist_ok=True)
    df = pd.read_csv(path_dataset_csv)
    result_rows = []
    # Compute for the first row (warmup)
    first_row = df.iloc[0]
    compute_features(first_row, dataset_path, config)
    # Now run the pipeline
    for idx, row in tqdm(df.iterrows(), "Processing rows", len(df)):
        try:
            result_row = compute_features(row, dataset_path, config)
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
    #     config=config.SJTUConfig(),
    #     dataset_path=BASE_PATH,
    #     output_csv="experiments/SJTU_MQAProject.csv",
    # )

    # run_experiment(
    #     path_dataset_csv="data/TSMD.csv",
    #     config=config.TSMDConfig(),
    #     dataset_path=BASE_PATH,
    #     output_csv="experiments/TSMD_MQAProject.csv",
    # )

    run_experiment(
        path_dataset_csv="data/TexturedMeshQuality.csv",
        config=config.TexturedMeshQualityConfig(),
        dataset_path=BASE_PATH,
        output_csv="experiments/TMQ_MQAProject.csv",
    )

    # run_experiment(
    #     path_dataset_csv="data/Yana_TAP_2021_ACR-HR.csv",
    #     config=config.YanaTAPConfig(),
    #     dataset_path=BASE_PATH,
    #     output_csv="experiments/Yana_TAP_2021_ACR-HR_MQAProject.csv",
    # )

    # run_experiment(
    #     path_dataset_csv="data/Yana_TAP_2021_DSIS.csv",
    #     config=config.YanaTAPConfig(),
    #     dataset_path=BASE_PATH,
    #     output_csv="experiments/Yana_TAP_2021_DSIS_MQAProject.csv",
    # )

    # run_experiment(
    #     path_dataset_csv="data/Yana_TAP_2021_SAMVIQ.csv",
    #     config=config.YanaTAPConfig(),
    #     dataset_path=BASE_PATH,
    #     output_csv="experiments/Yana_TAP_2021_SAMVIQ_MQAProject.csv",
    # )
