import os
import time
import warnings
from math import pi as PI

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait
from piq import DISTS

import mesh_projector as mp
from pointpca3 import compute_pointpca3


# Ignore tensorflow warnings
warnings.filterwarnings("ignore")
print(f"CUDA available: {torch.cuda.is_available()}")

# Project parameters
SEARCH_SIZE = 361
MAX_WORKERS = 10
PC_DENSITY = 0.3

# Rendering configuration
RENDER_CONFIG = mp.RenderConfig(
    resolution=512,
    ignore_opacity=True,
)

# Lighting configuration
LIGHT_CONFIG_IQA = mp.PointLights(
    positions=[
        (-1.0, 1.0, 1.0),
        (1.0, 1.0, -1.0),
    ],
    global_ambient=(0.6, 0.6, 0.6),
    global_diffuse=(0.6, 0.6, 0.6),
    global_specular=(0.0, 0.0, 0.0),
)
LIGHT_CONFIG_PCQA = LIGHT_CONFIG_IQA

# View angles (projected views)
VIEW_ANGLES = [
    (0.0, 0.0, 0.0),  # Front: Cam at +Z, looking -Z.
    (0.0, PI, 0.0),  # Back: Cam at -Z, looking +Z.
    (0.0, PI / 2.0, 0.0),  # Right: Cam at +X, looking -X.
    (0.0, -PI / 2.0, 0.0),  # Left: Cam at -X, looking +X.
    (PI / 2.0, 0.0, 0.0),  # Top: Cam at +Y, looking -Y.
    (-PI / 2.0, 0.0, 0.0),  # Bottom: Cam at -Y, looking +Y.
]


def init_dists(device: torch.device):
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


def process_iqa(ref_mesh, dst_mesh, dists, device):
    ref_projs, _, _, _ = mp.project_to_images(ref_mesh, VIEW_ANGLES, LIGHT_CONFIG_IQA)
    dst_projs, _, _, _ = mp.project_to_images(dst_mesh, VIEW_ANGLES, LIGHT_CONFIG_IQA)
    with torch.no_grad():
        ref_projs = prepare_single_tensor(ref_projs, device)
        dst_projs = prepare_single_tensor(dst_projs, device)
        iqa_indices = [i.item() for i in dists(ref_projs, dst_projs)]
    indices_data = {}
    for i, m in enumerate(iqa_indices):
        indices_data[f"QUALITY_INDEX_IQA_{i}"] = m
    return indices_data


def process_pcqa(ref_mesh, dst_mesh):
    ref_pc = mp.project_to_point_cloud(ref_mesh, LIGHT_CONFIG_PCQA, PC_DENSITY)
    dst_pc = mp.project_to_point_cloud(dst_mesh, LIGHT_CONFIG_PCQA, PC_DENSITY)
    pcqa_indices = compute_pointpca3(*ref_pc, *dst_pc, SEARCH_SIZE, MAX_WORKERS)
    indices_data = {
        "PCQA_REF_PC_NUM_POINTS": ref_pc[0].shape[0],
        "PCQA_DST_PC_NUM_POINTS": dst_pc[0].shape[0],
    }
    for i, m in enumerate(pcqa_indices):
        indices_data[f"QUALITY_INDEX_PCQA_{i}"] = m
    return indices_data


def compute_features(row: pd.Series, dataset_base_path: str) -> dict:
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
            ref_obj_location, dst_obj_location, RENDER_CONFIG, device
        )
        t1 = time.time()
        # Compute metrics
        device = "cuda" if torch.cuda.is_available() else "cpu"
        futures = [
            executor.submit(process_iqa, ref_mesh, dst_mesh, dists, device),
            executor.submit(process_pcqa, ref_mesh, dst_mesh),
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
        "RENDER_CONFIG": RENDER_CONFIG.model_dump(),
        "IQA_LIGHT_CONFIG": LIGHT_CONFIG_IQA.model_dump(),
        "IQA_VIEW_ANGLES": VIEW_ANGLES,
        "PCQA_SEARCH_SIZE": SEARCH_SIZE,
        "PCQA_PC_DENSITY": PC_DENSITY,
        "PCQA_LIGHT_CONFIG": LIGHT_CONFIG_PCQA.model_dump(),
    }
    result_row.update(iqa_indices)
    result_row.update(pcqa_indices)
    return result_row


def run_experiment(path_dataset_csv: str, dataset_path: str = "", output_csv: str = ""):
    if not output_csv:
        raise ValueError("Invalid output_csv (Empty str or None)")
    folder_path = os.path.dirname(output_csv)
    os.makedirs(folder_path, exist_ok=True)
    df = pd.read_csv(path_dataset_csv)
    result_rows = []
    # Compute for the first row (warmup)
    first_row = df.iloc[0]
    compute_features(first_row, dataset_path)
    # Now run the pipeline
    for idx, row in tqdm(df.iterrows(), "Processing rows", len(df)):
        try:
            result_row = compute_features(row, dataset_path)
        except Exception as e:
            print(f"Exception on row {idx}:")
            print(e)
            continue
        result_rows.append(result_row)
    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(output_csv)


if __name__ == "__main__":
    BASE_PATH = "/home/arthurc/meshes/"

    run_experiment(
        "data/SJTU-TMQA.csv",
        BASE_PATH,
        "experiments/SJTU_MQAProject.csv",
    )

    run_experiment(
        "data/TSMD.csv",
        BASE_PATH,
        "experiments/TSMD_MQAProject.csv",
    )

    run_experiment(
        "data/TexturedMeshQuality.csv",
        BASE_PATH,
        "experiments/TMQ_MQAProject.csv",
    )

    run_experiment(
        "data/Yana_TAP_2021_ACR-HR.csv",
        BASE_PATH,
        "experiments/Yana_TAP_2021_ACR-HR_MQAProject.csv",
    )

    run_experiment(
        "data/Yana_TAP_2021_DSIS.csv",
        BASE_PATH,
        "experiments/Yana_TAP_2021_DSIS_MQAProject.csv",
    )

    run_experiment(
        "data/Yana_TAP_2021_SAMVIQ.csv",
        BASE_PATH,
        "experiments/Yana_TAP_2021_SAMVIQ_MQAProject.csv",
    )
