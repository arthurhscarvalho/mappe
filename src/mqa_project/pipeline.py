import time
import warnings
from concurrent.futures import ThreadPoolExecutor, wait

import mesh_projector as mp
import numpy as np
import torch
from piq import DISTS
from pointpca3 import compute_pointpca3

from .schemas import MQAConfig

# Ignore tensorflow warnings
warnings.filterwarnings("ignore")
print(f"CUDA available: {torch.cuda.is_available()}")


def _init_dists(device: torch.device) -> DISTS:
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
                _init_dists,
                device,
            ),
        ]
        wait(futures)
        ref_mesh = futures[0].result()
        dst_mesh = futures[1].result()
        dists = futures[2].result()
    return ref_mesh, dst_mesh, dists


def _crop_to_occupancy_square(image: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    if image.shape[:2] != occupancy.shape[:2]:
        raise ValueError("Image and occupancy map must share spatial dimensions.")
    foreground = occupancy > 0
    if not np.any(foreground):
        return image
    rows, cols = np.where(foreground)
    row_min, row_max = rows.min(), rows.max() + 1
    col_min, col_max = cols.min(), cols.max() + 1
    height = row_max - row_min
    width = col_max - col_min
    side = max(height, width)
    image_height, image_width = image.shape[:2]
    if side > image_height or side > image_width:
        return image
    row_padding = side - height
    col_padding = side - width
    row_start = row_min - (row_padding // 2)
    col_start = col_min - (col_padding // 2)
    row_start = min(max(row_start, 0), image_height - side)
    col_start = min(max(col_start, 0), image_width - side)
    row_end = row_start + side
    col_end = col_start + side
    return image[row_start:row_end, col_start:col_end, ...]


def _prepare_single_tensor(array: np.ndarray, device: torch.device):
    tensor = torch.from_numpy(array)  # zero-copy
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
    tensor = tensor.to(device=device, dtype=torch.float32)  # move + cast once
    tensor = tensor.div_(255.0)  # in-place division
    return tensor


def process_iqa(
    ref_mesh: mp.Mesh,
    dst_mesh: mp.Mesh,
    mqa_config: MQAConfig,
    dists: DISTS,
    device: torch.device,
) -> dict:
    ref_projs, _, _, ref_ocps = mp.project_to_images(
        ref_mesh, mqa_config.iqa_projection_config, mqa_config.view_angles
    )
    dst_projs, _, _, dst_ocps = mp.project_to_images(
        dst_mesh, mqa_config.iqa_projection_config, mqa_config.view_angles
    )
    with torch.no_grad():
        iqa_indices = []
        for ref_proj, dst_proj, ref_ocp, dst_ocp in zip(
            ref_projs, dst_projs, ref_ocps, dst_ocps
        ):
            combined_occupancy = (ref_ocp > 0) | (dst_ocp > 0)
            ref_crop = _crop_to_occupancy_square(ref_proj, combined_occupancy)
            dst_crop = _crop_to_occupancy_square(dst_proj, combined_occupancy)
            ref_tensor = _prepare_single_tensor(ref_crop, device)
            dst_tensor = _prepare_single_tensor(dst_crop, device)
            iqa_indices.append(dists(ref_tensor, dst_tensor).item())
    indices_data = {}
    for i, m in enumerate(iqa_indices):
        indices_data[f"QUALITY_INDEX_IQA_{i}"] = m
    return indices_data


def process_pcqa(
    ref_mesh: mp.Mesh,
    dst_mesh: mp.Mesh,
    mqa_config: MQAConfig,
    max_workers: int,
) -> dict:
    ref_pc = mp.project_to_point_cloud(ref_mesh, mqa_config.pcqa_projection_config)
    dst_pc = mp.project_to_point_cloud(dst_mesh, mqa_config.pcqa_projection_config)
    pcqa_indices = compute_pointpca3(
        *ref_pc, *dst_pc, mqa_config.search_size, max_workers
    )
    indices_data = {
        "PCQA_REF_PC_NUM_POINTS": ref_pc[0].shape[0],
        "PCQA_DST_PC_NUM_POINTS": dst_pc[0].shape[0],
    }
    for i, m in enumerate(pcqa_indices):
        indices_data[f"QUALITY_INDEX_PCQA_{i}"] = m
    return indices_data


def process_mqa(
    ref_obj_path: str,
    dst_obj_path: str,
    mqa_config: MQAConfig,
    device: torch.device,
    max_workers: int,
) -> dict:
    with ThreadPoolExecutor() as executor:
        t0 = time.time()
        ref_mesh, dst_mesh, dists = process_io(
            ref_obj_path, dst_obj_path, mqa_config.render_config, device
        )
        t1 = time.time()
        futures = [
            executor.submit(process_iqa, ref_mesh, dst_mesh, mqa_config, dists, device),
            executor.submit(process_pcqa, ref_mesh, dst_mesh, mqa_config, max_workers),
        ]
        wait(futures)
        for future in futures:
            if future.exception():
                raise future.exception()
        future_results = [future.result() for future in futures]
        iqa_indices, pcqa_indices = future_results
        t2 = time.time()
    mqa_features = {
        "TIME_DELTA_IO": t1 - t0,
        "TIME_DELTA_INFERENCE": t2 - t1,
        "TIME_DELTA_END_TO_END": t2 - t0,
    }
    mqa_features.update(iqa_indices)
    mqa_features.update(pcqa_indices)
    return mqa_features
