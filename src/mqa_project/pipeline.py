import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Callable, Literal

import mesh_projector as mp
import torch
from numpy.typing import NDArray
from piq import DISTS
from pointpca2 import compute_pointpca2
from pointpca3 import compute_pointpca3

from .schemas import MQAConfig

logging.info(f"CUDA available: {torch.cuda.is_available()}")
PointPCABackend = Literal["pointpca3", "pointpca2_rs"]
PointPCAFunction = Callable[[NDArray, NDArray, NDArray, NDArray, int, int], NDArray]


def get_pointpca_backend(backend: PointPCABackend = "pointpca3") -> PointPCAFunction:
    if backend == "pointpca3":
        return compute_pointpca3
    if backend == "pointpca2_rs":
        return compute_pointpca2
    raise ValueError(
        f"Unsupported PointPCA backend '{backend}'. Expected 'pointpca3' or 'pointpca2_rs'."
    )


def _init_dists(device: torch.device) -> DISTS:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The parameter 'pretrained' is deprecated since 0\.13.*",
            category=UserWarning,
            module=r"torchvision\.models\._utils",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Arguments other than a weight enum or `None` for 'weights' are deprecated since 0\.13.*",
            category=UserWarning,
            module=r"torchvision\.models\._utils",
        )
        model = DISTS(reduction="none").to(device)
    return model


def _load_preprocessed_meshes(
    ref_obj_location: str,
    dst_obj_location: str,
    render_config: mp.RenderConfig,
) -> tuple[mp.Mesh, mp.Mesh]:
    ref_mesh = mp.read_mesh(
        obj_path=ref_obj_location,
        render_config=render_config,
    )
    preprocessing = mp.build_preprocessing(ref_mesh, render_config)
    dst_mesh = mp.read_mesh(
        obj_path=dst_obj_location,
        render_config=render_config,
    )
    ref_mesh = mp.apply_preprocessing(ref_mesh, preprocessing)
    dst_mesh = mp.apply_preprocessing(dst_mesh, preprocessing)
    return ref_mesh, dst_mesh


def process_io(
    ref_obj_location: str,
    dst_obj_location: str,
    render_config: mp.RenderConfig,
    device: torch.device,
) -> tuple[mp.Mesh, mp.Mesh, DISTS]:
    with ThreadPoolExecutor() as executor:
        mesh_future = executor.submit(
            _load_preprocessed_meshes,
            ref_obj_location,
            dst_obj_location,
            render_config,
        )
        dists_future = executor.submit(_init_dists, device)
        wait([mesh_future, dists_future])
        ref_mesh, dst_mesh = mesh_future.result()
        dists = dists_future.result()
    return ref_mesh, dst_mesh, dists


def prepare_projection_tensor(array: NDArray, device: torch.device):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The given NumPy array is not writable.*",
            category=UserWarning,
        )
        # No problem here since we won't write to this array
        tensor = torch.from_numpy(array)
    tensor = tensor.to(device=device, dtype=torch.float32)  # move + cast together
    tensor = tensor.narrow(-1, 0, 3)  # RGBA -> RGB
    tensor = tensor.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
    tensor = tensor.div_(255.0)  # in-place normalization
    return tensor


def process_iqa(
    ref_mesh: mp.Mesh,
    dst_mesh: mp.Mesh,
    mqa_config: MQAConfig,
    dists: DISTS,
    device: torch.device,
) -> dict:
    ref_projs, _, _, _ = mp.project_to_images(
        ref_mesh, mqa_config.iqa_projection_config, mqa_config.view_angles
    )
    dst_projs, _, _, _ = mp.project_to_images(
        dst_mesh, mqa_config.iqa_projection_config, mqa_config.view_angles
    )
    with torch.no_grad():
        ref_projs = prepare_projection_tensor(ref_projs, device)
        dst_projs = prepare_projection_tensor(dst_projs, device)
        iqa_indices = [i.item() for i in dists(ref_projs, dst_projs)]
    indices_data = {}
    for i, m in enumerate(iqa_indices):
        indices_data[f"QUALITY_INDEX_IQA_{i}"] = m
    return indices_data


def process_pcqa(
    ref_mesh: mp.Mesh,
    dst_mesh: mp.Mesh,
    mqa_config: MQAConfig,
    max_workers: int,
    pointpca_backend: PointPCABackend = "pointpca3",
) -> dict:
    ref_pc = mp.project_to_point_cloud(ref_mesh, mqa_config.pcqa_projection_config)
    dst_pc = mp.project_to_point_cloud(dst_mesh, mqa_config.pcqa_projection_config)
    compute_pointpca = get_pointpca_backend(pointpca_backend)
    pcqa_indices = compute_pointpca(
        *ref_pc, *dst_pc, mqa_config.search_size, max_workers
    )
    indices_data = {
        "POINTPCA_BACKEND": pointpca_backend,
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
    pointpca_backend: PointPCABackend = "pointpca3",
) -> dict:
    with ThreadPoolExecutor() as executor:
        t0 = time.time()
        ref_mesh, dst_mesh, dists = process_io(
            ref_obj_path, dst_obj_path, mqa_config.render_config, device
        )
        t1 = time.time()
        futures = [
            executor.submit(process_iqa, ref_mesh, dst_mesh, mqa_config, dists, device),
            executor.submit(
                process_pcqa,
                ref_mesh,
                dst_mesh,
                mqa_config,
                max_workers,
                pointpca_backend,
            ),
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
