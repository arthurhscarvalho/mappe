from __future__ import annotations

import argparse
import os
import resource
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mesh_projector as mp
import mqa_project
import numpy as np
import pandas as pd
import psutil
import torch
from dataset_configs import TSMDConfig
from mqa_project.pipeline import PointPCABackend, get_pointpca_backend
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_CSV = PROJECT_ROOT / "data" / "TSMD.csv"
DEFAULT_DATASET_BASE_PATH = Path("/home/arthurc/Documents/meshes/")
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "pointpca3_contribution_ablation_tsmd"
)
DEFAULT_MAX_WORKERS = 16

BACKEND_OUTPUTS = {
    "pointpca3": "TSMD_MAPPE_PointPCA3.csv",
    "pointpca2_rs": "TSMD_MAPPE_PointPCA2RS.csv",
}
KFOLD_OUTPUT = "pointpca3_contribution_ablation_tsmd_k_fold.xlsx"


@dataclass(frozen=True)
class MemorySample:
    baseline_mb: float
    peak_mb: float
    delta_mb: float


class RssSampler:
    def __init__(self, interval_seconds: float = 0.02) -> None:
        self.interval_seconds = interval_seconds
        self.process = psutil.Process(os.getpid()) if psutil is not None else None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_mb = 0.0
        self.peak_mb = 0.0

    def __enter__(self) -> "RssSampler":
        self.baseline_mb = self._rss_mb()
        self.peak_mb = self.baseline_mb
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_mb = max(self.peak_mb, self._rss_mb())

    def _rss_mb(self) -> float:
        if self.process is not None:
            return self.process.memory_info().rss / (1024 * 1024)
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    def _sample(self) -> None:
        while not self._stop_event.is_set():
            self.peak_mb = max(self.peak_mb, self._rss_mb())
            self._stop_event.wait(self.interval_seconds)

    def result(self) -> MemorySample:
        return MemorySample(
            baseline_mb=self.baseline_mb,
            peak_mb=self.peak_mb,
            delta_mb=max(0.0, self.peak_mb - self.baseline_mb),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TSMD PointPCA3 contribution ablation."
    )
    parser.add_argument(
        "--backend",
        choices=["pointpca3", "pointpca2_rs", "both"],
        default="both",
        help="PointPCA backend to process.",
    )
    parser.add_argument(
        "--dataset_csv",
        type=Path,
        default=DEFAULT_DATASET_CSV,
        help="TSMD metadata CSV.",
    )
    parser.add_argument(
        "--dataset_base_path",
        type=Path,
        default=DEFAULT_DATASET_BASE_PATH,
        help="Base directory for mesh paths in the TSMD CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where ablation outputs are written.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum PointPCA worker threads.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing backend CSVs.",
    )
    parser.add_argument(
        "--kfold",
        choices=["auto", "always", "never"],
        default="auto",
        help="Whether to run the GroupKFold workbook generation.",
    )
    parser.add_argument(
        "--no_warmup",
        action="store_true",
        help="Disable one-row backend warmup before measurement.",
    )
    return parser.parse_args()


def resolve_mesh_path(dataset_base_path: Path, relative_path: str) -> str:
    return os.path.normpath(os.path.join(dataset_base_path, relative_path))


def output_csv_for(output_dir: Path, backend: PointPCABackend) -> Path:
    return output_dir / BACKEND_OUTPUTS[backend]


def append_row(output_csv: Path, row: dict) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        output_csv,
        mode="a",
        index=False,
        header=not output_csv.exists(),
    )


def completed_ids(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    existing = pd.read_csv(output_csv, usecols=["ID"])
    return set(existing["ID"].astype(str))


def compute_backend_features(
    ref_pc: tuple,
    dst_pc: tuple,
    search_size: int,
    max_workers: int,
    backend: PointPCABackend,
) -> tuple[list[float], float, MemorySample]:
    compute_pointpca = get_pointpca_backend(backend)
    with RssSampler() as sampler:
        t0 = time.perf_counter()
        pcqa_indices = compute_pointpca(*ref_pc, *dst_pc, search_size, max_workers)
        elapsed = time.perf_counter() - t0
    return [float(value) for value in np.ravel(pcqa_indices)], elapsed, sampler.result()


def compute_features(
    row: pd.Series,
    dataset_base_path: Path,
    mqa_config: TSMDConfig,
    max_workers: int,
    backend: PointPCABackend,
    device: torch.device,
) -> dict:
    ref_obj_location = resolve_mesh_path(dataset_base_path, row["REF_OBJ_LOCATION"])
    dst_obj_location = resolve_mesh_path(dataset_base_path, row["SIGNAL_OBJ_LOCATION"])

    t0 = time.perf_counter()
    ref_mesh, dst_mesh, dists = mqa_project.process_io(
        ref_obj_location,
        dst_obj_location,
        mqa_config.render_config,
        device,
    )
    t1 = time.perf_counter()

    iqa_indices = mqa_project.process_iqa(
        ref_mesh,
        dst_mesh,
        mqa_config,
        dists,
        device,
    )
    t2 = time.perf_counter()

    ref_pc = mp.project_to_point_cloud(ref_mesh, mqa_config.pcqa_projection_config)
    dst_pc = mp.project_to_point_cloud(dst_mesh, mqa_config.pcqa_projection_config)
    t3 = time.perf_counter()

    pcqa_indices, pointpca_time, pointpca_memory = compute_backend_features(
        ref_pc,
        dst_pc,
        mqa_config.search_size,
        max_workers,
        backend,
    )

    result_row = {
        "ID": row["ID"],
        "SCORE": row["SCORE"],
        "REF": ref_obj_location,
        "SIGNAL": dst_obj_location,
        "POINTPCA_BACKEND": backend,
        "MAX_WORKERS": max_workers,
        "RENDER_CONFIG": mqa_config.render_config.model_dump(),
        "IQA_PROJECTION_CONFIG": mqa_config.iqa_projection_config.model_dump(),
        "IQA_LIGHT_CONFIG": mqa_config.iqa_projection_config.light_config.model_dump(),
        "IQA_VIEW_ANGLES": mqa_config.view_angles,
        "PCQA_SEARCH_SIZE": mqa_config.search_size,
        "PCQA_PROJECTION_CONFIG": mqa_config.pcqa_projection_config.model_dump(),
        "PCQA_PC_DENSITY": mqa_config.pcqa_projection_config.density_factor,
        "PCQA_LIGHT_CONFIG": mqa_config.pcqa_projection_config.light_config.model_dump(),
        "PCQA_REF_PC_NUM_POINTS": ref_pc[0].shape[0],
        "PCQA_DST_PC_NUM_POINTS": dst_pc[0].shape[0],
        "TIME_DELTA_IO": t1 - t0,
        "TIME_DELTA_IQA": t2 - t1,
        "TIME_DELTA_PC_PROJECTION": t3 - t2,
        "TIME_DELTA_POINTPCA_BACKEND": pointpca_time,
        "POINTPCA_BACKEND_RSS_BASELINE_MB": pointpca_memory.baseline_mb,
        "POINTPCA_BACKEND_RSS_PEAK_MB": pointpca_memory.peak_mb,
        "POINTPCA_BACKEND_RSS_DELTA_MB": pointpca_memory.delta_mb,
    }
    result_row.update(iqa_indices)
    for i, value in enumerate(pcqa_indices):
        result_row[f"QUALITY_INDEX_PCQA_{i}"] = value
    return result_row


def load_rows(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.dataset_csv)
    if args.limit is not None:
        df = df.head(args.limit)
    return df


def warmup(
    rows: pd.DataFrame,
    args: argparse.Namespace,
    mqa_config: TSMDConfig,
    backend: PointPCABackend,
    device: torch.device,
) -> None:
    if rows.empty or args.no_warmup:
        return
    compute_features(
        rows.iloc[0],
        args.dataset_base_path,
        mqa_config,
        args.max_workers,
        backend,
        device,
    )


def run_backend(args: argparse.Namespace, backend: PointPCABackend) -> Path:
    output_csv = output_csv_for(args.output_dir, backend)
    if args.overwrite and output_csv.exists():
        output_csv.unlink()

    df = load_rows(args)
    done = completed_ids(output_csv)
    rows_to_process = df[~df["ID"].astype(str).isin(done)].copy()
    mqa_config = TSMDConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not done:
        warmup(rows_to_process, args, mqa_config, backend, device)

    for idx, row in tqdm(
        rows_to_process.iterrows(),
        desc=f"Processing {backend}",
        total=len(rows_to_process),
    ):
        try:
            result_row = compute_features(
                row,
                args.dataset_base_path,
                mqa_config,
                args.max_workers,
                backend,
                device,
            )
        except Exception as exc:
            print(f"Exception on row index {idx}, ID {row.get('ID')}: {exc}")
            continue
        append_row(output_csv, result_row)

    print(f"{backend} CSV: {output_csv}")
    return output_csv


def child_command(args: argparse.Namespace, backend: PointPCABackend) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--backend",
        backend,
        "--dataset_csv",
        str(args.dataset_csv),
        "--dataset_base_path",
        str(args.dataset_base_path),
        "--output_dir",
        str(args.output_dir),
        "--max_workers",
        str(args.max_workers),
        "--kfold",
        "never",
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.overwrite:
        command.append("--overwrite")
    if args.no_warmup:
        command.append("--no_warmup")
    return command


def run_children(args: argparse.Namespace, backends: Iterable[PointPCABackend]) -> None:
    for backend in backends:
        subprocess.run(child_command(args, backend), check=True)


def validate_feature_csvs(output_dir: Path) -> list[Path]:
    paths = [output_csv_for(output_dir, backend) for backend in BACKEND_OUTPUTS]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing feature CSVs: {missing}")

    dataframes = [pd.read_csv(path) for path in paths]
    id_sets = [set(df["ID"].astype(str)) for df in dataframes]
    if id_sets[0] != id_sets[1]:
        raise ValueError("PointPCA3 and PointPCA2-RS CSVs do not have the same IDs.")

    for path, df in zip(paths, dataframes):
        iqa_cols = [col for col in df.columns if col.startswith("QUALITY_INDEX_IQA_")]
        pcqa_cols = [col for col in df.columns if col.startswith("QUALITY_INDEX_PCQA_")]
        if len(iqa_cols) != 6 or len(pcqa_cols) != 40:
            raise ValueError(
                f"{path} has {len(iqa_cols)} IQA and {len(pcqa_cols)} PCQA features; "
                "expected 6 and 40."
            )
        if df["REF"].nunique() < 5:
            raise ValueError(
                f"{path} has fewer than five REF groups; cannot run 5-fold."
            )
    return paths


def should_run_kfold(args: argparse.Namespace) -> bool:
    if args.kfold == "never":
        return False
    if args.kfold == "always":
        return True
    return args.backend == "both" and args.limit is None


def run_kfold(args: argparse.Namespace) -> None:
    feature_csvs = validate_feature_csvs(args.output_dir)
    output_xlsx = args.output_dir / KFOLD_OUTPUT
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "group_k_fold_test.py"),
        "--feature_csvs",
        *[str(path) for path in feature_csvs],
        "--output_xlsx",
        str(output_xlsx),
    ]
    subprocess.run(command, check=True)
    print(f"GroupKFold workbook: {output_xlsx}")


def main() -> None:
    args = parse_args()
    if args.backend == "both":
        run_children(args, ["pointpca3", "pointpca2_rs"])
    else:
        run_backend(args, args.backend)

    if should_run_kfold(args):
        run_kfold(args)


if __name__ == "__main__":
    main()
