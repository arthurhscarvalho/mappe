import math
import os
import time
from pathlib import Path

import mesh_projector as mp
import mqa_project
import pandas as pd
import torch
from dataset_configs import TexturedMeshQualityConfig
from mqa_project import MQAConfig
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

# Study configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_CSV = PROJECT_ROOT / "data" / "TexturedMeshQuality.csv"
DATASET_BASE_PATH = "/home/arthurc/Documents/meshes/"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "ablation_study"

VIEW_COUNTS = (1, 3, 6, 12, 20)
PC_DENSITIES = (0.3, 0.5, 1.0)
MAX_WORKERS = 10

RGB_07 = (0.7, 0.7, 0.7)
RGB_06 = (0.6, 0.6, 0.6)
RGB_00 = (0.0, 0.0, 0.0)


class LightVariant(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    iqa_light_config: mp.LightConfig
    pcqa_light_config: mp.LightConfig


LIGHT_VARIANTS = (
    LightVariant(
        name="baseline_tmq",
        iqa_light_config=mp.Headlight(
            global_ambient=RGB_07,
            global_diffuse=RGB_07,
            global_specular=RGB_00,
        ),
        pcqa_light_config=mp.AmbientLight(
            global_ambient=RGB_07,
            global_diffuse=RGB_07,
            global_specular=RGB_00,
        ),
    ),
    LightVariant(
        name="flat_ambient",
        iqa_light_config=mp.AmbientLight(
            global_ambient=RGB_07,
            global_diffuse=RGB_07,
            global_specular=RGB_00,
        ),
        pcqa_light_config=mp.AmbientLight(
            global_ambient=RGB_07,
            global_diffuse=RGB_07,
            global_specular=RGB_00,
        ),
    ),
    LightVariant(
        name="two_point_cross",
        iqa_light_config=mp.PointLights(
            positions=[
                (-1.0, 1.0, 1.0),
                (1.0, 1.0, -1.0),
            ],
            global_ambient=RGB_06,
            global_diffuse=RGB_06,
            global_specular=RGB_00,
        ),
        pcqa_light_config=mp.PointLights(
            positions=[
                (-1.0, 1.0, 1.0),
                (1.0, 1.0, -1.0),
            ],
            global_ambient=RGB_06,
            global_diffuse=RGB_06,
            global_specular=RGB_00,
        ),
    ),
)


class AblationRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    config: MQAConfig
    light_variant: str
    output_csv: Path


IqaKey = tuple[str, tuple[float, float, float]]
PcqaKey = tuple[str, float]


class IqaTask(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: IqaKey
    config: MQAConfig
    run_name: str
    view_number: int
    view_count: int


class PcqaTask(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: PcqaKey
    config: MQAConfig
    run_name: str
    density_factor: float


def direction_to_view_angle(
    direction: tuple[float, float, float],
) -> tuple[float, float, float]:
    x, y, z = direction
    length = math.sqrt(x * x + y * y + z * z)
    if length == 0.0:
        raise ValueError("Cannot convert a zero direction to a view angle")

    x /= length
    y /= length
    z /= length
    pitch = math.asin(max(-1.0, min(1.0, -y)))
    yaw = math.atan2(x, z)
    return (pitch, yaw, 0.0)


def directions_to_view_angles(
    directions: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    return [direction_to_view_angle(direction) for direction in directions]


def dodecahedron_face_normals() -> list[tuple[float, float, float]]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    directions = []
    for y in (-1.0, 1.0):
        for z in (-phi, phi):
            directions.append((0.0, y, z))
    for x in (-1.0, 1.0):
        for y in (-phi, phi):
            directions.append((x, y, 0.0))
    for x in (-phi, phi):
        for z in (-1.0, 1.0):
            directions.append((x, 0.0, z))
    return directions


def icosahedron_face_normals() -> list[tuple[float, float, float]]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    directions = []
    for x in (-1.0, 1.0):
        for y in (-1.0, 1.0):
            for z in (-1.0, 1.0):
                directions.append((x, y, z))
    for y in (-inv_phi, inv_phi):
        for z in (-phi, phi):
            directions.append((0.0, y, z))
    for x in (-inv_phi, inv_phi):
        for y in (-phi, phi):
            directions.append((x, y, 0.0))
    for x in (-phi, phi):
        for z in (-inv_phi, inv_phi):
            directions.append((x, 0.0, z))
    return directions


def build_view_angles(
    view_count: int, baseline_config: MQAConfig
) -> list[tuple[float, float, float]]:
    if view_count == 1:
        return [baseline_config.view_angles[0]]
    if view_count == 3:
        return directions_to_view_angles(
            [
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ]
        )
    if view_count == 6:
        return list(baseline_config.view_angles)
    if view_count == 12:
        return directions_to_view_angles(dodecahedron_face_normals())
    if view_count == 20:
        return directions_to_view_angles(icosahedron_face_normals())
    raise ValueError(f"Unsupported view_count: {view_count}")


def build_run_config(
    base_config: MQAConfig,
    view_angles: list[tuple[float, float, float]],
    light_variant: LightVariant,
    density_factor: float,
) -> MQAConfig:
    return build_mqa_config(base_config, view_angles, light_variant, density_factor)


def build_iqa_projection_config(
    base_config: MQAConfig,
    light_variant: LightVariant,
) -> mp.ImageProjectionConfig:
    base_projection = base_config.iqa_projection_config
    return mp.ImageProjectionConfig(
        filtering_radius=base_projection.filtering_radius,
        light_config=light_variant.iqa_light_config,
        camera_config=base_projection.camera_config,
        msaa_config=base_projection.msaa_config,
        ssao_config=base_projection.ssao_config,
        background_config=base_projection.background_config,
        verbose=base_projection.verbose,
    )


def build_pcqa_projection_config(
    base_config: MQAConfig,
    light_variant: LightVariant,
    density_factor: float,
) -> mp.PointCloudProjectionConfig:
    base_projection = base_config.pcqa_projection_config
    return mp.PointCloudProjectionConfig(
        density_factor=density_factor,
        light_config=light_variant.pcqa_light_config,
        verbose=base_projection.verbose,
    )


def build_mqa_config(
    base_config: MQAConfig,
    view_angles: list[tuple[float, float, float]],
    light_variant: LightVariant,
    density_factor: float,
) -> MQAConfig:
    return MQAConfig(
        search_size=base_config.search_size,
        render_config=base_config.render_config,
        iqa_projection_config=build_iqa_projection_config(
            base_config,
            light_variant,
        ),
        pcqa_projection_config=build_pcqa_projection_config(
            base_config,
            light_variant,
            density_factor,
        ),
        view_angles=view_angles,
    )


def build_runs() -> list[AblationRun]:
    baseline_config = TexturedMeshQualityConfig()
    baseline_light = LIGHT_VARIANTS[0]
    runs = []

    for view_count in VIEW_COUNTS:
        run_config = build_run_config(
            baseline_config,
            build_view_angles(view_count, baseline_config),
            baseline_light,
            baseline_config.pcqa_projection_config.density_factor,
        )
        runs.append(
            AblationRun(
                name=f"views_{view_count}",
                config=run_config,
                light_variant=baseline_light.name,
                output_csv=OUTPUT_DIR / f"views_{view_count}.csv",
            )
        )

    for light_variant in LIGHT_VARIANTS:
        run_config = build_run_config(
            baseline_config,
            list(baseline_config.view_angles),
            light_variant,
            baseline_config.pcqa_projection_config.density_factor,
        )
        runs.append(
            AblationRun(
                name=f"light_{light_variant.name}",
                config=run_config,
                light_variant=light_variant.name,
                output_csv=OUTPUT_DIR / f"light_{light_variant.name}.csv",
            )
        )

    for density_factor in PC_DENSITIES:
        run_config = build_run_config(
            baseline_config,
            list(baseline_config.view_angles),
            baseline_light,
            density_factor,
        )
        run_name = f"density_{str(density_factor).replace('.', '_')}"
        runs.append(
            AblationRun(
                name=run_name,
                config=run_config,
                light_variant=baseline_light.name,
                output_csv=OUTPUT_DIR / f"{run_name}.csv",
            )
        )

    return runs


def make_result_row(
    source_row: pd.Series,
    ref_obj_location: str,
    dst_obj_location: str,
    run: AblationRun,
    iqa_scores: dict[IqaKey, float],
    pcqa_scores: dict[PcqaKey, dict],
    io_time: float,
    iqa_time: float,
    pcqa_time: float,
) -> dict:
    result_row = {
        "ID": source_row["ID"],
        "SCORE": source_row["SCORE"],
        "REF": ref_obj_location,
        "SIGNAL": dst_obj_location,
        "TIME_DELTA_IO": io_time,
        "TIME_DELTA_IQA": iqa_time,
        "TIME_DELTA_PCQA": pcqa_time,
        "MAX_WORKERS": MAX_WORKERS,
        "RENDER_CONFIG": run.config.render_config.model_dump(),
        "IQA_VIEW_ANGLES_COUNT": len(run.config.view_angles),
        "IQA_VIEW_ANGLES": run.config.view_angles,
        "IQA_PROJECTION_CONFIG": run.config.iqa_projection_config.model_dump(),
        "IQA_LIGHT_CONFIG": run.config.iqa_projection_config.light_config.model_dump(),
        "PCQA_SEARCH_SIZE": run.config.search_size,
        "PCQA_PROJECTION_CONFIG": run.config.pcqa_projection_config.model_dump(),
        "PCQA_PC_DENSITY": run.config.pcqa_projection_config.density_factor,
        "PCQA_LIGHT_CONFIG": run.config.pcqa_projection_config.light_config.model_dump(),
    }

    for i, view_angle in enumerate(run.config.view_angles):
        result_row[f"QUALITY_INDEX_IQA_{i}"] = iqa_scores[
            (run.light_variant, view_angle)
        ]
    density_factor = run.config.pcqa_projection_config.density_factor
    result_row.update(pcqa_scores[(run.light_variant, density_factor)])
    return result_row


def resolve_mesh_paths(row: pd.Series) -> tuple[str, str]:
    ref_obj_location = os.path.normpath(
        os.path.join(DATASET_BASE_PATH, row["REF_OBJ_LOCATION"])
    )
    dst_obj_location = os.path.normpath(
        os.path.join(DATASET_BASE_PATH, row["SIGNAL_OBJ_LOCATION"])
    )
    return ref_obj_location, dst_obj_location


def load_row_inputs(
    row: pd.Series,
    render_config: mp.RenderConfig,
    device: str,
) -> tuple[mp.Mesh, mp.Mesh, object, str, str, float]:
    ref_obj_location, dst_obj_location = resolve_mesh_paths(row)

    started_at = time.time()
    ref_mesh, dst_mesh, dists = mqa_project.process_io(
        ref_obj_location,
        dst_obj_location,
        render_config,
        device,
    )
    io_time = time.time() - started_at
    return ref_mesh, dst_mesh, dists, ref_obj_location, dst_obj_location, io_time


def build_iqa_tasks(
    runs: list[AblationRun],
) -> dict[IqaKey, IqaTask]:
    tasks = {}
    for run in runs:
        total_views = len(run.config.view_angles)
        for view_index, view_angle in enumerate(run.config.view_angles):
            key = (run.light_variant, view_angle)
            if key not in tasks:
                single_view_config = MQAConfig(
                    search_size=run.config.search_size,
                    render_config=run.config.render_config,
                    iqa_projection_config=run.config.iqa_projection_config,
                    pcqa_projection_config=run.config.pcqa_projection_config,
                    view_angles=[view_angle],
                )
                tasks[key] = IqaTask(
                    key=key,
                    config=single_view_config,
                    run_name=run.name,
                    view_number=view_index + 1,
                    view_count=total_views,
                )
    return tasks


def build_pcqa_tasks(runs: list[AblationRun]) -> dict[PcqaKey, PcqaTask]:
    tasks = {}
    for run in runs:
        density_factor = run.config.pcqa_projection_config.density_factor
        key = (run.light_variant, density_factor)
        if key not in tasks:
            tasks[key] = PcqaTask(
                key=key,
                config=run.config,
                run_name=run.name,
                density_factor=density_factor,
            )
    return tasks


def compute_iqa_scores(
    ref_mesh: mp.Mesh,
    dst_mesh: mp.Mesh,
    dists: object,
    device: str,
    tasks: dict[IqaKey, IqaTask],
    progress_bar: tqdm | None = None,
) -> tuple[dict[IqaKey, float], float]:
    scores = {}
    total_time = 0.0

    for key, task in tasks.items():
        if progress_bar is not None:
            progress_bar.set_postfix_str(
                f"IQA {task.run_name} | view {task.view_number}/{task.view_count}"
            )

        started_at = time.time()
        iqa_data = mqa_project.process_iqa(
            ref_mesh,
            dst_mesh,
            task.config,
            dists,
            device,
        )
        total_time += time.time() - started_at
        scores[key] = iqa_data["QUALITY_INDEX_IQA_0"]

    return scores, total_time


def compute_pcqa_scores(
    ref_mesh: mp.Mesh,
    dst_mesh: mp.Mesh,
    tasks: dict[PcqaKey, PcqaTask],
    progress_bar: tqdm | None = None,
) -> tuple[dict[PcqaKey, dict], float]:
    scores = {}
    total_time = 0.0

    for key, task in tasks.items():
        if progress_bar is not None:
            progress_bar.set_postfix_str(
                f"PCQA {task.run_name} | density={task.density_factor}"
            )

        started_at = time.time()
        scores[key] = mqa_project.process_pcqa(
            ref_mesh, dst_mesh, task.config, MAX_WORKERS
        )
        total_time += time.time() - started_at

    return scores, total_time


def compute_cached_ablation_rows(
    row: pd.Series,
    runs: list[AblationRun],
    progress_bar: tqdm | None = None,
) -> dict[str, dict]:
    baseline_config = runs[0].config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_mesh, dst_mesh, dists, ref_obj_location, dst_obj_location, io_time = (
        load_row_inputs(row, baseline_config.render_config, device)
    )
    iqa_scores, iqa_time = compute_iqa_scores(
        ref_mesh,
        dst_mesh,
        dists,
        device,
        build_iqa_tasks(runs),
        progress_bar,
    )
    pcqa_scores, pcqa_time = compute_pcqa_scores(
        ref_mesh,
        dst_mesh,
        build_pcqa_tasks(runs),
        progress_bar,
    )

    return {
        run.name: make_result_row(
            row,
            ref_obj_location,
            dst_obj_location,
            run,
            iqa_scores,
            pcqa_scores,
            io_time,
            iqa_time,
            pcqa_time,
        )
        for run in runs
    }


def run_cached_ablation_study(runs: list[AblationRun]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_CSV)
    result_rows_by_run = {run.name: [] for run in runs}

    output_bar = tqdm(runs, desc="Running ablation outputs", unit="run")
    for run in output_bar:
        density_factor = run.config.pcqa_projection_config.density_factor
        output_bar.set_postfix_str(
            f"{run.name} | views={len(run.config.view_angles)} | "
            f"light={run.light_variant} | density={density_factor}"
        )

    row_bar = tqdm(df.iterrows(), desc="Running ablation study", total=len(df))
    for idx, row in row_bar:
        try:
            rows_by_run = compute_cached_ablation_rows(row, runs, row_bar)
        except Exception as e:
            print(f"Exception on row {idx}:")
            print(e)
            continue

        for run_name, result_row in rows_by_run.items():
            result_rows_by_run[run_name].append(result_row)

    write_bar = tqdm(runs, desc="Writing ablation outputs", unit="run")
    for run in write_bar:
        write_bar.set_postfix_str(run.name)
        pd.DataFrame(result_rows_by_run[run.name]).to_csv(run.output_csv)


def main() -> None:
    runs = build_runs()
    run_cached_ablation_study(runs)


if __name__ == "__main__":
    main()
