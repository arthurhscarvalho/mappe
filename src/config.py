from math import pi as PI
from typing import List, Tuple

import mesh_projector as mp
from pydantic import BaseModel


class DatasetConfig(BaseModel):
    search_size: int
    pc_density: float
    render_config: mp.RenderConfig
    light_config_iqa: mp.LightConfig
    light_config_pcqa: mp.LightConfig
    view_angles: List[Tuple[float, float, float]]


class TSMDConfig(DatasetConfig):
    search_size: int = 361
    pc_density: float = 0.3
    render_config: mp.RenderConfig = mp.RenderConfig(
        resolution=512,
        ignore_opacity=True,
    )
    light_config_iqa: mp.LightConfig = mp.PointLights(
        positions=[
            (-1.0, 1.0, 1.0),
            (1.0, 1.0, -1.0),
        ],
        global_ambient=(0.6, 0.6, 0.6),
        global_diffuse=(0.6, 0.6, 0.6),
        global_specular=(0.0, 0.0, 0.0),
    )
    light_config_pcqa: mp.LightConfig = mp.PointLights(
        positions=[
            (-1.0, 1.0, 1.0),
            (1.0, 1.0, -1.0),
        ],
        global_ambient=(0.6, 0.6, 0.6),
        global_diffuse=(0.6, 0.6, 0.6),
        global_specular=(0.0, 0.0, 0.0),
    )
    view_angles: List[Tuple[float, float, float]] = [
        (0.0, 0.0, 0.0),  # Front: Cam at +Z, looking -Z.
        (0.0, PI, 0.0),  # Back: Cam at -Z, looking +Z.
        (0.0, PI / 2.0, 0.0),  # Right: Cam at +X, looking -X.
        (0.0, -PI / 2.0, 0.0),  # Left: Cam at -X, looking +X.
        (PI / 2.0, 0.0, 0.0),  # Top: Cam at +Y, looking -Y.
        (-PI / 2.0, 0.0, 0.0),  # Bottom: Cam at -Y, looking +Y.
    ]


class SJTUConfig(DatasetConfig):
    search_size: int = 361
    pc_density: float = 0.3
    render_config: mp.RenderConfig = mp.RenderConfig(
        resolution=512,
        ignore_opacity=True,
        shading_mode=mp.ShadingMode.TEXTURE_MAPPED,
    )
    light_config_iqa: mp.LightConfig = mp.Headlight()
    light_config_pcqa: mp.LightConfig = mp.AmbientLight()
    view_angles: List[Tuple[float, float, float]] = [
        (0.0, 0.0, 0.0),  # Front: Cam at +Z, looking -Z.
        (0.0, PI, 0.0),  # Back: Cam at -Z, looking +Z.
        (0.0, PI / 2.0, 0.0),  # Right: Cam at +X, looking -X.
        (0.0, -PI / 2.0, 0.0),  # Left: Cam at -X, looking +X.
        (PI / 2.0, 0.0, 0.0),  # Top: Cam at +Y, looking -Y.
        (-PI / 2.0, 0.0, 0.0),  # Bottom: Cam at -Y, looking +Y.
    ]


class TexturedMeshQualityConfig(DatasetConfig):
    search_size: int = 289
    pc_density: float = 1.0
    render_config: mp.RenderConfig = mp.RenderConfig(
        resolution=512,
        ignore_opacity=True,
    )
    light_config_iqa: mp.LightConfig = mp.Headlight(
        global_ambient=(0.7, 0.7, 0.7),
        global_diffuse=(0.7, 0.7, 0.7),
        global_specular=(0.0, 0.0, 0.0),
    )
    light_config_pcqa: mp.LightConfig = mp.AmbientLight(
        global_ambient=(0.7, 0.7, 0.7),
        global_diffuse=(0.7, 0.7, 0.7),
        global_specular=(0.0, 0.0, 0.0),
    )
    view_angles: List[Tuple[float, float, float]] = [
        (0.0, 0.0, 0.0),  # Front: Cam at +Z, looking -Z.
        (0.0, PI, 0.0),  # Back: Cam at -Z, looking +Z.
        (0.0, PI / 2.0, 0.0),  # Right: Cam at +X, looking -X.
        (0.0, -PI / 2.0, 0.0),  # Left: Cam at -X, looking +X.
        (PI / 2.0, 0.0, 0.0),  # Top: Cam at +Y, looking -Y.
        (-PI / 2.0, 0.0, 0.0),  # Bottom: Cam at -Y, looking +Y.
    ]


class YanaTAPConfig(DatasetConfig):
    search_size: int = 361
    pc_density: float = 0.5
    render_config: mp.RenderConfig = mp.RenderConfig(
        resolution=512,
        ignore_opacity=True,
        shading_mode=mp.ShadingMode.VERTEX_COLOR,
    )
    light_config_iqa: mp.LightConfig = mp.AmbientLight()
    light_config_pcqa: mp.LightConfig = mp.AmbientLight()
    view_angles: List[Tuple[float, float, float]] = [
        (0.0, 0.0, 0.0),  # Front: Cam at +Z, looking -Z.
        (0.0, PI, 0.0),  # Back: Cam at -Z, looking +Z.
        (0.0, PI / 2.0, 0.0),  # Right: Cam at +X, looking -X.
        (0.0, -PI / 2.0, 0.0),  # Left: Cam at -X, looking +X.
        (PI / 2.0, 0.0, 0.0),  # Top: Cam at +Y, looking -Y.
        (-PI / 2.0, 0.0, 0.0),  # Bottom: Cam at -Y, looking +Y.
    ]
