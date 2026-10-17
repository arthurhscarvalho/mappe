from math import pi as PI
from typing import List, Tuple

import mesh_projector as mp
from mappe import MQAConfig

CUBE_VIEW_ANGLES: List[Tuple[float, float, float]] = [
    (0.0, 0.0, 0.0),  # Front: Cam at +Z, looking -Z.
    (0.0, PI, 0.0),  # Back: Cam at -Z, looking +Z.
    (0.0, PI / 2.0, 0.0),  # Right: Cam at +X, looking -X.
    (0.0, -PI / 2.0, 0.0),  # Left: Cam at -X, looking +X.
    (PI / 2.0, 0.0, 0.0),  # Top: Cam at +Y, looking -Y.
    (-PI / 2.0, 0.0, 0.0),  # Bottom: Cam at -Y, looking +Y.
]


class TSMDConfig(MQAConfig):
    search_size: int = 361
    render_config: mp.RenderConfig = mp.RenderConfig(
        resolution=512,
        ignore_opacity=True,
    )
    iqa_projection_config: mp.ImageProjectionConfig = mp.ImageProjectionConfig(
        light_config=mp.PointLights(
            positions=[
                (-1.0, 1.0, 1.0),
                (1.0, 1.0, -1.0),
            ],
            global_ambient=(0.6, 0.6, 0.6),
            global_diffuse=(0.6, 0.6, 0.6),
            global_specular=(0.0, 0.0, 0.0),
        )
    )
    pcqa_projection_config: mp.PointCloudProjectionConfig = mp.PointCloudProjectionConfig(
        density_factor=0.3,
        light_config=mp.PointLights(
            positions=[
                (-1.0, 1.0, 1.0),
                (1.0, 1.0, -1.0),
            ],
            global_ambient=(0.6, 0.6, 0.6),
            global_diffuse=(0.6, 0.6, 0.6),
            global_specular=(0.0, 0.0, 0.0),
        ),
    )
    view_angles: List[Tuple[float, float, float]] = CUBE_VIEW_ANGLES


class SJTUConfig(MQAConfig):
    search_size: int = 361
    render_config: mp.RenderConfig = mp.RenderConfig(
        resolution=512,
        ignore_opacity=True,
        shading_mode=mp.ShadingMode.TEXTURE_MAPPED,
    )
    iqa_projection_config: mp.ImageProjectionConfig = mp.ImageProjectionConfig(
        light_config=mp.Headlight()
    )
    pcqa_projection_config: mp.PointCloudProjectionConfig = mp.PointCloudProjectionConfig(
        density_factor=0.3,
        light_config=mp.AmbientLight(),
    )
    view_angles: List[Tuple[float, float, float]] = CUBE_VIEW_ANGLES


class TexturedMeshQualityConfig(MQAConfig):
    search_size: int = 289
    render_config: mp.RenderConfig = mp.RenderConfig(
        resolution=512,
        ignore_opacity=True,
    )
    iqa_projection_config: mp.ImageProjectionConfig = mp.ImageProjectionConfig(
        light_config=mp.Headlight(
            global_ambient=(0.7, 0.7, 0.7),
            global_diffuse=(0.7, 0.7, 0.7),
            global_specular=(0.0, 0.0, 0.0),
        )
    )
    pcqa_projection_config: mp.PointCloudProjectionConfig = mp.PointCloudProjectionConfig(
        density_factor=1.0,
        light_config=mp.AmbientLight(
            global_ambient=(0.7, 0.7, 0.7),
            global_diffuse=(0.7, 0.7, 0.7),
            global_specular=(0.0, 0.0, 0.0),
        ),
    )
    view_angles: List[Tuple[float, float, float]] = CUBE_VIEW_ANGLES


class YanaTAPConfig(MQAConfig):
    search_size: int = 361
    render_config: mp.RenderConfig = mp.RenderConfig(
        resolution=512,
        ignore_opacity=True,
        shading_mode=mp.ShadingMode.VERTEX_COLOR,
    )
    iqa_projection_config: mp.ImageProjectionConfig = mp.ImageProjectionConfig(
        light_config=mp.AmbientLight()
    )
    pcqa_projection_config: mp.PointCloudProjectionConfig = mp.PointCloudProjectionConfig(
        density_factor=0.5,
        light_config=mp.AmbientLight(),
    )
    view_angles: List[Tuple[float, float, float]] = CUBE_VIEW_ANGLES
