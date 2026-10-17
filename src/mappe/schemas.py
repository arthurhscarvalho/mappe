import mesh_projector as mp
from pydantic import BaseModel, ConfigDict


class MQAConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    search_size: int
    render_config: mp.RenderConfig
    iqa_projection_config: mp.ImageProjectionConfig
    pcqa_projection_config: mp.PointCloudProjectionConfig
    view_angles: list[tuple[float, float, float]]
