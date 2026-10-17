# MAPPE

## Purpose

MAPPE (Mesh Assessment via Point Cloud and Projection Estimators) is a full-reference mesh quality assessment framework. It combines image quality features from rendered views with PointPCA3 features from sampled point clouds. This repository is the entry point for the MAPPE experiments and regression-based evaluation.

The Python distribution and import are both named `mappe`.

## Related projects

| Project | Role |
| --- | --- |
| [mesh-projector-pylib](https://github.com/arthurhscarvalho/mesh-projector-pylib) | Python interface for rendering views and sampling colored point clouds. |
| [mesh-projector-rs](https://github.com/arthurhscarvalho/mesh-projector-rs) | Rust mesh processing, CPU rendering, and GPU rendering backend. |
| [pointpca3-pylib](https://github.com/arthurhscarvalho/pointpca3-pylib) | Python interface for PointPCA3 feature extraction. |
| [pointpca3-rs](https://github.com/arthurhscarvalho/pointpca3-rs) | Rust PointPCA3 backend with Voxelization-based Spatial Sampling (VoSS). |
| [pointpca2-pylib](https://github.com/arthurhscarvalho/pointpca2-pylib) | Python interface for the alternative PointPCA2-RS backend. |
| [pointpca2-rs](https://github.com/arthurhscarvalho/pointpca2-rs) | Rust PointPCA2 implementation, also used by PointPCA3. |

## Requirements

- Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).
- Git to clone the repository and network access to download Python packages.
- Mesh datasets, including OBJ files and any associated MTL files and texture images.

The package requires Python `>=3.11,<3.13`; its mesh-projector dependency uses `enum.StrEnum`, which requires Python 3.11 or later. A GPU is optional for the default CPU mesh renderer and CPU PyTorch execution. GPU execution requires compatible hardware and drivers.

## Installation

Clone MAPPE and install the default environment:

```bash
git clone https://github.com/arthurhscarvalho/mappe.git mappe
cd mappe
uv sync --python 3.12
```

MAPPE installs `mesh-projector==1.0.0` and `pointpca3==1.0.0` from PyPI. Their published wheels include the Rust extensions, so supported platforms require no Rust toolchain or sibling repository checkouts. `pointpca2` is also resolved from PyPI.

For the configured AMD ROCm environment, use:

```bash
uv sync --python 3.12 --no-default-groups --extra rocm
```

This selects the ROCm 6.3 package index configured in [pyproject.toml](pyproject.toml). Use a compatible ROCm installation and GPU. Run subsequent `uv run` commands with `--no-default-groups --extra rocm` as well, for example `uv run --no-default-groups --extra rocm python scripts/main_experiment.py`.

## Usage

```python
from math import pi

import mappe
import mesh_projector as mp

light_config = mp.PointLights(
    positions=[(-1.0, 1.0, 1.0), (1.0, 1.0, -1.0)],
    global_ambient=(0.6, 0.6, 0.6),
    global_diffuse=(0.6, 0.6, 0.6),
    global_specular=(0.0, 0.0, 0.0),
)
mqa_config = mappe.MQAConfig(
    search_size=361,
    render_config=mp.RenderConfig(resolution=512, ignore_opacity=True),
    iqa_projection_config=mp.ImageProjectionConfig(light_config=light_config),
    pcqa_projection_config=mp.PointCloudProjectionConfig(
        density_factor=0.3,
        light_config=light_config,
    ),
    view_angles=[
        (0.0, 0.0, 0.0),
        (0.0, pi, 0.0),
        (0.0, pi / 2, 0.0),
        (0.0, -pi / 2, 0.0),
        (pi / 2, 0.0, 0.0),
        (-pi / 2, 0.0, 0.0),
    ],
)

features = mappe.process_mqa(
    "reference.obj",
    "distorted.obj",
    mqa_config=mqa_config,
    device="cpu",
    max_workers=8,
)
print(features)
```

## Experiments

The [scripts/](scripts) directory contains dataset experiments, ablation studies, parameter searches, and regression evaluations. Dataset-specific configurations are in [scripts/dataset_configs.py](scripts/dataset_configs.py), and metadata CSVs are in [data/](data). Configure dataset paths and experiment parameters before running these scripts.

## Publications

1. Arthur Henrique Silva Carvalho and Pedro Garcia Freitas. “MAPPE: Mesh Assessment via Point Cloud and Projection Estimators.” *Signal Processing: Image Communication*, vol. 150, article 117722, 2027. [DOI: 10.1016/j.image.2026.117722](https://doi.org/10.1016/j.image.2026.117722). Available online 5 October 2026.

```bibtex
@article{carvalho2027mappe,
  author  = {Carvalho, Arthur Henrique Silva and Freitas, Pedro Garcia},
  title   = {{MAPPE}: Mesh Assessment via Point Cloud and Projection Estimators},
  journal = {Signal Processing: Image Communication},
  volume  = {150},
  pages   = {117722},
  year    = {2027},
  doi     = {10.1016/j.image.2026.117722}
}
```

## License

[MIT License](LICENSE).
