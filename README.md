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

- Python 3.11 or 3.12.
- Network access to download Python packages and pretrained image metric weights on first use.
- Reference and distorted OBJ meshes, including any associated MTL files and texture images.

A GPU is optional. CPU rendering and PyTorch execution are supported.

## Installation

### From PyPI

Install [MAPPE from PyPI](https://pypi.org/project/mappe/) in your Python environment:

```bash
pip install mappe
```

Dependencies are installed automatically, including the PointPCA and mesh-projector packages. Their published wheels include the Rust extensions.

For CUDA or ROCm, install a matching `torch` and `torchvision` pair using [PyTorch's installation instructions](https://pytorch.org/get-started/locally/) before installing MAPPE. Use `torchvision>=0.24.1` so pip can retain the installed pair. Both backends use `device="cuda"`.

### From the repository

For experiments or development, clone the repository and install its environment with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/arthurhscarvalho/mappe.git mappe
cd mappe
uv sync --python 3.12
```

For the repository's ROCm 6.3 environment:

```bash
uv sync --python 3.12 --no-default-groups --extra rocm
```

Include `--no-default-groups --extra rocm` in subsequent `uv run` commands as well.

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
