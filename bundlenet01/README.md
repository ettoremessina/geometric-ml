# bundlenet01

Procedural point-cloud dataset generator, desktop viewer, and PointNet classifier for geometric machine learning. First experiment in the bundlenet series.

## Features

- **10 object classes** built from parametric 3-D primitives (no external models required)
- **XYZ + surface normals** per point — normals sampled from the tangent bundle, rotated equivariantly during augmentation
- **ModelNet-style layout** — natively compatible with `torch_geometric.datasets.ModelNet`
- **Per-sample augmentation**: random SO(3) rotation and uniform scale
- **2048 points per cloud** sampled uniformly from mesh surfaces
- **80 / 20 train / test split** per class
- **Desktop viewer** (Open3D) with per-class colours, keyboard navigation and normal vector toggle
- **PointNet classifier** with T-Net alignment and feature T-Net regularisation

## Classes

| Class | Composition |
|-------|-------------|
| `chair` | seat + backrest + 4 legs |
| `table` | top panel + 4 legs |
| `mug` | cylinder body + torus handle |
| `bottle` | body + tapered neck + cap |
| `airplane` | fuselage + wings + stabilisers + tail fin |
| `car` | body + cabin + 4 wheels |
| `lamp` | disc base + pole + conical shade |
| `sofa` | seat + backrest + armrests + legs |
| `monitor` | screen panel + neck + base |
| `bookshelf` | outer frame + 2–4 inner shelves |

## Requirements

- Python 3.11+

```
trimesh==4.4.0
numpy==1.26.4
scipy==1.13.1
open3d==0.18.0
tqdm==4.66.4
torch==2.3.1
torch_geometric==2.5.3
matplotlib==3.9.0
scikit-learn==1.5.0
```

## Setup

```bash
git clone https://github.com/your-username/bundlenet01.git
cd bundlenet01
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Generate dataset

```bash
# All 10 classes, 100 samples each, 2048 pts/cloud (default)
python generator/generate.py

# Reproducible run
python generator/generate.py --seed 42

# Custom classes, sample count, output directory
python generator/generate.py --classes chair table airplane --n-samples 200 --output-dir data/MyDataset

# All options
python generator/generate.py --help
```

The dataset is written to `data/ModelNet10/` (or your chosen `--output-dir`) with the layout:

```
data/ModelNet10/
├── chair/
│   ├── train/   ← 80 .ply files  (XYZ + normals)
│   └── test/    ← 20 .ply files
├── table/
│   └── …
└── …
```

Each `.ply` file contains 2048 points in binary little-endian format with six properties per vertex: `x y z nx ny nz`.

### View point clouds

```bash
# Browse the whole dataset
python generator/viewer.py

# Filter by class
python generator/viewer.py --class chair

# Open a single file
python generator/viewer.py --file data/ModelNet10/car/train/car_0003.ply

# Custom dataset root
python generator/viewer.py --dataset data/MyDataset
```

Keyboard shortcuts:

| Key | Action |
|-----|--------|
| `N` | next cloud |
| `P` | previous cloud |
| `V` | toggle normal vectors |
| `Q` | quit |

### Train

```bash
# Default: 150 epochs, batch size 32, MPS/CUDA/CPU auto-detected
python classifier/train.py --seed 42 --run-name exp01

# All options
python classifier/train.py --help
```

### Evaluate

```bash
python classifier/evaluate.py --run-dir experiments/exp01
```

Produces in `experiments/exp01/`:
- `best_model.pth`
- `metrics.csv` — loss and accuracy per epoch
- `confusion_matrix.png` + `confusion_matrix.csv`
- `per_class_accuracy.csv`

## Project structure

```
bundlenet01/
├── requirements.txt
├── generator/
│   ├── generate.py          # CLI — generates the dataset
│   ├── viewer.py            # Desktop viewer (Open3D)
│   └── src/
│       ├── shapes/
│       │   ├── base.py      # Abstract ShapeGenerator
│       │   ├── chair.py
│       │   ├── table.py
│       │   ├── mug.py
│       │   ├── bottle.py
│       │   ├── airplane.py
│       │   ├── car.py
│       │   ├── lamp.py
│       │   ├── sofa.py
│       │   ├── monitor.py
│       │   └── bookshelf.py
│       ├── sampler.py       # Point sampling + augmentation (XYZ + normals)
│       └── io.py            # PLY I/O + train/test split
├── classifier/
│   ├── train.py             # CLI — training loop
│   ├── evaluate.py          # CLI — evaluation + reports
│   └── src/
│       ├── dataset.py       # PyG Dataset for .ply files
│       ├── model.py         # PointNet + T-Net
│       └── transforms.py    # NormalizePointCloud
├── experiments/             # Training runs (git-ignored)
│   └── <run_name>/
│       ├── best_model.pth
│       ├── metrics.csv
│       ├── confusion_matrix.png
│       └── per_class_accuracy.csv
└── data/                    # Generated dataset (git-ignored)
    └── ModelNet10/
```

## Using the dataset with PyTorch Geometric

```python
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints

dataset = ModelNet(root="data/", name="ModelNet10", train=True,
                   transform=SamplePoints(2048))
```

## References

Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). PointNet: Deep learning on point sets for 3D classification and segmentation. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 652–660. https://arxiv.org/abs/1612.00593
