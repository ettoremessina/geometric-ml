#!/usr/bin/env python3
"""Generate a classified point-cloud dataset in ModelNet-style .ply format.

Usage examples:
    # Generate all 10 classes, 100 samples each, 2048 pts/cloud
    python generate.py

    # Custom output dir, specific classes, reproducible seed
    python generate.py --output-dir data/MyDataset --classes chair table --seed 42

    # Override number of samples and points per cloud
    python generate.py --n-samples 200 --n-points 4096
"""

import argparse
import sys
import os

import numpy as np
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

_DEFAULT_OUTPUT = os.path.normpath(os.path.join(_HERE, "..", "data", "ModelNet10"))

from src.shapes import ALL_GENERATORS
from src.sampler import sample_points_and_normals, augment
from src.io import save_point_cloud, class_output_dir, sample_filename, train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a point-cloud dataset.")
    parser.add_argument(
        "--output-dir", default=_DEFAULT_OUTPUT,
        help="Root directory for the dataset (default: ../data/ModelNet10)"
    )
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="Classes to generate (default: all). E.g. --classes chair table"
    )
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Number of point clouds per class (default: 100)"
    )
    parser.add_argument(
        "--n-points", type=int, default=2048,
        help="Points per cloud (default: 2048)"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.2,
        help="Fraction reserved for test split (default: 0.2)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Global RNG seed for reproducibility"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    classes = args.classes if args.classes else list(ALL_GENERATORS.keys())
    unknown = [c for c in classes if c not in ALL_GENERATORS]
    if unknown:
        print(f"[ERROR] Unknown class(es): {unknown}. "
              f"Available: {list(ALL_GENERATORS.keys())}")
        sys.exit(1)

    global_rng = np.random.default_rng(args.seed)

    print(f"Dataset root : {args.output_dir}")
    print(f"Classes      : {classes}")
    print(f"Samples/class: {args.n_samples}  |  Points/cloud: {args.n_points}")
    print(f"Train/test   : {int((1 - args.test_ratio)*100)}/{int(args.test_ratio*100)}\n")

    for class_name in classes:
        GeneratorClass = ALL_GENERATORS[class_name]
        # Deterministic per-class seed derived from the global rng
        class_seed = int(global_rng.integers(0, 2**31))
        class_rng  = np.random.default_rng(class_seed)

        # Pre-compute train/test split indices
        train_idx, test_idx = train_test_split(
            args.n_samples, test_ratio=args.test_ratio, rng=class_rng
        )
        split_map = {i: "train" for i in train_idx}
        split_map.update({i: "test" for i in test_idx})

        gen = GeneratorClass(rng=class_rng)

        with tqdm(total=args.n_samples, desc=f"{class_name:>12}", unit="cloud") as pbar:
            for i in range(args.n_samples):
                mesh   = gen.generate()
                points = sample_points_and_normals(mesh, args.n_points)
                points = augment(points, class_rng)

                split    = split_map[i]
                out_dir  = class_output_dir(args.output_dir, class_name, split)
                filename = sample_filename(class_name, i)
                save_point_cloud(points, os.path.join(out_dir, filename))
                pbar.update(1)

    print(f"\nDone. Dataset saved to '{args.output_dir}'.")


if __name__ == "__main__":
    main()
