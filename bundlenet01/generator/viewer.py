#!/usr/bin/env python3
"""Desktop point-cloud viewer built on Open3D.

Usage:
    # Browse all clouds in the dataset (one at a time)
    python viewer.py

    # Browse a specific class
    python viewer.py --class chair

    # Open a single .ply file
    python viewer.py --file data/ModelNet10/chair/train/chair_0001.ply

    # Change dataset root
    python viewer.py --dataset data/ModelNet10

Keyboard shortcuts (shown in terminal on startup):
    N  →  next cloud
    P  →  previous cloud
    V  →  toggle normal vectors
    Q  →  quit
"""

import argparse
import glob
import os
import sys

import numpy as np
import open3d as o3d

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATASET = os.path.normpath(os.path.join(_HERE, "..", "data", "ModelNet10"))


# One visually distinct colour per class (RGB, 0-1)
CLASS_COLORS = {
    "chair":     [0.957, 0.263, 0.212],   # red
    "table":     [0.129, 0.588, 0.953],   # blue
    "mug":       [0.298, 0.686, 0.314],   # green
    "bottle":    [1.000, 0.596, 0.000],   # orange
    "airplane":  [0.612, 0.153, 0.690],   # purple
    "car":       [0.000, 0.737, 0.831],   # cyan
    "lamp":      [1.000, 0.922, 0.231],   # yellow
    "sofa":      [1.000, 0.341, 0.133],   # deep orange
    "monitor":   [0.620, 0.000, 0.000],   # dark red
    "bookshelf": [0.133, 0.400, 0.188],   # dark green
}
DEFAULT_COLOR = [0.6, 0.6, 0.6]

# Colour used for normal vectors (dark grey, visible on white background)
NORMAL_COLOR = [0.35, 0.35, 0.35]


def load_ply(path: str) -> o3d.geometry.PointCloud:
    """Load a PLY file via Open3D.  Normals (nx ny nz) are read automatically
    when present; check pcd.has_normals() after loading."""
    return o3d.io.read_point_cloud(path)


def collect_files(dataset_root: str, class_filter: str | None) -> list[tuple[str, str]]:
    """Return a list of (class_name, file_path) sorted by class then filename."""
    entries = []
    for class_name in sorted(os.listdir(dataset_root)):
        if class_filter and class_name != class_filter:
            continue
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for split in ("train", "test"):
            pattern = os.path.join(class_dir, split, "*.ply")
            for fp in sorted(glob.glob(pattern)):
                entries.append((class_name, fp))
    return entries


def parse_args():
    parser = argparse.ArgumentParser(description="Point-cloud viewer (Open3D).")
    parser.add_argument("--dataset", default=_DEFAULT_DATASET,
                        help="Root directory of the dataset (default: ../data/ModelNet10)")
    parser.add_argument("--class", dest="cls", default=None,
                        help="Show only this class")
    parser.add_argument("--file", default=None,
                        help="Open a single .ply file directly")
    return parser.parse_args()


def infer_class(path: str) -> str:
    """Guess the class name from the directory structure."""
    parts = path.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part in ("train", "test") and i > 0:
            return parts[i - 1]
    return os.path.splitext(os.path.basename(path))[0]


def run_viewer(entries: list[tuple[str, str]]):
    if not entries:
        print("[viewer] No point clouds found. Run generate.py first.")
        sys.exit(1)

    print("\nKeyboard shortcuts:")
    print("  N  →  next cloud")
    print("  P  →  previous cloud")
    print("  V  →  toggle normal vectors")
    print("  Q  →  quit\n")

    idx           = [0]      # mutable so callbacks can modify it
    show_normals  = [False]  # toggle state

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Point Cloud Viewer", width=1024, height=768)

    render_opt = vis.get_render_option()
    render_opt.point_size = 2.0
    render_opt.background_color = np.array([1.0, 1.0, 1.0])
    render_opt.point_show_normal = False

    pcd_geom = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_geom)

    def _load_current():
        class_name, path = entries[idx[0]]
        pcd  = load_ply(path)
        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)

        pcd_geom.points = pcd.points
        pcd_geom.paint_uniform_color(color)

        if pcd.has_normals():
            pcd_geom.normals = pcd.normals
        else:
            pcd_geom.normals = o3d.utility.Vector3dVector(
                np.zeros((len(pcd.points), 3), dtype=np.float32)
            )

        # Re-apply current normal visibility state
        render_opt.point_show_normal = show_normals[0]

        vis.update_geometry(pcd_geom)
        vis.reset_view_point(True)

        split       = "train" if "train" in path else "test"
        has_normals = "yes" if pcd.has_normals() else "no"
        print(f"[{idx[0]+1}/{len(entries)}]  class={class_name}  split={split}  "
              f"pts={len(pcd.points)}  normals={has_normals}  "
              f"{os.path.basename(path)}")

    def _next(vis_ref):
        idx[0] = (idx[0] + 1) % len(entries)
        _load_current()

    def _prev(vis_ref):
        idx[0] = (idx[0] - 1) % len(entries)
        _load_current()

    def _toggle_normals(vis_ref):
        show_normals[0] = not show_normals[0]
        render_opt.point_show_normal = show_normals[0]
        state = "ON" if show_normals[0] else "OFF"
        print(f"  normal vectors: {state}")

    def _quit(vis_ref):
        vis_ref.destroy_window()

    vis.register_key_callback(ord("N"), _next)
    vis.register_key_callback(ord("P"), _prev)
    vis.register_key_callback(ord("V"), _toggle_normals)
    vis.register_key_callback(ord("Q"), _quit)

    _load_current()
    vis.run()
    vis.destroy_window()


def main():
    args = parse_args()

    if args.file:
        class_name = infer_class(args.file)
        entries = [(class_name, args.file)]
    else:
        if not os.path.isdir(args.dataset):
            print(f"[viewer] Dataset not found at '{args.dataset}'. "
                  f"Run generate.py first.")
            sys.exit(1)
        entries = collect_files(args.dataset, args.cls)

    run_viewer(entries)


if __name__ == "__main__":
    main()
