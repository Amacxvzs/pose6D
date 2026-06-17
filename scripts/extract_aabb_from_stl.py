import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_binary_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError("File is too small for binary STL.")
    tri_count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + tri_count * 50
    if expected != len(data):
        raise ValueError("Binary STL size does not match triangle count.")

    vertices = []
    offset = 84
    for _ in range(tri_count):
        # normal: 12 bytes, vertices: 36 bytes, attr: 2 bytes
        offset += 12
        for _ in range(3):
            vertices.append(struct.unpack_from("<fff", data, offset))
            offset += 12
        offset += 2
    return np.asarray(vertices, dtype=np.float64)


def read_ascii_stl(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError("No vertices found in ASCII STL.")
    return np.asarray(vertices, dtype=np.float64)


def read_stl_vertices(path: Path) -> np.ndarray:
    try:
        return read_binary_stl(path)
    except Exception:
        return read_ascii_stl(path)


def make_corners(bounds_min: np.ndarray, bounds_max: np.ndarray) -> np.ndarray:
    xmin, ymin, zmin = bounds_min.tolist()
    xmax, ymax, zmax = bounds_max.tolist()
    return np.asarray(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract AABB corners from an STL CAD model.")
    parser.add_argument("--model", default=None, help="STL file path. If omitted, use the first .stl in models/plate.")
    parser.add_argument("--class-name", default="plate")
    parser.add_argument("--unit", default="mm")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.model:
        model_path = Path(args.model)
    else:
        candidates = sorted(Path("models") .joinpath(args.class_name).glob("*.stl"))
        if not candidates:
            raise RuntimeError(f"No .stl file found in models/{args.class_name}")
        model_path = candidates[0]

    vertices = read_stl_vertices(model_path)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    size = bounds_max - bounds_min
    center = (bounds_min + bounds_max) * 0.5
    corners = make_corners(bounds_min, bounds_max)

    out_path = Path(args.out) if args.out else model_path.with_name("aabb_corners.json")
    result = {
        "class": args.class_name,
        "model_file": str(model_path),
        "unit": args.unit,
        "vertex_count": int(vertices.shape[0]),
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
        "size": {
            "x": float(size[0]),
            "y": float(size[1]),
            "z": float(size[2]),
        },
        "center": center.tolist(),
        "corners_order": [
            "xmin,ymin,zmin",
            "xmax,ymin,zmin",
            "xmax,ymax,zmin",
            "xmin,ymax,zmin",
            "xmin,ymin,zmax",
            "xmax,ymin,zmax",
            "xmax,ymax,zmax",
            "xmin,ymax,zmax",
        ],
        "corners_3d": corners.tolist(),
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("model:", model_path)
    print("vertices:", vertices.shape[0])
    print("bounds_min:", bounds_min)
    print("bounds_max:", bounds_max)
    print("size:", size)
    print("center:", center)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
