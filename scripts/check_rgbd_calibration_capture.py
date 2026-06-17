from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_corners(image: np.ndarray, pattern: tuple[int, int]):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern)
    if not found:
        return False, corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def main() -> None:
    parser = argparse.ArgumentParser(description="Check RGB-D calibration captures saved by the MFC app.")
    parser.add_argument("--config", default="configs/calibration_config.json")
    parser.add_argument("--rgb-dir", default="data/calibration/rgbd/rgb_full")
    parser.add_argument("--depth-dir", default="data/calibration/rgbd/depth")
    parser.add_argument("--preview-dir", default="data/calibration/rgbd/preview")
    parser.add_argument("--output", default="data/calibration/rgbd/contact_sheet.jpg")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    pattern = (int(cfg["chessboard_cols"]), int(cfg["chessboard_rows"]))
    rgb_dir = PROJECT_ROOT / args.rgb_dir
    depth_dir = PROJECT_ROOT / args.depth_dir
    preview_dir = PROJECT_ROOT / args.preview_dir
    rows = []
    thumbs = []
    rgb_files = sorted(rgb_dir.glob("*.png"))
    for rgb_path in rgb_files:
        depth_path = depth_dir / rgb_path.name
        preview_path = preview_dir / rgb_path.name
        if not depth_path.exists():
            rows.append((rgb_path.name, "missing_depth", 0, 0, 0))
            continue
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            rows.append((rgb_path.name, "read_fail", 0, 0, 0))
            continue
        found, corners = find_corners(rgb, pattern)
        valid_depth = depth[depth > 0]
        depth_mean = float(valid_depth.mean()) if valid_depth.size else 0.0
        depth_valid_ratio = float(valid_depth.size / depth.size) if depth.size else 0.0
        status = "ok" if found else "no_rgb_corners"
        rows.append((rgb_path.name, status, int(valid_depth.size), depth_mean, depth_valid_ratio))

        view = rgb.copy()
        if found:
            cv2.drawChessboardCorners(view, pattern, corners, found)
        preview = cv2.imread(str(preview_path), cv2.IMREAD_COLOR) if preview_path.exists() else None
        if preview is None:
            depth8 = cv2.convertScaleAbs(depth, alpha=255.0 / 2500.0)
            preview = cv2.applyColorMap(depth8, cv2.COLORMAP_JET)
        view_small = cv2.resize(view, (320, 180), interpolation=cv2.INTER_AREA)
        prev_small = cv2.resize(preview, (240, 180), interpolation=cv2.INTER_AREA)
        panel = np.hstack([view_small, prev_small])
        color = (0, 255, 0) if found else (0, 0, 255)
        cv2.putText(panel, f"{rgb_path.stem[-18:]} {status}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(panel, f"depth_mean={depth_mean:.0f} valid={depth_valid_ratio:.2f}", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        thumbs.append(panel)

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if thumbs:
        cols = 1
        sheet = np.vstack(thumbs)
        cv2.imwrite(str(out_path), sheet)

    valid = sum(1 for r in rows if r[1] == "ok")
    print(f"pairs={len(rows)} valid_rgb_chessboard={valid} pattern={pattern}")
    for row in rows:
        print(f"{row[0]} {row[1]} depth_pixels={row[2]} depth_mean={row[3]:.1f} valid_ratio={row[4]:.3f}")
    if thumbs:
        print(out_path)


if __name__ == "__main__":
    main()
