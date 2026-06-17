import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_opencv_yaml(path: Path, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, image_size: tuple[int, int], rms: float) -> None:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("camera_matrix", camera_matrix)
    fs.write("distortion_coefficients", dist_coeffs)
    fs.write("rms_reprojection_error", float(rms))
    fs.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate RGB camera from chessboard images.")
    parser.add_argument("--config", default="configs/calibration_config.json")
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    image_dir = Path(args.image_dir or cfg["image_dir"])
    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cols = int(cfg["chessboard_cols"])
    rows = int(cfg["chessboard_rows"])
    square = float(cfg["square_size_mm"])
    pattern = (cols, rows)

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square

    objpoints = []
    imgpoints = []
    used_images = []
    image_size = None

    images = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")))
    if not images:
        raise RuntimeError(f"No calibration images found in {image_dir}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(gray, pattern)
        if not found:
            print("skip no corners:", path.name)
            continue
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(refined)
        used_images.append(path.name)

    if len(objpoints) < 8:
        raise RuntimeError(f"Need at least 8 valid chessboard images, got {len(objpoints)}")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    result = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "rms_reprojection_error": float(rms),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.reshape(-1).tolist(),
        "valid_image_count": len(objpoints),
        "used_images": used_images,
        "chessboard": {
            "cols": cols,
            "rows": rows,
            "square_size_mm": square
        }
    }

    json_path = output_dir / "camera_intrinsic.json"
    yaml_path = output_dir / "camera_intrinsic.yaml"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    write_opencv_yaml(yaml_path, camera_matrix, dist_coeffs, image_size, rms)

    print("Calibration done.")
    print("valid images:", len(objpoints))
    print("rms reprojection error:", rms)
    print("camera_matrix:\n", camera_matrix)
    print("dist_coeffs:", dist_coeffs.reshape(-1))
    print("saved:", json_path)
    print("saved:", yaml_path)


if __name__ == "__main__":
    main()
