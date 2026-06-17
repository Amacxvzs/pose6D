from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create camera intrinsics for the saved RGB crop images.")
    parser.add_argument("--input", default="data/calibration/camera_intrinsic.json")
    parser.add_argument("--output", default="data/calibration/camera_intrinsic_crop_720.json")
    parser.add_argument("--crop-x", type=float, default=280.0)
    parser.add_argument("--crop-y", type=float, default=0.0)
    parser.add_argument("--crop-width", type=int, default=720)
    parser.add_argument("--crop-height", type=int, default=720)
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output
    data = json.loads(input_path.read_text(encoding="utf-8"))

    camera_matrix = data["camera_matrix"]
    camera_matrix[0][2] = float(camera_matrix[0][2]) - args.crop_x
    camera_matrix[1][2] = float(camera_matrix[1][2]) - args.crop_y

    data["source_intrinsic"] = args.input
    data["crop"] = {
        "x": args.crop_x,
        "y": args.crop_y,
        "width": args.crop_width,
        "height": args.crop_height,
    }
    data["image_width"] = args.crop_width
    data["image_height"] = args.crop_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
