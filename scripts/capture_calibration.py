import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture chessboard images for camera calibration.")
    parser.add_argument("--config", default="configs/calibration_config.json")
    parser.add_argument("--camera-index", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    index = cfg["color_camera_index"] if args.camera_index is None else args.camera_index
    out_dir = Path(cfg["image_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = (int(cfg["chessboard_cols"]), int(cfg["chessboard_rows"]))

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg["frame_width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg["frame_height"]))

    print("Capture chessboard images. Space=save valid frame, q/Esc=quit.")
    print(f"Pattern inner corners: cols={pattern[0]}, rows={pattern[1]}")
    saved = len(list(out_dir.glob("*.png")))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern)
            view = frame.copy()
            if found:
                cv2.drawChessboardCorners(view, pattern, corners, found)
                cv2.putText(view, "FOUND - press Space to save", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(view, "NOT FOUND", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(view, f"saved={saved}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("calibration capture", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" ") and found:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                path = out_dir / f"calib_{stamp}_{saved:04d}.png"
                cv2.imwrite(str(path), frame)
                saved += 1
                print("saved", path)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
