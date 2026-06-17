import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_dirs(root: Path, split: str, class_name: str) -> dict:
    dirs = {
        "rgb": root / "rgb" / split / class_name,
        "depth": root / "depth" / split / class_name,
        "preview": root / "preview" / split / class_name,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def open_color(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open color camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def open_depth() -> cv2.VideoCapture | None:
    if not hasattr(cv2, "CAP_OPENNI2"):
        return None
    cap = cv2.VideoCapture(cv2.CAP_OPENNI2)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def read_depth(cap: cv2.VideoCapture | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if cap is None:
        return None, None
    ok = cap.grab()
    if not ok:
        return None, None
    ok_depth, depth = cap.retrieve(None, cv2.CAP_OPENNI_DEPTH_MAP)
    ok_bgr, bgr = cap.retrieve(None, cv2.CAP_OPENNI_BGR_IMAGE)
    if not ok_depth:
        depth = None
    if not ok_bgr:
        bgr = None
    return depth, bgr


def depth_preview(depth: np.ndarray | None) -> np.ndarray | None:
    if depth is None:
        return None
    depth_u16 = depth.astype(np.uint16)
    vis = cv2.convertScaleAbs(depth_u16, alpha=255.0 / 4000.0)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


def save_sample(dirs: dict, rgb: np.ndarray, depth: np.ndarray | None, scene: str, counter: int) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    base = f"{scene}_{stamp}_{counter:06d}"
    rgb_path = dirs["rgb"] / f"{base}.png"
    cv2.imwrite(str(rgb_path), rgb)
    if depth is not None:
        cv2.imwrite(str(dirs["depth"] / f"{base}.png"), depth.astype(np.uint16))
        prev = depth_preview(depth)
        if prev is not None:
            cv2.imwrite(str(dirs["preview"] / f"{base}_depth.png"), prev)
    print(f"saved {rgb_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture RGB and optional OpenNI depth frames for 6D pose data.")
    parser.add_argument("--config", default="configs/capture_config.json")
    parser.add_argument("--class-name", default=None)
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--scene", default="stack")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--auto", action="store_true", help="Auto-save frames every interval seconds.")
    parser.add_argument("--interval", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    class_name = args.class_name or cfg["default_class"]
    split = args.split or cfg["default_split"]
    index = cfg["color_camera_index"] if args.camera_index is None else args.camera_index
    interval = cfg["auto_interval_seconds"] if args.interval is None else args.interval
    root = Path(cfg["save_root"])
    dirs = make_dirs(root, split, class_name)

    color = open_color(index, cfg["frame_width"], cfg["frame_height"])
    depth_cap = open_depth() if cfg.get("depth", {}).get("enabled", True) else None
    if depth_cap is None:
        print("Depth stream not opened. Continuing in RGB-only mode.")
    else:
        print("Depth stream opened through OpenNI2.")

    print("Controls: Space=snapshot, a=toggle auto, q/Esc=quit.")
    auto = args.auto
    last_save = 0.0
    counter = len(list(dirs["rgb"].glob("*.png")))

    try:
        while True:
            ok, rgb = color.read()
            if not ok or rgb is None:
                print("Color frame read failed.")
                break
            depth, _ = read_depth(depth_cap)
            show = rgb.copy()
            cv2.putText(show, f"class={class_name} split={split} count={counter} auto={auto}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("RGB capture", show)
            dprev = depth_preview(depth)
            if dprev is not None:
                cv2.imshow("Depth preview", dprev)

            now = time.time()
            if auto and now - last_save >= interval:
                save_sample(dirs, rgb, depth, args.scene, counter)
                counter += 1
                last_save = now

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                save_sample(dirs, rgb, depth, args.scene, counter)
                counter += 1
            if key == ord("a"):
                auto = not auto
                print("auto:", auto)
    finally:
        color.release()
        if depth_cap is not None:
            depth_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
