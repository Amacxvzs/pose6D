from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def prepare(class_name: str, copy: bool) -> None:
    src_image_root = PROJECT_ROOT / "data" / "raw" / "rgb"
    src_label_root = PROJECT_ROOT / "data" / "labels" / "yolo"
    dst_root = PROJECT_ROOT / "data" / "yolo"

    total_images = 0
    total_labels = 0
    for split in SPLITS:
        image_dir = src_image_root / split / class_name
        label_dir = src_label_root / split / class_name
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing label directory: {label_dir}")

        images = sorted(image_dir.glob("*.png"))
        if not images:
            raise RuntimeError(f"No images found in {image_dir}")

        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")
            if not label_path.read_text(encoding="utf-8", errors="ignore").strip():
                raise RuntimeError(f"Empty label file: {label_path}")

            dst_image = dst_root / "images" / split / image_path.name
            dst_label = dst_root / "labels" / split / label_path.name
            link_or_copy(image_path, dst_image, copy)
            link_or_copy(label_path, dst_label, copy)
            total_images += 1
            total_labels += 1

        print(f"{split}: {len(images)} images and labels prepared")

    print(f"Done: {total_images} images, {total_labels} labels -> {dst_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a standard YOLO dataset layout.")
    parser.add_argument("--class-name", default="plate", help="Class folder name under raw/rgb and labels/yolo.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating hard links.")
    args = parser.parse_args()
    prepare(args.class_name, args.copy)


if __name__ == "__main__":
    main()
