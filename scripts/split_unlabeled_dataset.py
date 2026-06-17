import argparse
import random
import shutil
from pathlib import Path


MODALITIES = {
    "rgb": ".png",
    "rgb_full": ".png",
    "depth": ".png",
    "preview": ".png",
    "meta": ".json",
}


def parse_counts(text: str) -> tuple[int, int, int]:
    parts = [int(x) for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError("--counts must look like 200,30,30")
    return parts[0], parts[1], parts[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Split MFC-captured unlabeled samples into train/val/test class folders.")
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--counts", default="200,30,30", help="train,val,test counts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move", action="store_true", help="Move files instead of copying.")
    args = parser.parse_args()

    root = Path(args.root)
    train_count, val_count, test_count = parse_counts(args.counts)
    total_needed = train_count + val_count + test_count

    source_rgb = root / "rgb" / "train" / "unlabeled"
    stems = sorted(p.stem for p in source_rgb.glob("*.png"))
    if len(stems) < total_needed:
        raise RuntimeError(f"Need {total_needed} samples, found {len(stems)} in {source_rgb}")

    complete = []
    incomplete = []
    for stem in stems:
        ok = True
        for modality, ext in MODALITIES.items():
            p = root / modality / "train" / "unlabeled" / f"{stem}{ext}"
            if not p.exists():
                ok = False
                break
        if ok:
            complete.append(stem)
        else:
            incomplete.append(stem)

    if len(complete) < total_needed:
        raise RuntimeError(f"Need {total_needed} complete samples, found {len(complete)}; incomplete={len(incomplete)}")

    random.Random(args.seed).shuffle(complete)
    selected = complete[:total_needed]
    splits = {
        "train": selected[:train_count],
        "val": selected[train_count:train_count + val_count],
        "test": selected[train_count + val_count:],
    }

    operation = shutil.move if args.move else shutil.copy2
    for split, split_stems in splits.items():
        for stem in split_stems:
            for modality, ext in MODALITIES.items():
                src = root / modality / "train" / "unlabeled" / f"{stem}{ext}"
                dst = root / modality / split / args.class_name / f"{stem}{ext}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                operation(str(src), str(dst))

    print("Split done.")
    print("class:", args.class_name)
    print("operation:", "move" if args.move else "copy")
    for split, split_stems in splits.items():
        print(f"{split}: {len(split_stems)}")
    if incomplete:
        print("incomplete samples left untouched:", len(incomplete))


if __name__ == "__main__":
    main()
