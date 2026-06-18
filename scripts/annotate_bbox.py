import argparse
from pathlib import Path

import cv2


CLASS_ID = 0
CLASS_NAME = "plate"


class Annotator:
    def __init__(self, image_dir: Path, label_dir: Path, start: int = 0):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.images = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
        if not self.images:
            raise RuntimeError(f"No images found in {image_dir}")
        self.index = max(0, min(start, len(self.images) - 1))
        self.boxes: list[tuple[int, int, int, int]] = []
        self.dragging = False
        self.start_pt = (0, 0)
        self.current_pt = (0, 0)
        self.window = "plate bbox annotator"

    def label_path(self) -> Path:
        return self.label_dir / f"{self.images[self.index].stem}.txt"

    def load_boxes(self) -> None:
        self.boxes = []
        img = cv2.imread(str(self.images[self.index]))
        h, w = img.shape[:2]
        path = self.label_path()
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, xc, yc, bw, bh = parts
            xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
            x1 = int(round((xc - bw / 2) * w))
            y1 = int(round((yc - bh / 2) * h))
            x2 = int(round((xc + bw / 2) * w))
            y2 = int(round((yc + bh / 2) * h))
            self.boxes.append((x1, y1, x2, y2))

    def save_boxes(self) -> None:
        img = cv2.imread(str(self.images[self.index]))
        h, w = img.shape[:2]
        lines = []
        for x1, y1, x2, y2 in self.boxes:
            x1, x2 = sorted((max(0, min(w - 1, x1)), max(0, min(w - 1, x2))))
            y1, y2 = sorted((max(0, min(h - 1, y1)), max(0, min(h - 1, y2))))
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        self.label_path().write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print("saved", self.label_path(), "boxes", len(lines))

    def mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start_pt = (x, y)
            self.current_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.current_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            x1, y1 = self.start_pt
            x2, y2 = x, y
            if abs(x2 - x1) >= 3 and abs(y2 - y1) >= 3:
                self.boxes.append((x1, y1, x2, y2))

    def draw(self):
        img = cv2.imread(str(self.images[self.index]))
        view = img.copy()
        for x1, y1, x2, y2 in self.boxes:
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(view, CLASS_NAME, (min(x1, x2), max(20, min(y1, y2) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if self.dragging:
            cv2.rectangle(view, self.start_pt, self.current_pt, (0, 255, 255), 2)
        status = f"{self.index + 1}/{len(self.images)} {self.images[self.index].name} boxes={len(self.boxes)}"
        cv2.rectangle(view, (0, 0), (view.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(view, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return view

    def next_image(self, delta: int) -> None:
        self.index = max(0, min(len(self.images) - 1, self.index + delta))
        self.load_boxes()

    def run(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 900, 900)
        cv2.setMouseCallback(self.window, self.mouse)
        self.load_boxes()
        print("Mouse: drag left button to draw a bbox.")
        print("Keys: s/save, n/next, p/prev, u/undo, d/delete all, q/quit.")
        while True:
            cv2.imshow(self.window, self.draw())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                self.save_boxes()
            elif key == ord("n"):
                self.save_boxes()
                self.next_image(1)
            elif key == ord("p"):
                self.save_boxes()
                self.next_image(-1)
            elif key == ord("u"):
                if self.boxes:
                    self.boxes.pop()
            elif key == ord("d"):
                self.boxes = []
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple OpenCV YOLO bbox annotator for plate images.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--class-name", default="plate")
    parser.add_argument("--image-dir", type=Path, default=None, help="Override image directory.")
    parser.add_argument("--label-dir", type=Path, default=None, help="Override label output directory.")
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    if args.image_dir is not None:
        image_dir = args.image_dir
    else:
        image_dir = Path("data") / "raw" / "rgb" / args.split / args.class_name
    if args.label_dir is not None:
        label_dir = args.label_dir
    else:
        label_dir = Path("data") / "labels" / "yolo" / args.split / args.class_name
    Annotator(image_dir, label_dir, args.start).run()


if __name__ == "__main__":
    main()
