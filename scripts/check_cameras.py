import argparse
from pathlib import Path

import cv2


def try_open(index: int, width: int, height: int, backend: int) -> bool:
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    print(f"camera index={index}: ok, shape={frame.shape}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe OpenCV camera indexes.")
    parser.add_argument("--max-index", type=int, default=8)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    print("OpenCV:", cv2.__version__)
    print("Working directory:", Path.cwd())
    found = []
    for index in range(args.max_index + 1):
        if try_open(index, args.width, args.height, cv2.CAP_DSHOW):
            found.append(index)

    if not found:
        print("No camera opened through OpenCV/DirectShow.")
        return

    print("Found camera indexes:", found)
    if not args.preview:
        return

    cap = cv2.VideoCapture(found[0], cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print("Preview first camera. Press q or Esc to exit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("camera preview", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
