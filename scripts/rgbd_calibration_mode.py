from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLAG = PROJECT_ROOT / "CAPTURE_RGBD_CALIBRATION.flag"


def main() -> None:
    parser = argparse.ArgumentParser(description="Enable or disable MFC RGB-D calibration capture mode.")
    parser.add_argument("mode", choices=("on", "off", "status"))
    args = parser.parse_args()

    if args.mode == "on":
        FLAG.write_text("RGB-D calibration capture mode is enabled.\n", encoding="utf-8")
        print(f"enabled: {FLAG}")
    elif args.mode == "off":
        if FLAG.exists():
            FLAG.unlink()
        print(f"disabled: {FLAG}")
    else:
        print("enabled" if FLAG.exists() else "disabled")


if __name__ == "__main__":
    main()
