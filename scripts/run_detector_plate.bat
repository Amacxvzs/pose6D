@echo off
cd /d D:\1.sjcl\pose6d
D:\PY\python.exe scripts\make_crop_intrinsic.py
D:\PY\python.exe scripts\run_detector.py --split all --class-name plate --conf 0.8 --iou 0.45 --imgsz 640 --output-name detections_pose --min-size 30 --max-size 170 --min-aspect 0.55 --max-aspect 1.8 --border 3
pause
