@echo off
cd /d D:\1.sjcl\pose6d
D:\PY\python.exe scripts\train_yolov9_plate.py --epochs 100 --batch 4 --imgsz 640 --name plate_yolov9t
pause
