@echo off
cd /d D:\1.sjcl\pose6d
D:\PY\python.exe scripts\extract_rgbd_object_clouds.py --split all --detection-name detections_pose --class-name plate --depth-window-mm 80 --min-points 80
pause
