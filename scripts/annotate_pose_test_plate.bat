@echo off
cd /d D:\1.sjcl\pose6d
D:\PY\python.exe scripts\annotate_pose_keypoints.py --split test --class-name plate --detection-name detections_pose --scale 1.0
pause
