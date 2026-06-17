@echo off
cd /d D:\1.sjcl\pose6d
D:\PY\python.exe scripts\analyze_pose_quality.py --split all --class-name plate --detection-name detections_pose
pause
