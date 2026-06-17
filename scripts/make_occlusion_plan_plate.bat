@echo off
cd /d D:\1.sjcl\pose6d
D:\PY\python.exe scripts\make_occlusion_plan.py --split all --detection-name detections_pose
pause
