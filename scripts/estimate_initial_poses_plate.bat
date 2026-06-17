@echo off
cd /d D:\1.sjcl\pose6d
D:\PY\python.exe scripts\estimate_initial_poses.py --split all --class-name plate
pause
