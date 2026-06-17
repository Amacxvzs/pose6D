@echo off
cd /d D:\1.sjcl\pose6d
D:\PY\python.exe scripts\refine_poses_icp.py --split all --class-name plate --voxel-mm 3 --max-corr-mm 25 --icp-iters 40 --surface full
pause
