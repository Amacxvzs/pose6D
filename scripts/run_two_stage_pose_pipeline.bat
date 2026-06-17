@echo off
cd /d D:\1.sjcl\pose6d
D:\PY\python.exe scripts\make_occlusion_plan.py --split all --detection-name detections_pose
D:\PY\python.exe scripts\extract_stage2_occluded_clouds.py --split all --class-name plate --detection-name detections_pose --depth-window-mm 80 --min-points 80
D:\PY\python.exe scripts\estimate_initial_poses.py --split all --class-name plate --cloud-name object_clouds_stage2 --output-name initial_poses_stage2 --no-vis
D:\PY\python.exe scripts\refine_poses_icp.py --split all --class-name plate --init-name initial_poses_stage2 --cloud-name object_clouds_stage2 --output-name icp_poses_stage2 --voxel-mm 3 --max-corr-mm 25 --icp-iters 40 --surface full --no-vis
D:\PY\python.exe scripts\merge_two_stage_results.py
pause
