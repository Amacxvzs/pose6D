@echo off
cd /d D:\1.sjcl\pose6d
if not exist outputs\detections_unlabeled_20260617\train\manual_relabel_low_conf\labels_new mkdir outputs\detections_unlabeled_20260617\train\manual_relabel_low_conf\labels_new
D:\PY\python.exe scripts\annotate_bbox.py --image-dir outputs\detections_unlabeled_20260617\train\manual_relabel_low_conf\images --label-dir outputs\detections_unlabeled_20260617\train\manual_relabel_low_conf\labels_new
