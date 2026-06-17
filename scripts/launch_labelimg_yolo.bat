@echo off
cd /d D:\1.sjcl\pose6d
if not exist data\labels\yolo\train\plate mkdir data\labels\yolo\train\plate
python -m labelImg.labelImg data\raw\rgb\train\plate classes.txt data\labels\yolo\train\plate
