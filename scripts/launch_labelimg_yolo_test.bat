@echo off
cd /d D:\1.sjcl\pose6d
if not exist data\labels\yolo\test\plate mkdir data\labels\yolo\test\plate
python -m labelImg.labelImg data\raw\rgb\test\plate classes.txt data\labels\yolo\test\plate
