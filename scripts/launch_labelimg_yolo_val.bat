@echo off
cd /d D:\1.sjcl\pose6d
if not exist data\labels\yolo\val\plate mkdir data\labels\yolo\val\plate
python -m labelImg.labelImg data\raw\rgb\val\plate classes.txt data\labels\yolo\val\plate
