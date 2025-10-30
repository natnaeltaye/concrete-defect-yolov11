#!/bin/bash

# Activate the YOLO virtual environment
source ~/yolovenv/bin/activate

# Train the YOLOv11 segmentation model
yolo task=segment mode=train \
  model=yolo11l-seg.pt \
  data=/home/natnael/refined-concrete-defect-dataset-06142025/data.yaml \
  epochs=300 imgsz=640 batch=115 \
  lr0=0.001 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  patience=30 augment=True save=True warmup_epochs=3 \
  degrees=30 translate=0.1 scale=0.5 shear=2 perspective=0.0005 \
  flipud=0.3 fliplr=0.5 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4
