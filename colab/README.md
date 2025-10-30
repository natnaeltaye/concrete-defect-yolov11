# Google Colab Workflow

This folder contains a Colab notebook and helper scripts to reproduce training and inference without a VM.

## Open in Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/natnaeltaye/concrete-defect-yolov11/blob/add-colab-workflow/colab/YOLOv11-Concrete-defects-training-08202025.ipynb)

## Contents
- `YOLOv11-Concrete-defects-training-08202025.ipynb` – end-to-end steps (1–10).
- `scripts/`:
  - `step5_split_and_yaml.py`
  - `step8_patch_ultralytics_polygons.py`
  - `step9_predict_images.py`
  - `step10_predict_videos_bytetrack.py`

## Notes
- Results save to Drive under `/content/drive/MyDrive/yolo_runs`.
- Do not commit large outputs or model weights (`*.pt`, `*.onnx`). Use Zenodo or GitHub Releases and link them.
- Hyperparameters and data YAML are defined in the notebook and scripts for reproducibility.
