🏗️ Concrete Defect Detection wiht YOLOv11

This project develops and trains a YOLOv11 segmentation model to detect and outline eight types of defects in concrete bridge structures.
It is part of the NCRPC SMART Grant initiative for AI-enabled bridge inspection and asset management.

📁 Dataset

The dataset is stored in Google Drive and the SMART Data Management Platform.  
On this VM, it is located at:

```
/home/natnael/refined-concrete-defect-dataset-06142025/
```

> ⚠️ The dataset and annotations are not included in this GitHub repo due to size.

---

## 🧠 Classes

- Crack  
- ACrack (Alligator cracks)
- Efflorescence  
- WConccor (Water–Concrete corrosion) 
- Spalling  
- Wetspot  
- Rust  
- ExposedRebars  

---

## 🛠️ Custom YOLOv11 Code

This repo includes a **fully customized YOLOv11 implementation** inside:

```
yolov11_custom/
```


Key customizations:
- Polygon outlines only with class labels (no boxes, shaded masks, or confidences)
- Overlay of per-image defect counts directly on annotated outputs
- Structured JSON exports (labels, bounding boxes, polygons, confidence, pixel area)
- Per-image/video class count text files for quick summaries
- Cropped regions of defects for downstream VLM/AI analysis
- Video deduplication via hybrid ByteTrack + IoU filtering
- Frame-stride sampling for long drone sequences
- Full YOLOv11 functionality preserved (training, export, ONNX)

> The original YOLOv11 code was adapted and preserved for further training, export, and ONNX conversion.

---

## 🏋️ Training

To train the model, run:

```bash
bash scripts/train.sh
```

This script:
- Activates the YOLOv11 virtual environment
- Launches training using `yolo11x-seg.pt`
- Applies data augmentation (rotation, flipping, color jitter, etc.)

---

## 🧪 Inference (Prediction)

After training, the best model is saved here:

```
/home/natnael/refined-concrete-defect-dataset-06142025/runs/segment/train3/weights/best.pt
```

To predict on new **images**:

```bash
source yolovenv/bin/activate

yolo task=segment mode=predict \
  model="/home/natnael/refined-concrete-defect-dataset-06142025/runs/segment/train3/weights/best.pt" \
  source="/home/natnael/test cowley" \
  conf=0.25 save=True
```

---

## 🎞️ Inference on Videos

To predict on new **videos**:

```bash
source yolovenv/bin/activate

yolo task=segment mode=predict \
  model="/home/natnael/refined-concrete-defect-dataset-06142025/runs/segment/train3/weights/best.pt" \
  source="/home/natnael/test cowley videos 2" \
  conf=0.25 save=True
```

Predictions will be saved to:

```
runs/segment/predict/
```

---

## 🚀 Deployment (ONNX + FastAPI)
The model is also exported to ONNX format and served via a FastAPI web app for real-time inference.
Deployment scripts and backend code are available in the separate repo:

yolo11-fastapi-app/

---

## 🙌 Acknowledgments
Based on YOLOv11

Dataset collected and annotated as part of NCRPC SMART Grant

---

## 📦 Environment Setup

To recreate the environment:

```bash
python3 -m venv yolovenv
source yolovenv/bin/activate
pip install -r requirements.txt


## Reproduction Paths

- **VM workflow**: see `vm/` for training on a virtual machine.
- **Google Colab workflow**: see `colab/` for the end-to-end Colab notebook and helper scripts.
- One-click: use the Colab badge inside `colab/README.md`.
