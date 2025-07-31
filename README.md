# Concrete Defect Detection using YOLOv11

This project trains a YOLOv11 segmentation model to detect 8 types of defects in concrete bridge structures.

---

## 📁 Dataset

The dataset is stored in Google Drive and the SMART Data Management Platform.  
On this VM, it is located at:

```
/home/natnael/refined-concrete-defect-dataset-06142025/
```

> ⚠️ The dataset and annotations are not included in this GitHub repo due to size.

---

## 🧠 Classes

- Crack  
- ACrack  
- Efflorescence  
- WConccor  
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
- Draws **polygon outlines** only (no shaded masks or rectangles)
- **Class labels only**, without confidence scores
- Applied to both **image** and **video** predictions

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
- Applies extensive data augmentation

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


