from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
import math, torch, gc

MODEL       = "/content/drive/MyDrive/yolo_runs/train_colab_m640/weights/best.pt"
SRC_DIR     = Path("/content/drive/MyDrive/YOLOv11_Concrete-defect-dataset-08202025/trial test videos 2")
PROJECT     = "/content/drive/MyDrive/yolo_runs"
RUN_NAME    = "video_preds_polygons"
CONF        = 0.5

IMGSZ_VID      = 640
VID_STRIDE_VID = 1
IMGSZ_CNT      = 640
VID_STRIDE_CNT = 2
USE_FP16       = True
TRACKER        = "bytetrack.yaml"
PRINT_EVERY    = 50
VIDEO_EXTS     = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

videos = [p for p in SRC_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS]
if not videos: raise SystemExit(f"No videos found in {SRC_DIR}")

model = YOLO(MODEL)

print("Saving annotated videos...")
pred_results = model.predict(source=str(SRC_DIR), imgsz=IMGSZ_VID, conf=CONF, save=True, project=PROJECT, name=RUN_NAME, vid_stride=VID_STRIDE_VID, verbose=True)
if not pred_results: raise SystemExit("No prediction results returned.")
run_dir = Path(pred_results[0].save_dir); print(f"\n✅ Videos saved to: {run_dir}")

print("\nCounting unique defects (no video saving)...")
for vf in videos:
    print(f"\n▶ {vf.name}")
    unique_ids_per_class = defaultdict(set)
    names = None; had_ids=False; frame_idx=0

    for res in model.track(source=str(vf), imgsz=IMGSZ_CNT, conf=CONF, tracker=TRACKER, device=0, save=False, stream=True, stream_buffer=False, half=USE_FP16, vid_stride=VID_STRIDE_CNT, verbose=False):
        if names is None: names=res.names
        frame_idx += 1
        if frame_idx % PRINT_EVERY == 0: print(f"  processed frame {frame_idx}")

        boxes = getattr(res, "boxes", None)
        if boxes is None: continue

        cls_t = getattr(boxes, "cls", None)
        id_t  = getattr(boxes, "id",  None)
        if id_t is None or cls_t is None: continue

        had_ids = True
        clses = cls_t.detach().cpu().tolist() if isinstance(cls_t, torch.Tensor) else list(cls_t or [])
        ids   = id_t.detach().cpu().tolist()  if isinstance(id_t,  torch.Tensor) else list(id_t  or [])

        for c, tid in zip(clses, ids):
            if tid is None: continue
            if isinstance(tid, float) and (math.isnan(tid) or tid < 0): continue
            if isinstance(tid, (int, float)) and tid < 0: continue
            unique_ids_per_class[int(c)].add(int(tid))

        del res
        if frame_idx % 100 == 0: torch.cuda.empty_cache(); gc.collect()

    txt_path = run_dir / f"{vf.stem}_counts.txt"
    lines = [f"{names[c]}: {len(s)}" for c, s in sorted(unique_ids_per_class.items()) if len(s)>0] if (had_ids and unique_ids_per_class) else ["No detections"]
    with open(txt_path, "w") as f: f.write("\n".join(lines) + "\n")
    print("  ✅ Wrote counts:", txt_path); [print("   ", L) for L in lines]

print("\n🎬 Done. Videos and *_counts.txt are in:", run_dir)
