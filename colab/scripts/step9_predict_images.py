from ultralytics import YOLO
from pathlib import Path
from collections import Counter
import cv2, torch
import os, json, math, numpy as np

MODEL   = "/content/drive/MyDrive/yolo_runs/train_colab_m640/weights/best.pt"
SOURCE  = "/content/drive/MyDrive/YOLOv11_Concrete-defect-dataset-08202025/trial test 2"
PROJECT = "/content/drive/MyDrive/yolo_runs"
RUN_NAME = "trial_preds_polygons"
IMGSZ   = 640
CONF    = 0.5
BOX_ALPHA = 0.35

model = YOLO(MODEL)
results = model.predict(source=SOURCE, imgsz=IMGSZ, conf=CONF, save=True, max_det=300, project=PROJECT, name=RUN_NAME)

SAVE_JSON, SAVE_CROPS, PAD_PX = True, True, 8

def to_int_list(x): return [int(round(v)) for v in x]
def poly_area_px2(points_xy):
    if points_xy is None or len(points_xy) < 3: return 0.0
    return float(cv2.contourArea(np.asarray(points_xy, dtype=np.float32)))
def clamp_bbox(x1,y1,x2,y2,w,h):
    x1=max(0,min(int(math.floor(x1)),w-1)); y1=max(0,min(int(math.floor(y1)),h-1))
    x2=max(0,min(int(math.ceil(x2)), w-1)); y2=max(0,min(int(math.ceil(y2)), h-1))
    if x2<=x1:x2=min(x1+1,w-1)
    if y2<=y1:y2=min(y1+1,h-1)
    return x1,y1,x2,y2

def overlay_counts_on_image(img_path: Path, lines, box_alpha=BOX_ALPHA):
    img=cv2.imread(str(img_path))
    if img is None: print(f"⚠️ Could not read image for overlay: {img_path}"); return False
    sizes=[cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0] for t in lines] or [1]
    box_w=max(sizes)+20; line_h=28; box_h=line_h*max(1,len(lines))+20; x0,y0=5,5
    ov=img.copy(); cv2.rectangle(ov,(x0,y0),(x0+box_w,y0+box_h),(0,0,0),-1)
    img=cv2.addWeighted(ov, box_alpha, img, 1-box_alpha, 0)
    y=y0+20
    for t in lines:
        cv2.putText(img,t,(x0+10,y),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2,cv2.LINE_AA); y+=line_h
    ok=cv2.imwrite(str(img_path),img)
    if not ok: print(f"⚠️ Failed to write overlay to: {img_path}")
    return ok

def find_saved_image(save_dir: Path, src_path: Path):
    direct=save_dir/src_path.name
    if direct.exists(): return direct
    m=list(save_dir.glob(src_path.stem+".*"))
    return m[0] if m else None

if not results: print("No results returned.")
else:
    final_dir = Path(results[0].save_dir); json_dir=final_dir/"json"; crops_dir=final_dir/"crops"
    if SAVE_JSON: json_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_CROPS: crops_dir.mkdir(parents=True, exist_ok=True)
    print("Writing outputs in:", final_dir)

    total_images=0; total_dets=0
    for res in results:
        save_dir=Path(res.save_dir); src_path=Path(res.path)
        out_img=find_saved_image(save_dir, src_path)
        if out_img is None: print(f"⚠️ No saved image for {src_path.name} in {save_dir}"); continue

        names=res.names; dets=[]; cls_idxs=[]; confs=[]; bboxes=[]
        if getattr(res,"boxes",None) is not None:
            if getattr(res.boxes,"cls",None) is not None:
                cls_raw=res.boxes.cls; cls_idxs=(cls_raw.detach().cpu().tolist() if isinstance(cls_raw,torch.Tensor) else list(cls_raw)); cls_idxs=[int(x) for x in cls_idxs]
            if getattr(res.boxes,"conf",None) is not None:
                conf_raw=res.boxes.conf; confs=(conf_raw.detach().cpu().tolist() if isinstance(conf_raw,torch.Tensor) else list(conf_raw))
            if getattr(res.boxes,"xyxy",None) is not None:
                xyxy=res.boxes.xyxy; 
                if isinstance(xyxy,torch.Tensor): xyxy=xyxy.detach().cpu().numpy()
                bboxes=xyxy.tolist()

        polys_xy=None
        if getattr(res,"masks",None) is not None and getattr(res.masks,"xy",None) is not None:
            polys_xy=res.masks.xy

        N=len(polys_xy) if polys_xy is not None else len(bboxes)
        img_bgr=cv2.imread(str(src_path)); ih,iw=(img_bgr.shape[0],img_bgr.shape[1]) if img_bgr is not None else (None,None)

        for i in range(N):
            cls_name=names[cls_idxs[i]] if i<len(cls_idxs) else names[0]
            conf=float(confs[i]) if i<len(confs) else None
            if i<len(bboxes):
                x1,y1,x2,y2=bboxes[i]
            else:
                if polys_xy is not None and i<len(polys_xy) and len(polys_xy[i])>=3:
                    px,py=polys_xy[i][:,0],polys_xy[i][:,1]; x1,y1,x2,y2=float(px.min()),float(py.min()),float(px.max()),float(py.max())
                else:
                    x1=y1=0.0; x2=float(iw-1) if iw else 1.0; y2=float(ih-1) if ih else 1.0
            bbox=[int(round(x1)),int(round(y1)),int(round(x2)),int(round(y2))]

            poly_flat=[]; area_px2=None
            if polys_xy is not None and i<len(polys_xy) and len(polys_xy[i])>=3:
                pts=polys_xy[i]; area_px2=poly_area_px2(pts); poly_flat=to_int_list(pts.reshape(-1).tolist())
            else:
                if iw is not None and ih is not None: area_px2=max(1,(bbox[2]-bbox[0])*(bbox[3]-bbox[1]))

            # crops
            crop_rel_path=None
            if SAVE_CROPS and img_bgr is not None:
                x1p,y1p,x2p,y2p=clamp_bbox(bbox[0]-PAD_PX,bbox[1]-PAD_PX,bbox[2]+PAD_PX,bbox[3]+PAD_PX,iw,ih)
                crop=img_bgr[y1p:y2p, x1p:x2p]
                if crop.size>0:
                    cls_dir=crops_dir/cls_name; cls_dir.mkdir(parents=True, exist_ok=True)
                    crop_name=f"{src_path.stem}_{cls_name}_{i:02d}.jpg"
                    cv2.imwrite(str(cls_dir/crop_name), crop)
                    crop_rel_path=str(Path("crops")/cls_name/crop_name)

            dets.append({"class":cls_name,"bbox_xyxy":bbox,"poly_xy_flat":poly_flat,"area_px2":area_px2,"conf":conf,"crop_relpath":crop_rel_path})

        if SAVE_JSON:
            rec={"image_path":str(src_path),"save_path":str(out_img) if out_img else None,"detections":dets}
            (json_dir/f"{src_path.stem}.json").write_text(json.dumps(rec, indent=2))

        counts=Counter([d["class"] for d in dets]) if dets else {}
        lines=[f"{k}: {counts[k]}" for k in sorted(counts.keys())] if counts else ["No detections"]
        (save_dir/f"{src_path.stem}_counts.txt").write_text("\n".join(lines)+"\n")
        if out_img is not None: overlay_counts_on_image(out_img, lines)

        total_images=total_images+1 if 'total_images' in globals() else 1
        total_dets=total_dets+len(dets) if 'total_dets' in globals() else len(dets)
        print(f"✅ {src_path.name}: {len(dets)} detections | counts ->", ", ".join(lines))

    print("\n🎉 Done")
    if SAVE_JSON:  print("   JSON dir :", json_dir)
    if SAVE_CROPS: print("   Crops dir:", crops_dir)
    print(f"   Images processed: {total_images} | Total detections: {total_dets}")
