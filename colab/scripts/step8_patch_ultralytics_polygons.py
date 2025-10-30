# Polygon-only overlays (no boxes, no shaded masks, no confidences)
import cv2, numpy as np, torch
import torch.nn.functional as F
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.engine.results import Results

def _box_label_noop(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)): return
def _masks_noop(self, masks, colors, im_gpu, alpha: float = 0.5, retina_masks: bool = False): return

def _segmentation(self, mask, label='', color=(0, 255, 0), thresh: float = 0.30):
    if isinstance(mask, torch.Tensor):
        m = mask.unsqueeze(0).unsqueeze(0)
        m = F.interpolate(m, size=(self.im.shape[0], self.im.shape[1]), mode='bilinear', align_corners=False)
        m = (m.squeeze().detach().cpu().numpy() > thresh).astype(np.uint8)
    else:
        m = mask
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if len(cnt) < 3: continue
        cv2.polylines(self.im, [cnt], isClosed=True, color=color, thickness=self.lw)
        if label:
            x, y = cnt[0][0]
            cv2.putText(self.im, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

Annotator.box_label = _box_label_noop
Annotator.masks = _masks_noop
Annotator.segmentation = _segmentation

__orig_plot = Results.plot
def _plot_polygons_only(self, conf=True, boxes=True, masks=True, probs=False, labels=True, *args, **kwargs):
    im = (self.orig_img.copy() if getattr(self, "orig_img", None) is not None else self.plot_img.copy())
    annotator = Annotator(im, example=str(self.names))
    pred_masks = getattr(self, "masks", None)
    pred_boxes = getattr(self, "boxes", None)
    if getattr(pred_masks, "data", None) is not None:
        classes = (pred_boxes.cls.tolist() if (pred_boxes is not None and hasattr(pred_boxes, "cls"))
                   else [0] * len(pred_masks.data))
        for m, cls_idx in zip(pred_masks.data, classes):
            cls_idx = int(cls_idx)
            annotator.segmentation(m, label=self.names[cls_idx], color=colors(cls_idx, bgr=True))
    return annotator.result()

Results.plot = _plot_polygons_only
print("✅ Ultralytics patched: polygon-only overlays enabled.")
print("   Restart runtime to undo the patch.")
