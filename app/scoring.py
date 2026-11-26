"""Scoring helpers: boxes in, precision / recall / F1 / AP out.

Pure NumPy, no model needed, so every function here can be tested on its own.
Boxes are always [x1, y1, x2, y2] in pixels.
"""
import os
from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# A dataset class counts as a "vehicle" if its name contains one of these words
# (and is not a number plate). Works for names like car, auto, two_wheeler, threewheel, van...
VEHICLE_WORDS = ("car", "auto", "bus", "truck", "van", "bike", "motor", "wheel",
                 "rickshaw", "scooter", "vehicle", "lorry", "taxi", "jeep", "suv", "tempo", "tractor")


# --------------------------------------------------------------------------- boxes
def iou_matrix(a, b):
    """IoU between every box in a (N,4) and every box in b (M,4) -> (N,M)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def match_image(det_boxes, det_conf, gt_boxes, iou_thr=0.5):
    """Match detections to ground-truth boxes in ONE image.

    Greedy, highest confidence first: each detection takes the best-overlapping
    ground-truth box that is still free, if IoU >= iou_thr (then it is a true
    positive); otherwise it is a false positive.

    Returns (tp, conf, n_gt): tp[i] says whether detection i is a true positive,
    conf is sorted high to low, n_gt is the number of ground-truth boxes.
    """
    det_boxes = np.asarray(det_boxes, dtype=float).reshape(-1, 4)
    det_conf = np.asarray(det_conf, dtype=float).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    order = np.argsort(-det_conf)
    det_boxes, det_conf = det_boxes[order], det_conf[order]
    ious = iou_matrix(det_boxes, gt_boxes)
    tp = np.zeros(len(det_boxes), dtype=bool)
    used = []
    for i in range(len(det_boxes)):
        if ious.shape[1] == 0:
            break
        cand = ious[i].copy()
        if used:
            cand[used] = -1.0
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            tp[i] = True
            used.append(j)
    return tp, det_conf, len(gt_boxes)


# --------------------------------------------------------------------------- metrics
def precision_recall_f1(tp, conf, n_gt, conf_thr):
    """Precision, recall and F1 using only detections with confidence >= conf_thr.

    Because matching went highest-confidence-first, dropping the low-confidence
    detections afterwards gives exactly the result of never having produced them.
    """
    keep = conf >= conf_thr
    n_det = int(keep.sum())
    n_tp = int(tp[keep].sum())
    p = n_tp / n_det if n_det else 0.0
    r = n_tp / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def average_precision(tp, conf, n_gt):
    """Area under the precision-recall curve (AP at the IoU used for matching)."""
    if n_gt == 0 or len(tp) == 0:
        return 0.0
    order = np.argsort(-conf)
    tp = tp[order].astype(float)
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1.0 - tp)
    recall = ctp / n_gt
    precision = ctp / (ctp + cfp)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]  # make precision non-increasing
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# --------------------------------------------------------------------------- dataset files
def find_images(root, split=None):
    """All images under root; if split is given (e.g. 'test'), only those with that folder name in their path."""
    imgs = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT)
    if split:
        imgs = [p for p in imgs if split.lower() in [part.lower() for part in p.parts]]
    return imgs


def label_path_for(img_path):
    """YOLO layout: .../images/x.jpg -> .../labels/x.txt. Falls back to a .txt beside the image."""
    parts = list(Path(img_path).parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return Path(img_path).with_suffix(".txt")


def load_class_names(root):
    """Class names as {id: name}: from data.yaml, or from a classes.txt / obj.names file
    (one class name per line, as written by LabelImg). {} if none is found."""
    import yaml
    for p in sorted(Path(root).rglob("*.y*ml")):
        try:
            with open(p) as fh:
                cfg = yaml.safe_load(fh)
        except Exception:
            continue
        if isinstance(cfg, dict) and "names" in cfg:
            names = cfg["names"]
            if isinstance(names, list):
                return {i: str(n) for i, n in enumerate(names)}
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
    for fname in ("classes.txt", "obj.names", "classes.names", "names.txt"):
        for p in sorted(Path(root).rglob(fname)):
            lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
            if lines:
                return {i: n for i, n in enumerate(lines)}
    return {}


def is_vehicle_name(name):
    n = name.lower()
    return "plate" not in n and any(w in n for w in VEHICLE_WORDS)


def vehicle_class_ids(names, override=None):
    """Ids of vehicle classes, or None (keep everything) when no names are known.

    override: explicit list of class ids to use as vehicles (when the names are unusual).
    """
    if override:
        return {int(i) for i in override}
    if not names:
        return None
    ids = {i for i, n in names.items() if is_vehicle_name(n)}
    if not ids:
        raise ValueError(f"None of these class names look like vehicles: {names}. "
                         "Pass the vehicle class ids yourself (--vehicle-ids 0 1 2).")
    return ids


def read_yolo_labels(path, img_w, img_h, keep_ids=None):
    """Read one YOLO .txt label file -> (boxes xyxy in pixels, class ids).

    Handles normal rows (class cx cy w h) and polygon rows (class x1 y1 x2 y2 ...).
    A missing file means 'no objects in this image'.
    """
    boxes, classes = [], []
    if path and os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                if keep_ids is not None and cls not in keep_ids:
                    continue
                vals = [float(v) for v in parts[1:]]
                if len(vals) == 4:
                    cx, cy, w, h = vals
                    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                else:
                    xs, ys = vals[0::2], vals[1::2]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                boxes.append([x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h])
                classes.append(cls)
    return np.array(boxes, dtype=float).reshape(-1, 4), np.array(classes, dtype=int)
