"""Shared helpers for the benchmark and ROI experiments."""
import pickle
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = [1, 2, 3, 5, 7]   # bicycle, car, motorcycle, bus, truck


def load_frames(video_path, n=100, max_w=1280):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // n, 1)
    frames = []
    for idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        if w > max_w:
            f = cv2.resize(f, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
        frames.append(f)
        if len(frames) >= n:
            break
    cap.release()
    return frames


def jpeg_roundtrip(frame, q):
    """Encode at JPEG quality q, decode again. Returns (decoded image, size in bytes)."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), len(buf)


def detect(model, frame, imgsz, conf=0.25):
    r = model.predict(frame, imgsz=imgsz, conf=conf, classes=VEHICLE_CLASSES, verbose=False)[0]
    b = r.boxes
    return b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()


def iou_matrix(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(pb, pc, gb, gc, thr=0.5):
    """Greedy class-aware matching. Returns (tp, fp, fn)."""
    if len(pb) == 0 or len(gb) == 0:
        return 0, len(pb), len(gb)
    iou = iou_matrix(pb, gb)
    iou[pc[:, None] != gc[None, :]] = 0
    used_p, used_g, tp = set(), set(), 0
    for i, j in np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]:
        if iou[i, j] < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); tp += 1
    return tp, len(pb) - tp, len(gb) - tp


def score(preds, gts):
    """preds/gts: lists of (boxes, classes) per frame -> precision, recall, F1, count MAE."""
    tp = fp = fn = 0
    mae = []
    for (pb, pc), (gb, gc) in zip(preds, gts):
        a, b, c = match(pb, pc, gb, gc)
        tp += a; fp += b; fn += c
        mae.append(abs(len(pb) - len(gb)))
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9), float(np.mean(mae))


def get_pseudo_gt(frames, ref_name="yolo26l.pt", imgsz=1280, conf=0.3, cache="results/pseudo_gt.pkl"):
    p = Path(cache)
    if p.exists():
        d = pickle.loads(p.read_bytes())
        if d["n"] == len(frames) and d["ref"] == ref_name and d["imgsz"] == imgsz:
            print("Using cached reference detections")
            return d["gt"]
    print(f"Computing reference detections with {ref_name} @ {imgsz} ...")
    model = YOLO(ref_name)
    gt = [detect(model, f, imgsz, conf)[:2] for f in frames]
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(pickle.dumps({"n": len(frames), "ref": ref_name, "imgsz": imgsz, "gt": gt}))
    return gt
