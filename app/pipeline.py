"""Detection pipelines used by the command line and the web app.

Two jobs:
  evaluate_labeled - TRUE accuracy: run models on a labelled image dataset (YOLO-format labels).
  process_video    - any uploaded video: detections, counts, speed, file size, and AGREEMENT with a
                     bigger reference model. Agreement is a proxy for accuracy, because an
                     unlabelled video has no answer key.
"""
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from scoring import (average_precision, find_images, label_path_for, load_class_names,
                     match_image, precision_recall_f1, read_yolo_labels, vehicle_class_ids)

# COCO class ids the pretrained YOLO models use for vehicles
COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
WARMUP = 3  # the first few frames are slow while the model initialises, so their timing is dropped

_MODELS = {}


def get_model(name, tag=""):
    """Load a YOLO model once and reuse it. A different tag gives a separate copy, so the test model
    and the reference model never share tracker state even when they are the same weights."""
    key = (name, tag)
    if key not in _MODELS:
        from ultralytics import YOLO
        _MODELS[key] = YOLO(name)
    return _MODELS[key]


def jpeg_roundtrip(img, quality):
    """Compress a frame as JPEG and decode it again: what the detector would see after transmission.
    Returns (decoded frame, size in bytes)."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), len(buf)


def to_numpy(boxes):
    """Pull boxes, confidences, class ids and (if tracking) track ids out of an Ultralytics Boxes object."""
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    ids = None
    if getattr(boxes, "id", None) is not None:
        ids = boxes.id.cpu().numpy().astype(int)
    return xyxy, conf, cls, ids


def _stats(values):
    v = np.asarray(values[WARMUP:] if len(values) > WARMUP else values, dtype=float)
    if len(v) == 0:
        return None, None
    return float(v.mean()), float(np.percentile(v, 95))


# =========================================================================== labelled dataset
def evaluate_labeled(data_dir, models, sizes, jpeg_qualities=(None,), split=None, max_images=None,
                     conf_thr=0.25, iou_thr=0.5, device=None, progress=None, vehicle_ids=None):
    """True accuracy on an image dataset that comes with YOLO-format labels.

    vehicle_ids: optional list of the dataset's class ids to treat as vehicles (overrides the
    automatic choice made from the class names in data.yaml).

    Accuracy is class-agnostic: a detection is right if it overlaps ANY ground-truth vehicle box
    with IoU >= iou_thr. (The dataset's own classes, such as 'auto' or 'two_wheeler', do not exist
    in COCO, so matching on class would unfairly punish the pretrained model.)

    Returns a list of result rows, one per (model, size, jpeg quality).
    """
    images = find_images(data_dir, split)
    if not images:
        raise FileNotFoundError(f"No images found under {data_dir} (split={split!r}).")
    if max_images and len(images) > max_images:
        idx = np.linspace(0, len(images) - 1, int(max_images)).astype(int)
        images = [images[i] for i in idx]

    names = load_class_names(data_dir)
    keep_ids = vehicle_class_ids(names, vehicle_ids)
    print(f"All classes in the dataset: {names or 'unknown (no data.yaml found)'}")
    if keep_ids is not None:
        print(f"Treated as vehicles (ground truth): {sorted(keep_ids)} "
              f"= {[names.get(i, i) for i in sorted(keep_ids)]}")
        print("Check this list. If it is wrong, pass --vehicle-ids with the right class numbers.")
    else:
        print("No class names found: using ALL labelled boxes as vehicles.")

    n_label_files = sum(label_path_for(p).exists() for p in images)
    if n_label_files == 0:
        raise FileNotFoundError(
            "Found images but no label .txt files. Expected .../images/x.jpg with .../labels/x.txt. "
            "Run `find <data_dir> | head -30` and check the folder layout.")
    print(f"{len(images)} images, {n_label_files} with a label file.")

    gt_cache = {}
    rows = []
    combos = [(m, s, q) for m in models for s in sizes for q in jpeg_qualities]
    for ci, (mname, size, q) in enumerate(combos):
        model = get_model(mname)
        all_tp, all_conf, lat, kbs = [], [], [], []
        n_gt = 0
        for ii, path in enumerate(images):
            img = cv2.imread(str(path))
            if img is None:
                continue
            h, w = img.shape[:2]
            if path not in gt_cache:
                gt_cache[path] = read_yolo_labels(label_path_for(path), w, h, keep_ids)[0]
            if q:
                img, nbytes = jpeg_roundtrip(img, q)
                kbs.append(nbytes / 1024)
            else:
                kbs.append(path.stat().st_size / 1024)

            t0 = time.perf_counter()
            r = model.predict(img, imgsz=int(size), conf=0.05, classes=list(COCO_VEHICLES),
                              device=device, verbose=False)[0]
            lat.append((time.perf_counter() - t0) * 1000)

            xyxy, conf, _, _ = to_numpy(r.boxes)
            tp, c, ng = match_image(xyxy, conf, gt_cache[path], iou_thr)
            all_tp.append(tp)
            all_conf.append(c)
            n_gt += ng
            if progress and ii % 10 == 0:
                progress((ci + ii / len(images)) / len(combos), f"{mname} @ {size}px  {ii}/{len(images)}")

        tp = np.concatenate(all_tp) if all_tp else np.zeros(0, bool)
        conf = np.concatenate(all_conf) if all_conf else np.zeros(0)
        p, r_, f1 = precision_recall_f1(tp, conf, n_gt, conf_thr)
        lat_mean, lat_p95 = _stats(lat)
        rows.append({
            "model": mname, "imgsz": int(size), "jpeg_q": q if q else "none",
            "images": len(images), "gt_boxes": n_gt,
            "precision": round(p, 3), "recall": round(r_, 3), "f1": round(f1, 3),
            "ap50": round(average_precision(tp, conf, n_gt), 3),
            "latency_ms_mean": round(lat_mean, 1) if lat_mean is not None else None,
            "latency_ms_p95": round(lat_p95, 1) if lat_p95 is not None else None,
            "kb_per_image": round(float(np.mean(kbs)), 1) if kbs else None,
        })
        print(rows[-1])
    return rows


# =========================================================================== any video
def process_video(video_path, out_path, model_name="yolo26n.pt", imgsz=640, jpeg_q=None, conf=0.25,
                  max_frames=300, line_frac=0.5, ref_name=None, ref_imgsz=960, iou_thr=0.5,
                  device=None, progress=None):
    """Track vehicles in a video, count line crossings, time the model, and optionally score it
    against a bigger reference model on the same frames.

    Returns a dict: the annotated video path, a summary, counts per class and per-frame counts.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    n_target = min(total, int(max_frames))

    model = get_model(model_name)
    model.predictor = None  # forget the previous video's tracker so IDs start fresh
    ref_model = get_model(ref_name, tag="ref") if ref_name else None

    tmp_path = str(Path(out_path).with_suffix(".raw.mp4"))
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    line_y = int(height * line_frac)
    prev_cy, counted = {}, set()
    counts = {n: {"down": 0, "up": 0} for n in COCO_VEHICLES.values()}
    lat, kbs, n_test, n_ref = [], [], [], []
    ref_tp, ref_conf, ref_ngt = [], [], 0

    frame_idx = 0
    while frame_idx < n_target:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        model_in = frame
        if jpeg_q:
            model_in, nbytes = jpeg_roundtrip(frame, jpeg_q)
            kbs.append(nbytes / 1024)

        t0 = time.perf_counter()
        r = model.track(model_in, imgsz=int(imgsz), conf=float(conf), classes=list(COCO_VEHICLES),
                        persist=True, device=device, verbose=False)[0]
        lat.append((time.perf_counter() - t0) * 1000)
        xyxy, confs, cls, ids = to_numpy(r.boxes)
        n_test.append(len(xyxy))

        # line crossing: a tracked vehicle is counted once, when its centre passes the line
        if ids is not None:
            for k in range(len(xyxy)):
                tid, cy = int(ids[k]), float((xyxy[k, 1] + xyxy[k, 3]) / 2)
                name = COCO_VEHICLES.get(int(cls[k]), "other")
                if tid not in counted and tid in prev_cy and name in counts:
                    if prev_cy[tid] < line_y <= cy:
                        counts[name]["down"] += 1
                        counted.add(tid)
                    elif prev_cy[tid] >= line_y > cy:
                        counts[name]["up"] += 1
                        counted.add(tid)
                prev_cy[tid] = cy

        # agreement with the reference model, which always sees the ORIGINAL frame
        if ref_model is not None:
            rr = ref_model.predict(frame, imgsz=int(ref_imgsz), conf=float(conf),
                                   classes=list(COCO_VEHICLES), device=device, verbose=False)[0]
            ref_xyxy = to_numpy(rr.boxes)[0]
            n_ref.append(len(ref_xyxy))
            tp, c, ng = match_image(xyxy, confs, ref_xyxy, iou_thr)
            ref_tp.append(tp)
            ref_conf.append(c)
            ref_ngt += ng

        annotated = r.plot()
        cv2.line(annotated, (0, line_y), (width, line_y), (0, 0, 255), 2)
        total_down = sum(v["down"] for v in counts.values())
        total_up = sum(v["up"] for v in counts.values())
        cv2.putText(annotated, f"in view: {len(xyxy)}   down: {total_down}   up: {total_up}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        writer.write(annotated)
        if progress and frame_idx % 5 == 0:
            progress(frame_idx / n_target, f"frame {frame_idx}/{n_target}")

    cap.release()
    writer.release()

    # re-encode to H.264 so the browser can play it
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_path, "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p", str(out_path)], check=True)
        Path(tmp_path).unlink(missing_ok=True)
    else:
        Path(tmp_path).replace(out_path)

    lat_mean, lat_p95 = _stats(lat)
    summary = {
        "frames_processed": frame_idx,
        "video_size": f"{width}x{height}",
        "model": model_name, "imgsz": int(imgsz), "jpeg_q": jpeg_q if jpeg_q else "none",
        "latency_ms_mean": round(lat_mean, 1) if lat_mean else None,
        "latency_ms_p95": round(lat_p95, 1) if lat_p95 else None,
        "model_fps": round(1000 / lat_mean, 1) if lat_mean else None,
        "kb_per_frame": round(float(np.mean(kbs)), 1) if kbs else None,
        "vehicles_per_frame": round(float(np.mean(n_test)), 2) if n_test else 0,
    }
    if ref_model is not None and ref_tp:
        tp, c = np.concatenate(ref_tp), np.concatenate(ref_conf)
        p, r_, f1 = precision_recall_f1(tp, c, ref_ngt, 0.0)
        summary.update({"reference": ref_name, "reference_imgsz": int(ref_imgsz),
                        "agreement_precision": round(p, 3), "agreement_recall": round(r_, 3),
                        "agreement_f1": round(f1, 3)})
    return {"video": str(out_path), "summary": summary, "counts": counts,
            "per_frame_test": n_test, "per_frame_ref": n_ref}
