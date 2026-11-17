"""
Vehicle detection, tracking, sub-type classification and line-crossing counting
(YOLO26 + OpenCV). Based on khayyam90/yolo-vehicules-detection (MIT).
"""
import argparse
import colorsys
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO_URL = "https://youtu.be/MNn9qKG2UFI?si=cu4PjC_5t4L3BF_E"
VIDEO_PATH = Path("traffic.mp4")
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COUNT_LINE_FRAC = 0.5
_GOLDEN_RATIO = 0.618033988749895


def track_color(track_id: int):
    hue = (track_id * _GOLDEN_RATIO) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def download_video() -> None:
    if VIDEO_PATH.exists():
        print(f"Video already present: {VIDEO_PATH}")
        return
    print(f"Downloading video from {VIDEO_URL} ...")
    subprocess.run([sys.executable, "-m", "yt_dlp", "-f", "best[ext=mp4]/best",
                    "-o", str(VIDEO_PATH), VIDEO_URL], check=True)


def detect_cars(model_name, imgsz, conf, save_output, display, use_subtypes):
    model = YOLO(model_name)
    voter = None
    if use_subtypes:
        from vehicle_types import SubtypeClassifier, TrackTypeVoter
        voter = TrackTypeVoter(SubtypeClassifier())

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.1f} fps ({total_frames} frames)")

    writer = None
    if save_output:
        writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    line_y = int(height * COUNT_LINE_FRAC)
    prev_cy, counted = {}, set()
    down_by_type, up_by_type = Counter(), Counter()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        annotated = frame.copy()
        cv2.line(annotated, (0, line_y), (width, line_y), (0, 0, 255), 2)

        results = model.track(frame, persist=True, conf=conf, imgsz=imgsz,
                              classes=list(VEHICLE_CLASSES), tracker="bytetrack.yaml",
                              verbose=False)[0]

        n_vehicles = 0
        for box in results.boxes:
            cls_id = int(box.cls)
            n_vehicles += 1
            c = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tid = int(box.id) if box.id is not None else None
            name = VEHICLE_CLASSES[cls_id]
            if voter and tid is not None:
                name = voter.label(tid, frame, (x1, y1, x2, y2), name)

            color = track_color(tid) if tid is not None else (0, 255, 0)
            label = f"{name} #{tid} {c:.2f}" if tid is not None else f"{name} {c:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if tid is not None and tid not in counted:
                cy = (y1 + y2) // 2
                if tid in prev_cy:
                    p = prev_cy[tid]
                    if p < line_y <= cy:
                        down_by_type[name] += 1
                        counted.add(tid)
                    elif p >= line_y > cy:
                        up_by_type[name] += 1
                        counted.add(tid)
                prev_cy[tid] = cy

        by_type = down_by_type + up_by_type
        cv2.putText(annotated, f"In frame: {n_vehicles}  Frame: {frame_idx}/{total_frames}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(annotated, f"Down: {sum(down_by_type.values())}  Up: {sum(up_by_type.values())}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
        for i, (nm, n) in enumerate(by_type.most_common(8)):
            cv2.putText(annotated, f"{nm}: {n}", (10, 88 + 26 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if writer:
            writer.write(annotated)
        if display:
            cv2.imshow("Vehicle analytics", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                cv2.imwrite(f"frame_{frame_idx:06d}.jpg", annotated)
        if frame_idx % 100 == 0:
            print(f"  processed {frame_idx}/{total_frames} frames")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    Path("results").mkdir(exist_ok=True)
    with open("results/counts_by_type.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "down", "up", "total"])
        for nm in sorted(set(down_by_type) | set(up_by_type)):
            d, u = down_by_type[nm], up_by_type[nm]
            w.writerow([nm, d, u, d + u])
    print(f"Done: {frame_idx} frames. Down {sum(down_by_type.values())} | Up {sum(up_by_type.values())}")
    print("Counts by type:", dict(down_by_type + up_by_type))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Vehicle detection, typing and counting.")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--subtypes", action="store_true", help="Enable second-stage sub-type classifier")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--no-display", action="store_true", help="Headless (use this if the window fails in WSL)")
    a = ap.parse_args()
    download_video()
    detect_cars(a.model, a.imgsz, a.conf, not a.no_save, not a.no_display, a.subtypes)
