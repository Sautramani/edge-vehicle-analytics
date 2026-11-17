"""
Car detection using YOLO26 on a traffic video.
Downloads the video if not already present, then processes it frame by frame.
"""

import cv2
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO

#VIDEO_URL = "https://youtu.be/KBsqQez-O4w?si=AlZTy4ahNV2EvsLp"
VIDEO_URL = "https://youtu.be/MNn9qKG2UFI?si=cu4PjC_5t4L3BF_E"
VIDEO_PATH = Path("traffic.mp4")
MODEL_NAME = "yolo26n.pt"

# COCO class IDs for vehicles
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CONF_THRESHOLD = 0.2

# Fraction of frame height where the counting line sits (0.0 = top, 1.0 = bottom)
COUNT_LINE_FRAC = 0.5

# Box colours per class (BGR) — fallback when no track ID is available
CLASS_COLORS = {
    2: (0, 255, 0),    # car     — green
    3: (255, 0, 255),  # moto    — magenta
    5: (0, 165, 255),  # bus     — orange
    7: (255, 255, 0),  # truck   — cyan
}

# Golden-ratio hue stepping gives maximally distinct colours for sequential IDs
_GOLDEN_RATIO = 0.618033988749895

def track_color(track_id: int) -> tuple[int, int, int]:
    """Return a unique BGR colour for the given track ID."""
    import colorsys
    hue = (track_id * _GOLDEN_RATIO) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def download_video() -> None:
    if VIDEO_PATH.exists():
        print(f"Video already present: {VIDEO_PATH}")
        return
    print(f"Downloading video from {VIDEO_URL} …")
    subprocess.run(
        [
            sys.executable, "-m", "yt_dlp",
            "-f", "best[ext=mp4]/best",
            "-o", str(VIDEO_PATH),
            VIDEO_URL,
        ],
        check=True,
    )
    print("Download complete.")


def detect_cars(save_output: bool = True) -> None:
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.1f} fps  ({total_frames} frames)")

    writer = None
    if save_output:
        out_path = Path("output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        print(f"Writing annotated output to {out_path}")

    count_line_y = int(height * COUNT_LINE_FRAC)
    prev_centroids: dict[int, int] = {}   # track_id -> previous centroid y
    counted_ids: set[int] = set()         # track IDs already counted (prevent re-count)
    count_down = 0   # crossed line top→bottom (increasing y)
    count_up = 0     # crossed line bottom→top (decreasing y)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        annotated = frame.copy()

        # Draw the counting line
        cv2.line(annotated, (0, count_line_y), (width, count_line_y), (0, 0, 255), 2)
        cv2.putText(
            annotated, "counting line",
            (4, count_line_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
        )

        results = model.track(frame, persist=True, verbose=False)[0]

        car_count = 0
        for box in results.boxes:
            cls_id = int(box.cls)
            if cls_id not in VEHICLE_CLASSES:
                continue
            conf = float(box.conf)
            if conf < CONF_THRESHOLD:
                continue

            car_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tid = int(box.id) if box.id is not None else None
            color = track_color(tid) if tid is not None else CLASS_COLORS[cls_id]
            label = f"{VEHICLE_CLASSES[cls_id]} #{tid} {conf:.2f}" if tid is not None else f"{VEHICLE_CLASSES[cls_id]} {conf:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, label,
                (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

            # Count line crossing for tracked vehicles only
            if tid is not None and tid not in counted_ids:
                cy = (y1 + y2) // 2
                if tid in prev_centroids:
                    prev_cy = prev_centroids[tid]
                    if prev_cy < count_line_y <= cy:
                        count_down += 1
                        counted_ids.add(tid)
                    elif prev_cy >= count_line_y > cy:
                        count_up += 1
                        counted_ids.add(tid)
                prev_centroids[tid] = cy

        cv2.putText(
            annotated, f"Vehicles: {car_count}  Frame: {frame_idx}/{total_frames}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2,
        )
        cv2.putText(
            annotated, f"Down (v): {count_down}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2,
        )
        cv2.putText(
            annotated, f"Up   (^): {count_up}",
            (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 140, 0), 2,
        )

        if writer:
            writer.write(annotated)

        cv2.imshow("Car Detection (YOLO)", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Interrupted by user.")
            break
        if key == ord("s"):
            snapshot_path = f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(snapshot_path, annotated)
            print(f"Saved snapshot: {snapshot_path}")

        if frame_idx % 100 == 0:
            pct = frame_idx / total_frames * 100 if total_frames else 0
            print(f"  processed {frame_idx} frames ({pct:.1f}%)")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Done — processed {frame_idx} frames.")
    print(f"  Down (↓): {count_down}  |  Up (↑): {count_up}  |  Total: {count_down + count_up}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect cars in a traffic video using YOLO.")
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write an annotated output video.",
    )
    args = parser.parse_args()

    download_video()
    detect_cars(save_output=not args.no_save)
