# OpenCV Cars Tracking

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Ultralytics YOLO26](https://img.shields.io/badge/Ultralytics-YOLO26-FF6F00?logo=pytorch&logoColor=white)](https://docs.ultralytics.com/)
[![yt-dlp](https://img.shields.io/badge/yt--dlp-latest-FF0000?logo=youtube&logoColor=white)](https://github.com/yt-dlp/yt-dlp)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e)](LICENSE)

Real-time vehicle detection, tracking, and **line-crossing counting** on traffic footage using **YOLO26** and **OpenCV**. Each vehicle (car, motorcycle, bus, truck) gets a unique colour tied to its track ID, and a horizontal counting line tallies how many vehicles cross in each direction.

---

## Preview

![Annotated frame — car counter in action](frame_001208.jpg)

*Frame 1208: bounding boxes with per-track colours, counting line (red), and live up/down counters (top-left).*

---

## Features

- Detects **cars, motorcycles, buses, and trucks** (COCO classes 2, 3, 5, 7)
- Persistent **multi-object tracking** — each vehicle keeps its ID across frames
- **Bidirectional line-crossing counter** — counts vehicles crossing a configurable horizontal line, separately for downward (↓) and upward (↑) directions; each vehicle is counted only once
- **Per-track colour** using golden-ratio hue stepping for maximum visual separation
- Live **HUD overlay** — vehicle count in frame, frame progress, and running up/down totals
- Saves an annotated **output video** alongside real-time preview window
- Auto-downloads the source video with `yt-dlp` if not already present
- Press **`q`** to quit early, **`s`** to save a snapshot of the current frame

---

## Car Counter

A red horizontal line is drawn across the middle of the frame (configurable via `COUNT_LINE_FRAC` at the top of `detect_cars.py`).

Every tracked vehicle's centroid is compared frame-to-frame:

| Direction | Condition | Counter |
|-----------|-----------|---------|
| Down (↓)  | centroid was **above** the line, now **at or below** | `count_down += 1` |
| Up (↑)    | centroid was **at or below** the line, now **above** | `count_up += 1` |

Each track ID is counted **at most once** — re-crossing the same line does not double-count. Vehicles without a track ID (rare edge case) are detected but not counted.

The final tally is printed to the console when processing finishes:

```
Done — processed 1834 frames.
  Down (↓): 42  |  Up (↑): 3  |  Total: 45
```

---

## Requirements

- Python 3.10+
- A CUDA-capable GPU is optional but speeds up inference significantly

---

## Installation

```bash
git clone https://github.com/khayyam90/yolo-vehicules-detection.git
cd yolo-vehicules-detection

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

```bash
# Download the video (if needed) and process it with annotated output saved
python detect_cars.py

# Process without saving output.mp4
python detect_cars.py --no-save
```

| Argument    | Default | Description                              |
|-------------|---------|------------------------------------------|
| `--no-save` | off     | Skip writing the annotated `output.mp4`  |

**Keyboard shortcuts during playback:**

| Key | Action |
|-----|--------|
| `q` | Quit early |
| `s` | Save current annotated frame as `frame_NNNNNN.jpg` |

---

## How It Works

1. **Download** — `yt-dlp` fetches the traffic video from YouTube if `traffic.mp4` is missing.
2. **Load model** — YOLO26n (nano) is loaded via the Ultralytics library; weights are downloaded automatically on first run.
3. **Track** — Each frame is passed to `model.track()` with `persist=True` so the tracker maintains object identities across frames.
4. **Filter** — Only detections belonging to vehicle classes with confidence ≥ 0.2 are kept.
5. **Count** — Each tracked vehicle's centroid is checked against the counting line; the first crossing in either direction increments the corresponding counter.
6. **Annotate** — Bounding boxes, labels, the counting line, and the live HUD are drawn using a unique colour per track ID (golden-ratio HSV stepping).
7. **Output** — Annotated frames are written to `output.mp4` and displayed in a live window.

---

## Configuration

All tuneable constants are at the top of `detect_cars.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `VIDEO_URL` | YouTube URL | Source video to download |
| `MODEL_NAME` | `yolo26n.pt` | YOLO26 model variant (n/s/m/l/x) |
| `CONF_THRESHOLD` | `0.2` | Minimum detection confidence |
| `COUNT_LINE_FRAC` | `0.5` | Counting line position (0.0 = top, 1.0 = bottom) |

---

## Project Structure

```
opencv-cars-tracking/
├── detect_cars.py   # main script
├── requirements.txt
├── traffic.mp4      # source video (git-ignored, downloaded at runtime)
└── output.mp4       # annotated output (git-ignored, generated at runtime)
```

---

## License

[MIT](LICENSE)
