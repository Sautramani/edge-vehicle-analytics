"""Latency / accuracy / bandwidth benchmark over model size, input resolution and JPEG quality."""
import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

from common import detect, get_pseudo_gt, jpeg_roundtrip, load_frames, score


def run_config(model, frames, gts, imgsz, q):
    detect(model, frames[0], imgsz); detect(model, frames[0], imgsz)      # warm-up
    lat, kb, preds = [], [], []
    for f in frames:
        img, nbytes = jpeg_roundtrip(f, q)
        t = time.perf_counter()
        b, c, _ = detect(model, img, imgsz)
        lat.append((time.perf_counter() - t) * 1000)
        preds.append((b, c)); kb.append(nbytes / 1024)
    p, r, f1, mae = score(preds, gts)
    return dict(latency_ms=float(np.mean(lat)), p95_ms=float(np.percentile(lat, 95)),
                fps=1000 / float(np.mean(lat)), kb_per_frame=float(np.mean(kb)),
                precision=p, recall=r, f1=f1, count_mae=mae)


def plot(rows, out):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    spec = [("model", "latency_ms", "Latency (ms/frame)", "model", "Model size (640 px, JPEG q95)"),
            ("resolution", "latency_ms", "Latency (ms/frame)", "imgsz", "Input resolution (yolo26n)"),
            ("jpeg", "kb_per_frame", "KB per frame", "jpeg_q", "JPEG quality (yolo26n, 640 px)")]
    for a, (sweep, xk, xl, lk, title) in zip(ax, spec):
        rs = [r for r in rows if r["sweep"] == sweep]
        a.plot([r[xk] for r in rs], [r["f1"] for r in rs], "o-")
        for r in rs:
            a.annotate(str(r[lk]), (r[xk], r["f1"]), textcoords="offset points", xytext=(4, 5))
        a.set_xlabel(xl); a.set_ylabel("F1 vs reference"); a.set_title(title); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="traffic.mp4")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--ref", default="yolo26l.pt")
    args = ap.parse_args()
    print("CUDA available:", torch.cuda.is_available())

    frames = load_frames(args.video, args.frames)
    print(f"Loaded {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}")
    gts = get_pseudo_gt(frames, args.ref)

    configs = [("model", m, 640, 95) for m in ["yolo26n", "yolo26s", "yolo26m"]]
    configs += [("resolution", "yolo26n", r, 95) for r in [320, 480, 640, 960, 1280]]
    configs += [("jpeg", "yolo26n", 640, q) for q in [95, 75, 50, 30, 15]]

    models, cache, rows = {}, {}, []
    for sweep, m, imgsz, q in configs:
        key = (m, imgsz, q)
        if key not in cache:
            if m not in models:
                models[m] = YOLO(f"{m}.pt")
            print(f"Running {m} imgsz={imgsz} jpeg_q={q}")
            cache[key] = run_config(models[m], frames, gts, imgsz, q)
        rows.append(dict(sweep=sweep, model=m, imgsz=imgsz, jpeg_q=q, **cache[key]))

    Path("results").mkdir(exist_ok=True)
    with open("results/benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    plot(rows, "results/benchmark.png")
    print("\nsweep       model     imgsz  q    ms/frame   KB/frame   F1     countMAE")
    for r in rows:
        print(f"{r['sweep']:<11} {r['model']:<9} {r['imgsz']:<6} {r['jpeg_q']:<4} "
              f"{r['latency_ms']:<10.1f} {r['kb_per_frame']:<10.1f} {r['f1']:<6.3f} {r['count_mae']:.2f}")


if __name__ == "__main__":
    main()
