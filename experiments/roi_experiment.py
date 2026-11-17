"""ROI-aware encoding: sharp vehicles, low-detail background, vs uniform JPEG quality."""
import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from common import detect, get_pseudo_gt, jpeg_roundtrip, load_frames, score


def padded_boxes(boxes, shape, pad=0.15):
    h, w = shape[:2]
    out = []
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        out.append((max(int(x1 - pad * bw), 0), max(int(y1 - pad * bh), 0),
                    min(int(x2 + pad * bw), w), min(int(y2 + pad * bh), h)))
    return out


def roi_mask(shape, rois, feather=5):
    m = np.zeros(shape[:2], np.float32)
    for x1, y1, x2, y2 in rois:
        m[y1:y2, x1:x2] = 1.0
    return cv2.GaussianBlur(m, (0, 0), feather)[..., None]


def roi_encode(frame, rois, down, q):
    """Background low-passed by downscale factor `down`, vehicles kept sharp, then ONE JPEG."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(w // down, 1), max(h // down, 1)), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    m = roi_mask(frame.shape, rois)
    comp = (frame.astype(np.float32) * m + bg.astype(np.float32) * (1 - m)).astype(np.uint8)
    return jpeg_roundtrip(comp, q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="traffic.mp4")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--ref", default="yolo26l.pt")
    ap.add_argument("--edge-model", default="yolo26n.pt", help="cheap detector that proposes ROIs")
    ap.add_argument("--eval-model", default="yolo26s.pt", help="detector run on the received frame")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    frames = load_frames(args.video, args.frames)
    gts = get_pseudo_gt(frames, args.ref)
    edge, cloud = YOLO(args.edge_model), YOLO(args.eval_model)

    print("Proposing ROIs with the edge model ...")
    rois = [padded_boxes(detect(edge, f, 640, conf=0.15)[0], f.shape) for f in frames]

    def evaluate(encode):
        preds, kbs = [], []
        for i, f in enumerate(frames):
            img, nbytes = encode(f, rois[i])
            b, c, _ = detect(cloud, img, args.imgsz)
            preds.append((b, c)); kbs.append(nbytes / 1024)
        p, r, f1, mae = score(preds, gts)
        return float(np.mean(kbs)), p, r, f1, mae

    rows = []
    for q in [95, 90, 70, 50, 30, 15, 5]:
        print(f"uniform q={q}")
        kb, p, r, f1, mae = evaluate(lambda f, rs, q=q: jpeg_roundtrip(f, q))
        rows.append(dict(method="uniform", setting=f"q{q}", kb_per_frame=kb, precision=p, recall=r, f1=f1, count_mae=mae))
    for d in [4, 8, 16]:
        for q in [90, 70, 50]:
            print(f"roi down={d} q={q}")
            kb, p, r, f1, mae = evaluate(lambda f, rs, d=d, q=q: roi_encode(f, rs, d, q))
            rows.append(dict(method="roi", setting=f"d{d}/q{q}", kb_per_frame=kb, precision=p, recall=r, f1=f1, count_mae=mae))

    Path("results").mkdir(exist_ok=True)
    with open("results/roi_experiment.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    plt.figure(figsize=(7, 5))
    for method, style in [("uniform", "o-"), ("roi", "s")]:
        rs = [r for r in rows if r["method"] == method]
        plt.plot([r["kb_per_frame"] for r in rs], [r["f1"] for r in rs], style, label=method)
        for r in rs:
            plt.annotate(r["setting"], (r["kb_per_frame"], r["f1"]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    plt.xlabel("KB per frame (bandwidth)"); plt.ylabel("F1 vs reference"); plt.grid(alpha=.3)
    plt.title("Uniform JPEG vs ROI-aware encoding"); plt.legend(); plt.tight_layout()
    plt.savefig("results/roi_experiment.png", dpi=150)

    print("\nmethod   setting   KB/frame   F1     recall")
    for r in rows:
        print(f"{r['method']:<8} {r['setting']:<9} {r['kb_per_frame']:<10.1f} {r['f1']:<6.3f} {r['recall']:.3f}")


if __name__ == "__main__":
    main()
