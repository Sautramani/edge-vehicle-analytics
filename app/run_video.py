"""Run the detector on ANY traffic video and save the results. No web page needed.

Example:
  python app/run_video.py my_video.mp4 --model yolo26n.pt --imgsz 640 --jpeg 50 --ref yolo26l.pt

Saves, in --out-dir:
  annotated.mp4         the video with boxes, track IDs and the counting line
  summary.json          speed, frame size, vehicles per frame, agreement with the reference model
  counts.csv            line-crossing counts per vehicle class
  per_frame.csv         vehicles detected in every frame
  vehicles_per_frame.png plot of the above
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pipeline import process_video


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="path to any traffic video (mp4, avi, mov ...)")
    ap.add_argument("--model", default="yolo26n.pt", help="detector: yolo26n/s/m/l .pt")
    ap.add_argument("--imgsz", type=int, default=640, help="image size the model works at")
    ap.add_argument("--jpeg", default="none", help="JPEG quality 1-100 to simulate compression, or 'none'")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--max-frames", type=int, default=300, help="only process this many frames")
    ap.add_argument("--line", type=float, default=0.5, help="counting line height, 0 = top, 1 = bottom")
    ap.add_argument("--ref", default="none", help="reference model for the agreement score, or 'none'")
    ap.add_argument("--ref-imgsz", type=int, default=960)
    ap.add_argument("--out-dir", default="results/video_run")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    q = None if args.jpeg.lower() == "none" else int(args.jpeg)
    ref = None if args.ref.lower() == "none" else args.ref

    def show(frac, desc):
        print(f"\r{desc}", end="", flush=True)

    res = process_video(args.video, str(out / "annotated.mp4"), args.model, args.imgsz, q, args.conf,
                        args.max_frames, args.line, ref, args.ref_imgsz, progress=show)
    print()

    s = res["summary"]
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    counts = pd.DataFrame([{"class": k, "down": v["down"], "up": v["up"], "total": v["down"] + v["up"]}
                           for k, v in res["counts"].items()])
    counts.to_csv(out / "counts.csv", index=False)
    per_frame = pd.DataFrame({"frame": range(1, len(res["per_frame_test"]) + 1),
                              "vehicles_test": res["per_frame_test"]})
    if res["per_frame_ref"]:
        per_frame["vehicles_reference"] = res["per_frame_ref"]
    per_frame.to_csv(out / "per_frame.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(per_frame["frame"], per_frame["vehicles_test"], label=f"{args.model} (test)")
    if "vehicles_reference" in per_frame:
        ax.plot(per_frame["frame"], per_frame["vehicles_reference"], label=f"{ref} (reference)", alpha=0.7)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Vehicles detected")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out / "vehicles_per_frame.png", dpi=150, bbox_inches="tight")

    print(json.dumps(s, indent=2))
    print("\nLine-crossing counts:")
    print(counts.to_string(index=False))
    if "agreement_f1" in s:
        print("\nNote: the agreement score compares against the reference model, not human labels.")
    print(f"\nSaved everything in {out}/")


if __name__ == "__main__":
    main()
