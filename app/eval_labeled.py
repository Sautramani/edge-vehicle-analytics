"""Command-line true-accuracy check on a labelled image dataset (YOLO-format labels).

Example:
  python eval_labeled.py --data /content/traffic-vehicles-object-detection --split test \
      --models yolo26n.pt yolo26m.pt --imgsz 320 640 --jpeg none 50 --max-images 300
"""
import argparse
from pathlib import Path

import pandas as pd

from pipeline import evaluate_labeled


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset folder (images + labels + data.yaml)")
    ap.add_argument("--split", default=None, help="only images whose path contains this folder name, e.g. test")
    ap.add_argument("--models", nargs="+", default=["yolo26n.pt", "yolo26m.pt"])
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640])
    ap.add_argument("--jpeg", nargs="+", default=["none"], help="JPEG qualities to test, or 'none'")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold for precision/recall/F1")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU needed to count a detection as correct")
    ap.add_argument("--vehicle-ids", nargs="+", type=int, default=None,
                    help="dataset class numbers to treat as vehicles (default: chosen from class names)")
    ap.add_argument("--out", default="results/labeled_eval.csv")
    args = ap.parse_args()

    qualities = [None if q.lower() == "none" else int(q) for q in args.jpeg]
    rows = evaluate_labeled(args.data, args.models, args.imgsz, qualities, args.split,
                            args.max_images, args.conf, args.iou, vehicle_ids=args.vehicle_ids)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nSaved {args.out}")

    if len(df) > 1 and df["latency_ms_mean"].notna().all():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for _, r in df.iterrows():
            ax.scatter(r.latency_ms_mean, r.f1)
            ax.annotate(f"{r.model[:-3]}@{r.imgsz} q{r.jpeg_q}", (r.latency_ms_mean, r.f1), fontsize=8)
        ax.set_xlabel("Mean latency per image (ms)")
        ax.set_ylabel("F1 against human labels")
        ax.set_title("Latency vs true accuracy")
        ax.grid(alpha=0.3)
        png = str(Path(args.out).with_suffix(".png"))
        fig.savefig(png, dpi=150, bbox_inches="tight")
        print(f"Saved {png}")


if __name__ == "__main__":
    main()
