"""Web app: upload any traffic video, or point at a labelled dataset, and see accuracy and speed.

Run:  python app.py            (then open http://localhost:7860)
      python app.py --share    (Colab: prints a public link)
"""
import argparse
import tempfile

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pipeline import evaluate_labeled, process_video

MODELS = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt"]
SIZES = [320, 480, 640, 960, 1280]
JPEG = ["none", "95", "75", "50", "30", "15"]


def run_video(video, model, size, jpeg, conf, max_frames, line_frac, compare, ref_model, ref_size,
              progress=gr.Progress()):
    if video is None:
        raise gr.Error("Upload a video first.")
    q = None if jpeg == "none" else int(jpeg)
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    res = process_video(video, out, model, int(size), q, float(conf), int(max_frames), float(line_frac),
                        ref_model if compare else None, int(ref_size),
                        progress=lambda f, d: progress(f, desc=d))
    s = res["summary"]

    lines = [f"**{s['frames_processed']} frames** ({s['video_size']}) with `{s['model']}` at {s['imgsz']} px, "
             f"JPEG quality: {s['jpeg_q']}",
             f"- Speed: {s['latency_ms_mean']} ms per frame on average (p95 {s['latency_ms_p95']} ms), "
             f"about {s['model_fps']} FPS",
             f"- Vehicles in view per frame: {s['vehicles_per_frame']}"]
    if s["kb_per_frame"]:
        lines.append(f"- Average frame size after JPEG: {s['kb_per_frame']} KB")
    if "agreement_f1" in s:
        lines += [f"- **Agreement with `{s['reference']}` at {s['reference_imgsz']} px: "
                  f"F1 {s['agreement_f1']}** (precision {s['agreement_precision']}, "
                  f"recall {s['agreement_recall']})",
                  "",
                  "_This is agreement with a bigger model, not accuracy against human labels, because an "
                  "uploaded video has no answer key. Use the **Labelled dataset** tab for true accuracy._"]
    else:
        lines += ["", "_Tick **Compare with a reference model** to get an agreement score._"]

    df = pd.DataFrame([{"class": k, "down": v["down"], "up": v["up"], "total": v["down"] + v["up"]}
                       for k, v in res["counts"].items()])

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(res["per_frame_test"], label=f"{s['model']} (test)")
    if res["per_frame_ref"]:
        ax.plot(res["per_frame_ref"], label=f"{s['reference']} (reference)", alpha=0.7)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Vehicles detected")
    ax.legend()
    ax.grid(alpha=0.3)
    return res["video"], "\n".join(lines), df, fig


def run_labeled(data_dir, split, models, sizes, jpegs, max_images, conf, vehicle_ids_text,
                progress=gr.Progress()):
    if not data_dir:
        raise gr.Error("Enter the dataset folder path.")
    if not models or not sizes or not jpegs:
        raise gr.Error("Pick at least one model, one image size and one JPEG option.")
    qualities = [None if j == "none" else int(j) for j in jpegs]
    ids = [int(x) for x in vehicle_ids_text.replace(",", " ").split()] if vehicle_ids_text else None
    try:
        rows = evaluate_labeled(data_dir, models, [int(s) for s in sizes], qualities, split or None,
                                int(max_images), float(conf), progress=lambda f, d: progress(f, desc=d),
                                vehicle_ids=ids)
    except (FileNotFoundError, ValueError) as e:
        raise gr.Error(str(e))
    return pd.DataFrame(rows)


def build():
    with gr.Blocks(title="Edge Vehicle Analytics") as demo:
        gr.Markdown("# Edge Vehicle Analytics\nHow do model size, image size and JPEG compression change "
                    "vehicle-detection accuracy, speed and bandwidth?")
        with gr.Tab("Upload a video"):
            with gr.Row():
                with gr.Column():
                    video = gr.Video(label="Traffic video")
                    model = gr.Dropdown(MODELS, value="yolo26n.pt", label="Model")
                    size = gr.Dropdown(SIZES, value=640, label="Image size (px)")
                    jpeg = gr.Dropdown(JPEG, value="none", label="JPEG quality (lower = smaller frames)")
                    conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
                    max_frames = gr.Slider(30, 2000, value=300, step=10, label="Max frames to process")
                    line = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Counting line position (0 = top)")
                    compare = gr.Checkbox(value=True, label="Compare with a reference model (slower)")
                    ref_model = gr.Dropdown(MODELS, value="yolo26l.pt", label="Reference model")
                    ref_size = gr.Dropdown(SIZES, value=960, label="Reference image size (px)")
                    go = gr.Button("Run", variant="primary")
                with gr.Column():
                    out_video = gr.Video(label="Annotated result")
                    out_md = gr.Markdown()
                    out_df = gr.Dataframe(label="Line-crossing counts by vehicle class")
                    out_plot = gr.Plot(label="Vehicles per frame")
            go.click(run_video, [video, model, size, jpeg, conf, max_frames, line, compare, ref_model, ref_size],
                     [out_video, out_md, out_df, out_plot])

        with gr.Tab("Labelled dataset (true accuracy)"):
            gr.Markdown("Point this at a folder holding images, YOLO-format `labels/` and a `data.yaml` "
                        "(for example the Kaggle *traffic-vehicles-object-detection* dataset). "
                        "Accuracy here is measured against human labels.")
            data_dir = gr.Textbox(label="Dataset folder", placeholder="/content/traffic-vehicles-object-detection")
            split = gr.Textbox(label="Split folder name (optional)", value="test")
            models = gr.CheckboxGroup(MODELS[:3], value=["yolo26n.pt", "yolo26m.pt"], label="Models")
            sizes = gr.CheckboxGroup(SIZES[:4], value=[320, 640], label="Image sizes")
            jpegs = gr.CheckboxGroup(JPEG, value=["none", "50"], label="JPEG qualities")
            max_images = gr.Slider(20, 1000, value=200, step=10, label="Max images")
            conf_l = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold for P / R / F1")
            veh_ids = gr.Textbox(label="Vehicle class numbers (optional, e.g. 0 1 2)",
                                 info="Leave empty to choose vehicles automatically from the class names in data.yaml.")
            go2 = gr.Button("Evaluate", variant="primary")
            table = gr.Dataframe(label="Results")
            go2.click(run_labeled, [data_dir, split, models, sizes, jpegs, max_images, conf_l, veh_ids], table)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public link (use on Colab)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build().launch(share=args.share, server_name="0.0.0.0", server_port=args.port)
