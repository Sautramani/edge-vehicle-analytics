
## Run it on your own traffic video

Everything below lives in the `app/` folder.

**Colab (T4 GPU):** open `vehicle_analytics_colab.ipynb`, set the runtime to T4 GPU, and run the cells.
It uploads your video, runs the detector, and shows the annotated video, counts, speed and frame size.

**Command line (WSL / Linux):**

```bash
pip install -r requirements.txt -r app/requirements_app.txt
python app/run_video.py my_video.mp4 --model yolo26n.pt --imgsz 640 --jpeg 50 --ref yolo26l.pt
```

Results go to `results/video_run/`: `annotated.mp4`, `summary.json`, `counts.csv`, `per_frame.csv` and a plot.

**Web app:** `python app/app.py` and open http://localhost:7860 (on Colab: `python app/app.py --share`).

## Two ways of measuring accuracy

| | Uploaded video | Labelled dataset |
|---|---|---|
| Answer key | None | Human labels |
| What the score means | **Agreement** with a bigger reference model (`yolo26l`) | **True accuracy** (precision, recall, F1, AP at IoU 0.5) |
| Command | `app/run_video.py` | `app/eval_labeled.py` |

An uploaded video has no labels, so its score only says how closely a smaller or more compressed setting matches the reference model, mistakes included.
True accuracy comes from the labelled Kaggle dataset
[Traffic Road Object Detection Dataset using YOLO](https://www.kaggle.com/datasets/boukraailyesali/traffic-road-object-detection-dataset-using-yolo).
Matching is class-agnostic: a detection is correct if it overlaps any labelled vehicle with IoU of at least 0.5,
because the pretrained COCO models have no class for some local vehicle types.

```bash
python app/eval_labeled.py --data path/to/dataset --split test --models yolo26n.pt yolo26m.pt --imgsz 320 640 --jpeg none 50
```

The script prints which dataset classes it treated as vehicles; check that list, and use `--vehicle-ids` to correct it.
