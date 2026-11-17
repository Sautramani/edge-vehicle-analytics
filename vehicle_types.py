"""Second-stage vehicle sub-type classifier (ImageNet MobileNetV3, restricted per COCO class)."""
import os
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch
from torchvision import models

# ImageNet category -> label shown on screen
SUBTYPE_MAP = {
    "sports car": "sports car", "racer": "sports car", "convertible": "convertible",
    "limousine": "sedan", "cab": "taxi", "jeep": "SUV/jeep", "minivan": "van",
    "beach wagon": "wagon", "pickup": "pickup", "police van": "police van",
    "ambulance": "ambulance", "moving van": "van", "trailer truck": "trailer truck",
    "garbage truck": "garbage truck", "tow truck": "tow truck", "fire engine": "fire engine",
    "minibus": "minibus", "school bus": "bus", "trolleybus": "bus",
    "motor scooter": "scooter", "moped": "moped", "tractor": "tractor",
    "mountain bike": "bicycle", "bicycle-built-for-two": "bicycle", "tricycle": "tricycle",
}

# Sub-types allowed for each COCO class reported by the detector
ALLOWED = {
    "car": {"sports car", "convertible", "sedan", "taxi", "SUV/jeep", "van", "wagon"},
    "truck": {"pickup", "van", "trailer truck", "garbage truck", "tow truck",
              "fire engine", "ambulance", "police van", "tractor"},
    "bus": {"bus", "minibus"},
    "motorcycle": {"scooter", "moped"},
    "bicycle": {"bicycle", "tricycle"},
}


class SubtypeClassifier:
    def __init__(self, min_mass=0.05, min_share=0.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        w = models.MobileNet_V3_Large_Weights.DEFAULT
        self.model = models.mobilenet_v3_large(weights=w).eval().to(self.device)
        self.prep = w.transforms()
        cats = w.meta["categories"]
        self.idx = [i for i, c in enumerate(cats) if c in SUBTYPE_MAP]
        self.names = [SUBTYPE_MAP[cats[i]] for i in self.idx]
        self.min_mass = min_mass     # total probability the net gives to allowed vehicle classes
        self.min_share = min_share   # winner's share among allowed classes
        print(f"Sub-type classifier: {len(self.idx)} vehicle classes on {self.device}")

    @torch.no_grad()
    def classify(self, bgr_crop, coco_name):
        allowed = ALLOWED.get(coco_name)
        if not allowed:
            return None, 0.0
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        x = self.prep(torch.from_numpy(rgb).permute(2, 0, 1)).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0][self.idx].tolist()
        agg = defaultdict(float)
        for n, p in zip(self.names, probs):
            if n in allowed:
                agg[n] += p
        total = sum(agg.values())
        if total < self.min_mass:
            return None, 0.0
        best = max(agg, key=agg.get)
        share = agg[best] / total
        return (best, share) if share >= self.min_share else (None, share)


class TrackTypeVoter:
    """Classifies each track a few times and keeps the majority label, e.g. 'car (sedan)'."""

    def __init__(self, clf, max_votes=5, every=5, min_side=48):
        self.clf, self.max_votes, self.every, self.min_side = clf, max_votes, every, min_side
        self.votes = defaultdict(Counter)
        self.n_votes = defaultdict(int)
        self.seen = defaultdict(int)
        self.dump = os.environ.get("DUMP_CROPS")      # set DUMP_CROPS=1 to save sample crops
        self.dumped = Counter()

    def label(self, tid, frame, box, fallback):
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
        self.seen[tid] += 1
        if (self.n_votes[tid] < self.max_votes and self.seen[tid] % self.every == 1
                and x2 - x1 >= self.min_side and y2 - y1 >= self.min_side):
            crop = frame[y1:y2, x1:x2]
            name, _ = self.clf.classify(crop, fallback)
            if name:
                self.votes[tid][name] += 1
                if self.dump and self.dumped[name] < 20:
                    d = Path("results/crops") / f"{fallback}_{name.replace('/', '_')}"
                    d.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(d / f"{tid}_{self.dumped[name]}.jpg"), crop)
                    self.dumped[name] += 1
            self.n_votes[tid] += 1
        if self.votes[tid]:
            return f"{fallback} ({self.votes[tid].most_common(1)[0][0]})"
        return fallback
