"""Builds one contact sheet per sub-type folder for quick visual checking."""
from pathlib import Path

import cv2
import numpy as np

src, out = Path("results/crops"), Path("results/crop_sheets")
out.mkdir(parents=True, exist_ok=True)
T, COLS = 160, 5
for d in sorted(p for p in src.iterdir() if p.is_dir()):
    files = sorted(d.glob("*.jpg"))[:20]
    if not files:
        continue
    tiles = [cv2.resize(cv2.imread(str(f)), (T, T)) for f in files]
    while len(tiles) % COLS:
        tiles.append(np.zeros((T, T, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]
    sheet = np.vstack(rows)
    cv2.putText(sheet, d.name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(out / f"{d.name}.jpg"), sheet)
    print("wrote", out / f"{d.name}.jpg", f"({len(files)} crops)")
