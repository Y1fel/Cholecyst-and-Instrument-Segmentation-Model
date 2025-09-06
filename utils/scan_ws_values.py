# utils/scan_ws_values.py
import os, glob
import cv2
import numpy as np
from collections import Counter

def scan(root):
    pats = ["*_endo_watershed_mask.png", "*_watershed*.png"]
    c = Counter()
    files = []
    for p in pats:
        files += glob.glob(os.path.join(root, "**", p), recursive=True)
    for i, f in enumerate(files[:1000]):  # 先扫前1000张
        m = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if m is None: 
            continue
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        u, cnt = np.unique(m, return_counts=True)
        for ui, ci in zip(u.tolist(), cnt.tolist()):
            c[ui] += ci
        if i % 100 == 0:
            print(f"[scan] {i}/{len(files)}")
    print("\n=== Unique Gray Values (value: count) ===")
    for v, cnt in sorted(c.items()):
        print(f"{v:>3}: {cnt}")

if __name__ == "__main__":
    scan("data/seg8k")