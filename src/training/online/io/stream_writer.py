from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class StreamWriter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_mask(self, idx: int, mask: np.ndarray) -> None:
        out = self.output_dir / f"pred_{idx:06d}.png"
        cv2.imwrite(str(out), mask.astype("uint8"))
