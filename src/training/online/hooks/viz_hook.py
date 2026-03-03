from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VizHook:
    def __init__(self, output_dir: str, interval: int = 50):
        self.interval = max(1, interval)
        self.output_dir = Path(output_dir) / "viz"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_prediction(self, step: int, image_bgr: np.ndarray, mask: np.ndarray) -> None:
        if step % self.interval != 0:
            return
        colored = cv2.applyColorMap((mask * 40).astype("uint8"), cv2.COLORMAP_JET)
        mix = cv2.addWeighted(image_bgr, 0.7, colored, 0.3, 0)
        cv2.imwrite(str(self.output_dir / f"viz_{step:06d}.jpg"), mix)
