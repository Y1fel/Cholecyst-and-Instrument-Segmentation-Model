from __future__ import annotations

import cv2
import torch

from src.training.online.strategies.frame_selector import FrameItem


class VideoStreamReader:
    def __init__(self, video_path: str, img_size: int = 256):
        self.cap = cv2.VideoCapture(video_path)
        self.img_size = img_size

    def __iter__(self):
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            resized = cv2.resize(frame, (self.img_size, self.img_size))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            yield FrameItem(index=idx, image_bgr=resized, tensor_chw=tensor)
            idx += 1

    def close(self):
        self.cap.release()
