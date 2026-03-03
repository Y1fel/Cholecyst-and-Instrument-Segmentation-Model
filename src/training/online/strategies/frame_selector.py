from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class FrameItem:
    index: int
    image_bgr: any
    tensor_chw: any


class FrameSelector(Protocol):
    def should_train(self, frame: FrameItem) -> bool:
        ...


class EveryFrameSelector:
    def should_train(self, frame: FrameItem) -> bool:
        return True


class IntervalFrameSelector:
    def __init__(self, interval: int = 5):
        self.interval = max(1, interval)

    def should_train(self, frame: FrameItem) -> bool:
        return frame.index % self.interval == 0
