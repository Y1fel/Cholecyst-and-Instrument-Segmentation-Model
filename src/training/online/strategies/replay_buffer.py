from __future__ import annotations

import random
from collections import deque
from typing import Deque, List


class ReplayBuffer:
    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._buf: Deque = deque(maxlen=capacity)

    def add(self, item) -> None:
        self._buf.append(item)

    def sample(self, n: int):
        if not self._buf:
            return []
        n = min(n, len(self._buf))
        return random.sample(list(self._buf), n)

    def __len__(self) -> int:
        return len(self._buf)
