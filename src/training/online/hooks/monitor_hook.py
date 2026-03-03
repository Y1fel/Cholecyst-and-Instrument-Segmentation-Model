from __future__ import annotations


class MonitorHook:
    def __init__(self, interval: int = 20):
        self.interval = max(1, interval)

    def on_step_end(self, step: int, loss: float) -> None:
        if step % self.interval == 0:
            print(f"[online] step={step} loss={loss:.4f}")
