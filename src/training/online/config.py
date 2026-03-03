from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:
    yaml = None


@dataclass
class OnlineConfig:
    video_root: str
    student_model_path: str
    student_model_name: str = "adaptive_unet"
    student_num_classes: int = 3
    teacher_model_path: str | None = None
    teacher_model_name: str = "unet_plus_plus"
    teacher_num_classes: int = 3
    img_size: int = 256
    lr: float = 1e-4
    online_steps: int = -1
    device: str = "cuda"

    frame_selector: str = "none"
    frame_select_interval: int = 5
    replay_policy: str = "fifo"
    replay_capacity: int = 256
    update_policy: str = "bn"
    update_last_n: int = 2
    pseudo_label_policy: str = "teacher_student"
    teacher_weight: float = 0.7

    monitor_interval: int = 20
    save_viz: bool = False
    viz_interval: int = 50
    output_dir: str = "runs/online"


def _load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for --config but is not installed.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping.")
    return data


def parse_args() -> OnlineConfig:
    parser = argparse.ArgumentParser("Online training entrypoint")
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--video-root", type=str, default=None)
    parser.add_argument("--student-model-path", type=str, default=None)
    parser.add_argument("--student-model-name", type=str, default=None)
    parser.add_argument("--student-num-classes", type=int, default=None)
    parser.add_argument("--teacher-model-path", type=str, default=None)
    parser.add_argument("--teacher-model-name", type=str, default=None)
    parser.add_argument("--teacher-num-classes", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--online-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--frame-selector", type=str, default=None)
    parser.add_argument("--frame-select-interval", type=int, default=None)
    parser.add_argument("--replay-policy", type=str, default=None)
    parser.add_argument("--replay-capacity", type=int, default=None)
    parser.add_argument("--update-policy", type=str, default=None)
    parser.add_argument("--update-last-n", type=int, default=None)
    parser.add_argument("--pseudo-label-policy", type=str, default=None)
    parser.add_argument("--teacher-weight", type=float, default=None)

    parser.add_argument("--monitor-interval", type=int, default=None)
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument("--viz-interval", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    ns = vars(parser.parse_args())
    base = OnlineConfig(video_root="", student_model_path="").__dict__.copy()
    if ns.get("config"):
        base.update(_load_yaml(ns["config"]))

    for k, v in ns.items():
        if k == "config":
            continue
        key = k.replace("-", "_")
        if v is not None:
            base[key] = v

    cfg = OnlineConfig(**{k: base[k] for k in OnlineConfig.__dataclass_fields__.keys()})
    _validate(cfg)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def _validate(cfg: OnlineConfig) -> None:
    if not cfg.video_root:
        raise ValueError("video_root is required")
    if not cfg.student_model_path:
        raise ValueError("student_model_path is required")
    if cfg.frame_selector not in {"none", "interval"}:
        raise ValueError("frame_selector must be one of: none, interval")
    if cfg.update_policy not in {"bn", "all", "last_n"}:
        raise ValueError("update_policy must be one of: bn, all, last_n")
