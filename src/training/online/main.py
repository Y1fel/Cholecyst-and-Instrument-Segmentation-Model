from __future__ import annotations

import torch

from src.models.model_zoo import build_model
from src.training.online.config import parse_args
from src.training.online.engine import OnlineTrainer
from src.training.online.hooks.monitor_hook import MonitorHook
from src.training.online.hooks.viz_hook import VizHook
from src.training.online.io.stream_reader import VideoStreamReader
from src.training.online.io.stream_writer import StreamWriter
from src.training.online.strategies.frame_selector import EveryFrameSelector, IntervalFrameSelector
from src.training.online.strategies.pseudo_label import PseudoLabelStrategy
from src.training.online.strategies.replay_buffer import ReplayBuffer
from src.training.online.strategies.update_policy import configure_trainable_params


def _load_checkpoint(model, path: str):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    return model


def _build_models(cfg):
    student = build_model(cfg.student_model_name, num_classes=cfg.student_num_classes)
    _load_checkpoint(student, cfg.student_model_path)

    teacher = None
    if cfg.teacher_model_path:
        teacher = build_model(cfg.teacher_model_name, num_classes=cfg.teacher_num_classes)
        _load_checkpoint(teacher, cfg.teacher_model_path)
    return student, teacher


def main() -> None:
    cfg = parse_args()

    student, teacher = _build_models(cfg)
    trainable = configure_trainable_params(student, cfg.update_policy, cfg.update_last_n)
    optimizer = torch.optim.Adam(trainable, lr=cfg.lr)

    selector = IntervalFrameSelector(cfg.frame_select_interval) if cfg.frame_selector == "interval" else EveryFrameSelector()
    trainer = OnlineTrainer(
        cfg=cfg,
        student=student,
        teacher=teacher,
        reader=VideoStreamReader(cfg.video_root, cfg.img_size),
        writer=StreamWriter(cfg.output_dir),
        frame_selector=selector,
        replay_buffer=ReplayBuffer(cfg.replay_capacity),
        pseudo_label_strategy=PseudoLabelStrategy(cfg.pseudo_label_policy, cfg.teacher_weight),
        optimizer=optimizer,
        monitor_hook=MonitorHook(cfg.monitor_interval),
        viz_hook=VizHook(cfg.output_dir, cfg.viz_interval) if cfg.save_viz else None,
    )
    trainer.run()


if __name__ == "__main__":
    main()
