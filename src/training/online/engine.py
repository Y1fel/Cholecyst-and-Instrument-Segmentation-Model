from __future__ import annotations

import torch
import torch.nn.functional as F

from src.training.online.config import OnlineConfig


class OnlineTrainer:
    def __init__(
        self,
        cfg: OnlineConfig,
        student,
        teacher,
        reader,
        writer,
        frame_selector,
        replay_buffer,
        pseudo_label_strategy,
        optimizer,
        monitor_hook,
        viz_hook=None,
    ):
        self.cfg = cfg
        self.student = student
        self.teacher = teacher
        self.reader = reader
        self.writer = writer
        self.frame_selector = frame_selector
        self.replay_buffer = replay_buffer
        self.pseudo_label_strategy = pseudo_label_strategy
        self.optimizer = optimizer
        self.monitor_hook = monitor_hook
        self.viz_hook = viz_hook
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    def run(self) -> None:
        self.student.to(self.device).train()
        if self.teacher is not None:
            self.teacher.to(self.device).eval()

        for frame in self.reader:
            if 0 <= self.cfg.online_steps <= frame.index:
                break
            if not self.frame_selector.should_train(frame):
                continue

            x = frame.tensor_chw.unsqueeze(0).to(self.device)
            s_logits = self.student(x)
            t_logits = self.teacher(x) if self.teacher is not None else None

            pseudo = self.pseudo_label_strategy.build(s_logits.detach(), t_logits)
            if isinstance(pseudo, tuple):
                target, quality_mask = pseudo
                if quality_mask.sum() == 0:
                    continue
                loss_map = F.cross_entropy(s_logits, target, reduction="none")
                loss = (loss_map * quality_mask.float()).sum() / quality_mask.float().sum().clamp_min(1.0)
            else:
                target = pseudo
                loss = F.cross_entropy(s_logits, target)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            pred = torch.argmax(s_logits.detach(), dim=1).squeeze(0).cpu().numpy()
            self.writer.save_mask(frame.index, pred)
            self.monitor_hook.on_step_end(frame.index, float(loss.item()))
            if self.viz_hook is not None:
                self.viz_hook.on_prediction(frame.index, frame.image_bgr, pred)

            self.replay_buffer.add((x.detach().cpu(), target.detach().cpu()))

        self.reader.close()
