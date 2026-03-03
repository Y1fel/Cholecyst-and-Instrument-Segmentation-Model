from __future__ import annotations

import torch


class PseudoLabelStrategy:
    def __init__(self, mode: str = "teacher_student", teacher_weight: float = 0.7):
        self.mode = mode
        self.teacher_weight = teacher_weight

    @torch.no_grad()
    def build(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor | None):
        if self.mode == "teacher" or teacher_logits is None:
            return student_logits.argmax(1) if teacher_logits is None else teacher_logits.argmax(1)

        s_prob = torch.softmax(student_logits, dim=1)
        t_prob = torch.softmax(teacher_logits, dim=1)
        merged = self.teacher_weight * t_prob + (1.0 - self.teacher_weight) * s_prob
        conf, label = merged.max(dim=1)
        quality_mask = conf > 0.6
        return label, quality_mask
