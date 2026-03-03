from __future__ import annotations

import torch.nn as nn


def configure_trainable_params(model: nn.Module, policy: str, last_n: int = 2):
    for p in model.parameters():
        p.requires_grad = False

    if policy == "all":
        for p in model.parameters():
            p.requires_grad = True
    elif policy == "bn":
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = True
    elif policy == "last_n":
        modules = [m for m in model.modules() if any(True for _ in m.parameters(recurse=False))]
        for m in modules[-max(1, last_n):]:
            for p in m.parameters(recurse=False):
                p.requires_grad = True
    else:
        raise ValueError(f"Unknown update policy: {policy}")

    return [p for p in model.parameters() if p.requires_grad]
