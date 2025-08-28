import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

#知识蒸馏
class DistillationLoss(nn.Module):
    def __init__(self,
                 temperature: float = 4.0,
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 feature_weight: float = 0.1):
        """
        Args:
            temperature: 蒸馏温度
            alpha: 蒸馏损失权重
            beta: 任务损失权重
            feature_weight: 特征蒸馏损失权重
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.feature_weight = feature_weight

        self.task_loss = nn.BCELoss()            # 任务损失 (学生 vs 真实标签)
        self.distill_loss = nn.KLDivLoss(reduction='batchmean')  # 蒸馏损失 (学生 vs 教师)
        self.feature_loss = nn.MSELoss()         # 特征蒸馏损失

    def forward(self,
                student_outputs: torch.Tensor,
                teacher_outputs: torch.Tensor,
                targets: torch.Tensor,
                student_features: Optional[List[torch.Tensor]] = None,
                teacher_features: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """计算蒸馏总损失"""
        task_loss = self.task_loss(student_outputs, targets)

        # 蒸馏损失 (带温度缩放)
        student_logits = torch.log(student_outputs + 1e-8) / self.temperature
        teacher_probs = (teacher_outputs + 1e-8)  # 教师已经是 sigmoid 输出
        distill_loss = self.distill_loss(student_logits, teacher_probs)

        # 特征蒸馏损失
        feature_loss = torch.tensor(0.0, device=student_outputs.device)
        if student_features and teacher_features:
            feature_loss = self._compute_feature_loss(student_features, teacher_features)

        total_loss = (self.alpha * distill_loss +
                      self.beta * task_loss +
                      self.feature_weight * feature_loss)

        return {
            'total_loss': total_loss,
            'task_loss': task_loss,
            'distill_loss': distill_loss,
            'feature_loss': feature_loss
        }

    def _compute_feature_loss(self,
                              student_features: List[torch.Tensor],
                              teacher_features: List[torch.Tensor]) -> torch.Tensor:
        feature_loss = 0.0
        min_len = min(len(student_features), len(teacher_features))
        for i in range(min_len):
            if student_features[i].shape != teacher_features[i].shape:
                student_feat = F.interpolate(
                    student_features[i],
                    size=teacher_features[i].shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                student_feat = student_features[i]
            feature_loss += self.feature_loss(student_feat, teacher_features[i])
        return feature_loss / min_len

#特征提取
class FeatureExtractor:
    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.features = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(
                    lambda m, i, o, name=name: self._hook_fn(name, o)
                )
                self.hooks.append(hook)

    def _hook_fn(self, name: str, output: torch.Tensor):
        self.features[name] = output

    def get_features(self) -> List[torch.Tensor]:
        return [self.features.get(name) for name in self.layer_names if name in self.features]

    def clear_features(self):
        self.features.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
