import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

# 知识蒸馏
class DistillationLoss(nn.Module):
    def __init__(self,
                 num_classes: int = 2,  # 类别数
                 temperature: float = 4.0,
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 feature_weight: float = 0.1):
        """
        Args:
            num_classes: 类别数 (2 = 二分类, >2 = 多分类)
            temperature: 蒸馏温度
            alpha: 蒸馏损失权重
            beta: 任务损失权重
            feature_weight: 特征蒸馏损失权重
        """
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.feature_weight = feature_weight

        # 任务损失
        if num_classes == 2:
            self.task_loss = nn.BCEWithLogitsLoss()  # 二分类更推荐
        else:
            self.task_loss = nn.CrossEntropyLoss()

        self.distill_loss = nn.KLDivLoss(reduction='batchmean')  # 蒸馏损失
        self.feature_loss = nn.MSELoss()                         # 特征蒸馏损失

    def forward(self,
                student_outputs: torch.Tensor,
                teacher_outputs: torch.Tensor,
                targets: torch.Tensor,
                student_features: Optional[List[torch.Tensor]] = None,
                teacher_features: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """计算蒸馏总损失"""

        # ========= 1. 任务损失 =========
        if self.num_classes == 2:
            # student_outputs: [N, 1] logits
            student_logits = student_outputs.view(-1)
            targets = targets.float().view(-1)
            task_loss = self.task_loss(student_logits, targets)
        else:
            # student_outputs: [N, C] logits
            task_loss = self.task_loss(student_outputs, targets)

        # ========= 2. 蒸馏损失 =========
        if self.num_classes == 2:
            student_probs = F.log_softmax(student_outputs / self.temperature, dim=1)
            teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=1)
        else:
            student_probs = F.log_softmax(student_outputs / self.temperature, dim=1)
            teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=1)

        distill_loss = self.distill_loss(student_probs, teacher_probs) * (self.temperature ** 2)

        # ========= 3. 特征蒸馏损失 =========
        feature_loss = torch.tensor(0.0, device=student_outputs.device)
        if student_features and teacher_features:
            feature_loss = self._compute_feature_loss(student_features, teacher_features)

        # ========= 4. 总损失 =========
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

# 伪标签方法
class PseudoLabelLoss(nn.Module):
    def __init__(self,
                 num_classes: int = 2,
                 use_soft: bool = False,
                 confidence_threshold: float = 0.7):
        """
        Args:
            num_classes: 类别数 (2 = 二分类, >2 = 多分类)
            use_soft: 是否使用软伪标签 (False = 硬标签CE, True = soft标签)
            confidence_threshold: 伪标签置信度阈值, 低于阈值的样本不算损失
        """
        super().__init__()
        self.num_classes = num_classes
        self.use_soft = use_soft
        self.confidence_threshold = confidence_threshold

        if num_classes == 2:
            self.ce_loss = nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self,
                student_outputs: torch.Tensor,
                teacher_outputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算伪标签损失"""

        if self.num_classes == 2:
            # teacher_probs: [N, 2]
            teacher_probs = torch.sigmoid(teacher_outputs)
            student_logits = student_outputs.view(-1)

            if self.use_soft:
                # ========= 软伪标签 =========
                pseudo_labels = teacher_probs.detach()
                # BCE with soft targets
                loss = F.binary_cross_entropy_with_logits(
                    student_logits, pseudo_labels.view(-1), reduction="none"
                )
            else:
                # ========= 硬伪标签 =========
                pseudo_labels = (teacher_probs > 0.5).float()
                loss = self.ce_loss(student_logits, pseudo_labels.view(-1))
        else:
            # 多分类
            teacher_probs = F.softmax(teacher_outputs, dim=1)

            if self.use_soft:
                # ========= 软伪标签 =========
                pseudo_labels = teacher_probs.detach()
                student_log_probs = F.log_softmax(student_outputs, dim=1)
                # soft CE (相当于 KL + 常数)
                loss = -(pseudo_labels * student_log_probs).sum(dim=1)
            else:
                # ========= 硬伪标签 =========
                conf, pseudo_labels = torch.max(teacher_probs, dim=1)
                loss = self.ce_loss(student_outputs, pseudo_labels)

                # 筛掉低置信度伪标签
                mask = conf >= self.confidence_threshold
                if mask.sum() > 0:
                    loss = loss[mask]
                else:
                    loss = torch.tensor(0.0, device=student_outputs.device)

        return {
            "pseudo_loss": loss.mean()
        }
