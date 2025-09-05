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
                 feature_weight: float = 0.1,
                 ignore_index: int = 255):
        """
        Args:
            num_classes: 类别数 (2 = 二分类, >2 = 多分类)
            temperature: 蒸馏温度
            alpha: 蒸馏损失权重
            beta: 任务损失权重
            feature_weight: 特征蒸馏损失权重
            ignore_index: 忽略索引，用于屏蔽无效像素
        """
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.feature_weight = feature_weight
        self.ignore_index = ignore_index

        # 任务损失
        if num_classes == 2:
            self.task_loss = nn.BCEWithLogitsLoss()  # 二分类更推荐
        else:
            self.task_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

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
            # 二分类: 期望模型输出[N, 2, H, W]，转换为BCE格式
            # 取第1个通道作为正类logits
            student_logits = student_outputs[:, 1].flatten()  # [N*H*W]
            targets_flat = targets.float().flatten()          # [N*H*W]
            task_loss = self.task_loss(student_logits, targets_flat)
        else:
            # 多分类: [N, C, H, W] -> targets [N, H, W]
            targets_long = targets.long()
            task_loss = self.task_loss(student_outputs, targets_long)

        # ========= 2. 蒸馏损失 =========
        # 统一处理：都使用softmax在类别维度
        student_probs = F.log_softmax(student_outputs / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=1)

        # 处理ignore_index：在计算KL散度时排除这些像素
        if self.num_classes > 2:
            # 多分类情况：创建valid mask排除ignore_index像素
            valid_mask = (targets != self.ignore_index)  # [N, H, W]
            
            if valid_mask.any():
                # 只对有效像素计算蒸馏损失
                # 修复：正确的mask扩展和索引方式
                batch_size, num_classes, height, width = student_probs.shape
                valid_mask_expanded = valid_mask.unsqueeze(1).expand(batch_size, num_classes, height, width)
                
                # 安全的索引方式
                student_probs_flat = student_probs.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
                teacher_probs_flat = teacher_probs.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
                
                # 只选择有效像素
                student_probs_valid = student_probs_flat[valid_mask]  # [N_valid, C]
                teacher_probs_valid = teacher_probs_flat[valid_mask]  # [N_valid, C]
                
                # 添加数值稳定性检查
                if student_probs_valid.size(0) > 0:
                    # 检查是否有NaN或Inf
                    if torch.isfinite(student_probs_valid).all() and torch.isfinite(teacher_probs_valid).all():
                        distill_loss = self.distill_loss(student_probs_valid, teacher_probs_valid) * (self.temperature ** 2)
                    else:
                        print("[WARNING] NaN/Inf detected in distillation probs, using fallback loss")
                        distill_loss = torch.tensor(0.0, device=student_outputs.device, requires_grad=True)
                else:
                    distill_loss = torch.tensor(0.0, device=student_outputs.device, requires_grad=True)
            else:
                print("[WARNING] No valid pixels found, setting distill_loss to 0")
                distill_loss = torch.tensor(0.0, device=student_outputs.device, requires_grad=True)
        else:
            # 二分类情况：按原来的方式处理
            distill_loss = self.distill_loss(student_probs, teacher_probs) * (self.temperature ** 2)

        # ========= 3. 特征蒸馏损失 =========
        feature_loss = torch.tensor(0.0, device=student_outputs.device)
        if student_features and teacher_features:
            feature_loss = self._compute_feature_loss(student_features, teacher_features)

        # ========= 4. 总损失 =========
        # 添加数值稳定性检查
        if not torch.isfinite(task_loss):
            print(f"[WARNING] Task loss is NaN/Inf: {task_loss.item()}, setting to 0")
            task_loss = torch.tensor(0.0, device=student_outputs.device, requires_grad=True)
        
        if not torch.isfinite(distill_loss):
            print(f"[WARNING] Distill loss is NaN/Inf: {distill_loss.item()}, setting to 0")
            distill_loss = torch.tensor(0.0, device=student_outputs.device, requires_grad=True)
        
        total_loss = (self.alpha * distill_loss +
                      self.beta * task_loss +
                      self.feature_weight * feature_loss)

        # ========= 5. 调试输出 (第一个batch) =========
        if not hasattr(self, '_debug_logged'):
            print(f"[DISTILL DEBUG] Task Loss: {task_loss.item():.4f}")
            print(f"[DISTILL DEBUG] Distill Loss: {distill_loss.item():.4f}")
            print(f"[DISTILL DEBUG] Feature Loss: {feature_loss.item():.4f}")
            print(f"[DISTILL DEBUG] Total Loss: {total_loss.item():.4f}")
            print(f"[DISTILL DEBUG] Alpha: {self.alpha}, Beta: {self.beta}, Feature_weight: {self.feature_weight}")
            if self.num_classes > 2:
                print(f"[DISTILL DEBUG] Valid pixels ratio: {valid_mask.float().mean().item():.4f}")
            self._debug_logged = True

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
