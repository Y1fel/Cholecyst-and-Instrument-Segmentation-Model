"""
伪标签生成模块 - pseudo_label_generator.py
实现熵反转置信度计算和自适应权重衰减的伪标签生成
"""

import torch
import numpy as np
from collections import deque
import warnings


class AdaptiveWeightScheduler:
    """自适应教师权重调度器"""

    def __init__(self, initial_weight=0.8, min_weight=0.1, window_size=20):
        self.initial_weight = initial_weight
        self.min_weight = min_weight
        self.current_weight = initial_weight
        self.window_size = window_size

        # 性能追踪窗口
        self.student_conf_history = deque(maxlen=window_size)
        self.teacher_conf_history = deque(maxlen=window_size)
        self.loss_history = deque(maxlen=window_size)
        self.disagreement_history = deque(maxlen=window_size)

        # 状态计数器
        self.consecutive_student_wins = 0
        self.consecutive_teacher_wins = 0
        self.plateau_count = 0
        self.total_steps = 0

    def _compute_disagreement_rate(self, teacher_pred, student_pred):
        """计算教师-学生预测差异率"""
        if teacher_pred.shape != student_pred.shape:
            return 0.5  # 默认中等差异

        disagreement = (teacher_pred != student_pred).float().mean().item()
        return disagreement

    def _evaluate_student_superiority(self, student_conf, teacher_conf,
                                      student_loss, disagreement_rate):
        """评估学生是否超越教师"""

        # 指标1: 置信度优势 (权重40%)
        conf_advantage = student_conf - teacher_conf
        conf_score = min(1.0, max(0.0, (conf_advantage + 0.1) / 0.3))

        # 指标2: 学习趋势 (权重30%)
        if len(self.loss_history) >= 5:
            recent_losses = list(self.loss_history)[-5:]
            loss_trend = (recent_losses[0] - recent_losses[-1]) / (recent_losses[0] + 1e-8)
            loss_score = min(1.0, max(0.0, loss_trend * 2))
        else:
            loss_score = 0.0

        # 指标3: 预测稳定性 (权重20%)
        stability_score = max(0.0, 1.0 - disagreement_rate / 0.4)

        # 指标4: 综合置信度水平 (权重10%)
        avg_conf = (student_conf + teacher_conf) / 2
        conf_level_bonus = 0.1 if avg_conf > 0.7 else 0.0

        # 综合评分
        total_score = (conf_score * 0.4 + loss_score * 0.3 +
                       stability_score * 0.2 + conf_level_bonus)

        is_superior = total_score > 0.70
        return is_superior, total_score

    def _compute_decay_rate(self, student_superior, performance_score, current_loss):
        """计算自适应衰减率"""

        # 基础衰减率
        base_decay = 0.995

        if student_superior:
            self.consecutive_student_wins += 1
            self.consecutive_teacher_wins = 0

            # 连续优势加速衰减
            if self.consecutive_student_wins >= 7:
                acceleration = min(0.05, (self.consecutive_student_wins - 5) * 0.01)
                decay_rate = max(0.92, base_decay - 0.03 - acceleration)
            elif self.consecutive_student_wins >= 3:
                decay_rate = 0.98
            else:
                decay_rate = 0.99

        else:
            self.consecutive_teacher_wins += 1
            self.consecutive_student_wins = 0

            # 教师仍优秀时减缓衰减
            if self.consecutive_teacher_wins >= 5:
                decay_rate = 0.999
            elif self.consecutive_teacher_wins >= 3:
                decay_rate = 0.998
            else:
                decay_rate = base_decay

        # 检测学习平台期
        if len(self.loss_history) >= 10:
            recent_losses = list(self.loss_history)[-10:]
            loss_variance = np.var(recent_losses)

            if loss_variance < 0.0001:  # 损失变化很小
                self.plateau_count += 1
                if self.plateau_count > 8:
                    # 强制探索：加速衰减
                    decay_rate = min(decay_rate, 0.96)
                    print(f"[AdaptiveWeight] 检测到学习平台期，强制衰减: decay_rate={decay_rate:.4f}")
            else:
                self.plateau_count = 0

        return decay_rate

    def update(self, student_conf, teacher_conf, student_loss=None,
               teacher_pred=None, student_pred=None):
        """更新教师权重"""

        self.total_steps += 1

        # 计算预测差异
        disagreement_rate = 0.2  # 默认值
        if teacher_pred is not None and student_pred is not None:
            disagreement_rate = self._compute_disagreement_rate(teacher_pred, student_pred)

        # 更新历史记录
        self.student_conf_history.append(student_conf)
        self.teacher_conf_history.append(teacher_conf)
        if student_loss is not None:
            self.loss_history.append(student_loss)
        self.disagreement_history.append(disagreement_rate)

        # 评估学生表现
        student_superior, perf_score = self._evaluate_student_superiority(
            student_conf, teacher_conf, student_loss or 1.0, disagreement_rate
        )

        # 计算衰减率
        decay_rate = self._compute_decay_rate(student_superior, perf_score, student_loss or 1.0)

        # 应用衰减
        self.current_weight *= decay_rate
        self.current_weight = max(self.min_weight, self.current_weight)

        # 调试信息
        if self.total_steps % 10 == 0:
            status = "STUDENT_LEAD" if student_superior else "TEACHER_LEAD"
            print(f"[AdaptiveWeight] Step {self.total_steps}: {status} "
                  f"weight={self.current_weight:.4f} decay={decay_rate:.4f} "
                  f"perf_score={perf_score:.3f} student_wins={self.consecutive_student_wins}")

        return self.current_weight, student_superior, perf_score


class PseudoLabelGenerator:
    """伪标签生成器 - 集成置信度计算和自适应权重管理"""

    def __init__(self, initial_teacher_weight=0.8, min_teacher_weight=0.2):
        self.weight_scheduler = AdaptiveWeightScheduler(
            initial_weight=initial_teacher_weight,
            min_weight=min_teacher_weight
        )

    def compute_entropy_confidence(self, logits):
        """使用熵反转法计算置信度"""
        if logits is None or logits.numel() == 0:
            return 0.0

        try:
            # 转换为概率分布
            probs = torch.softmax(logits, dim=1)

            # 计算熵
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

            # 最大熵 (完全不确定时)
            max_entropy = torch.log(torch.tensor(float(probs.shape[1])))

            # 熵反转置信度: 熵越小，置信度越高
            confidence = (1.0 - entropy / max_entropy).mean().item()

            return max(0.0, min(1.0, confidence))

        except Exception as e:
            warnings.warn(f"置信度计算异常: {e}, 返回默认值")
            return 0.5

    def generate_hybrid_labels(self, teacher_logits, student_logits,
                               current_loss=None, binary_task=False):
        """
        生成混合伪标签

        Args:
            teacher_logits: 教师模型输出 logits
            student_logits: 学生模型输出 logits
            current_loss: 当前损失值 (用于权重调度)
            binary_task: 是否为二分类任务

        Returns:
            hybrid_labels: 混合伪标签
            teacher_confidence: 教师置信度
            student_confidence: 学生置信度
            current_teacher_weight: 当前教师权重
            debug_info: 调试信息字典
        """

        # 计算置信度
        teacher_conf = self.compute_entropy_confidence(teacher_logits)
        student_conf = self.compute_entropy_confidence(student_logits)

        # 生成预测用于权重更新
        with torch.no_grad():
            if binary_task:
                teacher_pred = (torch.sigmoid(teacher_logits) > 0.5).long()
                student_pred = (torch.sigmoid(student_logits) > 0.5).long()
            else:
                teacher_pred = torch.argmax(teacher_logits, dim=1)
                student_pred = torch.argmax(student_logits, dim=1)

        # 更新教师权重
        current_teacher_weight, student_superior, perf_score = self.weight_scheduler.update(
            student_conf, teacher_conf, current_loss, teacher_pred, student_pred
        )

        # 生成概率分布
        if binary_task:
            teacher_probs = torch.sigmoid(teacher_logits)
            student_probs = torch.sigmoid(student_logits)
        else:
            teacher_probs = torch.softmax(teacher_logits, dim=1)
            student_probs = torch.softmax(student_logits, dim=1)

        # 动态权重融合
        total_conf = teacher_conf + student_conf + 1e-8
        teacher_weight = (current_teacher_weight * teacher_conf) / total_conf
        student_weight = ((1 - current_teacher_weight) * student_conf) / total_conf

        # 归一化权重
        total_weight = teacher_weight + student_weight + 1e-8
        teacher_weight_norm = teacher_weight / total_weight
        student_weight_norm = student_weight / total_weight

        # 概率融合
        hybrid_probs = (teacher_weight_norm * teacher_probs +
                        student_weight_norm * student_probs)

        # 生成最终标签
        if binary_task:
            hybrid_labels = (hybrid_probs > 0.5).float()
        else:
            hybrid_labels = torch.argmax(hybrid_probs, dim=1)

        # 调试信息
        debug_info = {
            'teacher_weight_scheduled': current_teacher_weight,
            'teacher_weight_actual': teacher_weight_norm,
            'student_weight_actual': student_weight_norm,
            'student_superior': student_superior,
            'performance_score': perf_score,
            'teacher_pred': teacher_pred,
            'student_pred': student_pred,
            'disagreement_rate': self.weight_scheduler.disagreement_history[
                -1] if self.weight_scheduler.disagreement_history else 0.0
        }

        return hybrid_labels, teacher_conf, student_conf, current_teacher_weight, debug_info

    def should_use_pseudo_labels(self, teacher_conf, student_conf, min_confidence=0.8):
        """判断是否应该使用生成的伪标签进行训练"""
        max_conf = max(teacher_conf, student_conf)
        return max_conf >= min_confidence

    def get_current_teacher_weight(self):
        """获取当前教师权重"""
        return self.weight_scheduler.current_weight

    def get_scheduler_stats(self):
        """获取调度器统计信息"""
        return {
            'current_weight': self.weight_scheduler.current_weight,
            'total_steps': self.weight_scheduler.total_steps,
            'consecutive_student_wins': self.weight_scheduler.consecutive_student_wins,
            'consecutive_teacher_wins': self.weight_scheduler.consecutive_teacher_wins,
            'plateau_count': self.weight_scheduler.plateau_count,
            'avg_student_conf': np.mean(list(
                self.weight_scheduler.student_conf_history)) if self.weight_scheduler.student_conf_history else 0.0,
            'avg_teacher_conf': np.mean(
                list(self.weight_scheduler.teacher_conf_history)) if self.weight_scheduler.teacher_conf_history else 0.0
        }
