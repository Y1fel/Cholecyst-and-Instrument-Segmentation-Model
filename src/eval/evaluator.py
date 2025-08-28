"""
精简评估模块 - 基础版本
计算核心分割指标，预留扩展接口
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class Evaluator:
    def __init__(self, device: str = "cuda", threshold: float = 0.5):
        self.device = device
        self.threshold = threshold
    
    def evaluate(self, model, val_loader, criterion) -> Dict:
        # 
        model.eval()
        val_loss = 0.0

        # confusion matrix
        TP = FP = FN = TN =0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device, non_blocking=True)
                masks  = masks.to(self.device, non_blocking=True)

                # forward
                logits = model(images)
                loss   = criterion(logits, masks)
                val_loss += loss.item()

                # gain predicted results (binary segmentation)
                probs = torch.sigmoid(logits)
                pred  = (probs > self.threshold).int()

                # dimension processing for ground truth mask, ensuring a format of [B, H, W]
                if masks.dim() == 4 and masks.shape[1] == 1:
                    groundtruth_mask = masks.squeeze(1)  # [B, H, W]
                else:
                    groundtruth_mask = masks.int()

                # dimension processing for predicted
                if pred.dim() == 4 and pred.shape[1] == 1:
                    pred = pred.squeeze(1)  # [B, H, W]

                # update new confusion matrix
                TP += ((pred == 1) & (groundtruth_mask == 1)).sum().item()
                FP += ((pred == 1) & (groundtruth_mask == 0)).sum().item()
                FN += ((pred == 0) & (groundtruth_mask == 1)).sum().item()
                TN += ((pred == 0) & (groundtruth_mask == 0)).sum().item()

        # compute metrics
        eps = 1e-7  # prevent division by zero
        metrics = {
            "val_loss": val_loss / max(1, len(val_loader)),
            "iou": TP / (TP + FP + FN + eps),
            "dice": 2 * TP / (2 * TP + FP + FN + eps),
            "accuracy": (TP + TN) / (TP + FP + FN + TN + eps),
            "precision": TP / (TP + FP + eps),
            "recall": TP / (TP + FN + eps)
        }

        return metrics
    
    def logits_to_preds(logits, threshold: float = 0.5):
        """
        统一把训练期 logits 转为离散预测：
        - 二类: sigmoid -> 阈值 -> [B,H,W]
        - 多类: argmax(logits, dim=1) -> [B,H,W]
        """
        import torch
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            return (probs > threshold).long().squeeze(1)
        else:
            return torch.argmax(logits, dim=1)
        
    def auto_detect_task_type(self, logits, masks):
        """预留接口：自动检测任务类型（二分类/多分类）"""
        # TODO: 自动检测任务类型（二分类/多分类）
        pass

    def evaluate_universal(self, model, val_loader, criterion, task_type="auto"):
        """通用评估接口，支持不同模型架构"""
        # TODO: 通用评估接口，支持不同模型架构
    
    # 预留扩展接口
    def evaluate_with_predictions(self, model, val_loader, criterion):
        """预留接口：返回预测结果用于可视化"""
        # TODO: 未来实现返回预测数据
        pass
    
    def compute_advanced_metrics(self, predictions, targets):
        """预留接口：计算高级指标"""
        # TODO: 未来添加更多指标 (hausdorff distance, surface dice等)
        pass