"""
精简评估模块 - 基础版本
计算核心分割指标，预留扩展接口
"""
import torch, cv2
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd

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

    @torch.inference_mode()
    def evaluate_multiclass(self, model, val_loader, criterion, num_classes: int, ignore_index: int) -> Dict:
        """多类评估接口"""
        model.eval()
        total_loss  = 0.0
        total_count = 0

        # accmulate confusion matrix
        confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long, device=self.device)

        for images, targets in val_loader:
            images  = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True).long()
            logits  = model(images)
            loss    = criterion(logits, targets)
            total_loss  += float(loss.item()) * images.size(0)
            total_count += images.size(0)

            # predict
            preds = logits.argmax(dim=1)

            # ignore index
            valid = (targets != ignore_index)
            if valid.sum() == 0:
                continue

            preds_v   = preds[valid].view(-1)
            targets_v = targets[valid].view(-1)

            # calculate
            idx               = targets_v * num_classes + preds_v
            bincount          = torch.bincount(idx, minlength=num_classes * num_classes)
            confusion_matrix += bincount.view(num_classes, num_classes)

        # from confusion matrix calculate
        tp = confusion_matrix.diag().float()
        fp = confusion_matrix.sum(dim=0).float() - tp
        fn = confusion_matrix.sum(dim=1).float() - tp
        denom_iou  = tp + fp + fn
        denom_dice = (2 * tp + fp + fn)

        iou_per_class   = torch.where(denom_iou > 0, tp / denom_iou, torch.zeros_like(tp))
        dice_per_class  = torch.where(denom_dice > 0, 2 * tp / denom_dice, torch.zeros_like(tp))
        acc_per_class   = torch.where((tp + fn) > 0, tp / (tp + fn), torch.zeros_like(tp))

        metrics = {
            "val_loss":       total_loss / total_count,
            "miou":           iou_per_class.mean().item(),
            "mdice":          dice_per_class.mean().item(),
            "macc":           acc_per_class.mean().item(),
            "iou_per_class":  iou_per_class.tolist(),
            "dice_per_class": dice_per_class.tolist(),
            "acc_per_class":  acc_per_class.tolist()
        }

        return metrics

    def compute_boundary_f1(self, predictions, targets, tolerance=2):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        # ensure dimensions
        if predictions.ndim == 3:
            batch_boundary_f1 = []
            for i in range(predictions.shape[0]):
                bf1 = self._compute_single_boundary_f1(predictions[i], targets[i], tolerance)
                batch_boundary_f1.append(bf1)
            return np.mean(batch_boundary_f1)
        else:
            return self._compute_single_boundary_f1(predictions, targets, tolerance)
    
    # 辅助函数：提取边界
    def _compute_single_boundary_f1(self, prediction, target, tolerance):
        # extract boundaries
        pred_boundary = self._extract_boundary(prediction)
        target_boundary = self._extract_boundary(target)

        if pred_boundary.sum() == 0 and target_boundary.sum() == 0:
            return 1.0
        if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
            return 0.0
        
        # compute change in distance
        target_dist = ndimage.distance_transform_edt(~target_boundary.astype(bool))
        pred_dist   = ndimage.distance_transform_edt(~pred_boundary.astype(bool))

        # compute matched boundary pixels
        pred_match = (target_dist <= tolerance) & (pred_boundary > 0)
        target_match = (pred_dist <= tolerance) & (target_boundary > 0)

        # compute F1 score
        tp = pred_match.sum()
        fp = (pred_boundary > 0).sum() - tp
        fn = (target_boundary > 0).sum() - target_match.sum()

        if tp + fp + fn == 0:
            return 1.0
            
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        return f1
    
    def _extract_boundary(self, mask):
        kernal = np.ones((3, 3), dtype=np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernal, iterations=1)
        boundary = mask.astype(np.uint8) - eroded
        return boundary
    
    def compute_ece_nll(self, logits, targets, n_bins=15):
        if isinstance(logits, torch.Tensor):
            logits = logits.detach()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach()

        # dealing with classification
        if logits.dim() == 3 or (logits.dim() == 4 and logits.shape[1] == 1):
            if logits.dim() == 4:
                logits = logits.squeeze(1)
            probs = torch.sigmoid(logits)
            targets = targets.float()

        else: # multi-class classification
            probs = torch.softmax(logits, dim=1)
            max_probs, pred_classes = torch.max(probs, dim=1)
            probs = max_probs
            targets = (pred_classes == targets).float()

        # flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # compute NLL
        nll = F.binary_cross_entropy(probs_flat, targets_flat, reduction='mean').item()
        # compute ECE
        ece = self._compute_ece(probs_flat, targets_flat, n_bins)

        return {"ece": ece, "nll": nll}
    
    def _compute_ece(self, probs, targets, n_bins):
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        total_samples = len(probs)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower.item()) & (probs <= bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = targets[in_bin].float().mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece.item()
    
    def generate_reliability_diagram(self, logits, targets, save_path, num_bins=15):
        if isinstance(logits, torch.Tensor):
            logits = logits.detach()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach()

        if logits.dim() == 3 or (logits.dim() == 4 and logits.shape[1] == 1):
            if logits.dim() == 4:
                logits = logits.squeeze(1)
            probs = torch.sigmoid(logits)
            targets = targets.float()
        else: # multi-class classification
            probs = torch.softmax(logits, dim=1)
            max_probs, pred_classes = torch.max(probs, dim=1)
            probs = max_probs
            targets = (pred_classes == targets).float()

        probs_flat = probs.view(-1).cpu().numpy()
        targets_flat = targets.view(-1).cpu().numpy()

        # Calculate statistics for each bin
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (probs_flat > bin_lower) & (probs_flat <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(targets_flat[in_bin].mean())
                bin_confidences.append(probs_flat[in_bin].mean())
                bin_counts.append(in_bin.sum())
            else:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(0)
                bin_confidences.append((bin_lower + bin_upper) / 2)
                bin_counts.append(0)
        
        # 绘制可靠性图
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)
        
        # 绘制条形图表示每个bin的样本数量
        plt.bar(bin_centers, bin_accuracies, width=1.0/num_bins, 
                alpha=0.3, edgecolor='black', label='Accuracy')
        
        # 绘制置信度vs准确率的连线
        plt.plot(bin_confidences, bin_accuracies, 'ro-', 
                label='Model Calibration', markersize=6)
        
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy') 
        plt.title('Reliability Diagram')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        # 计算并显示ECE
        ece = self._compute_ece(torch.tensor(probs_flat), torch.tensor(targets_flat), num_bins)
        plt.text(0.02, 0.98, f'ECE: {ece:.4f}', transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'bin_centers': bin_centers,
            'bin_accuracies': bin_accuracies, 
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts,
            'ece': ece
        }
    
    def evaluate_with_full_metrics(self, model, val_loader, criterion, num_classes=None, 
                                  ignore_index=255, task_type="auto", save_reliability_path=None):
        """
        完整指标评估 - 包含所有新指标
        Args:
            model: 待评估模型
            val_loader: 验证数据加载器
            criterion: 损失函数
            num_classes: 类别数（多分类时需要）
            ignore_index: 忽略的标签值
            task_type: "binary", "multiclass", "auto"
            save_reliability_path: 可靠性图保存路径
        Returns:
            dict: 包含所有指标的字典
        """
        model.eval()
        
        # 收集所有logits和targets用于校准指标计算
        all_logits = []
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            # 首先检测任务类型
            if task_type == "auto":
                sample_batch = next(iter(val_loader))
                sample_images, sample_masks = sample_batch
                sample_images = sample_images.to(self.device)
                sample_logits = model(sample_images)
                
                if sample_logits.shape[1] == 1:
                    task_type = "binary"
                    if num_classes is None:
                        num_classes = 2
                else:
                    task_type = "multiclass"
                    if num_classes is None:
                        num_classes = sample_logits.shape[1]
            
            # 执行相应的评估
            if task_type == "binary":
                base_metrics = self.evaluate(model, val_loader, criterion)
            else:
                base_metrics = self.evaluate_multiclass(model, val_loader, criterion, 
                                                       num_classes, ignore_index)
            
            # 重新遍历数据收集用于高级指标计算的数据
            for images, targets in val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                logits = model(images)
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())
                
                # 生成预测用于边界F1计算
                if task_type == "binary":
                    if logits.dim() == 4 and logits.shape[1] == 1:
                        probs = torch.sigmoid(logits).squeeze(1)
                        preds = (probs > self.threshold).int()
                    else:
                        probs = torch.sigmoid(logits)
                        preds = (probs > self.threshold).int()
                else:
                    preds = torch.argmax(logits, dim=1)
                
                all_predictions.append(preds.cpu())
        
        # 合并所有批次数据
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        
        # 计算校准指标
        calibration_metrics = self.compute_ece_nll(all_logits, all_targets)
        
        # 计算边界F1
        boundary_f1 = self.compute_boundary_f1(all_predictions, all_targets)
        
        # 生成可靠性图（如果指定了路径）
        reliability_stats = None
        if save_reliability_path:
            reliability_stats = self.generate_reliability_diagram(
                all_logits, all_targets, save_reliability_path
            )
        
        # 合并所有指标
        full_metrics = {
            **base_metrics,
            "boundary_f1": boundary_f1,
            "ece": calibration_metrics["ece"],
            "nll": calibration_metrics["nll"],
            "task_type": task_type,
            "num_classes": num_classes
        }
        
        if reliability_stats:
            full_metrics["reliability_stats"] = reliability_stats
            
        return full_metrics
    
    def export_kd_comparison_csv(self, metrics_dict, regime_name, teacher_model_name=None, 
                               student_model_name=None, epochs=None, training_time=None,
                               model_params=None, fps=None, notes="", save_path=None):
        """
        Export CSV table for KD comparison experiments
        Args:
            metrics_dict: Metrics dictionary returned by evaluate_with_full_metrics
            regime_name: Experiment group name ("S-Equal", "KD-Student", "S-Long")
            teacher_model_name: Teacher model name (used in KD mode)
            student_model_name: Student model name
            epochs: Number of training epochs
            training_time: Training time (hours)
            model_params: Model parameter count (M)
            fps: Inference speed
            notes: Additional notes
            save_path: CSV save path
        Returns:
            dict: Formatted row data
        """

        row_data = {
            "Regime": regime_name,
            "Teacher": teacher_model_name if teacher_model_name else "—",
            "Student": student_model_name if student_model_name else "—", 
            "Budget (ep/steps)": epochs if epochs else "—",
            "Time (h)": f"{training_time:.2f}" if training_time else "—",
            "Params (M)": f"{model_params:.1f}" if model_params else "—",
            "mIoU ↑": f"{metrics_dict.get('miou', metrics_dict.get('iou', 0)):.3f}",
            "Dice ↑": f"{metrics_dict.get('mdice', metrics_dict.get('dice', 0)):.3f}",
            "Boundary-F1 ↑": f"{metrics_dict.get('boundary_f1', 0):.3f}",
            "ECE ↓": f"{metrics_dict.get('ece', 0):.4f}",
            "NLL ↓": f"{metrics_dict.get('nll', 0):.4f}",
            "FPS (Student)": f"{fps:.1f}" if fps else "—",
            "Notes": notes
        }
        
        # 如果指定了保存路径，则保存到CSV
        if save_path:
            import pandas as pd
            import os

            # Check if file exists to decide whether to write header
            file_exists = os.path.exists(save_path)
            
            df = pd.DataFrame([row_data])
            df.to_csv(save_path, mode='a', header=not file_exists, index=False)
            
            print(f"KD comparison data appended to: {save_path}")
        
        return row_data


