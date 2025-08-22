# 
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional

class Visualizer:
    def __init__(self):
        self.colors = {
            'pred_overlay': (0, 255, 0),  # Green
            'groundtruth_overlay': (255, 0, 0)  # Red
        }
                       
    def create_overlay_image(
            self, 
            image: np.ndarray, 
            pred_mask: np.ndarray, 
            grounftruth_mask: Optional[np.ndarray] = None
        ) -> np.ndarray:
        
        # ensure image is in a [H, W, 3] format
        if image.shape[0] == 3: # [3, H, W] -> [H, W, 3]
            image = np.transpose(image, (1, 2, 0))
        
        # validate range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        overlay = image.copy()

        # shape of predicted
        pred_contours, _ = cv2.findContours(
            pred_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            overlay,
            pred_contours,
            -1,
            self.colors['pred_overlay'],
            2
        )

        # shape of ground truth
        if grounftruth_mask is not None:
            grounftruth_contours, _ = cv2.findContours(
                grounftruth_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                overlay,
                grounftruth_contours,
                -1,
                self.colors['groundtruth_overlay'],
                2
            )

        return overlay

    def save_basic_predictions(
            self,
            model,
            val_loader,
            save_dir: str,
            max_samples: int = 10,
            device: str = "cuda",
            threshold: float = None
        ):

        # create & check dir
        os.makedirs(save_dir, exist_ok=True)

        # create & check subdir
        predmask_dir = os.path.join(save_dir, "predicted_masks")
        overlay_dir  = os.path.join(save_dir, "overlay_images")

        os.makedirs(predmask_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)

        model.eval()
        saved_count = 0

        # Detect optimal threshold if not provided
        if threshold is None:
            print("Detecting optimal threshold...")
            threshold = self._detect_optimal_threshold(model, val_loader, device)
            print(f"Using threshold: {threshold:.4f}")

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                if saved_count >= max_samples: # stop when reaching limited counts
                    break

                images = images.to(device, non_blocking=True)
                masks  = masks.to(device, non_blocking=True)

                # forward
                logits = model(images)
                probs  = torch.sigmoid(logits)
                
                if batch_idx == 0:  # print debug info only at the first batch
                    print(f"🔍 Debug Info:")
                    print(f"  Logits - min: {logits.min():.4f}, max: {logits.max():.4f}, mean: {logits.mean():.4f}")
                    print(f"  Probs  - min: {probs.min():.4f}, max: {probs.max():.4f}, mean: {probs.mean():.4f}")
                    print(f"  GT mask unique values: {torch.unique(masks)}")

                preds  = (probs > threshold).float()

                # 🆕 检查预测效果
                pred_ratio = preds.mean().item()
                if batch_idx == 0:
                    print(f"  预测前景比例: {pred_ratio:.3f}")
                    
                # check predicted result
                pred_positive_ratio = (preds > 0).float().mean()
                print(f"Batch {batch_idx}: {pred_positive_ratio:.2%} pixels predicted as positive")

                # save each sample
                batch_size = images.shape[0]
                for i in range(min(batch_size, max_samples - saved_count)):
                    # convert to numpy format
                    img_np         = images[i].cpu().numpy()
                    groundtruth_np = masks[i].cpu().numpy()
                    pred_np        = preds[i].cpu().numpy()

                    # processing dimensions
                    if groundtruth_np.shape[0] == 1:
                        groundtruth_np = groundtruth_np[0] # [1, H, W] -> [H, W]
                    if pred_np.shape[0] == 1:
                        pred_np = pred_np[0]
                
                # save predicted mask
                pred_mask_path = os.path.join(
                    predmask_dir, f"pred_mask_{saved_count:03d}.png")
                plt.imsave(pred_mask_path, pred_np, cmap='gray')

                overlay = self.create_overlay_image(
                    img_np, pred_np, groundtruth_np)
                overlay_path = os.path.join(
                    overlay_dir, f"overlay_{saved_count:03d}.png")
                plt.imsave(overlay_path, overlay)

                saved_count += 1
                if saved_count >= max_samples:
                    break
        
        
        print(f"Saved {saved_count} samples to {save_dir}")
        print(f"Predicted masks saved to: {predmask_dir}")
        print(f"Overlay images saved to: {overlay_dir}")

    def create_comparison_panel(
        self, 
        image: np.ndarray, 
        gt_mask: np.ndarray, 
        pred_mask: np.ndarray,
        sample_name: str = "sample"
    ) -> np.ndarray:
        """
        创建三面板对比图：原图 | 真实标注 | 预测结果
        """
        # 确保图像格式正确
        if image.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
            image = np.transpose(image, (1, 2, 0))
        
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # 确保mask是2D
        if gt_mask.ndim == 3 and gt_mask.shape[0] == 1:
            gt_mask = gt_mask[0]
        if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
            pred_mask = pred_mask[0]
        
        # 转换mask为0-255范围用于显示
        gt_mask_vis = (gt_mask * 255).astype(np.uint8)
        pred_mask_vis = (pred_mask * 255).astype(np.uint8)
        
        # 创建RGB版本的mask（用于更好的可视化）
        gt_mask_rgb = cv2.cvtColor(gt_mask_vis, cv2.COLOR_GRAY2RGB)
        pred_mask_rgb = cv2.cvtColor(pred_mask_vis, cv2.COLOR_GRAY2RGB)
        
        # 获取图像尺寸
        h, w = image.shape[:2]
        
        # 创建组合图像
        combined = np.zeros((h, w * 3, 3), dtype=np.uint8)
        
        # 放置三个面板
        combined[:, 0:w] = image                    # 原图
        combined[:, w:2*w] = gt_mask_rgb           # 真实标注
        combined[:, 2*w:3*w] = pred_mask_rgb       # 预测结果
        
        # 添加分割线
        combined[:, w-1:w+1] = [255, 255, 255]     # 白色分割线
        combined[:, 2*w-1:2*w+1] = [255, 255, 255]
        
        # 添加文字标签（可选）
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Ground Truth", (w + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Prediction", (2*w + 10, 30), font, 1, (255, 255, 255), 2)
        
        return combined

    def save_comparison_predictions(
        self,
        model,
        val_loader,
        save_dir: str,
        max_samples: int = 10,
        device: str = "cuda",
        threshold: float = 0.5  # 🆕 可调节阈值
    ):
        """
        保存对比预测结果（包含三面板对比图）
        """
        # 创建目录
        os.makedirs(save_dir, exist_ok=True)
        comparison_dir = os.path.join(save_dir, "comparison_panels")
        os.makedirs(comparison_dir, exist_ok=True)
        
        model.eval()
        saved_count = 0
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                if saved_count >= max_samples:
                    break
                    
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                # 前向传播
                logits = model(images)
                probs = torch.sigmoid(logits)
                
                # 🆕 使用可调节阈值
                preds = (probs > threshold).float()
                
                # 🆕 添加调试信息
                print(f"Batch {batch_idx}: Logits range [{logits.min():.4f}, {logits.max():.4f}]")
                print(f"Batch {batch_idx}: Probs range [{probs.min():.4f}, {probs.max():.4f}]")
                print(f"Batch {batch_idx}: Pred unique values: {torch.unique(preds)}")
                
                batch_size = images.shape[0]
                for i in range(min(batch_size, max_samples - saved_count)):
                    # 转换为numpy
                    img_np = images[i].cpu().numpy()
                    gt_np = masks[i].cpu().numpy()
                    pred_np = preds[i].cpu().numpy()
                    
                    # 处理维度
                    if gt_np.shape[0] == 1:
                        gt_np = gt_np[0]
                    if pred_np.shape[0] == 1:
                        pred_np = pred_np[0]
                    
                    # 创建对比图
                    comparison_panel = self.create_comparison_panel(
                        img_np, gt_np, pred_np, f"sample_{saved_count:03d}"
                    )
                    
                    # 保存对比图
                    comparison_path = os.path.join(
                        comparison_dir, f"comparison_{saved_count:03d}.png"
                    )
                    plt.imsave(comparison_path, comparison_panel)
                    
                    saved_count += 1
                    if saved_count >= max_samples:
                        break
        
        print(f"Saved {saved_count} comparison panels to {comparison_dir}")

    def _detect_optimal_threshold(self, model, val_loader, device, num_samples=100):
        """
        检测最优阈值 - 基于IoU最大化
        """
        model.eval()
        thresholds = np.arange(0.1, 0.9, 0.1)
        best_threshold = 0.5
        best_iou = 0
        
        sample_count = 0
        all_probs = []
        all_masks = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                if sample_count >= num_samples:
                    break
                    
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                logits = model(images)
                probs = torch.sigmoid(logits)
                
                all_probs.append(probs.cpu())
                all_masks.append(masks.cpu())
                
                sample_count += len(images)
        
        # 合并所有样本
        all_probs = torch.cat(all_probs, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        # 测试不同阈值
        for threshold in thresholds:
            preds = (all_probs > threshold).float()
            
            # 计算IoU
            intersection = (preds * all_masks).sum()
            union = (preds + all_masks).clamp(0, 1).sum()
            iou = intersection / (union + 1e-7)
            
            if iou > best_iou:
                best_iou = iou
                best_threshold = threshold
        
        return best_threshold
    

    # 预留扩展接口
    def auto_format_inputs(self, images, masks, predictions):
        """自动格式化输入数据为标准格式"""
        # TODO: 自动检测和转换维度格式
        # TODO: 统一数据类型和数值范围

    def create_multiclass_overlay(self, image, pred_mask, gt_mask=None, num_classes=2):
        """支持多类别的覆盖图生成"""
        # TODO: 根据类别数生成不同颜色映射
    
    def create_multi_panel(self, image, gt_mask, pred_mask, **kwargs):
        """预留接口：创建多面板对比图"""
        # TODO: 未来实现四面板或更复杂的可视化
        pass
    
    def plot_metrics_curves(self, metrics_history, save_path):
        """预留接口：绘制指标曲线"""
        # TODO: 未来实现训练曲线可视化
        pass
    
    def create_error_analysis(self, predictions, targets, save_dir):
        """预留接口：错误分析可视化"""
        # TODO: 未来实现错误案例分析
        pass