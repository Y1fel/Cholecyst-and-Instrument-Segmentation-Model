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
            groundtruth_mask: Optional[np.ndarray] = None
        ) -> np.ndarray:
        """
        创建叠加图像，支持二分类和多分类
        使用填充色彩和轮廓线增强可视化效果
        """
        
        # ensure image is in a [H, W, 3] format
        if image.shape[0] == 3: # [3, H, W] -> [H, W, 3]
            image = np.transpose(image, (1, 2, 0))
        
        # validate range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        overlay = image.copy()

        # 检测是否为多分类
        pred_unique = np.unique(pred_mask)
        is_multiclass = len(pred_unique) > 2

        if is_multiclass:
            # 多分类：为每个类别创建彩色填充和轮廓
            from src.common.constants import PALETTE, GT_PALETTE
            
            # 创建半透明叠加层
            alpha_overlay = image.copy().astype(np.float32)
            
            # 处理预测结果
            for class_id in pred_unique:
                if class_id == 0:  # 跳过背景
                    continue
                    
                # 创建当前类别的二值mask
                class_mask = (pred_mask == class_id).astype(np.uint8)
                
                # 获取颜色
                color = PALETTE.get(int(class_id), (255, 255, 255))
                
                # 添加半透明填充
                alpha_overlay[class_mask > 0] = alpha_overlay[class_mask > 0] * 0.6 + np.array(color) * 0.4
                
                # 找到轮廓并绘制
                contours, _ = cv2.findContours(
                    class_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(alpha_overlay, contours, -1, color, 3)

            # 如果有真值标签，用虚线轮廓绘制
            if groundtruth_mask is not None:
                gt_unique = np.unique(groundtruth_mask)
                for class_id in gt_unique:
                    if class_id == 0:  # 跳过背景
                        continue
                        
                    # 创建当前类别的二值mask
                    class_mask = (groundtruth_mask == class_id).astype(np.uint8)
                    
                    # 找到轮廓
                    contours, _ = cv2.findContours(
                        class_mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    # 使用GT专用颜色绘制虚线轮廓
                    color = GT_PALETTE.get(int(class_id), (128, 128, 128))
                    
                    # 绘制虚线效果（通过间隔绘制实现）
                    for contour in contours:
                        for i in range(0, len(contour), 10):  # 每10个点绘制一段
                            if i + 5 < len(contour):
                                cv2.polylines(alpha_overlay, [contour[i:i+5]], False, color, 2)
            
            overlay = alpha_overlay.astype(np.uint8)
        else:
            # 二分类：使用原有逻辑
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
            if groundtruth_mask is not None:
                groundtruth_contours, _ = cv2.findContours(
                    groundtruth_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(
                    overlay,
                    groundtruth_contours,
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
        """
        保存基础预测结果
        支持二分类和多分类
        """

        # create & check dir
        os.makedirs(save_dir, exist_ok=True)

        # create & check subdir
        predmask_dir = os.path.join(save_dir, "predicted_masks")
        overlay_dir  = os.path.join(save_dir, "overlay_images")

        os.makedirs(predmask_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)

        model.eval()
        saved_count = 0

        # 检查任务类型
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            sample_images, sample_masks = sample_batch
            sample_images = sample_images.to(device, non_blocking=True)
            sample_logits = model(sample_images)
            is_binary = sample_logits.shape[1] == 1
            
        # 只对二分类检测阈值
        if is_binary and threshold is None:
            print("Detecting optimal threshold for binary task...")
            threshold = self._detect_optimal_threshold(model, val_loader, device)
            print(f"Using threshold: {threshold:.4f}")
        elif not is_binary:
            print("Multiclass task detected, using argmax for predictions")
            threshold = None  # 多分类不需要阈值

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                if saved_count >= max_samples: # stop when reaching limited counts
                    break

                images = images.to(device, non_blocking=True)
                masks  = masks.to(device, non_blocking=True)

                # forward
                logits = model(images)
                
                if is_binary:
                    # 二分类：使用sigmoid + 阈值
                    probs = torch.sigmoid(logits)
                    preds = (probs > threshold).float()
                    
                    if batch_idx == 0:  # print debug info only at the first batch
                        print(f"Binary Debug Info:")
                        print(f"Logits - min: {logits.min():.4f}, max: {logits.max():.4f}, mean: {logits.mean():.4f}")
                        print(f"Probs  - min: {probs.min():.4f}, max: {probs.max():.4f}, mean: {probs.mean():.4f}")
                        print(f"GT mask unique values: {torch.unique(masks)}")
                else:
                    # 多分类：使用softmax + argmax
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(logits, dim=1).float()  # [B, H, W]
                    
                    if batch_idx == 0:  # print debug info only at the first batch
                        print(f"Multiclass Debug Info:")
                        print(f"Logits shape: {logits.shape}")
                        print(f"Logits - min: {logits.min():.4f}, max: {logits.max():.4f}, mean: {logits.mean():.4f}")
                        print(f"Pred unique values: {torch.unique(preds)}")
                        print(f"GT mask unique values: {torch.unique(masks)}")

                #检查预测效果
                if is_binary:
                    pred_ratio = preds.mean().item()
                    if batch_idx == 0:
                        print(f"预测前景比例: {pred_ratio:.3f}")
                    
                    # check predicted result
                    pred_positive_ratio = (preds > 0).float().mean()
                    print(f"Batch {batch_idx}: {pred_positive_ratio:.2%} pixels predicted as positive")
                else:
                    # 多分类：显示各类别比例
                    if batch_idx == 0:
                        for class_id in torch.unique(preds):
                            class_ratio = (preds == class_id).float().mean()
                            print(f"Class {int(class_id)}: {class_ratio:.2%}")

                # save each sample
                batch_size = images.shape[0]
                for i in range(min(batch_size, max_samples - saved_count)):
                    # convert to numpy format
                    img_np         = images[i].cpu().numpy()
                    groundtruth_np = masks[i].cpu().numpy()
                    pred_np        = preds[i].cpu().numpy()

                    # processing dimensions
                    if is_binary:
                        if groundtruth_np.shape[0] == 1:
                            groundtruth_np = groundtruth_np[0] # [1, H, W] -> [H, W]
                        if pred_np.shape[0] == 1:
                            pred_np = pred_np[0]
                    else:
                        # 多分类：确保维度正确
                        if groundtruth_np.ndim == 3:
                            groundtruth_np = groundtruth_np.squeeze()
                        if pred_np.ndim == 3:
                            pred_np = pred_np.squeeze()
                
                    # save predicted mask
                    pred_mask_path = os.path.join(
                        predmask_dir, f"pred_mask_{saved_count:03d}.png")
                    
                    if is_binary:
                        # 二分类：保存为灰度图
                        plt.imsave(pred_mask_path, pred_np, cmap='gray')
                    else:
                        # 多分类：保存为类别ID图（可选择使用颜色映射）
                        plt.imsave(pred_mask_path, pred_np, cmap='viridis')

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
        支持二分类和多分类，使用鲜艳颜色映射
        """
        # 确保图像格式正确
        if image.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
            image = np.transpose(image, (1, 2, 0))
        
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # 确保mask是2D
        if gt_mask.ndim == 3:
            gt_mask = gt_mask.squeeze()
        if pred_mask.ndim == 3:
            pred_mask = pred_mask.squeeze()
        
        # 检测是否为多分类（有超过2个唯一值）
        gt_unique = np.unique(gt_mask)
        pred_unique = np.unique(pred_mask)
        is_multiclass = len(gt_unique) > 2 or len(pred_unique) > 2
        
        if is_multiclass:
            # 多分类：使用鲜艳颜色映射
            from src.common.constants import PALETTE, GT_PALETTE
            
            def mask_to_color(mask, is_gt=False):
                """将分类mask转换为彩色图像，使用鲜艳颜色"""
                h, w = mask.shape
                color_mask = np.zeros((h, w, 3), dtype=np.uint8)
                
                # 选择调色板
                palette = GT_PALETTE if is_gt else PALETTE
                
                for class_id in np.unique(mask):
                    if class_id in palette:
                        color = palette[class_id]
                        color_mask[mask == class_id] = color
                    else:
                        # 默认颜色（灰色用于未知类别）
                        color_mask[mask == class_id] = (128, 128, 128)
                
                return color_mask
            
            gt_mask_rgb = mask_to_color(gt_mask, is_gt=True)
            pred_mask_rgb = mask_to_color(pred_mask, is_gt=False)
            
        else:
            # 二分类：转换为灰度图再转RGB
            # 归一化到0-255范围
            gt_mask_norm = ((gt_mask - gt_mask.min()) / (gt_mask.max() - gt_mask.min() + 1e-8) * 255).astype(np.uint8)
            pred_mask_norm = ((pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min() + 1e-8) * 255).astype(np.uint8)
            
            # 确保是2D数组
            if gt_mask_norm.ndim != 2:
                gt_mask_norm = gt_mask_norm.squeeze()
            if pred_mask_norm.ndim != 2:
                pred_mask_norm = pred_mask_norm.squeeze()
            
            # 转换为RGB
            gt_mask_rgb = cv2.cvtColor(gt_mask_norm, cv2.COLOR_GRAY2RGB)
            pred_mask_rgb = cv2.cvtColor(pred_mask_norm, cv2.COLOR_GRAY2RGB)
        
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
        
        # 添加文字标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Ground Truth", (w + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Prediction", (2*w + 10, 30), font, 1, (255, 255, 255), 2)
        
        # 如果是多分类，添加颜色图例
        if is_multiclass:
            from src.common.constants import CLASSES
            legend_start_y = h - 150
            legend_box_size = 15
            
            # 在每个面板底部添加图例
            for panel_idx in range(3):
                legend_x = panel_idx * w + 10
                
                # 遍历所有出现的类别
                all_classes = set(gt_unique) | set(pred_unique)
                for i, class_id in enumerate(sorted(all_classes)):
                    if class_id == 0:  # 跳过背景
                        continue
                        
                    y_pos = legend_start_y + i * 25
                    
                    # 选择合适的颜色
                    if panel_idx == 1:  # Ground Truth面板
                        color = GT_PALETTE.get(int(class_id), (128, 128, 128))
                    else:  # 其他面板使用预测颜色
                        color = PALETTE.get(int(class_id), (128, 128, 128))
                    
                    # 绘制颜色方块
                    cv2.rectangle(combined, 
                                (legend_x, y_pos), 
                                (legend_x + legend_box_size, y_pos + legend_box_size), 
                                color, -1)
                    
                    # 添加类别名称
                    class_name = CLASSES.get(int(class_id), f"Class_{class_id}")
                    cv2.putText(combined, class_name, 
                              (legend_x + legend_box_size + 5, y_pos + 12), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return combined

    def save_comparison_predictions(
        self,
        model,
        val_loader,
        save_dir: str,
        max_samples: int = 10,
        device: str = "cuda",
        threshold: float = 0.5  #可调节阈值
    ):
        """
        保存对比预测结果（包含三面板对比图）
        支持二分类和多分类
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
                
                # 判断任务类型
                if logits.shape[1] == 1:
                    # 二分类任务
                    probs = torch.sigmoid(logits)
                    preds = (probs > threshold).float()
                    
                    # 只在第一个batch显示调试信息
                    if batch_idx == 0:
                        print(f"[DEBUG] Binary - Logits range [{logits.min():.4f}, {logits.max():.4f}]")
                        print(f"[DEBUG] Binary - Probs range [{probs.min():.4f}, {probs.max():.4f}]")
                        print(f"[DEBUG] Binary - Pred unique values: {torch.unique(preds)}")
                else:
                    # 多分类任务
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(logits, dim=1).float()  # [B, H, W]
                    
                    # 只在第一个batch显示调试信息
                    if batch_idx == 0:
                        print(f"[DEBUG] Multiclass - Logits shape: {logits.shape}")
                        print(f"[DEBUG] Multiclass - Logits range [{logits.min():.4f}, {logits.max():.4f}]")
                        print(f"[DEBUG] Multiclass - Pred unique values: {torch.unique(preds)}")
                
                batch_size = images.shape[0]
                for i in range(min(batch_size, max_samples - saved_count)):
                    # 转换为numpy
                    img_np = images[i].cpu().numpy()
                    gt_np = masks[i].cpu().numpy()
                    pred_np = preds[i].cpu().numpy()
                    
                    # 处理维度 - 确保都是2D
                    if img_np.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
                        img_np = np.transpose(img_np, (1, 2, 0))
                    
                    # 对于二分类，移除通道维度
                    if logits.shape[1] == 1:
                        if gt_np.shape[0] == 1:
                            gt_np = gt_np[0]  # [1, H, W] -> [H, W]
                        if pred_np.shape[0] == 1:
                            pred_np = pred_np[0]  # [1, H, W] -> [H, W]
                    
                    # 对于多分类，gt_np已经是[H, W]，pred_np也是[H, W]
                    # 但需要确保维度正确
                    if gt_np.ndim == 3:
                        gt_np = gt_np.squeeze()  # 移除多余维度
                    if pred_np.ndim == 3:
                        pred_np = pred_np.squeeze()  # 移除多余维度
                    
                    # 调试：打印维度信息
                    if saved_count == 0:
                        print(f"Debug shapes - img: {img_np.shape}, gt: {gt_np.shape}, pred: {pred_np.shape}")
                    
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
        """绘制训练过程中的指标曲线"""
        import matplotlib.pyplot as plt
        import pandas as pd
        
        if not metrics_history:
            print("No metrics history available for plotting")
            return
        
        # 转换为DataFrame便于处理
        df = pd.DataFrame(metrics_history)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics Over Time', fontsize=16)
        
        # 1. Loss曲线
        if 'train_loss' in df.columns and 'val_loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
            axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', color='red')
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # 2. IoU曲线
        if 'iou' in df.columns:  # 二分类
            axes[0, 1].plot(df['epoch'], df['iou'], label='IoU', color='green')
            axes[0, 1].set_title('IoU Over Time')
        elif 'miou' in df.columns:  # 多分类
            axes[0, 1].plot(df['epoch'], df['miou'], label='mIoU', color='green')
            axes[0, 1].set_title('mIoU Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Dice曲线
        if 'dice' in df.columns:  # 二分类
            axes[1, 0].plot(df['epoch'], df['dice'], label='Dice', color='orange')
            axes[1, 0].set_title('Dice Score Over Time')
        elif 'mdice' in df.columns:  # 多分类
            axes[1, 0].plot(df['epoch'], df['mdice'], label='mDice', color='orange')
            axes[1, 0].set_title('mDice Score Over Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. 准确率曲线
        if 'accuracy' in df.columns:  # 二分类
            axes[1, 1].plot(df['epoch'], df['accuracy'], label='Accuracy', color='purple')
            axes[1, 1].set_title('Accuracy Over Time')
        elif 'macc' in df.columns:  # 多分类
            axes[1, 1].plot(df['epoch'], df['macc'], label='mAccuracy', color='purple')
            axes[1, 1].set_title('mAccuracy Over Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # 保存图形
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Metrics curves saved to: {save_path}")

    def create_error_analysis(self, predictions, targets, save_dir):
        """预留接口：错误分析可视化"""
        # TODO: 未来实现错误案例分析
        pass