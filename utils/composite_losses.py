"""
Advanced Loss Functions for Medical Image Segmentation
支持 CE+Dice 复合损失、Focal Loss、以及自动类别权重计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    适用于多分类和二分类分割任务
    """
    def __init__(self, smooth: float = 1e-5, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [N, C, H, W] 模型预测logits
            target: [N, H, W] 真实标签
        """
        # 获取预测概率
        pred_probs = F.softmax(pred, dim=1)  # [N, C, H, W]
        
        # 创建有效像素掩码
        valid_mask = (target != self.ignore_index)  # [N, H, W]
        
        # 将target转换为one-hot编码
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(
            target.clamp(0, num_classes-1), 
            num_classes=num_classes
        ).permute(0, 3, 1, 2).float()  # [N, C, H, W]
        
        # 只在有效像素上计算Dice
        valid_mask = valid_mask.unsqueeze(1)  # [N, 1, H, W]
        pred_probs = pred_probs * valid_mask
        target_one_hot = target_one_hot * valid_mask
        
        # 计算每个类别的Dice系数
        intersection = torch.sum(pred_probs * target_one_hot, dim=(2, 3))  # [N, C]
        union = torch.sum(pred_probs, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))  # [N, C]
        
        dice_scores = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [N, C]
        
        # 计算平均Dice Loss (1 - Dice)
        dice_loss = 1.0 - dice_scores.mean()
        
        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    改进版本，支持自动权重和更好的参数调整
    """
    def __init__(self, 
                 alpha: Union[float, torch.Tensor] = 1.0, 
                 gamma: float = 2.0, 
                 ignore_index: int = 255,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.class_weights = class_weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [N, C, H, W] 模型预测logits
            target: [N, H, W] 真实标签
        """
        # 标准交叉熵损失
        ce_loss = F.cross_entropy(
            pred, target, 
            weight=self.class_weights,
            ignore_index=self.ignore_index, 
            reduction='none'
        )  # [N, H, W]
        
        # 计算概率
        pt = torch.exp(-ce_loss)  # [N, H, W]
        
        # Focal weight
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # 应用Focal weight
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    CE + Dice 复合损失函数
    结合交叉熵损失和Dice损失的优点
    """
    def __init__(self, 
                 dice_weight: float = 0.4,
                 class_weights: Optional[torch.Tensor] = None,
                 ignore_index: int = 255,
                 use_focal: bool = False,
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = 1.0 - dice_weight
        self.ignore_index = ignore_index
        self.use_focal = use_focal
        
        # 选择基础损失函数
        if use_focal:
            self.base_loss = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                ignore_index=ignore_index,
                class_weights=class_weights
            )
        else:
            self.base_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=ignore_index
            )
        
        # Dice损失
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [N, C, H, W] 模型预测logits
            target: [N, H, W] 真实标签
        """
        # 基础损失 (CE或Focal)
        base_loss = self.base_loss(pred, target)
        
        # Dice损失
        dice_loss = self.dice_loss(pred, target)
        
        # 组合损失
        total_loss = self.ce_weight * base_loss + self.dice_weight * dice_loss
        
        return total_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    减少过拟合，提高泛化能力
    """
    def __init__(self, 
                 smoothing: float = 0.1, 
                 class_weights: Optional[torch.Tensor] = None,
                 ignore_index: int = 255):
        super().__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [N, C, H, W] 模型预测logits
            target: [N, H, W] 真实标签
        """
        num_classes = pred.shape[1]
        
        # 创建平滑标签
        with torch.no_grad():
            # 创建one-hot编码
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            
            # 处理有效像素
            valid_mask = (target != self.ignore_index)
            target_valid = target.clone()
            target_valid[~valid_mask] = 0  # 临时设置为0，避免索引错误
            
            true_dist.scatter_(1, target_valid.unsqueeze(1), 1.0 - self.smoothing)
            
            # 对忽略像素设置为0
            invalid_mask = ~valid_mask
            true_dist[invalid_mask.unsqueeze(1).expand_as(true_dist)] = 0
        
        # 计算KL散度损失
        log_pred = F.log_softmax(pred, dim=1)
        loss = -true_dist * log_pred
        
        # 只在有效像素上计算损失
        valid_mask = valid_mask.unsqueeze(1)  # [N, 1, H, W]
        loss = loss * valid_mask
        
        # 应用类别权重
        if self.class_weights is not None:
            weight_expanded = self.class_weights.view(1, -1, 1, 1)
            loss = loss * weight_expanded
        
        return loss.sum() / valid_mask.sum()


def compute_auto_class_weights(dataset, num_classes: int, ignore_index: int = 255, 
                              sample_ratio: float = 0.1) -> torch.Tensor:
    """
    自动计算类别权重以处理类别不平衡
    
    Args:
        dataset: 数据集对象
        num_classes: 类别数量
        ignore_index: 要忽略的像素值
        sample_ratio: 采样比例，避免计算过慢
    
    Returns:
        torch.Tensor: 类别权重 [num_classes]
    """
    print(f"Computing automatic class weights from dataset (sampling {sample_ratio:.1%})...")
    
    class_counts = np.zeros(num_classes, dtype=np.float64)
    total_pixels = 0
    
    # 确定采样数量
    dataset_size = len(dataset)
    sample_size = max(1, int(dataset_size * sample_ratio))
    sample_indices = np.random.choice(dataset_size, sample_size, replace=False)
    
    print(f"Sampling {sample_size} images from {dataset_size} total images...")
    
    for i, idx in enumerate(sample_indices):
        if i % 50 == 0:
            print(f"  Processing sample {i+1}/{sample_size}...")
        
        try:
            _, mask = dataset[idx]
            
            # 转换为numpy数组
            if hasattr(mask, 'numpy'):
                mask_np = mask.numpy()
            elif torch.is_tensor(mask):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            # 统计每个类别的像素数
            for class_id in range(num_classes):
                class_counts[class_id] += np.sum(mask_np == class_id)
            
            # 统计有效像素总数
            total_pixels += np.sum(mask_np != ignore_index)
            
        except Exception as e:
            print(f"  Warning: Error processing sample {idx}: {e}")
            continue
    
    # 处理没有样本的类别
    class_counts = np.maximum(class_counts, 1.0)  # 避免0计数
    
    # 计算权重
    if total_pixels > 0:
        # 使用倒数权重 + 归一化
        class_weights = total_pixels / (num_classes * class_counts)
        # 归一化使权重和为类别数
        class_weights = class_weights / np.sum(class_weights) * num_classes
    else:
        # 如果没有有效像素，使用均匀权重
        class_weights = np.ones(num_classes, dtype=np.float32)
    
    # 打印统计信息
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")
    print(f"Total valid pixels: {total_pixels}")
    
    return torch.FloatTensor(class_weights)


def create_loss_function(loss_config: dict, num_classes: int, device: torch.device) -> nn.Module:
    """
    根据配置创建损失函数的工厂函数
    
    Args:
        loss_config: 损失函数配置字典
        num_classes: 类别数量
        device: 设备
    
    Returns:
        nn.Module: 配置好的损失函数
    """
    ignore_index = loss_config.get('ignore_index', 255)
    
    # 处理类别权重
    class_weights = None
    if loss_config.get('auto_class_weights', False):
        # 这里需要传入dataset，实际使用时需要在外部计算
        print("Note: auto_class_weights requires dataset, will be computed externally")
    elif 'class_weights' in loss_config and loss_config['class_weights'] is not None:
        class_weights = torch.tensor(loss_config['class_weights'], device=device)
    
    # 选择损失函数类型
    loss_type = loss_config.get('loss_type', 'ce')  # 默认使用交叉熵
    
    if loss_type == 'combined' or loss_config.get('dice_weight', 0) > 0:
        # CE + Dice 复合损失
        return CombinedLoss(
            dice_weight=loss_config.get('dice_weight', 0.4),
            class_weights=class_weights,
            ignore_index=ignore_index,
            use_focal=loss_config.get('use_focal_base', False),
            focal_alpha=loss_config.get('focal_alpha', 1.0),
            focal_gamma=loss_config.get('focal_gamma', 2.0)
        )
    
    elif loss_type == 'focal':
        # Focal Loss
        return FocalLoss(
            alpha=loss_config.get('focal_alpha', 1.0),
            gamma=loss_config.get('focal_gamma', 2.0),
            ignore_index=ignore_index,
            class_weights=class_weights
        )
    
    elif loss_type == 'label_smoothing':
        # Label Smoothing Cross Entropy
        return LabelSmoothingCrossEntropy(
            smoothing=loss_config.get('label_smoothing', 0.1),
            class_weights=class_weights,
            ignore_index=ignore_index
        )
    
    elif loss_type == 'dice':
        # 纯Dice Loss
        return DiceLoss(ignore_index=ignore_index)
    
    else:
        # 标准交叉熵损失
        return nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )


if __name__ == "__main__":
    # 测试代码
    print("Testing composite loss functions...")
    
    # 测试参数
    batch_size, num_classes, height, width = 2, 3, 64, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成测试数据
    pred = torch.randn(batch_size, num_classes, height, width, device=device)
    target = torch.randint(0, num_classes, (batch_size, height, width), device=device)
    
    # 测试各种损失函数
    print("\n1. Testing DiceLoss...")
    dice_loss = DiceLoss()
    dice_result = dice_loss(pred, target)
    print(f"   Dice Loss: {dice_result.item():.4f}")
    
    print("\n2. Testing FocalLoss...")
    focal_loss = FocalLoss()
    focal_result = focal_loss(pred, target)
    print(f"   Focal Loss: {focal_result.item():.4f}")
    
    print("\n3. Testing CombinedLoss...")
    combined_loss = CombinedLoss(dice_weight=0.4)
    combined_result = combined_loss(pred, target)
    print(f"   Combined Loss: {combined_result.item():.4f}")
    
    print("\n4. Testing LabelSmoothingCrossEntropy...")
    ls_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    ls_result = ls_loss(pred, target)
    print(f"   Label Smoothing Loss: {ls_result.item():.4f}")
    
    print("\nAll tests completed successfully!")