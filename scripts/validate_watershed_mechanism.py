# scripts/validate_watershed_mechanism.py
"""
Validate Watershed Region Learning Mechanism with Real Model Prediction
验证Watershed区域学习机制的具体演示脚本（使用真实模型预测）
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import Counter

# 导入模型和相关模块
import sys
sys.path.append('.')
from src.models.model_zoo import build_model
from src.viz.colorize import id_to_color
from src.common.constants import PALETTE, GT_PALETTE

def load_trained_model(checkpoint_path, num_classes=10, device='cuda'):
    """
    加载训练好的模型
    """
    print(f"Loading trained model from: {checkpoint_path}")
    
    # 检查checkpoint文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
        return None
    
    # 创建模型
    model = build_model('unet_min', num_classes=num_classes, stage='offline')
    
    # 加载checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Model epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"✓ Validation loss: {checkpoint.get('val_loss', 'unknown')}")
        
        return model
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None

def preprocess_image(image_rgb, img_size=512):
    """
    预处理图像用于模型推理
    """
    # 调整大小
    image_resized = cv2.resize(image_rgb, (img_size, img_size))
    
    # 转换为tensor并归一化
    image_tensor = torch.from_numpy(image_resized).float()
    image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
    image_tensor = image_tensor / 255.0  # 归一化到[0,1]
    image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度 [1, 3, H, W]
    
    return image_tensor

@torch.no_grad()
def predict_with_model(model, image_tensor, device='cuda'):
    """
    使用模型进行预测
    """
    image_tensor = image_tensor.to(device)
    
    # 模型推理
    logits = model(image_tensor)  # [1, num_classes, H, W]
    
    # 获取预测结果
    predictions = torch.argmax(logits, dim=1)  # [1, H, W]
    predictions = predictions.squeeze(0).cpu().numpy()  # [H, W]
    
    # 获取概率
    probs = torch.softmax(logits, dim=1)
    max_probs = torch.max(probs, dim=1)[0].squeeze(0).cpu().numpy()
    
    return predictions, max_probs
    
def analyze_specific_sample_with_model(sample_path, model_path, device='cuda'):
    """
    分析指定样本的Watershed处理过程，并使用训练好的模型生成预测
    Args:
        sample_path: 样本图像的完整路径
        model_path: 训练好的模型checkpoint路径
        device: 计算设备
    """
    # 从路径中提取目录和文件名信息
    sample_dir = os.path.dirname(sample_path)
    base_name = os.path.basename(sample_path).replace('_endo.png', '')
    
    print("="*80)
    print("WATERSHED REGION LEARNING MECHANISM WITH REAL PREDICTION")
    print("="*80)
    print(f"Analyzing sample: {sample_path}")
    print(f"Using model: {model_path}")
    print(f"Base name: {base_name}")
    
    # 构建相关文件路径
    image_path = sample_path
    watershed_path = os.path.join(sample_dir, f"{base_name}_endo_watershed_mask.png")
    color_mask_path = os.path.join(sample_dir, f"{base_name}_endo_color_mask.png")
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"ERROR: Original image not found: {image_path}")
        return None
    
    if not os.path.exists(watershed_path):
        print(f"ERROR: Watershed mask not found: {watershed_path}")
        return None
    
    print(f"Original image: {os.path.basename(image_path)}")
    print(f"Watershed mask: {os.path.basename(watershed_path)}")
    if os.path.exists(color_mask_path):
        print(f"Color mask: {os.path.basename(color_mask_path)}")
    
    # 加载训练好的模型
    model = load_trained_model(model_path, num_classes=10, device=device)
    if model is None:
        return None
    
    # 读取图像
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 读取watershed mask
    watershed_mask = cv2.imread(watershed_path, cv2.IMREAD_GRAYSCALE)
    
    # 读取color mask（如果存在）
    color_mask = None
    if os.path.exists(color_mask_path):
        color_mask = cv2.imread(color_mask_path)
        color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)
    
    print(f"\nImage shape: {image_rgb.shape}")
    print(f"Watershed mask shape: {watershed_mask.shape}")
    
    # 分析watershed mask中的区域
    unique_regions = np.unique(watershed_mask)
    print(f"\nOriginal Watershed Regions Analysis:")
    print(f"Unique region values: {unique_regions}")
    print(f"Number of regions: {len(unique_regions)}")
    
    # 模拟训练时的重新编号过程（来自seg_dataset_min.py的逻辑）
    print(f"\nSimulating training-time region remapping:")
    region_mapping = {}
    region_mapping[0] = 0  # 背景始终为0
    
    class_id = 1
    for region_id in sorted(unique_regions):
        if region_id != 0 and region_id != 255 and class_id < 10:  # 跳过背景和ignore_index
            region_mapping[region_id] = class_id
            print(f"  Original region {region_id:3d} -> Training class {class_id}")
            class_id += 1
    
    print(f"\nFinal mapping: {region_mapping}")
    print(f"Total training classes: {class_id - 1} (plus background)")
    
    # 应用重新编号生成Ground Truth
    gt_mask = np.full_like(watershed_mask, fill_value=255, dtype=np.uint8)
    for original_id, new_id in region_mapping.items():
        gt_mask[watershed_mask == original_id] = new_id
    
    # 使用模型进行预测
    print(f"\nGenerating model prediction...")
    image_tensor = preprocess_image(image_rgb, img_size=512)
    pred_mask, pred_confidence = predict_with_model(model, image_tensor, device)
    
    # 调整预测结果尺寸以匹配原图
    if pred_mask.shape != image_rgb.shape[:2]:
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), 
                              (image_rgb.shape[1], image_rgb.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
    
    pred_unique = np.unique(pred_mask)
    print(f"Prediction unique values: {pred_unique}")
    print(f"Prediction confidence range: [{pred_confidence.min():.3f}, {pred_confidence.max():.3f}]")
    
    # 生成6面板可视化
    create_six_panel_visualization(image_rgb, watershed_mask, gt_mask, color_mask, 
                                  pred_mask, pred_confidence, region_mapping, base_name)
    
    return {
        'original_regions': unique_regions,
        'region_mapping': region_mapping,
        'training_classes': class_id - 1,
        'gt_mask': gt_mask,
        'pred_mask': pred_mask,
        'pred_confidence': pred_confidence
    }

def create_six_panel_visualization(image_rgb, watershed_mask, gt_mask, color_mask, 
                                  pred_mask, pred_confidence, region_mapping, base_name):
    """
    创建6面板可视化：
    第一行：原图，watershed mask（原图），染色后的watershed
    第二行：colormask，predict（原图），染色后的predict
    """
    
    # 创建可视化布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Watershed Learning with Real Model Prediction - {base_name}', 
                fontsize=16, fontweight='bold')
    
    # 第一行：原图，watershed mask（原图），染色后的watershed
    
    # 1. 原图
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Endoscopic Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Watershed mask（原图，灰度显示）
    axes[0, 1].imshow(watershed_mask, cmap='gray')
    axes[0, 1].set_title('Watershed Mask (Raw)', fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. 染色后的watershed（Ground Truth）
    gt_colored = id_to_color(gt_mask, GT_PALETTE)
    axes[0, 2].imshow(gt_colored)
    axes[0, 2].set_title('Watershed (Colored GT)', fontweight='bold')
    axes[0, 2].axis('off')
    
    # 第二行：colormask，predict（原图），染色后的predict
    
    # 4. Color mask（如果有的话）
    if color_mask is not None:
        axes[1, 0].imshow(color_mask)
        axes[1, 0].set_title('Color Mask', fontweight='bold')
    else:
        # 如果没有color mask，显示GT的另一种颜色版本
        gt_colored_alt = id_to_color(gt_mask, PALETTE)
        axes[1, 0].imshow(gt_colored_alt)
        axes[1, 0].set_title('GT (Alternative Colors)', fontweight='bold')
    axes[1, 0].axis('off')
    
    # 5. Predict（原图，灰度显示）
    axes[1, 1].imshow(pred_mask, cmap='gray', vmin=0, vmax=9)
    axes[1, 1].set_title('Prediction (Raw)', fontweight='bold')
    axes[1, 1].axis('off')
    
    # 6. 染色后的predict
    pred_colored = id_to_color(pred_mask, PALETTE)
    axes[1, 2].imshow(pred_colored)
    axes[1, 2].set_title('Prediction (Colored)', fontweight='bold')
    axes[1, 2].axis('off')
    
    # 添加颜色图例和统计信息
    add_comprehensive_legend(fig, region_mapping, gt_mask, pred_mask)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.20)
    
    # 保存图像
    output_path = f'watershed_real_prediction_{base_name}.png'
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
        plt.show()
    except Exception as e:
        print(f"Error saving visualization: {e}")
        print("Continuing without saving...")

def add_comprehensive_legend(fig, region_mapping, gt_mask, pred_mask):
    """添加综合图例和统计信息"""
    
    # 统计GT和预测的类别分布
    gt_unique, gt_counts = np.unique(gt_mask, return_counts=True)
    pred_unique, pred_counts = np.unique(pred_mask, return_counts=True)
    
    # 创建图例文本
    legend_text = "Color Legend & Statistics:\n"
    legend_text += "GT Classes: "
    
    class_names = ['Background', 'Instrument', 'Gallbladder', 'Liver', 'Fat', 
                  'Tissue', 'Vessel', 'Organ', 'Structure', 'Region']
    
    for i, (class_id, count) in enumerate(zip(gt_unique, gt_counts)):
        if class_id < len(class_names):
            class_name = class_names[class_id]
            percentage = count / gt_mask.size * 100
            legend_text += f"Class {class_id}({class_name}): {percentage:.1f}%  "
    
    legend_text += "\nPred Classes: "
    for i, (class_id, count) in enumerate(zip(pred_unique, pred_counts)):
        if class_id < len(class_names):
            class_name = class_names[class_id]
            percentage = count / pred_mask.size * 100
            legend_text += f"Class {class_id}({class_name}): {percentage:.1f}%  "
    
    # 计算IoU（简单版本）
    intersection = np.sum((gt_mask == pred_mask) & (gt_mask != 255))
    total_valid = np.sum(gt_mask != 255)
    accuracy = intersection / total_valid if total_valid > 0 else 0
    
    legend_text += f"\nPixel Accuracy: {accuracy:.3f}"
    
    # 在图的底部添加文本
    fig.text(0.5, 0.02, legend_text, ha='center', va='bottom', 
             fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

def analyze_specific_sample(sample_path):
    """
    分析指定样本的Watershed处理过程
    Args:
        sample_path: 样本图像的完整路径
    """
    # 从路径中提取目录和文件名信息
    sample_dir = os.path.dirname(sample_path)
    base_name = os.path.basename(sample_path).replace('_endo.png', '')
    
    print("="*80)
    print("WATERSHED REGION LEARNING MECHANISM ANALYSIS")
    print("="*80)
    print(f"Analyzing sample: {sample_path}")
    print(f"Base name: {base_name}")
    
    # 构建相关文件路径
    image_path = sample_path
    watershed_path = os.path.join(sample_dir, f"{base_name}_endo_watershed_mask.png")
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"ERROR: Original image not found: {image_path}")
        return None
    
    if not os.path.exists(watershed_path):
        print(f"ERROR: Watershed mask not found: {watershed_path}")
        return None
    
    print(f"Original image: {os.path.basename(image_path)}")
    print(f"Watershed mask: {os.path.basename(watershed_path)}")
    
    # 读取图像
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 读取watershed mask
    watershed_mask = cv2.imread(watershed_path, cv2.IMREAD_GRAYSCALE)
    
    print(f"\nImage shape: {image_rgb.shape}")
    print(f"Watershed mask shape: {watershed_mask.shape}")
    
    # 分析watershed mask中的区域
    unique_regions = np.unique(watershed_mask)
    print(f"\nOriginal Watershed Regions Analysis:")
    print(f"Unique region values: {unique_regions}")
    print(f"Number of regions: {len(unique_regions)}")
    
    # 统计每个区域的像素数量
    region_counts = Counter(watershed_mask.flatten())
    print(f"\nRegion pixel statistics:")
    for region_id in sorted(unique_regions):
        count = region_counts[region_id]
        percentage = count / watershed_mask.size * 100
        print(f"  Region {region_id:3d}: {count:6d} pixels ({percentage:5.2f}%)")
    
    # 模拟训练时的重新编号过程（来自seg_dataset_min.py的逻辑）
    print(f"\nSimulating training-time region remapping:")
    region_mapping = {}
    region_mapping[0] = 0  # 背景始终为0
    
    class_id = 1
    for region_id in sorted(unique_regions):
        if region_id != 0 and region_id != 255 and class_id < 10:  # 跳过背景和ignore_index
            region_mapping[region_id] = class_id
            print(f"  Original region {region_id:3d} -> Training class {class_id}")
            class_id += 1
    
    print(f"\nFinal mapping: {region_mapping}")
    print(f"Total training classes: {class_id - 1} (plus background)")
    
    # 应用重新编号
    remapped_mask = np.full_like(watershed_mask, fill_value=255, dtype=np.uint8)
    for original_id, new_id in region_mapping.items():
        remapped_mask[watershed_mask == original_id] = new_id
    
    # 验证重新编号结果
    remapped_unique = np.unique(remapped_mask)
    print(f"\nRemapped mask unique values: {remapped_unique}")
    
    # 生成可视化
    create_visualization(image_rgb, watershed_mask, remapped_mask, 
                        unique_regions, region_mapping, base_name)
    
    return {
        'original_regions': unique_regions,
        'region_mapping': region_mapping,
        'training_classes': class_id - 1,
        'remapped_mask': remapped_mask
    }

def create_visualization(image_rgb, watershed_mask, remapped_mask, 
                        unique_regions, region_mapping, base_name):
    """保留原有的可视化函数以防向后兼容 - 已弃用"""
    print("Warning: This function is deprecated. Use create_six_panel_visualization instead.")
    pass

def add_color_legend(fig, region_mapping, training_colors):
    """原有的添加颜色图例函数 - 已弃用"""
    print("Warning: This function is deprecated. Use add_comprehensive_legend instead.")
    pass

def print_learning_explanation():
    """打印学习机制的详细解释"""
    print("\n" + "="*80)
    print("UNDERSTANDING THE LEARNING MECHANISM")
    print("="*80)
    
    print("\n1. WATERSHED MASK GENERATION:")
    print("   - Watershed algorithm segments image into regions based on gradients")
    print("   - Each region gets an arbitrary grayscale value (e.g., 0, 17, 45, 78...)")
    print("   - These values represent distinct anatomical regions")
    
    print("\n2. TRAINING-TIME REMAPPING:")
    print("   - Original watershed values are remapped to consecutive class IDs")
    print("   - Example: Region 17 -> Class 1, Region 45 -> Class 2, etc.")
    print("   - This creates consistent training labels across all samples")
    
    print("\n3. MODEL LEARNING PROCESS:")
    print("   - Model learns: Visual Features -> Semantic Class Mapping")
    print("   - Input: RGB image pixels")
    print("   - Output: Class probability for each pixel")
    print("   - Loss: CrossEntropyLoss between predicted and ground truth classes")
    
    print("\n4. WHAT THE MODEL ACTUALLY LEARNS:")
    print("   - Spatial feature patterns (color, texture, shape)")
    print("   - Anatomical structure recognition (gallbladder, instrument, liver)")
    print("   - Region boundary detection")
    print("   - Semantic understanding (not memorizing grayscale values!)")
    
    print("\n5. INFERENCE AND VISUALIZATION:")
    print("   - Model predicts class probabilities for each pixel")
    print("   - argmax() selects the most probable class")
    print("   - Color mapping converts class IDs to visualization colors")
    print("   - Result: Semantically meaningful colored segmentation")
    
    print("\n" + "="*80)
    print("KEY INSIGHT: The model learns SEMANTIC UNDERSTANDING, not grayscale memorization!")
    print("="*80)

def main():
    """主函数"""
    # 指定要分析的样本和模型
    sample_path = "data/seg8k/video20/video20_03260/frame_3260_endo.png"
    model_path = "outputs/baseline_multi_20250829_105216/checkpoints/baseline_multi_epoch_003.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Starting Watershed mechanism validation with real model prediction...")
    print(f"Target sample: {sample_path}")
    print(f"Model checkpoint: {model_path}")
    print(f"Device: {device}")
    
    try:
        # 分析样本并生成预测
        result = analyze_specific_sample_with_model(sample_path, model_path, device)
        
        if result:
            print_learning_explanation()
            
            print(f"\nANALYSIS SUMMARY:")
            print(f"- Original watershed regions: {len(result['original_regions'])}")
            print(f"- Training classes generated: {result['training_classes']}")
            print(f"- Region mapping: {result['region_mapping']}")
            print(f"- Model predicted {len(np.unique(result['pred_mask']))} different classes")
            print(f"- Prediction confidence range: [{result['pred_confidence'].min():.3f}, {result['pred_confidence'].max():.3f}]")
        else:
            print("Analysis failed - please check the file paths")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
