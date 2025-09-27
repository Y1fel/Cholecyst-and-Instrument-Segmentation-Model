#!/usr/bin/env python3
"""
最终效果图生成脚本
支持多种分类方案：3class_org, 6class, detailed (13类)
与 train_offline_universal.py 和 SegDatasetMin 的训练机制完全一致
"""

import os
import sys
import argparse
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.dataio.datasets.seg_dataset_min import SegDatasetMin
from src.models.model_zoo import build_model
from src.viz.visualizer import Visualizer
from src.eval.evaluator import Evaluator
from src.common.constants import CLASSIFICATION_SCHEMES, compose_mapping
from src.common.output_manager import OutputManager

def parse_args():
    """参数解析"""
    parser = argparse.ArgumentParser(description="Generate final segmentation results")
    
    # 基础参数
    parser.add_argument("--model_path", type=str, required=True,
                       help="训练好的模型权重路径")
    parser.add_argument("--data_root", type=str, default="data/seg8k",
                       help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default="final_results",
                       help="结果输出目录")
    
    # 分类方案参数
    parser.add_argument("--classification_scheme", type=str, default="3class_org",
                       choices=["3class_org", "6class", "detailed"],
                       help="分类方案: 3class_org(3类), 6class(6类), detailed(13类)")
    
    # 模型参数
    parser.add_argument("--model", type=str, default="unet_plus_plus",
                       choices=["unet_min", "unet_plus_plus", "deeplabv3_plus", "hrnet", "mobile_unet", "adaptive_unet"],
                       help="模型架构")
    parser.add_argument("--img_size", type=int, default=384,
                       help="输入图像尺寸")
    
    # 可视化参数
    parser.add_argument("--max_samples", type=int, default=20,
                       help="最大可视化样本数")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批处理大小")
    
    # FOV参数
    parser.add_argument("--apply_fov_mask", action="store_true",
                       help="是否应用FOV遮罩")
    
    return parser.parse_args()

def get_classification_info(scheme_name: str):
    """获取分类方案信息"""
    if scheme_name not in CLASSIFICATION_SCHEMES:
        raise ValueError(f"Unsupported classification scheme: {scheme_name}")
    
    scheme = CLASSIFICATION_SCHEMES[scheme_name]
    return {
        "num_classes": scheme["num_classes"],
        "target_classes": scheme["target_classes"],
        "description": scheme["description"]
    }

def create_dataset(args):
    """创建数据集，与训练时完全一致"""
    # 使用compose_mapping生成映射，与train_offline_universal.py一致
    class_id_map = compose_mapping(
        classification_scheme=args.classification_scheme,
        custom_mapping=None,
        target_classes=None
    )
    
    print(f"[DATASET] Using classification scheme: {args.classification_scheme}")
    print(f"[DATASET] Class mapping: {class_id_map}")
    
    # 创建数据集，参数与训练时一致
    dataset = SegDatasetMin(
        data_root=args.data_root,
        dtype="train",  # 使用训练集进行效果展示
        img_size=args.img_size,
        return_multiclass=True,  # 多分类模式
        class_id_map=class_id_map,  # 直接传入映射
        apply_fov_mask=args.apply_fov_mask
    )
    
    print(f"[DATASET] Loaded {len(dataset)} samples")
    print(f"[DATASET] Number of classes: {dataset.num_classes}")
    
    return dataset

def load_model(args, num_classes: int):
    """加载训练好的模型"""
    print(f"[MODEL] Building {args.model} with {num_classes} classes")
    
    # 构建模型，与训练时一致
    model = build_model(
        args.model, 
        num_classes=num_classes, 
        in_ch=3, 
        stage="auto"
    ).to(args.device)
    
    # 加载权重
    print(f"[MODEL] Loading weights from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    
    # 处理不同的checkpoint格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[MODEL] Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"[MODEL] Model metrics: {metrics}")
    else:
        model.load_state_dict(checkpoint)
        print(f"[MODEL] Loaded raw state dict")
    
    model.eval()
    return model

def create_color_mapping(num_classes: int):
    """为不同分类数创建颜色映射"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 使用不同的colormap for不同分类数
    if num_classes <= 3:
        cmap_name = 'Set1'  # 明亮的离散颜色
    elif num_classes <= 6:
        cmap_name = 'Set2'  # 柔和的离散颜色
    else:
        cmap_name = 'tab20'  # 20种区分度高的颜色
    
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i) for i in np.linspace(0, 1, num_classes)]
    
    return cmap_name, colors

def generate_detailed_comparison(model, dataset, args, output_dir: str):
    """生成详细的对比可视化"""
    print(f"\n[VISUALIZATION] Generating detailed comparison visualizations...")
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    # 创建可视化器
    visualizer = Visualizer()
    
    # 保存基础预测结果
    print(f"[VISUALIZATION] Generating basic predictions...")
    visualizer.save_basic_predictions(
        model=model,
        val_loader=dataloader,
        save_dir=output_dir,
        max_samples=args.max_samples,
        device=args.device
    )
    
    # 保存对比预测结果（三面板对比图）
    print(f"[VISUALIZATION] Generating comparison panels...")
    visualizer.save_comparison_predictions(
        model=model,
        val_loader=dataloader,
        save_dir=output_dir,
        max_samples=args.max_samples,
        device=args.device
    )
    
    print(f"[VISUALIZATION] Saved visualizations to: {output_dir}")

def evaluate_model_performance(model, dataset, args):
    """评估模型性能"""
    print(f"\n[EVALUATION] Evaluating model performance...")
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    # 创建评估器
    evaluator = Evaluator(device=args.device)
    
    # 执行评估
    metrics = evaluator.evaluate_with_full_metrics(
        model=model,
        val_loader=dataloader,
        num_classes=dataset.num_classes,
        binary_mode=False,  # 多分类模式
        max_samples=args.max_samples * args.batch_size
    )
    
    print(f"[EVALUATION] Performance metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return metrics

def save_experiment_info(args, classification_info, metrics, output_dir: str):
    """保存实验信息"""
    info = {
        "experiment_info": {
            "model": args.model,
            "model_path": str(args.model_path),
            "classification_scheme": args.classification_scheme,
            "num_classes": classification_info["num_classes"],
            "target_classes": classification_info["target_classes"],
            "description": classification_info["description"],
            "img_size": args.img_size,
            "apply_fov_mask": args.apply_fov_mask,
            "max_samples": args.max_samples
        },
        "performance_metrics": metrics,
        "dataset_info": {
            "data_root": args.data_root,
            "total_samples": "loaded_from_dataset"
        }
    }
    
    # 保存为JSON
    info_path = os.path.join(output_dir, "experiment_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    # 保存为可读文本
    txt_path = os.path.join(output_dir, "experiment_summary.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("SEGMENTATION MODEL FINAL RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("EXPERIMENT CONFIGURATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Model Architecture: {args.model}\n")
        f.write(f"Model Weights: {args.model_path}\n")
        f.write(f"Classification Scheme: {args.classification_scheme}\n")
        f.write(f"Number of Classes: {classification_info['num_classes']}\n")
        f.write(f"Class Description: {classification_info['description']}\n")
        f.write(f"Target Classes: {', '.join(classification_info['target_classes'])}\n")
        f.write(f"Input Image Size: {args.img_size}x{args.img_size}\n")
        f.write(f"FOV Mask Applied: {args.apply_fov_mask}\n")
        f.write(f"Visualization Samples: {args.max_samples}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write(f"\nGenerated on: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}\n")
    
    print(f"[INFO] Saved experiment info to: {info_path}")
    print(f"[INFO] Saved experiment summary to: {txt_path}")

def main():
    """主函数"""
    args = parse_args()
    
    # 验证输入
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    
    # 获取分类方案信息
    classification_info = get_classification_info(args.classification_scheme)
    print(f"[CONFIG] Classification Scheme: {args.classification_scheme}")
    print(f"[CONFIG] {classification_info['description']}")
    print(f"[CONFIG] Classes: {', '.join(classification_info['target_classes'])}")
    
    # 创建输出目录
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{args.classification_scheme}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[OUTPUT] Results will be saved to: {output_dir}")
    
    # 创建数据集
    dataset = create_dataset(args)
    
    # 加载模型
    model = load_model(args, classification_info["num_classes"])
    
    # 生成可视化
    generate_detailed_comparison(model, dataset, args, output_dir)
    
    # 评估性能
    metrics = evaluate_model_performance(model, dataset, args)
    
    # 保存实验信息
    save_experiment_info(args, classification_info, metrics, output_dir)
    
    print(f"\n" + "=" * 60)
    print(f"FINAL RESULTS GENERATION COMPLETED!")
    print(f"Results saved to: {output_dir}")
    print(f"Classification: {args.classification_scheme} ({classification_info['num_classes']} classes)")
    print(f"Best mIoU: {metrics.get('miou', 'N/A'):.4f}")
    print(f"=" * 60)

if __name__ == "__main__":
    main()