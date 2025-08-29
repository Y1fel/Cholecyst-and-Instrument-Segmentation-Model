#!/usr/bin/env python3
"""
训练后处理脚本 - 专门用于完成训练后的可视化和分析
当训练完成但后处理被中断时使用
"""

import os
import sys
import argparse
import torch
import yaml
from torch.utils.data import DataLoader

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.dataio.datasets.seg_dataset_min import SegDatasetMin
from src.models.model_zoo import build_model
from src.viz.visualizer import Visualizer
from src.common.output_manager import OutputManager
from src.common.train_monitor import TrainMonitor


def parse_args():
    parser = argparse.ArgumentParser("Post Training Visualization")
    
    # 必需参数
    parser.add_argument("--run_dir", type=str, required=True, 
                       help="训练输出目录路径 (例如: outputs/baseline_improved_20250823_015907)")
    parser.add_argument("--data_root", type=str, required=True,
                       help="数据集根目录路径")
    
    # 可选参数 - 如果不提供会尝试从config.json读取
    parser.add_argument("--model", type=str, default=None,
                       help="模型类型 (如果不提供，从config.json读取)")
    parser.add_argument("--binary", action="store_true", default=None,
                       help="是否二分类模式 (如果不提供，从config.json读取)")
    parser.add_argument("--num_classes", type=int, default=None,
                       help="类别数量 (如果不提供，从config.json读取)")
    parser.add_argument("--img_size", type=int, default=None,
                       help="图像尺寸 (如果不提供，从config.json读取)")
    parser.add_argument("--batch_size", type=int, default=6,
                       help="批次大小")
    parser.add_argument("--val_ratio", type=float, default=None,
                       help="验证集比例 (如果不提供，从config.json读取)")
    parser.add_argument("--viz_samples", type=int, default=50,
                       help="可视化样本数量")
    
    # 高级选项
    parser.add_argument("--best_model", action="store_true", default=True,
                       help="使用最佳模型进行可视化")
    parser.add_argument("--force_recreate", action="store_true",
                       help="强制重新生成所有可视化")
    
    return parser.parse_args()


def load_training_config(run_dir):
    """从训练输出目录加载配置"""
    config_path = os.path.join(run_dir, "config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    import json
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # 提取实际的训练配置
    if "config" in config_data:
        return config_data["config"]
    else:
        return config_data


def validate_run_directory(run_dir):
    """验证运行目录的完整性"""
    required_items = [
        "config.json",
        "checkpoints",
        "visualizations"
    ]
    
    missing = []
    for item in required_items:
        item_path = os.path.join(run_dir, item)
        if not os.path.exists(item_path):
            missing.append(item)
    
    if missing:
        print(f"WARNING: Missing items in run directory: {missing}")
        # 创建缺失的目录
        for item in missing:
            if item == "visualizations":
                os.makedirs(os.path.join(run_dir, item), exist_ok=True)
                print(f"Created missing directory: {item}")
    
    return True


def find_best_model(run_dir, model_type):
    """查找最佳模型文件"""
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    
    # 优先查找最佳模型
    best_model_patterns = [
        f"{model_type}_best.pth",
        "best.pth",
        "*_best.pth"
    ]
    
    import glob
    for pattern in best_model_patterns:
        matches = glob.glob(os.path.join(checkpoints_dir, pattern))
        if matches:
            return matches[0]
    
    # 如果没有最佳模型，查找最新的checkpoint
    all_checkpoints = glob.glob(os.path.join(checkpoints_dir, "*.pth"))
    if all_checkpoints:
        # 按修改时间排序，返回最新的
        latest = max(all_checkpoints, key=os.path.getmtime)
        print(f"WARNING: Best model not found, using latest checkpoint: {os.path.basename(latest)}")
        return latest
    
    raise FileNotFoundError(f"No model checkpoints found in {checkpoints_dir}")


def recreate_output_manager(run_dir):
    """从现有运行目录重建OutputManager"""
    # 解析目录名获取模型类型和时间戳
    dir_name = os.path.basename(run_dir)
    parts = dir_name.split('_')
    
    if len(parts) >= 2:
        # 假设格式为 model_type_timestamp
        timestamp = parts[-1]
        model_type = '_'.join(parts[:-1])
    else:
        model_type = "unknown"
        timestamp = "unknown"
    
    # 创建一个特殊的OutputManager实例，指向现有目录
    class ExistingOutputManager(OutputManager):
        def __init__(self, existing_run_dir):
            self.run_dir = existing_run_dir
            self.model_type = model_type
            self.timestamp = timestamp
            self.output_dir = os.path.dirname(existing_run_dir)
    
    return ExistingOutputManager(run_dir)


def main():
    args = parse_args()
    
    print("=" * 80)
    print("POST-TRAINING VISUALIZATION SCRIPT")
    print("=" * 80)
    
    # 验证运行目录
    if not os.path.exists(args.run_dir):
        raise FileNotFoundError(f"Run directory not found: {args.run_dir}")
    
    validate_run_directory(args.run_dir)
    print(f"✓ Using run directory: {args.run_dir}")
    
    # 加载训练配置
    try:
        training_config = load_training_config(args.run_dir)
        print(f"✓ Loaded training configuration")
    except Exception as e:
        print(f"ERROR: Failed to load training config: {e}")
        return
    
    # 合并配置 - 命令行参数优先，否则使用训练时的配置
    model = args.model or training_config.get("model", "unet_min")
    binary = args.binary if args.binary is not None else training_config.get("binary", False)
    num_classes = args.num_classes or training_config.get("num_classes", 10)
    img_size = args.img_size or training_config.get("img_size", 512)
    val_ratio = args.val_ratio or training_config.get("val_ratio", 0.2)
    stage = training_config.get("stage", "offline")
    
    print(f"✓ Configuration: model={model}, binary={binary}, num_classes={num_classes}, img_size={img_size}")
    
    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Using device: {device}")
    
    # 重建数据集（使用与训练时相同的分割）
    print("-- Recreating Dataset --")
    is_multiclass = (not binary) and (num_classes >= 2)
    full_dataset = SegDatasetMin(
        args.data_root, dtype="train", img_size=img_size,
        return_multiclass=is_multiclass    
    )
    
    # 使用相同的随机种子分割数据集
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    seed = 42  # 与训练时使用相同的种子
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, 
        num_workers=0, pin_memory=True
    )
    print(f"✓ Dataset recreated: Train={len(train_ds)}, Val={len(val_ds)}")
    
    # 重建模型
    print("-- Recreating Model --")
    if binary:
        model_instance = build_model(model, num_classes=1, in_ch=3, stage=stage).to(device)
    else:
        model_instance = build_model(model, num_classes=num_classes, in_ch=3, stage=stage).to(device)
    
    # 查找并加载最佳模型
    try:
        model_path = find_best_model(args.run_dir, model)
        checkpoint = torch.load(model_path, map_location=device)
        model_instance.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model from: {os.path.basename(model_path)}")
        
        # 显示模型信息
        if 'epoch' in checkpoint:
            print(f"  Model epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            if binary:
                print(f"  Model metrics: Loss={metrics.get('val_loss', 'N/A'):.4f}, "
                      f"IoU={metrics.get('iou', 'N/A'):.4f}, Dice={metrics.get('dice', 'N/A'):.4f}")
            else:
                print(f"  Model metrics: Loss={metrics.get('val_loss', 'N/A'):.4f}, "
                      f"mIoU={metrics.get('miou', 'N/A'):.4f}, mDice={metrics.get('mdice', 'N/A'):.4f}")
                
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return
    
    # 重建OutputManager
    output_mgr = recreate_output_manager(args.run_dir)
    viz_dir = output_mgr.get_vis_dir()
    
    # 执行可视化
    print("-- Generating Visualizations --")
    visualizer = Visualizer()
    
    try:
        # 保存对比预测结果
        print("  Generating comparison predictions...")
        visualizer.save_comparison_predictions(
            model_instance, val_loader, viz_dir, 
            max_samples=args.viz_samples, device=device
        )
        
        # 保存基础预测结果
        print("  Generating basic predictions...")
        visualizer.save_basic_predictions(
            model_instance, val_loader, viz_dir, 
            max_samples=args.viz_samples, device=device
        )
        
        print(f"✓ Visualizations saved to: {viz_dir}")
        
    except Exception as e:
        print(f"ERROR during visualization: {e}")
        import traceback
        traceback.print_exc()
    
    # 生成训练曲线（如果有指标历史）
    try:
        print("-- Generating Training Curves --")
        metrics_csv_path = os.path.join(args.run_dir, "metrics.csv")
        
        if os.path.exists(metrics_csv_path):
            import pandas as pd
            df = pd.read_csv(metrics_csv_path)
            metrics_history = df.to_dict('records')
            
            curves_path = output_mgr.get_viz_path("training_curves.png")
            visualizer.plot_metrics_curves(metrics_history, curves_path)
            print(f"✓ Training curves saved to: {os.path.basename(curves_path)}")
        else:
            print("  No metrics.csv found, skipping training curves")
            
    except Exception as e:
        print(f"WARNING: Failed to generate training curves: {e}")
    
    print("=" * 80)
    print("POST-TRAINING PROCESSING COMPLETED")
    print(f"All outputs saved to: {args.run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
