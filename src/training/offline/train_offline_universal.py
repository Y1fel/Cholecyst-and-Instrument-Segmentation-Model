# src/training/train_offline_universal.py
"""
通用训练模板 - 集成监控、可视化、评估功能
基于train_offline_min改进，适用于各种模型的训练
"""

# $env:PYTHONPATH="F:\Documents\Courses\CIS\Cholecyst-and-Instrument-Segmentation-Model"

import os, argparse, yaml, torch, sys, json
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# 导入通用模块
from src.eval.evaluator import Evaluator
from src.viz.visualizer import Visualizer
from src.common.output_manager import OutputManager
from src.common.train_monitor import TrainMonitor

# 示例模型导入 - 根据实际模型替换
from src.dataio.datasets.seg_dataset_min import SegDatasetMin
from src.models.baseline.unet_min import UNetMin

from src.models.model_zoo import build_model
from src.common.constants import compose_mapping

# 蒸馏相关导入
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from utils.class_distillation import DistillationLoss
    from src.viz.distillation_visualizer import DistillationVisualizer
    DISTILLATION_AVAILABLE = True
except ImportError:
    DISTILLATION_AVAILABLE = False
    print("WARNING: Distillation module not available")

# 高级损失函数导入
try:
    from utils.composite_losses import (
        CombinedLoss, DiceLoss, FocalLoss, LabelSmoothingCrossEntropy,
        compute_auto_class_weights, create_loss_function
    )
    COMPOSITE_LOSSES_AVAILABLE = True
except ImportError:
    COMPOSITE_LOSSES_AVAILABLE = False
    print("WARNING: Composite losses module not available")

# process arguments
def parse_args():
    """参数配置 - 可根据不同模型需求调整"""
    p = argparse.ArgumentParser("Offline Universal Trainer")
    
    # 基础训练参数
    p.add_argument("--config", "--cfg", type=str, default=None, help="Optional YAML config file path")
    p.add_argument("--data_root", type=str, required=True, help="Dataset root path")
    # p.add_argument("--model_type", type=str, default="universal", help="Model type identifier")
    
    # 数据参数
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=0)
    
    # 训练参数
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    # p.add_argument("--num_classes", type=int, default=2)
    
    # 监控和输出参数
    p.add_argument("--monitor_interval", type=int, default=10, help="Progress update interval (batches)")
    p.add_argument("--enable_gpu_monitor", action='store_true', default=True, help="Enable GPU monitoring")
    p.add_argument("--save_viz", action='store_true', help="Save visualizations")
    p.add_argument("--viz_samples", type=int, default=50, help="Number of visualization samples")
    p.add_argument("--exclude_background_metrics", action='store_true', default=False, help="Exclude background class when computing aggregate metrics (mIoU/mDice/mAcc)")
    
    # 调试和高级选项
    p.add_argument("--debug", action='store_true', help="Enable debug mode")
    p.add_argument("--save_best_only", action='store_true', default=True, help="Only save best checkpoints")

    # 任务定义
    p.add_argument("--binary", action="store_true",
                   help="二分类（胆囊+器械=前景=1）。若关闭则按多类训练。")
     # 灵活分类配置
    p.add_argument("--classification_scheme", type=str, default=None,
                   choices=["binary", "3class_org", "3class_balanced", "5class", "detailed", "custom"],
                   help="分类方案：binary(2类), 3class(3类), 3class_balanced(3类平衡版), 5class(5类), detailed(13类), custom(自定义)")
    
    p.add_argument("--target_classes", nargs="+", default=None,
                   help="指定目标类别列表，例如：--target_classes background instrument target_organ")
    
    p.add_argument("--custom_mapping_file", type=str, default=None,
                   help="自定义映射JSON文件路径")
    
    p.add_argument("--num_classes", type=int, default=10,
                   help="多类时>=2；--binary 生效时忽略此项。watershed模式建议使用10。")
    
    # 模型插拔
    p.add_argument("--model", type=str, default="unet_min",
                   choices=["unet_min", "unet_plus_plus", "deeplabv3_plus", "hrnet", "mobile_unet", "adaptive_unet"])
    
    # 兼容 OutputManager 的模型类型标记（用于run目录命名）
    p.add_argument("--model_type", type=str, default=None,
                   help="若不指定，将自动使用 --model 的值。")
    
    # 知识蒸馏参数
    p.add_argument("--enable_distillation", action="store_true",
                   help="启用知识蒸馏训练模式")
    p.add_argument("--teacher_model", type=str, default="unet_plus_plus",
                   choices=["unet_min", "unet_plus_plus", "deeplabv3_plus", "hrnet"],
                   help="Teacher模型架构")
    p.add_argument("--teacher_checkpoint", type=str, default=None,
                   help="Teacher模型预训练权重路径")
    p.add_argument("--student_model", type=str, default="mobile_unet", 
                   choices=["mobile_unet", "adaptive_unet", "unet_min"],
                   help="Student模型架构")
    p.add_argument("--distill_temperature", type=float, default=4.0,
                   help="蒸馏温度参数")
    p.add_argument("--distill_alpha", type=float, default=0.7,
                   help="蒸馏损失权重")
    p.add_argument("--distill_beta", type=float, default=0.3,
                   help="任务损失权重")
    p.add_argument("--distill_feature_weight", type=float, default=0.1,
                   help="特征蒸馏损失权重")
    
    # 训练阶段选择
    p.add_argument("--stage", type=str, default="auto",
                   choices=["offline", "online", "auto"], help="模型训练阶段")
    
    # 优化器选择
    p.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adam", "adamw", "sgd", "rmsprop"], help="Optimizer type")
    # SGD动量
    p.add_argument("--sgd_momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    
    # 学习率调度
    p.add_argument("--scheduler", type=str, default="cosine",
                   choices=["none", "step", "cosine", "plateau"], help="Learning rate scheduler type")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")

    # 数据增强
    p.add_argument("--augment", action='store_true', default=True, help="Enable data augmentation")
    p.add_argument("--flip_prob", type=float, default=0.5, help="Horizontal flip probability")
    p.add_argument("--rotation_degree", type=int, default=15, help="random rotation range")

    # FOV处理
    p.add_argument("--apply_fov_mask", action='store_true', default=False, 
                   help="Apply FOV (Field of View) mask to remove black border regions")

    # 验证和保存
    p.add_argument("--val_interval", type=int, default=1, help="Validation interval (epochs)")
    p.add_argument("--save_interval", type=int, default=5, help="Checkpoint save interval (epochs)")
    p.add_argument("--loss_threshold", type=float, default=0.02, 
                   help="Loss improvement threshold for hybrid evaluation (default: 0.02)")
    p.add_argument("--loss_degradation_threshold", type=float, default=0.05,
                   help="Loss degradation threshold - if loss degrades more than this, ignore mIoU improvements (default: 0.05)")

    # early stopping
    p.add_argument("--early_stopping", action='store_true', help="Enable early stopping")
    p.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    p.add_argument("--early_stopping_metric", type=str, default="loss", 
                   choices=["loss", "miou"], help="Metric for early stopping: loss (minimize) or miou (maximize)")

    # Advanced Loss Functions
    p.add_argument("--loss_type", type=str, default="ce", 
                   choices=["ce", "focal", "dice", "combined", "label_smoothing"],
                   help="Loss function type")
    p.add_argument("--dice_weight", type=float, default=0.0, 
                   help="Dice loss weight in combined loss (0.0 = pure CE, 0.5 = balanced)")
    p.add_argument("--use_focal_loss", action='store_true', 
                   help="Use Focal Loss to handle class imbalance")
    p.add_argument("--focal_alpha", type=float, default=1.0, 
                   help="Focal loss alpha parameter (class balancing)")
    p.add_argument("--focal_gamma", type=float, default=2.0, 
                   help="Focal loss gamma parameter (focusing)")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)")
    p.add_argument("--auto_class_weights", action='store_true',
                   help="Automatically compute class weights from dataset")
    p.add_argument("--class_weight_sample_ratio", type=float, default=0.1,
                   help="Sampling ratio for computing class weights (0.1 = 10% of dataset)")

    p.add_argument("--mode", choices=["standard", "kd"], default="standard",
                    help="standard: 仅GT训练Teacher；kd: Teacher+Student知识蒸馏")
    p.add_argument("--teacher_ckpt", type=str, default="",
                    help="KD模式下Teacher的权重路径")
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.7,
                    help="总损失: alpha*CE + (1-alpha)*KD")
    
    # KD Evidence Package System
    p.add_argument("--generate_evidence_package", action='store_true',
                   help="Generate comprehensive KD evidence package after training")
    p.add_argument("--evidence_samples", type=int, default=500,
                   help="Number of samples for evidence package evaluation")
    p.add_argument("--evidence_experiment_name", type=str, default=None,
                   help="Experiment name for evidence package (auto-detected if None)")
    
    # 恢复训练参数
    p.add_argument("--resume", type=str, default=None,
                   help="Resume training from checkpoint directory (e.g., outputs/model_20250911_230321/checkpoints)")
    p.add_argument("--resume_from_best", action='store_true', default=False,
                   help="Resume from best checkpoint instead of latest epoch checkpoint")
    p.add_argument("--resume_new_run", action='store_true', default=False,
                   help="When resuming, create a new output run instead of reusing the original directory")
    
    # 恢复时可选的参数覆盖（预留扩展）
    p.add_argument("--resume_lr", type=float, default=None,
                   help="Override learning rate when resuming (optional)")
    p.add_argument("--resume_epochs", type=int, default=None,
                   help="Override total epochs when resuming (optional)")

    # 数据划分策略选择
    p.add_argument("--split_strategy", type=str, default="video_aware",
                   choices=["video_aware", "frame_random", "from_file"],
                   help="Data split strategy: video_aware (防泄漏), frame_random (旧逻辑), from_file (加载指定文件)")
    p.add_argument("--split_file", type=str, default=None,
                   help="Path to split YAML file (for --split_strategy=from_file)")
    p.add_argument("--save_split", action='store_true', default=True,
                   help="Save split result to YAML file for reproducibility")

    return p.parse_args()

# validate arguments
def validate_args(args):
    if getattr(args, "exclude_background_metrics", False):
        print("NOTE: Aggregate metrics will exclude background class (id=0)")
    errors = []    # storage for error messages
    warnings = []  # storage for warning messages
    if getattr(args, "resume_new_run", False) and not args.resume:
        warnings.append("--resume_new_run is set but --resume is missing; option will be ignored")

    # Basic parameter validation
    if args.epochs <= 0:
        errors.append("epochs must be positive")
    if args.batch_size <= 0:
        errors.append("batch_size must be positive")
    if args.lr <= 0:
        errors.append("learning rate must be positive")
    if args.num_classes < 1:
        errors.append("num_classes must be >= 1")
    
    # Data parameter validation
    # 只在非from_file模式下校验val_ratio
    if getattr(args, 'split_strategy', 'video_aware') != "from_file":
        if not (0 < args.val_ratio < 1):
            errors.append("val_ratio must be between 0 and 1 when not using split file")
    if args.img_size < 32:
        warnings.append("img_size < 32 may cause issues")
    
    # Binary classification consistency check
    if args.binary and args.num_classes != 2:
        warnings.append("binary=True but num_classes != 2, using binary mode")
    
    # Monitor parameter validation
    if args.monitor_interval <= 0:
        errors.append("monitor_interval must be positive")
    if args.viz_samples <= 0:
        warnings.append("viz_samples <= 0, visualization disabled")
    
    # Early stopping parameter validation
    if args.early_stopping and args.patience <= 0:
        errors.append("patience must be positive when early_stopping is enabled")

    # Knowledge distillation parameter validation
    if args.enable_distillation and not args.teacher_checkpoint:
        errors.append("KD模式必须提供 --teacher_checkpoint，禁止使用随机Teacher。")
    
    # Split strategy parameter validation
    valid_strategies = ["video_aware", "from_file", "frame_random"]
    if hasattr(args, 'split_strategy') and args.split_strategy not in valid_strategies:
        errors.append(f"split_strategy must be one of {valid_strategies}")
    
    if hasattr(args, 'split_strategy') and args.split_strategy == "from_file":
        if not hasattr(args, 'split_file') or not args.split_file:
            errors.append("split_file is required when split_strategy is 'from_file'")
        elif not os.path.exists(args.split_file):
            errors.append(f"Split file not found: {args.split_file}")
    
    # Output validation results
    if warnings:
        print("WARNING: Parameter Warnings:")
        for w in warnings:
            print(f"   - {w}")
    
    if errors:
        print("ERROR: Parameter Errors:")
        for e in errors:
            print(f"   - {e}")
        raise ValueError("Invalid parameters detected")
    
    print("OK: Parameter validation passed")
    return True

# use config
def load_config(config_path):
    """加载和验证YAML配置文件"""
    if not config_path:
        return {}
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"OK: Loaded config from: {config_path}")
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML config: {e}")
    except Exception as e:
        raise ValueError(f"Error loading config: {e}")

# get default parser value
# combine config with arguments
def merge_config_with_args(args, config):
    if not config:
        return args

    cli_keys = set()
    raw_cli = sys.argv[1:]
    skip_next = False
    for idx, token in enumerate(raw_cli):
        if skip_next:
            skip_next = False
            continue
        if token.startswith('--'):
            key = token[2:]
            if '=' in key:
                key = key.split('=', 1)[0]
            else:
                next_index = idx + 1
                if next_index < len(raw_cli) and not raw_cli[next_index].startswith('--'):
                    skip_next = True
            cli_keys.add(key.replace('-', '_'))

    overridden = []

    for key, value in config.items():
        if not hasattr(args, key):
            continue
        if key in cli_keys:
            continue

        current_value = getattr(args, key)
        if current_value != value:
            setattr(args, key, value)
            overridden.append(f"{key}: {current_value} -> {value}")

    if overridden:
        print('CONFIG OVERRIDES:')
        for item in overridden:
            print(f'   - {item}')
    else:
        print('CONFIG: Using default values, no overrides needed')

    return args

# video-aware train/val split to prevent video-level leakage
def video_aware_train_val_split(dataset_pairs, val_ratio=0.25, seed=42):
    """
    按视频级别划分训练/验证集，避免视频级泄漏
    
    Args:
        dataset_pairs: List of (img_path, mask_path) tuples from dataset
        val_ratio: Validation ratio (0.25 = 25%)
        seed: Random seed for reproducibility
    
    Returns:
        train_pairs, val_pairs: Two lists of (img_path, mask_path) tuples
    """
    import random
    import re
    
    print(f"[VIDEO SPLIT] Performing video-aware train/val split with seed={seed}")
    
    # 1. 提取视频分组信息 - 使用正则表达式更稳定地提取video ID
    video_groups = {}
    for img_path, mask_path in dataset_pairs:
        # 从路径提取视频ID：使用正则匹配 video\d+ 模式
        normalized_path = img_path.replace('\\', '/')
        m = re.search(r"(video\d+)", normalized_path)
        video_id = m.group(1) if m else None
        
        if video_id is None:
            raise ValueError(f"Cannot extract video ID from path: {img_path}")
        
        if video_id not in video_groups:
            video_groups[video_id] = []
        video_groups[video_id].append((img_path, mask_path))
    
    # 2. 统计视频信息
    video_ids = sorted(video_groups.keys())
    total_videos = len(video_ids)
    total_frames = sum(len(frames) for frames in video_groups.values())
    
    print(f"[VIDEO SPLIT] Found {total_videos} videos with {total_frames} total frames")
    for video_id in video_ids[:10]:  # 显示前10个视频的帧数
        frame_count = len(video_groups[video_id])
        print(f"[VIDEO SPLIT]   {video_id}: {frame_count} frames")
    if total_videos > 10:
        print(f"[VIDEO SPLIT]   ... and {total_videos - 10} more videos")
    
    # 3. 按视频随机划分（确保可重复性）
    random.Random(seed).shuffle(video_ids)
    
    val_video_count = max(1, int(total_videos * val_ratio))  # 至少1个视频用于验证
    val_videos = video_ids[:val_video_count]
    train_videos = video_ids[val_video_count:]
    
    # 4. 收集训练和验证样本
    train_pairs = []
    val_pairs = []
    
    for video_id in train_videos:
        train_pairs.extend(video_groups[video_id])
    
    for video_id in val_videos:
        val_pairs.extend(video_groups[video_id])
    
    # 5. 统计和验证结果
    actual_val_ratio = len(val_pairs) / (len(train_pairs) + len(val_pairs))
    
    print(f"[VIDEO SPLIT] ✅ Video-level split completed:")
    print(f"[VIDEO SPLIT]   Training videos: {len(train_videos)} ({train_videos[:5]}{'...' if len(train_videos) > 5 else ''})")
    print(f"[VIDEO SPLIT]   Validation videos: {len(val_videos)} ({val_videos})")
    print(f"[VIDEO SPLIT]   Training frames: {len(train_pairs)}")
    print(f"[VIDEO SPLIT]   Validation frames: {len(val_pairs)}")
    print(f"[VIDEO SPLIT]   Actual val ratio: {actual_val_ratio:.3f} (target: {val_ratio:.3f})")
    
    # 6. 验证无视频重叠
    train_video_set = set(train_videos) 
    val_video_set = set(val_videos)
    overlap = train_video_set & val_video_set
    
    if overlap:
        raise ValueError(f"Video overlap detected: {overlap}")
    else:
        print(f"[VIDEO SPLIT] ✅ No video overlap - split is valid!")
    
    # 7. 保存分割结果用于复现和归档
    import yaml
    import os
    
    os.makedirs("splits", exist_ok=True)
    split_record = {
        "train": [img_path for img_path, _ in train_pairs],
        "val": [img_path for img_path, _ in val_pairs],
        "train_videos": sorted(train_videos),
        "val_videos": sorted(val_videos),
        "metadata": {
            "total_videos": total_videos,
            "total_frames": total_frames,
            "train_frames": len(train_pairs),
            "val_frames": len(val_pairs),
            "target_val_ratio": val_ratio,
            "actual_val_ratio": actual_val_ratio,
            "seed": seed,
            "split_method": "video_aware"
        }
    }
    
    split_file = "splits/seg8k_video_split.yaml"
    with open(split_file, "w") as f:
        yaml.safe_dump(split_record, f, sort_keys=False)
    print(f"[VIDEO SPLIT] ✅ Split record saved to: {split_file}")
    
    return train_pairs, val_pairs

def load_split_from_file(split_file, dataset_pairs):
    """
    从YAML文件加载预定义的训练/验证划分
    
    Args:
        split_file: YAML文件路径
        dataset_pairs: 原始数据集pairs，用于匹配路径
    
    Returns:
        train_pairs, val_pairs: 训练和验证的(img_path, mask_path)列表
    """
    import yaml
    
    print(f"[SPLIT FROM FILE] Loading split from: {split_file}")
    
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    with open(split_file, 'r') as f:
        split_data = yaml.safe_load(f)
    
    # 路径归一化函数，提高匹配稳定性
    def normalize_path(path):
        return os.path.normpath(path).replace("\\", "/")
    
    train_img_paths = {normalize_path(p) for p in split_data['train']}
    val_img_paths = {normalize_path(p) for p in split_data['val']}
    
    # 根据图像路径匹配对应的pairs
    train_pairs = []
    val_pairs = []
    unmatched_count = 0
    
    for img_path, mask_path in dataset_pairs:
        normalized_img_path = normalize_path(img_path)
        if normalized_img_path in train_img_paths:
            train_pairs.append((img_path, mask_path))
        elif normalized_img_path in val_img_paths:
            val_pairs.append((img_path, mask_path))
        else:
            unmatched_count += 1
            if unmatched_count <= 5:  # 只显示前5个未匹配的
                print(f"[WARNING] Path not found in split file: {img_path}")
            elif unmatched_count == 6:
                print(f"[WARNING] ... and {unmatched_count - 5} more unmatched samples")
    
    print(f"[SPLIT FROM FILE] ✅ Loaded split:")
    print(f"[SPLIT FROM FILE]   Train samples: {len(train_pairs)}")
    print(f"[SPLIT FROM FILE]   Val samples: {len(val_pairs)}")
    
    # 显示元数据（如果有）
    if 'metadata' in split_data:
        metadata = split_data['metadata']
        print(f"[SPLIT FROM FILE]   Split method: {metadata.get('split_method', 'unknown')}")
        print(f"[SPLIT FROM FILE]   Original seed: {metadata.get('seed', 'unknown')}")
        if 'train_videos' in split_data and 'val_videos' in split_data:
            print(f"[SPLIT FROM FILE]   Train videos: {len(split_data['train_videos'])} ({split_data['train_videos'][:3]}{'...' if len(split_data['train_videos']) > 3 else ''})")
            print(f"[SPLIT FROM FILE]   Val videos: {len(split_data['val_videos'])} ({split_data['val_videos']})")
    
    return train_pairs, val_pairs

def frame_random_split(dataset_pairs, val_ratio=0.25, seed=42):
    """
    传统的帧级随机划分（可能存在视频泄漏，仅用于复现旧结果）
    
    Args:
        dataset_pairs: List of (img_path, mask_path) tuples
        val_ratio: Validation ratio
        seed: Random seed
    
    Returns:
        train_pairs, val_pairs: 训练和验证的pairs列表
    """
    import torch
    
    print(f"[FRAME RANDOM SPLIT] ⚠️  Using frame-level random split (may have video leakage)")
    print(f"[FRAME RANDOM SPLIT] Total samples: {len(dataset_pairs)}, val_ratio: {val_ratio}, seed: {seed}")
    
    # 创建临时索引用于random_split
    total_size = len(dataset_pairs)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    indices = list(range(total_size))
    g = torch.Generator().manual_seed(seed)
    train_indices, val_indices = torch.utils.data.random_split(indices, [train_size, val_size], generator=g)
    
    train_pairs = [dataset_pairs[i] for i in train_indices.indices]
    val_pairs = [dataset_pairs[i] for i in val_indices.indices]
    
    print(f"[FRAME RANDOM SPLIT] ✅ Split completed:")
    print(f"[FRAME RANDOM SPLIT]   Train samples: {len(train_pairs)}")
    print(f"[FRAME RANDOM SPLIT]   Val samples: {len(val_pairs)}")
    
    return train_pairs, val_pairs

def save_video_aware_split(train_pairs, val_pairs, args, output_mgr):
    """
    保存视频级分割结果到YAML文件
    
    Args:
        train_pairs: 训练样本对列表
        val_pairs: 验证样本对列表  
        args: 命令行参数
        output_mgr: 输出管理器
    """
    import yaml
    from datetime import datetime
    import re
    
    try:
        # 收集训练和验证的图像路径
        train_img_paths = [pair[0] for pair in train_pairs]
        val_img_paths = [pair[0] for pair in val_pairs]
        
        # 提取视频信息
        video_pattern = re.compile(r"(video\d+)")
        train_videos = set()
        val_videos = set()
        
        for img_path in train_img_paths:
            match = video_pattern.search(img_path)
            if match:
                train_videos.add(match.group(1))
        
        for img_path in val_img_paths:
            match = video_pattern.search(img_path)
            if match:
                val_videos.add(match.group(1))
        
        # 构建保存数据
        split_data = {
            'train': train_img_paths,
            'val': val_img_paths,
            'train_videos': sorted(list(train_videos)),
            'val_videos': sorted(list(val_videos)),
            'metadata': {
                'split_method': 'video_aware',
                'seed': 42,
                'val_ratio': args.val_ratio,
                'total_samples': len(train_pairs) + len(val_pairs),
                'train_samples': len(train_pairs),
                'val_samples': len(val_pairs),
                'train_video_count': len(train_videos),
                'val_video_count': len(val_videos),
                'created_at': datetime.now().isoformat(),
                'experiment_name': getattr(args, 'evidence_experiment_name', 'unnamed'),
                'model': args.model,
                'img_size': args.img_size
            }
        }
        
        # 保存到outputs目录下的splits子目录
        splits_dir = os.path.join(output_mgr.get_run_dir(), "splits")
        os.makedirs(splits_dir, exist_ok=True)
        
        split_filename = f"video_aware_split_{getattr(args, 'evidence_experiment_name', 'exp')}.yaml"
        split_path = os.path.join(splits_dir, split_filename)
        
        with open(split_path, 'w') as f:
            yaml.dump(split_data, f, default_flow_style=False, indent=2)
        
        print(f"💾 Video-aware split saved to: {split_path}")
        print(f"   📁 {len(train_videos)} train videos, {len(val_videos)} val videos")
        
    except Exception as e:
        print(f"⚠️ Failed to save split: {e}")

# compute class weights
def compute_class_weights(dataset, num_classes, ignore_index=255):
    """计算类别权重以处理类不平衡"""
    print("Computing class weights from dataset...")
    
    class_counts = np.zeros(num_classes)
    total_pixels = 0
    
    # 统计前100个样本的类别分布（避免过慢）
    sample_size = min(100, len(dataset))
    for i in range(sample_size):
        _, mask = dataset[i]
        mask_np = mask.numpy() if hasattr(mask, 'numpy') else np.array(mask)
        
        for class_id in range(num_classes):
            class_counts[class_id] += np.sum(mask_np == class_id)
        total_pixels += np.sum(mask_np != ignore_index)
    
    # 处理没有样本的类别
    class_counts = np.maximum(class_counts, 1)  # 避免0计数
    
    # 计算权重（倒数 + 平滑）
    if total_pixels > 0:
        class_weights = total_pixels / (num_classes * class_counts)
        class_weights = class_weights / np.sum(class_weights) * num_classes  # 归一化
    else:
        # 如果没有有效像素，使用均匀权重
        class_weights = np.ones(num_classes)
    
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")
    
    return torch.FloatTensor(class_weights)

# build loss function for segmentation class
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, 
                                 ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


def create_advanced_loss_function(args, dataset, device):
    """
    创建高级损失函数 - 支持多种损失类型和自动权重
    """
    print(f"=== Creating Advanced Loss Function ===")
    
    # 处理类别权重
    class_weights = None
    if args.auto_class_weights and COMPOSITE_LOSSES_AVAILABLE:
        print("Computing automatic class weights...")
        class_weights = compute_auto_class_weights(
            dataset, 
            args.num_classes, 
            ignore_index=255,
            sample_ratio=args.class_weight_sample_ratio
        ).to(device)
    elif not args.auto_class_weights:
        # 使用原有的类别权重计算方法作为fallback
        print("Computing class weights using legacy method...")
        class_weights = compute_class_weights(dataset, args.num_classes, ignore_index=255).to(device)
    
    # 确定损失函数类型
    loss_type = "ce"  # 默认
    if args.dice_weight > 0:
        loss_type = "combined"
    elif args.use_focal_loss:
        loss_type = "focal"  
    elif args.label_smoothing > 0:
        loss_type = "label_smoothing"
    elif hasattr(args, 'loss_type'):
        loss_type = args.loss_type
    
    print(f"Using loss type: {loss_type}")
    if class_weights is not None:
        print(f"Class weights: {class_weights}")
    
    # 创建损失函数
    if COMPOSITE_LOSSES_AVAILABLE and loss_type != "ce":
        # 使用高级损失函数
        loss_config = {
            'loss_type': loss_type,
            'dice_weight': getattr(args, 'dice_weight', 0.0),
            'focal_alpha': getattr(args, 'focal_alpha', 1.0),
            'focal_gamma': getattr(args, 'focal_gamma', 2.0),
            'label_smoothing': getattr(args, 'label_smoothing', 0.0),
            'ignore_index': 255,
            'class_weights': class_weights.cpu().numpy().tolist() if class_weights is not None else None,
            'auto_class_weights': False  # 已经计算过了
        }
        
        if loss_type == "combined":
            print(f"Using Combined Loss (CE + Dice) with dice_weight={args.dice_weight}")
            criterion = CombinedLoss(
                dice_weight=args.dice_weight,
                class_weights=class_weights,
                ignore_index=255,
                use_focal=args.use_focal_loss,
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma
            )
        elif loss_type == "focal":
            print(f"Using Focal Loss with alpha={args.focal_alpha}, gamma={args.focal_gamma}")
            criterion = FocalLoss(
                alpha=args.focal_alpha,
                gamma=args.focal_gamma,
                ignore_index=255,
                class_weights=class_weights
            )
        elif loss_type == "label_smoothing":
            print(f"Using Label Smoothing CrossEntropy with smoothing={args.label_smoothing}")
            criterion = LabelSmoothingCrossEntropy(
                smoothing=args.label_smoothing,
                class_weights=class_weights,
                ignore_index=255
            )
        elif loss_type == "dice":
            print("Using pure Dice Loss")
            criterion = DiceLoss(ignore_index=255)
        else:
            # Fallback to standard CE
            print("Fallback to standard CrossEntropy Loss")
            criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    else:
        # 使用标准损失函数或原有的Focal Loss
        if args.use_focal_loss:
            print(f"Using legacy Focal Loss with alpha={args.focal_alpha}, gamma={args.focal_gamma}")
            criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, ignore_index=255)
        else:
            print("Using standard CrossEntropy Loss")
            criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    
    print(f"Loss function created successfully: {type(criterion).__name__}")
    return criterion
    
# build optimizer
def create_optimizer(model, args):
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.sgd_momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    print(f"OK: Created {args.optimizer} optimizer with lr={args.lr}, weight_decay={args.weight_decay}")
    return optimizer

# build scheduler
def create_scheduler(optimizer, args):
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

    if args.scheduler == "none":
        return None
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs//3, gamma=0.5)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif args.scheduler == "cosine_warmup":
        warmup = max(2, int(0.1 * args.epochs))  # 前 10% epoch 线性升温
        sched1 = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
        sched2 = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup)
        return SequentialLR(optimizer, schedulers=[sched1, sched2], milestones=[warmup])
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")
    
    print(f"OK: Created {args.scheduler} scheduler")
    return scheduler

# one epoch training
def train_one_epoch(
    model, loader, criterion, optimizer, device, monitor, epoch_index, args, teacher_model=None
):
    model.train()
    if teacher_model is not None:
        teacher_model.eval()  # Teacher保持评估模式

    running_loss = 0.0
    running_distill_loss = 0.0
    running_task_loss = 0.0
    total = len(loader)

    for step, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True) # [path, 3, H, W]
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        

        # Forward pass: images -> logits
        if args.enable_distillation and teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            
            student_logits = model(images)

            # KD 通道对齐断言：防止 Teacher/Student 通道不一致导致蒸馏错误
            assert teacher_logits.shape[1] == student_logits.shape[1] == args.num_classes, \
                f"Teacher/Student/num_classes不一致: Teacher={teacher_logits.shape} vs Student={student_logits.shape} vs num_classes={args.num_classes}"

            # 使用蒸馏损失
            loss_dict = criterion(student_logits, teacher_logits, masks)
            loss = loss_dict['total_loss']

            running_distill_loss += loss_dict['distill_loss'].item() * images.size(0)
            running_task_loss += loss_dict['task_loss'].item() * images.size(0)

        else:
            logits = model(images)
            if args.binary:
                target = masks.float()
                if target.dim() == 3:
                    target = target.unsqueeze(1)
                loss = criterion(logits, target)
            else:
                targets = masks.long()
                loss = criterion(logits, targets)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # monitor progress
        if (step % args.monitor_interval) == 0:
            avg_loss = running_loss / max(1, (step + 1) * args.batch_size)

            if args.enable_distillation and teacher_model is not None:
                avg_distill = running_distill_loss / max(1, (step + 1) * args.batch_size)
                avg_task = running_task_loss / max(1, (step + 1) * args.batch_size)

                monitor.print_progress(
                    epoch_index + 1, args.epochs,
                    step + 1, total,
                    {"total": avg_loss, "distill": avg_distill, "task": avg_task},
                    refresh=True
                )
            else:
                monitor.print_progress(
                    epoch_index + 1, args.epochs,
                    step + 1, total,
                    {"loss": avg_loss},
                    refresh=True
                )

        # Force output every 50 batches to avoid long periods without output
        # if step > 0 and (step % 50) == 0:
        #     avg = running_loss / max(1, (step + 1) * args.batch_size)
        #     print(f"\n[Checkpoint] Epoch {epoch_index + 1}/{args.epochs} Batch {step + 1}/{total} | Loss: {avg:.4f}")
        #     sys.stdout.flush()

    # 返回损失信息
    dataset_size = len(loader.dataset) if hasattr(loader, 'dataset') else (total * args.batch_size)
    avg_total_loss = running_loss / dataset_size
    
    if args.enable_distillation and teacher_model is not None:
        avg_distill_loss = running_distill_loss / dataset_size
        avg_task_loss = running_task_loss / dataset_size
        return {
            'total_loss': avg_total_loss,
            'distill_loss': avg_distill_loss,
            'task_loss': avg_task_loss
        }
    else:
        return avg_total_loss

# Validation
@torch.inference_mode()
def validate(model, loader, criterion, device, args):
    if args.enable_distillation:
        if args.binary:
            val_criterion = nn.BCEWithLogitsLoss()
        else:
            val_criterion = nn.CrossEntropyLoss(ignore_index=255)
    else:
        val_criterion = criterion

    exclude_metric_classes = [0] if getattr(args, "exclude_background_metrics", False) else None

    if args.binary:
        evaluator = Evaluator(device=device, threshold=0.5)
        return evaluator.evaluate(model, loader, val_criterion)
    else:
        evaluator = Evaluator(device=device)
        return evaluator.evaluate_multiclass(
            model, loader, val_criterion,
            num_classes = args.num_classes,
            ignore_index = 255,
            exclude_metric_classes = exclude_metric_classes
        )

# Main function
def generate_kd_evidence_package(args, teacher_model, student_model, val_loader, output_mgr, device):
    """
    Generate comprehensive KD evidence package for analysis
    
    This function creates the unified evidence package that proves KD effectiveness:
    1. Full metrics evaluation (standard + calibration + boundary)
    2. Unified CSV export for comparison tables
    3. KD-specific visualizations (teacher-student analysis, reliability diagrams)
    4. Four-panel KD analysis for presentation
    """
    print("\n" + "="*60)
    print("🔬 GENERATING KD EVIDENCE PACKAGE")
    print("="*60)
    
    # Determine experiment type from config or model names
    experiment_name = args.evidence_experiment_name
    if experiment_name is None:
        if hasattr(args, 'teacher_model') and hasattr(args, 'student_model'):
            experiment_name = f"KD_{args.teacher_model}_to_{args.student_model}"
        else:
            experiment_name = "KD_Experiment"
    
    # Initialize components
    evaluator = Evaluator(device=device)
    distill_visualizer = DistillationVisualizer(output_mgr.get_vis_dir(), device, classification_scheme=args.classification_scheme)
    visualizer = Visualizer(classification_scheme=args.classification_scheme)
    exclude_metric_classes = [0] if getattr(args, "exclude_background_metrics", False) else None
    
    print(f"📊 Experiment: {experiment_name}")
    print(f"📁 Output Directory: {output_mgr.get_run_dir()}")
    
    # Step 1: Full metrics evaluation for both models
    print("\n🔍 Step 1: Comprehensive Metrics Evaluation")
    print("-" * 40)
    
    # Teacher evaluation
    print("   📚 Evaluating Teacher Model...")
    teacher_metrics = evaluator.evaluate_with_full_metrics(
        teacher_model, val_loader, 
        num_classes=args.num_classes,
        binary_mode=args.binary,
        max_samples=args.evidence_samples,
        exclude_metric_classes=exclude_metric_classes
    )
    print(f"      ✅ Teacher - IoU: {teacher_metrics.get('iou', teacher_metrics.get('miou', 0)):.4f}")
    
    # Student evaluation  
    print("   🎓 Evaluating Student Model...")
    student_metrics = evaluator.evaluate_with_full_metrics(
        student_model, val_loader,
        num_classes=args.num_classes, 
        binary_mode=args.binary,
        max_samples=args.evidence_samples,
        exclude_metric_classes=exclude_metric_classes
    )
    print(f"      ✅ Student - IoU: {student_metrics.get('iou', student_metrics.get('miou', 0)):.4f}")
    
    # Step 2: Generate unified evidence package
    print("\n📦 Step 2: Generating Unified Evidence Package")
    print("-" * 40)
    
    evidence_data = {
        'teacher_metrics': teacher_metrics,
        'student_metrics': student_metrics,
        'experiment_name': experiment_name,
        'training_config': {
            'teacher_model': args.teacher_model if hasattr(args, 'teacher_model') else 'Unknown',
            'student_model': args.student_model if hasattr(args, 'student_model') else 'Unknown', 
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'distill_temperature': getattr(args, 'distill_temperature', args.temperature),
            'distill_alpha': getattr(args, 'distill_alpha', args.alpha)
        }
    }
    
    # Generate evidence package with distillation visualizer (unified format)
    package_paths = distill_visualizer.generate_unified_kd_evidence_package(
        evidence_data, save_prefix=f"{experiment_name}_evidence"
    )
    
    print(f"   ✅ CSV Summary: {os.path.basename(package_paths['csv_path'])}")
    print(f"   ✅ Performance Plot: {os.path.basename(package_paths['performance_plot'])}")
    print(f"   ✅ Reliability Diagrams: {os.path.basename(package_paths['reliability_diagrams'])}")
    
    # Step 3: Four-panel KD analysis
    print("\n📈 Step 3: Four-Panel KD Analysis")
    print("-" * 40)
    
    try:
        four_panel_path = distill_visualizer.create_kd_four_panel_analysis(
            teacher_model, student_model, val_loader,
            temperature=getattr(args, 'distill_temperature', args.temperature),
            max_samples=min(200, args.evidence_samples),  # Limit for memory
            save_name=f"{experiment_name}_four_panel_analysis.png"
        )
        print(f"   ✅ Four-Panel Analysis: {os.path.basename(four_panel_path)}")
    except Exception as e:
        print(f"   ⚠️  Four-Panel Analysis failed: {str(e)}")
    
    # Step 4: Additional KD-specific analyses
    print("\n🔬 Step 4: KD-Specific Analysis")
    print("-" * 40)
    
    try:
        # Teacher-Student comparison
        comparison_path = distill_visualizer.visualize_prediction_comparison(
            teacher_model, student_model, val_loader,
            num_samples=min(6, args.evidence_samples // 100),
            save_name=f"{experiment_name}_prediction_comparison.png"
        )
        print(f"   ✅ Prediction Comparison: {os.path.basename(comparison_path)}")
        
        # Knowledge transfer analysis
        kd_stats = distill_visualizer.visualize_knowledge_transfer(
            teacher_model, student_model, val_loader,
            temperature=getattr(args, 'distill_temperature', args.temperature),
            max_samples=min(500, args.evidence_samples),
            save_name=f"{experiment_name}_knowledge_transfer.png"
        )
        print(f"   ✅ Knowledge Transfer Analysis: Generated")
        
    except Exception as e:
        print(f"   ⚠️  Additional analysis failed: {str(e)}")
    
    print("\n" + "="*60)
    print("✅ KD EVIDENCE PACKAGE GENERATION COMPLETE")
    print(f"📁 All files saved in: {output_mgr.get_vis_dir()}")
    print("="*60)
    
    return package_paths

# setup resume training
def setup_resume_training(args):
    """
    设置恢复训练
    Returns:
        (resume_manager, resume_info, start_epoch) 或 (None, None, 0)
    """
    if not args.resume:
        return None, None, 0
    
    from src.training.resume_manager import ResumeManager
    
    print("=== RESUME MODE ACTIVATED ===")
    resume_manager = ResumeManager(args.resume)
    resume_info = resume_manager.get_resume_info(args.resume_from_best)
    
    # 从checkpoint信息中获取起始epoch
    start_epoch = resume_info['checkpoint_info']['epoch']
    
    # 可选：使用原始配置覆盖当前args（预留功能）
    if resume_info['original_config'] and hasattr(args, 'use_original_config'):
        if args.use_original_config:
            original_args = resume_info['original_config']
            for key, value in original_args.items():
                if hasattr(args, key) and getattr(args, key) is None:
                    setattr(args, key, value)
    
    return resume_manager, resume_info, start_epoch

# use resume overrides
def apply_resume_overrides(args):
    """应用恢复时的参数覆盖"""
    if args.resume_lr is not None:
        print(f"Override learning rate: {args.lr} -> {args.resume_lr}")
        args.lr = args.resume_lr
    
    if args.resume_epochs is not None:
        print(f"Override total epochs: {args.epochs} -> {args.resume_epochs}")
        args.epochs = args.resume_epochs

# load resume states
def load_resume_states(model, optimizer, scheduler, resume_info, device):
    """
    加载恢复状态到模型、优化器、调度器
    """
    checkpoint_info = resume_info['checkpoint_info']
    
    # 加载模型状态
    if checkpoint_info['model_state_dict']:
        model.load_state_dict(checkpoint_info['model_state_dict'])
        print(f"✓ Model state loaded from epoch {checkpoint_info['epoch']}")
    
    # 加载优化器状态
    if checkpoint_info['optimizer_state_dict'] and optimizer:
        optimizer.load_state_dict(checkpoint_info['optimizer_state_dict'])
        print(f"✓ Optimizer state loaded")
    
    # 加载调度器状态
    if checkpoint_info['scheduler_state_dict'] and scheduler:
        scheduler.load_state_dict(checkpoint_info['scheduler_state_dict'])
        print(f"✓ Scheduler state loaded")

# main function
def main():
    args = parse_args()

    config = load_config(args.config)
    args = merge_config_with_args(args, config)

    resume_manager, resume_info, start_epoch = setup_resume_training(args)

    if resume_manager:
        apply_resume_overrides(args)

    validate_args(args)

    # Binary/Multiclass strong consistency: fix num_classes when needed
    if args.binary:
        if args.num_classes != 2:
            print(f"[WARN] binary=True but num_classes={args.num_classes} != 2, forcing num_classes to 2")
            args.num_classes = 2

    # Print save strategy description
    print(f"SAVE STRATEGY:")
    print(f"   - Best model: Save when validation improves")
    print(f"   - Checkpoints: Every {args.save_interval} epochs")
    print(f"   - Validation: Every {args.val_interval} epoch(s)")
    if args.early_stopping:
        print(f"   - Early stopping: Enabled (patience={args.patience})")
    else:
        print(f"   - Early stopping: Disabled")

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # train monitor
    monitor = TrainMonitor(enable_gpu_monitor=args.enable_gpu_monitor)
    monitor.start_timing()

    # output manager
    if args.enable_distillation:
        model_tag = f"distill_{args.teacher_model}_to_{args.student_model}"
    else:
        model_tag = args.model if args.model_type is None else args.model_type

    resume_source_run_dir = None
    if resume_manager:
        original_run_dir = resume_info["run_dir"]
        resume_source_run_dir = original_run_dir
        if getattr(args, "resume_new_run", False):
            output_mgr = OutputManager(model_type=model_tag)
            output_mgr.save_config(vars(args))
            print(f"=== RESUME: Starting new output directory: {output_mgr.get_run_dir()} (source: {original_run_dir}) ===")
        else:
            output_mgr = OutputManager(model_type=model_tag, run_dir=original_run_dir)
            print(f"=== RESUME: Using original output directory: {original_run_dir} ===")
    else:
        output_mgr = OutputManager(model_type=model_tag)
        output_mgr.save_config(vars(args))  # save config for fresh training
    # load custom mapping
    custom_mapping = None
    if args.custom_mapping_file:
        with open(args.custom_mapping_file, 'r') as f:
            custom_mapping = json.load(f)
    
    # 使用compose_mapping生成最终的watershed -> target class映射
    class_id_map = compose_mapping(
        classification_scheme=args.classification_scheme,
        custom_mapping=custom_mapping,
        target_classes=args.target_classes
    )

    print(f"[MAPPING] Generated class_id_map with {len(class_id_map)} entries")
    print(f"[MAPPING] Target classes: {list(set(class_id_map.values()))}")

    # Dataloader  
    is_multiclass = (not args.binary) and (args.num_classes >= 2)

    # Dataset configuration
    dataset_config = {
        # "classification_scheme": args.classification_scheme,
        # "custom_mapping": custom_mapping,
        # "target_classes": args.target_classes,
        "class_id_map": class_id_map,  # 直接传入最终映射
        "return_multiclass": is_multiclass, # 保持兼容性
        "apply_fov_mask": args.apply_fov_mask  # FOV mask处理
    }

    full_dataset = SegDatasetMin(
        args.data_root, dtype=args.split, img_size=args.img_size,
        **dataset_config
    )

    # 用数据集真实类数回写 args.num_classes
    # 防止CE/BCE通道数与实际映射不一致导致的全255/全背景/颜色错乱问题
    if not args.binary:
        original_num_classes = args.num_classes
        args.num_classes = getattr(full_dataset, "num_classes", args.num_classes)
        if args.num_classes != original_num_classes:
            print(f"[DATASET] Updated num_classes: {original_num_classes} -> {args.num_classes} (from dataset)")
        else:
            print(f"[DATASET] Confirmed num_classes: {args.num_classes} (matches dataset)")

    # 标签健康检查：快速检测是否所有标签都被映射为255（ignore）
    print("[HEALTH CHECK] Checking label distribution in first 20 samples...")
    from collections import Counter
    valid_counter = Counter()
    sample_size = min(200, len(full_dataset))
    
    # check first N samples' labels
    for i in range(sample_size):
        try:
            _, mask = full_dataset[i]  # 假设 __getitem__ 返回 image, mask
            mask_tensor = mask if torch.is_tensor(mask) else torch.tensor(mask)
            unique_values = torch.unique(mask_tensor)
            valid_values = unique_values[unique_values != 255]  # 排除ignore标签
            valid_counter.update(valid_values.cpu().tolist())
            
            lab = mask_tensor.numpy()
            valid = lab[lab != 255]
            # if valid.size == 0:
            #     print(f"[HEALTH CHECK] sample#{i}: only ignore")
            # else:
            #     u, c = np.unique(valid, return_counts=True)
            #     print(f"[HEALTH CHECK] sample#{i}: {dict(zip(u.tolist(), c.tolist()))}")
        except Exception as e:
            print(f"[HEALTH CHECK] Warning: Failed to check sample {i}: {e}")
            continue
    
    if len(valid_counter) == 0:
        raise ValueError(
            "[FAILED] 所有样本的标签都只有 255（ignore），请检查 class_id_map / 映射流程。\n"
            "可能原因：\n"
            "  1. class_id_map 映射错误，所有原始标签都被映射为255\n"
            "  2. 数据集路径错误或标签文件损坏\n"
            "  3. 映射函数逻辑错误，未正确处理目标类别"
        )
    else:
        valid_classes = sorted(valid_counter.keys())
        total_valid_pixels = sum(valid_counter.values())
        print(f"[PASS] [HEALTH CHECK] 发现有效标签: {valid_classes}")
        print(f"[PASS] [HEALTH CHECK] 标签分布: {dict(valid_counter)} (共 {total_valid_pixels:,} 个有效像素)")
        
        # 额外检查：确保有效类别数与预期匹配
        if not args.binary and len(valid_classes) > args.num_classes:
            print(f"[WARN] [HEALTH CHECK] Warning: 发现 {len(valid_classes)} 个有效类别，但 num_classes={args.num_classes}")

    # 🚨 CRITICAL: Configurable data split strategy to prevent video-level leakage
    print("=" * 60)
    print(f"🎬 APPLYING DATA SPLIT STRATEGY: {args.split_strategy.upper()}")
    print("=" * 60)
    
    # 获取数据集的原始pairs用于划分
    dataset_pairs = full_dataset.pairs  # [(img_path, mask_path), ...]
    val_ratio = args.val_ratio
    seed = 42  # 保持固定种子确保可重复性
    
    # 根据策略选择不同的划分方法
    if args.split_strategy == "from_file":
        # 优先：从指定文件加载分割（用于复现Teacher等）
        if not args.split_file:
            raise ValueError("--split_strategy=from_file requires --split_file argument")
        train_pairs, val_pairs = load_split_from_file(args.split_file, dataset_pairs)
        
    elif args.split_strategy == "video_aware":
        # 推荐：视频级划分避免泄漏
        train_pairs, val_pairs = video_aware_train_val_split(
            dataset_pairs, val_ratio=val_ratio, seed=seed
        )
        
    elif args.split_strategy == "frame_random":
        # 兼容：帧级随机划分（可能有泄漏，仅用于复现旧结果）
        train_pairs, val_pairs = frame_random_split(
            dataset_pairs, val_ratio=val_ratio, seed=seed
        )
        
    else:
        raise ValueError(f"Unknown split_strategy: {args.split_strategy}")
    
    # 创建新的数据集实例用于训练和验证
    # 训练数据集 - 带增强
    train_dataset = SegDatasetMin(
        args.data_root, dtype="train", img_size=args.img_size,
        **dataset_config
    )
    train_dataset.pairs = train_pairs  # 覆盖为训练pairs
    
    # 验证数据集 - 不带增强，显式设置为val模式
    val_dataset_config = dataset_config.copy()
    val_dataset = SegDatasetMin(
        args.data_root, dtype="val", img_size=args.img_size,
        **val_dataset_config
    )
    val_dataset.pairs = val_pairs  # 覆盖为验证pairs
    
    # 显式关闭验证集的数据增强（如果数据集支持）
    if hasattr(val_dataset, 'augment'):
        val_dataset.augment = False
        print(f"[DATASET] Validation augmentation disabled")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print(f"✅ Data Split Complete ({args.split_strategy}):")
    print(f"   Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"   Actual split ratio: {len(val_dataset)/(len(train_dataset)+len(val_dataset)):.3f}")
    
    # 如果是视频级分割，显示视频分布信息
    # 显示视频分布统计（适用于所有分割策略）
    try:
        # 统计训练和验证集中的视频
        import re
        video_pattern = re.compile(r"(video\d+)")
        train_videos = set()
        val_videos = set()
        
        # 从数据集中获取文件路径统计视频
        for i in range(len(train_dataset)):
            # 尝试多种方式获取图像路径
            img_path = None
            if hasattr(train_dataset, 'img_files'):
                img_path = train_dataset.img_files[i]
            elif hasattr(train_dataset, 'all_file_paths'):
                img_path = train_dataset.all_file_paths[i]
            elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'img_files'):
                # 处理Subset情况
                actual_idx = train_dataset.indices[i]
                img_path = train_dataset.dataset.img_files[actual_idx]
            elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'all_file_paths'):
                # 处理Subset情况
                actual_idx = train_dataset.indices[i]
                img_path = train_dataset.dataset.all_file_paths[actual_idx]
            
            if img_path:
                match = video_pattern.search(str(img_path))
                if match:
                    train_videos.add(match.group(1))
        
        for i in range(len(val_dataset)):
            # 尝试多种方式获取图像路径
            img_path = None
            if hasattr(val_dataset, 'img_files'):
                img_path = val_dataset.img_files[i]
            elif hasattr(val_dataset, 'all_file_paths'):
                img_path = val_dataset.all_file_paths[i]
            elif hasattr(val_dataset, 'dataset') and hasattr(val_dataset.dataset, 'img_files'):
                # 处理Subset情况
                actual_idx = val_dataset.indices[i]
                img_path = val_dataset.dataset.img_files[actual_idx]
            elif hasattr(val_dataset, 'dataset') and hasattr(val_dataset.dataset, 'all_file_paths'):
                # 处理Subset情况
                actual_idx = val_dataset.indices[i]
                img_path = val_dataset.dataset.all_file_paths[actual_idx]
            
            if img_path:
                match = video_pattern.search(str(img_path))
                if match:
                    val_videos.add(match.group(1))
        
        if train_videos or val_videos:
            print(f"   📹 Video distribution:")
            if train_videos:
                print(f"      Train videos: {len(train_videos)} ({sorted(list(train_videos))[:3]}{'...' if len(train_videos) > 3 else ''})")
            if val_videos:
                print(f"      Val videos: {len(val_videos)} ({sorted(list(val_videos))})")
            if train_videos and val_videos:
                overlap = train_videos & val_videos
                print(f"      Video overlap: {len(overlap)} {'(should be 0)' if args.split_strategy == 'video_aware' else '(expected for frame-level split)'}")
                if overlap and len(overlap) <= 5:
                    print(f"         Overlapping videos: {sorted(list(overlap))}")
        else:
            print(f"   📹 Video information not available from dataset structure")
    except Exception as e:
        print(f"   ⚠️ Video statistics unavailable: {e}")
    
    # 记录关键配置信息用于实验追踪
    config_summary = {
        "model_arch": args.model,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "augment": args.augment,
        "flip_prob": args.flip_prob if args.augment else None,
        "rotation_degree": args.rotation_degree if args.augment else None,
        "apply_fov_mask": args.apply_fov_mask,
        "classification_scheme": args.classification_scheme,
        "split_strategy": args.split_strategy,
        "split_file": args.split_file if args.split_strategy == "from_file" else None,
        "val_ratio": args.val_ratio,
        "seed": 42,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "exclude_background_metrics": getattr(args, "exclude_background_metrics", False),
        "resume_source_run_dir": resume_source_run_dir,
        "num_classes": args.num_classes
    }
    
    # 蒸馏相关配置
    if args.enable_distillation:
        config_summary.update({
            "distillation": True,
            "teacher_model": args.teacher_model,
            "student_model": args.student_model,
            "distill_temperature": args.distill_temperature,
            "distill_alpha": args.distill_alpha,
            "distill_beta": args.distill_beta
        })
    else:
        config_summary["distillation"] = False
    
    print(f"\n📋 Experiment Configuration Summary:")
    for key, value in config_summary.items():
        if value is not None:
            print(f"   {key}: {value}")
    
    # 保存分割结果到文件（用于重现和分析）
    if getattr(args, 'save_split', True):
        save_video_aware_split(train_pairs, val_pairs, args, output_mgr)
    
    print("=" * 60)

    # Model 和 蒸馏设置
    if args.enable_distillation:
        if not DISTILLATION_AVAILABLE:
            raise ImportError("Distillation requested but class_distillation module not available")

        print(f"=== Knowledge Distillation Mode Enabled ===")
        print(f"Teacher: {args.teacher_model}, Student: {args.student_model}")
        print(f"Temperature: {args.distill_temperature}, Alpha: {args.distill_alpha}, Beta: {args.distill_beta}")
        
        # Build Teacher and Student models  
        teacher_model = build_model(args.teacher_model, num_classes=args.num_classes, in_ch=3, stage="offline").to(device)
        student_model = build_model(args.student_model, num_classes=args.num_classes, in_ch=3, stage="online").to(device)

        # Load pretrained Teacher weights if provided
        if args.teacher_checkpoint:
            print(f"Loading Teacher model weights from: {args.teacher_checkpoint}")
            try:
                if not os.path.exists(args.teacher_checkpoint):
                    raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_checkpoint}")
                
                teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in teacher_checkpoint:
                    state_dict = teacher_checkpoint['model_state_dict']
                    print(f"[TEACHER] Loading from 'model_state_dict' format")
                elif 'state_dict' in teacher_checkpoint:
                    state_dict = teacher_checkpoint['state_dict']
                    print(f"[TEACHER] Loading from 'state_dict' format")
                else:
                    state_dict = teacher_checkpoint
                    print(f"[TEACHER] Loading from direct state_dict format")
                
                # 检查模型兼容性
                teacher_state_keys = set(teacher_model.state_dict().keys())
                checkpoint_keys = set(state_dict.keys())
                
                missing_keys = teacher_state_keys - checkpoint_keys
                unexpected_keys = checkpoint_keys - teacher_state_keys
                
                if missing_keys:
                    print(f"[TEACHER] Missing keys: {len(missing_keys)} (will be randomly initialized)")
                    if len(missing_keys) <= 5:
                        print(f"[TEACHER] Missing: {list(missing_keys)}")
                
                if unexpected_keys:
                    print(f"[TEACHER] Unexpected keys: {len(unexpected_keys)} (will be ignored)")
                    if len(unexpected_keys) <= 5:
                        print(f"[TEACHER] Unexpected: {list(unexpected_keys)}")
                
                # 加载权重（忽略不匹配的键）
                teacher_model.load_state_dict(state_dict, strict=False)
                
                # 验证加载结果
                loaded_params = sum(p.numel() for p in teacher_model.parameters())
                print(f"[PASS] Teacher model weights loaded successfully")
                print(f"[TEACHER] Total parameters: {loaded_params:,}")
                
                # 冻结Teacher模型参数
                for param in teacher_model.parameters():
                    param.requires_grad = False
                teacher_model.eval()  # 设置为评估模式
                print(f"[TEACHER] Model frozen and set to eval mode")
                
                # 验证Teacher模型是否可以正常前向推理
                with torch.no_grad():
                    test_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)
                    test_output = teacher_model(test_input)
                    output_stats = {
                        'shape': test_output.shape,
                        'min': test_output.min().item(),
                        'max': test_output.max().item(),
                        'mean': test_output.mean().item(),
                        'std': test_output.std().item()
                    }
                    print(f"[TEACHER] Test forward pass successful:")
                    print(f"[TEACHER]   Input: {test_input.shape} -> Output: {test_output.shape}")
                    print(f"[TEACHER]   Output stats: min={output_stats['min']:.4f}, max={output_stats['max']:.4f}, mean={output_stats['mean']:.4f}, std={output_stats['std']:.4f}")
                    
                    # 检查输出是否有意义（不是全零或全NaN）
                    if torch.all(test_output == 0):
                        print(f"[WARN] [TEACHER] WARNING: Model outputs all zeros!")
                    elif torch.isnan(test_output).any():
                        print(f"[WARN] [TEACHER] WARNING: Model outputs contain NaN!")
                    else:
                        print(f"[PASS] [TEACHER] Model output appears normal")
                    
            except Exception as e:
                print(f"[FAILED] Error loading Teacher weights: {e}")
                print("[WARN] Continuing with randomly initialized Teacher model")
                print("[WARN] This will result in poor distillation performance")
                
                # 即使加载失败，也要冻结Teacher
                for param in teacher_model.parameters():
                    param.requires_grad = False
                teacher_model.eval()
        else:
            print("[WARN] No Teacher checkpoint provided - using randomly initialized Teacher model")
            print("[WARN] This will result in poor distillation performance")
            # 冻结随机初始化的Teacher
            for param in teacher_model.parameters():
                param.requires_grad = False
            teacher_model.eval()

        # Use distillation loss
        criterion = DistillationLoss(
            num_classes=args.num_classes,  # 直接使用args.num_classes，不再根据binary判断
            temperature=args.distill_temperature,
            alpha=args.distill_alpha,
            beta=args.distill_beta,
            feature_weight=args.distill_feature_weight,
            ignore_index=255  # 添加ignore_index以忽略无效像素
        )

        # Train the Student model primarily
        model = student_model
    else:
        # 原有的单模型训练模式
        print(f"=== Standard Training Mode ===")
        print(f"Model: {args.model}")    
        # Model
        if args.binary:
            # 二分类：模型输出1个通道，用于BCEWithLogitsLoss
            model = build_model(args.model, num_classes=1, in_ch=3, stage=args.stage).to(device)
            criterion = nn.BCEWithLogitsLoss()  # 二分类用BCE
        else:
            # 多分类：模型输出num_classes个通道，用于高级损失函数
            model = build_model(args.model, num_classes=args.num_classes, in_ch=3, stage=args.stage).to(device)
            
            # 智能损失函数选择
            criterion = create_advanced_loss_function(args, full_dataset, device)
            
        teacher_model = None  # 标准模式下没有Teacher模型

    # Optimizer and Scheduler
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)
    
    best_val_loss = float("inf")
    best_miou = 0.0  # mIoU越大越好
    patience_counter = 0  # Initialize early stopping counter

    print("=" * 80) # Training start
    if args.enable_distillation:
        print(f" Knowledge Distillation Training: {args.teacher_model} → {args.student_model}")
        print(f"     Teacher: {args.teacher_model} (frozen, providing soft targets)")
        print(f"     Student: {args.student_model} (learning, will be deployed)")
        
        # 初始化蒸馏指标收集
        distillation_metrics = {
            'total_loss': [],
            'task_loss': [],
            'distill_loss': [],
            'val_loss': [],
            'miou': [],
            'mdice': [],
            'macc': []
        }
    else:
        print(f" Standard Training: {args.model}")
    
    # 改进epoch显示逻辑
    if resume_manager:
        remaining_epochs = max(0, args.epochs - start_epoch)
        print(f"Training: Resume from epoch {start_epoch} → Continue to epoch {args.epochs-1} (总共{remaining_epochs}个新epoch)")
        if remaining_epochs == 0:
            print(f"⚠️  WARNING: 已到达目标epoch ({args.epochs})，无需继续训练！")
            print(f"   建议使用 --epochs {start_epoch + 5} 或更高的值来继续训练")
    else:
        print(f"Training: Start from epoch 0 → Train to epoch {args.epochs-1} (总共{args.epochs}个epoch)")

    # if resume, load states
    if resume_manager:
        load_resume_states(model, optimizer, scheduler, resume_info, device)
        print(f"=== RESUMING FROM EPOCH {start_epoch} ===")

    # Training loop - 检查是否有epoch需要训练
    if start_epoch >= args.epochs:
        print(f"🛑 训练已完成！当前epoch ({start_epoch}) >= 目标epoch ({args.epochs})")
        print(f"   如需继续训练，请使用 --epochs {start_epoch + 10} 等更高的值")
        return
    
    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch
        if args.enable_distillation:
            train_results = train_one_epoch(model, train_loader, criterion, optimizer, device, monitor, epoch, args, teacher_model)
            avg_train = train_results['total_loss']
            
            # 收集蒸馏指标
            distillation_metrics['total_loss'].append(train_results['total_loss'])
            distillation_metrics['task_loss'].append(train_results['task_loss'])
            distillation_metrics['distill_loss'].append(train_results['distill_loss'])
        else:
            avg_train = train_one_epoch(model, train_loader, criterion, optimizer, device, monitor, epoch, args)

        print(f"Epoch [{epoch + 1}/{args.epochs}], Train Loss: {avg_train:.4f}")

        val_metrics = validate(model, val_loader, criterion, device, args) # Validate model
        
        # 在蒸馏模式下收集验证指标
        if args.enable_distillation:
            distillation_metrics['val_loss'].append(val_metrics['val_loss'])
            if 'miou' in val_metrics:
                distillation_metrics['miou'].append(val_metrics['miou'])
            if 'mdice' in val_metrics:
                distillation_metrics['mdice'].append(val_metrics['mdice'])
            if 'macc' in val_metrics:
                distillation_metrics['macc'].append(val_metrics['macc'])

        if args.binary: # Binary classification
            print(f"[Epoch {epoch+1}] "
                f"Val loss: {val_metrics['val_loss']:.4f} | "
                f"IoU: {val_metrics['iou']:.4f} | Dice: {val_metrics['dice']:.4f} | "
                f"Acc: {val_metrics['accuracy']:.4f} | Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f}")
        else: # Multi-class classification
            print(f"[Epoch {epoch+1}] "
                f"Val loss: {val_metrics['val_loss']:.4f} | "
                f"mIoU: {val_metrics['miou']:.4f} | mDice: {val_metrics['mdice']:.4f} | mAcc: {val_metrics['macc']:.4f}")

        # Combine metrics for logging
        combined_metrics = {"train_loss": avg_train}
        combined_metrics.update(val_metrics)
        output_mgr.save_metrics_csv(combined_metrics, epoch + 1)

        if scheduler is not None: # Learning rate scheduler
            if args.scheduler == "plateau": # Plateau scheduler
                scheduler.step(val_metrics['val_loss'])
            else:
                scheduler.step()

        # Early stopping logic (before saving) - 支持多种指标
        current_val_loss = val_metrics['val_loss']
        current_miou = val_metrics.get('miou', 0.0)
        
        # 根据选择的指标判断是否改善
        if args.early_stopping_metric == "loss":
            is_improvement = current_val_loss < best_val_loss
            monitor_value = current_val_loss
            best_monitor_value = best_val_loss
            print(f"   Early stopping monitoring: loss = {monitor_value:.4f} (best: {best_monitor_value:.4f})")
        elif args.early_stopping_metric == "miou":
            is_improvement = current_miou > best_miou
            monitor_value = current_miou
            best_monitor_value = best_miou
            print(f"   Early stopping monitoring: mIoU = {monitor_value:.4f} (best: {best_monitor_value:.4f})")
        else:
            # Fallback to loss
            is_improvement = current_val_loss < best_val_loss
            monitor_value = current_val_loss
            best_monitor_value = best_val_loss
        
        if args.early_stopping:
            if is_improvement:
                patience_counter = 0
                print(f"   ✓ Improvement detected, resetting patience counter")
            else:
                patience_counter += 1
                print(f"   ✗ No improvement for {patience_counter}/{args.patience} epochs")
                if patience_counter >= args.patience:
                    print(f"\n🛑 Early stopping triggered! No improvement in {args.early_stopping_metric} for {args.patience} epochs")
                    print(f"   Final {args.early_stopping_metric}: {monitor_value:.4f} (best: {best_monitor_value:.4f})")
                    break

        # save checkpoints using the unified method
        if args.enable_distillation:
            # 知识蒸馏模式：只保存Student模型（Teacher是冻结的，无需保存）
            # 保存Student模型（主要训练模型） - 使用混合评估策略
            saved_path, best_val_loss, best_miou = output_mgr.save_checkpoint_with_hybrid_evaluation(
                model=student_model,
                epoch=epoch + 1,
                metrics=val_metrics,
                current_best_loss=best_val_loss,
                current_best_miou=best_miou,
                loss_threshold=args.loss_threshold,
                loss_degradation_threshold=args.loss_degradation_threshold,
                save_interval=args.save_interval,
                model_suffix="student"
            )
            
            # Teacher模型在第一个epoch时保存一次作为参考，后续不再保存
            if epoch == 0:
                teacher_reference_path = output_mgr.save_model(
                    teacher_model, 
                    epoch + 1, 
                    val_metrics, 
                    is_best=False, 
                    model_suffix="teacher_reference"
                )
                print(f"SAVED: Teacher reference model (frozen): {os.path.basename(teacher_reference_path)}")
        else:
            # 标准训练模式：只保存单个模型 - 使用混合评估策略
            saved_path, best_val_loss, best_miou = output_mgr.save_checkpoint_with_hybrid_evaluation(
                model=model,
                epoch=epoch + 1,
                metrics=val_metrics,
                current_best_loss=best_val_loss,
                current_best_miou=best_miou,
                loss_threshold=args.loss_threshold,
                loss_degradation_threshold=args.loss_degradation_threshold,
                save_interval=args.save_interval
            )

        # Add epoch summary (migrated from train_offline_min)
        train_metrics = {"loss": avg_train}
        monitor.print_epoch_summary(epoch + 1, train_metrics, val_metrics)


    # Visualization after training (outside training loop)
    if args.save_viz and 'val_metrics' in locals():
        print("\n-- Loading Best Model for Visualization --")
        
        # Find the best model file
        checkpoints_dir = output_mgr.get_checkpoints_dir()
        
        if args.enable_distillation:
            # 蒸馏模式：加载Student最佳模型
            best_model_path = os.path.join(checkpoints_dir, f"{model_tag}_student_best.pth")
        else:
            best_model_path = os.path.join(checkpoints_dir, f"{model_tag}_best.pth")
        
        if os.path.exists(best_model_path): # Load best model
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded best model from {best_model_path}")
        else:
            print("WARNING: Best model not found, using current model for visualization")
        
        print("-- Saving Visualizations --")
        visualizer = Visualizer(classification_scheme=args.classification_scheme)
        viz_dir = output_mgr.get_vis_dir() # Get visualization directory

        # Save visualization results
        visualizer.save_comparison_predictions(
            model, val_loader, viz_dir, max_samples=args.viz_samples, device=device
        )
        visualizer.save_basic_predictions(
            model, val_loader, viz_dir, max_samples=args.viz_samples, device=device
        )
        
        # 蒸馏模式特有的可视化
        if args.enable_distillation and DISTILLATION_AVAILABLE:
            print("-- Generating Distillation Analysis --")
            distill_visualizer = DistillationVisualizer(viz_dir, device, classification_scheme=args.classification_scheme)
            
            # Teacher-Student预测对比
            distill_visualizer.visualize_prediction_comparison(
                teacher_model, student_model, val_loader, 
                num_samples=6, save_name="teacher_student_comparison.png"
            )
            
            # 知识传递分析
            distillation_stats = distill_visualizer.visualize_knowledge_transfer(
                teacher_model, student_model, val_loader,
                temperature=args.distill_temperature,
                max_samples=1000,  # Limit samples to prevent memory overflow
                save_name="knowledge_transfer_analysis.png"
            )
            
            # 生成蒸馏总结报告
            distill_visualizer.create_distillation_summary_report(
                distillation_metrics, distillation_stats,
                args.teacher_model, args.student_model,
                save_name="distillation_summary_report.png"
            )
            
            # 保存蒸馏指标表格
            distill_visualizer.save_distillation_metrics_table(
                distillation_metrics, distillation_stats,
                save_name="distillation_metrics.csv"
            )
            
            print(f"[PASS] 蒸馏分析完成！所有结果保存在: {viz_dir}/distillation_analysis/")

    # Final model saving (using the last epoch's metrics)
    if 'val_metrics' in locals():
        if args.enable_distillation:
            # 蒸馏模式：只保存最终的Student模型（Teacher已在第一轮保存过参考版本）
            student_final_path = output_mgr.save_model(student_model, args.epochs, val_metrics, is_best=False, model_suffix="student_final")
            print(f"SAVED: Final student model saved: {os.path.basename(student_final_path)}")
            print(f"NOTE: Teacher model was saved as reference in epoch 1 (frozen model, no updates)")
        else:
            # 标准模式：保存单个模型
            final_path = output_mgr.save_model(model, args.epochs, val_metrics, is_best=False)
            print(f"SAVED: Final model saved: {os.path.basename(final_path)}")

    # Print summary
    summary = output_mgr.get_run_summary()
    print(f"\n--> Train Completed <--")
    print(f"Results saved to: {summary['run_dir']}")
    
    # KD Evidence Package Generation (if enabled)
    if args.generate_evidence_package and args.enable_distillation and DISTILLATION_AVAILABLE:
        print("\n🔬 Generating KD Evidence Package...")
        try:
            evidence_paths = generate_kd_evidence_package(
                args, teacher_model, student_model, val_loader, output_mgr, device
            )
            print(f"✅ Evidence package generated successfully!")
            print(f"📊 Key outputs:")
            print(f"   - Metrics CSV: {os.path.basename(evidence_paths['csv_path'])}")
            print(f"   - Performance Analysis: {os.path.basename(evidence_paths['performance_plot'])}")
            print(f"   - Reliability Diagrams: {os.path.basename(evidence_paths['reliability_diagrams'])}")
        except Exception as e:
            print(f"⚠️  Evidence package generation failed: {str(e)}")
            print("   Training completed successfully, but evidence package could not be generated.")
    elif args.generate_evidence_package and not args.enable_distillation:
        print("⚠️  Evidence package requested but distillation not enabled. Skipping evidence generation.")
    elif args.generate_evidence_package and not DISTILLATION_AVAILABLE:
        print("⚠️  Evidence package requested but distillation modules not available. Skipping evidence generation.")
    
    # 保存指标曲线图
    if 'monitor' in locals():
        viz_visualizer = Visualizer(classification_scheme=args.classification_scheme)
        metrics_history = monitor.get_metrics_history()
        if metrics_history:
            curves_path = output_mgr.get_viz_path("training_curves.png")
            viz_visualizer.plot_metrics_curves(metrics_history, curves_path)
            print(f"Training curves saved to: {curves_path}")
    
    if 'val_metrics' in locals():
        if args.binary:
            print(f"Final Metrics - Loss: {val_metrics['val_loss']:.4f}, IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        else:
            print(f"Final Metrics - Loss: {val_metrics['val_loss']:.4f}, mIoU: {val_metrics['miou']:.4f}, mDice: {val_metrics['mdice']:.4f}")
    else:
        print("Training completed but no validation metrics available")

if __name__ == "__main__":
    main()
