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
    p.add_argument("--stage", type=str, default="offline",
                   choices=["offline", "online"], help="模型训练阶段")
    
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

    # early stopping
    p.add_argument("--early_stopping", action='store_true', help="Enable early stopping")
    p.add_argument("--patience", type=int, default=5, help="Patience for early stopping")

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
    
    # 恢复时可选的参数覆盖（预留扩展）
    p.add_argument("--resume_lr", type=float, default=None,
                   help="Override learning rate when resuming (optional)")
    p.add_argument("--resume_epochs", type=int, default=None,
                   help="Override total epochs when resuming (optional)")

    return p.parse_args()

# validate arguments
def validate_args(args):
    errors = []    # storage for error messages
    warnings = []  # storage for warning messages

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
    if not (0 < args.val_ratio < 1):
        errors.append("val_ratio must be between 0 and 1")
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
def get_parser_default(param_name):
    # Create a temporary parser to get default values
    temp_parser = argparse.ArgumentParser()

    # Redefine all parameters here (only key ones listed)
    # Add more as needed based on actual usage
    defaults = {
        'epochs': 5,
        'batch_size': 6,
        'lr': 0.0003,
        'num_classes': 10,
        'val_ratio': 0.2,
        'num_workers': 4,
        'img_size': 512,
        'monitor_interval': 5,
        'viz_samples': 50,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'weight_decay': 0.0001,
        'early_stopping': False,    # Short-term training default: disabled
        'patience': 5,              # Long-term training recommended value
        'val_interval': 1,          # Validate every epoch
        'save_interval': 1,         # Save checkpoint every x epochs
        'flip_prob': 0.5,
        'rotation_degree': 15,
        'apply_fov_mask': False,
        # 添加模型相关默认值
        'model': 'unet_min',
        # 添加蒸馏相关默认值
        'enable_distillation': False,
        'teacher_model': 'unet_plus_plus',
        'teacher_checkpoint': None,
        'student_model': 'mobile_unet',
        'distill_temperature': 4.0,
        'distill_alpha': 0.7,
        'distill_beta': 0.3,
        'distill_feature_weight': 0.1,
        # 添加分类相关默认值
        'binary': False,
        'classification_scheme': None,
        'target_classes': None,
        'custom_mapping_file': None,
        # 添加可视化相关默认值
        'save_viz': False,
        'enable_gpu_monitor': False,
        'augment': False,
        'debug': False,
        'save_best_only': True,
    }
    
    return defaults.get(param_name, None)

# combine config with arguments
def merge_config_with_args(args, config):
    if not config:
        return args

    # Record which parameters are overridden by the config file
    overridden = []
    
    for key, value in config.items():
        if hasattr(args, key):
            # Only use config file value if command line argument is default
            current_value = getattr(args, key)
            parser_default = get_parser_default(key)

            # Only override if current value is default AND config value is different
            if current_value == parser_default and current_value != value:
                setattr(args, key, value)
                overridden.append(f"{key}: {current_value} -> {value}")
    
    if overridden:
        print("CONFIG OVERRIDES:")
        for override in overridden:
            print(f"   - {override}")
    else:
        print("CONFIG: Using default values, no overrides needed")
    
    return args

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
    if args.scheduler == "none":
        return None
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs//3, gamma=0.5)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
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
                loss = criterion(logits, masks) # BCEWithLogitsLoss(logits, targets)
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
    # 在蒸馏模式下，创建标准验证损失
    if args.enable_distillation:
        if args.binary:
            val_criterion = nn.BCEWithLogitsLoss()
        else:
            val_criterion = nn.CrossEntropyLoss(ignore_index=255)
    else:
        val_criterion = criterion
    
    if args.binary:
        evaluator = Evaluator(device=device, threshold=0.5)
        return evaluator.evaluate(model, loader, val_criterion)
    else:
        evaluator = Evaluator(device=device)
        return evaluator.evaluate_multiclass(
            model, loader, val_criterion,
            num_classes = args.num_classes,
            ignore_index = 255
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
    distill_visualizer = DistillationVisualizer(output_mgr.get_viz_dir(), device)
    visualizer = Visualizer()
    
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
        max_samples=args.evidence_samples
    )
    print(f"      ✅ Teacher - IoU: {teacher_metrics.get('iou', teacher_metrics.get('miou', 0)):.4f}")
    
    # Student evaluation  
    print("   🎓 Evaluating Student Model...")
    student_metrics = evaluator.evaluate_with_full_metrics(
        student_model, val_loader,
        num_classes=args.num_classes, 
        binary_mode=args.binary,
        max_samples=args.evidence_samples
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
    print(f"📁 All files saved in: {output_mgr.get_viz_dir()}")
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

    # setup resume training
    resume_manager, resume_info, start_epoch = setup_resume_training(args)

    # apply resume overrides
    if resume_manager:
        apply_resume_overrides(args)

    # load config and validate args
    config = load_config(args.config)
    args = merge_config_with_args(args, config)
    validate_args(args)

    # Binary/Multiclass 强一致性保护：自动修正 num_classes
    if args.binary:
        if args.num_classes != 2:
            print(f"[WARN] binary=True 但 num_classes={args.num_classes} != 2，自动将 num_classes 置为 2")
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
    
    # 恢复训练时使用原来的输出目录
    if resume_manager:
        # 使用原来的run目录
        original_run_dir = resume_info['run_dir']
        output_mgr = OutputManager(model_type=model_tag)
        # 覆盖run_dir为原来的目录
        output_mgr.run_dir = original_run_dir
        output_mgr._setup_directories()  # 确保目录存在
        print(f"=== RESUME: Using original output directory: {original_run_dir} ===")
    else:
        # 正常训练：创建新的输出目录
        output_mgr = OutputManager(model_type=model_tag)
        output_mgr.save_config(vars(args))  # 正常训练时保存配置

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

    # split ratio
    val_ratio  = args.val_ratio
    val_size   = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    # create datasets with random split and seed
    seed = 42
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"Dataset: Train={len(train_ds)}, Val={len(val_ds)}")

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
            # 多分类：模型输出num_classes个通道，用于CrossEntropyLoss  
            model = build_model(args.model, num_classes=args.num_classes, in_ch=3, stage=args.stage).to(device)
            
            # 计算类别权重以处理类别不平衡
            print("Computing class weights to handle class imbalance...")
            class_weights = compute_class_weights(full_dataset, args.num_classes, ignore_index=255)
            class_weights = class_weights.to(device)
            print(f"Class weights: {class_weights}")
            
            # 使用加权CrossEntropyLoss来处理类别不平衡（推荐方案）
            criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
            # criterion = FocalLoss(alpha=1.0, gamma=2.0, ignore_index=255)  # 备选：Focal Loss
        teacher_model = None  # 标准模式下没有Teacher模型

    # Optimizer and Scheduler
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)
    
    best_val_loss = float("inf")
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

        # Early stopping logic (before saving)
        current_val_loss = val_metrics['val_loss']
        is_improvement = current_val_loss < best_val_loss
        
        if args.early_stopping:
            if is_improvement:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered after {args.patience} epochs without improvement")
                    break

        # save checkpoints using the unified method
        if args.enable_distillation:
            # 知识蒸馏模式：只保存Student模型（Teacher是冻结的，无需保存）
            # 保存Student模型（主要训练模型）
            saved_path, best_val_loss = output_mgr.save_checkpoint_if_needed(
                model=student_model,
                epoch=epoch + 1,
                metrics=val_metrics,
                current_best_metric=best_val_loss,
                metric_name='val_loss',
                minimize=True,
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
            # 标准训练模式：只保存单个模型
            saved_path, best_val_loss = output_mgr.save_checkpoint_if_needed(
                model=model,
                epoch=epoch + 1,
                metrics=val_metrics,
                current_best_metric=best_val_loss,
                metric_name='val_loss',
                minimize=True,
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
        visualizer = Visualizer()
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
            distill_visualizer = DistillationVisualizer(viz_dir, device)
            
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
        viz_visualizer = Visualizer()
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
