# src/training/train_offline_universal.py
"""
通用训练模板 - 集成监控、可视化、评估功能
基于train_offline_min改进，适用于各种模型的训练
"""

# $env:PYTHONPATH="F:\Documents\Courses\CIS\Cholecyst-and-Instrument-Segmentation-Model"

import os, argparse, yaml, torch, sys, json
from torch import nn
from torch.utils.data import DataLoader

# 导入通用模块
from src.eval.evaluator import Evaluator
from src.viz.visualizer import Visualizer
from src.common.output_manager import OutputManager
from src.common.train_monitor import TrainMonitor

# 示例模型导入 - 根据实际模型替换
from src.dataio.datasets.seg_dataset_min import SegDatasetMin
from src.models.baseline.unet_min import UNetMin

from src.models.model_zoo import build_model

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
                   choices=["binary", "3class", "5class", "detailed", "custom"],
                   help="分类方案：binary(2类), 3class(3类), 5class(5类), detailed(13类), custom(自定义)")
    
    p.add_argument("--target_classes", nargs="+", default=None,
                   help="指定目标类别列表，例如：--target_classes background instrument target_organ")
    
    p.add_argument("--custom_mapping_file", type=str, default=None,
                   help="自定义映射JSON文件路径")
    
    p.add_argument("--num_classes", type=int, default=10,
                   help="多类时>=2；--binary 生效时忽略此项。watershed模式建议使用10。")
    
    # 模型插拔
    p.add_argument("--model", type=str, default="unet_min",
                   choices=["unet_min", "mobile_unet", "adaptive_unet"])
    
    # 兼容 OutputManager 的模型类型标记（用于run目录命名）
    p.add_argument("--model_type", type=str, default=None,
                   help="若不指定，将自动使用 --model 的值。")

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

    # 验证和保存
    p.add_argument("--val_interval", type=int, default=1, help="Validation interval (epochs)")
    p.add_argument("--save_interval", type=int, default=5, help="Checkpoint save interval (epochs)")

    # early stopping
    p.add_argument("--early_stopping", action='store_true', help="Enable early stopping")
    p.add_argument("--patience", type=int, default=5, help="Patience for early stopping")

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
    model, loader, criterion, optimizer, device, monitor, epoch_index, args
):
    model.train()
    running_loss = 0.0
    total = len(loader)

    for step, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True) # [path, 3, H, W]
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)

        # Forward pass: images -> logits
        if args.binary:
            loss = criterion(logits, masks) # BCEWithLogitsLoss(logits, targets)
        else:
            targets = masks.long()
            # if logits.shape[1] == 1:
            #     raise RuntimeError("Multiclass training requires model output C>1")
            loss = criterion(logits, targets)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # monitor progress
        if (step % args.monitor_interval) == 0:
            avg = running_loss / max(1, (step + 1) * args.batch_size)
            monitor.print_progress(
                    epoch_index + 1, args.epochs,
                    step + 1, total,
                    {"loss": avg},
                    refresh=True
                )

        # Force output every 50 batches to avoid long periods without output
        # if step > 0 and (step % 50) == 0:
        #     avg = running_loss / max(1, (step + 1) * args.batch_size)
        #     print(f"\n[Checkpoint] Epoch {epoch_index + 1}/{args.epochs} Batch {step + 1}/{total} | Loss: {avg:.4f}")
        #     sys.stdout.flush()

    return running_loss / (len(loader.dataset) if hasattr(loader, 'dataset') else (total * args.batch_size))

# Validation
@torch.inference_mode()
def validate(model, loader, criterion, device, args):
    if args.binary:
        evaluator = Evaluator(device=device, threshold=0.5)
        return evaluator.evaluate(model, loader, criterion)
    else:
        evaluator = Evaluator(device=device)
        return evaluator.evaluate_multiclass(
            model, loader, criterion,
            num_classes = args.num_classes,
            ignore_index = 255
        )

# Main function
def main():
    args = parse_args()

    # load config and validate args
    config = load_config(args.config)
    args = merge_config_with_args(args, config)
    validate_args(args)

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
    model_tag  = args.model if args.model_type is None else args.model_type
    output_mgr = OutputManager(model_type=model_tag)
    output_mgr.save_config(vars(args))

    # 
    custom_mapping = None
    if args.custom_mapping_file:
        with open(args.custom_mapping_file, 'r') as f:
            custom_mapping = json.load(f)

    # Dataset configuration
    dataset_config = {
        "classification_scheme": args.classification_scheme,
        "custom_mapping": custom_mapping,
        "target_classes": args.target_classes,
        "return_multiclass": is_multiclass # 保持兼容性
    }

    # Dataloader
    is_multiclass = (not args.binary) and (args.num_classes >= 2)
    full_dataset = SegDatasetMin(
        args.data_root, dtype=args.split, img_size=args.img_size,
        **dataset_config
    )

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
    
    # Model
    if args.binary:
        # 二分类：模型输出1个通道，用于BCEWithLogitsLoss
        model = build_model(args.model, num_classes=1, in_ch=3, stage=args.stage).to(device)
        criterion = nn.BCEWithLogitsLoss()  # 二分类用BCE
    else:
        # 多分类：模型输出num_classes个通道，用于CrossEntropyLoss  
        model = build_model(args.model, num_classes=args.num_classes, in_ch=3, stage=args.stage).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=255)   # 多分类用CE

    # Optimizer and Scheduler
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)
    
    best_val_loss = float("inf")
    patience_counter = 0  # Initialize early stopping counter

    print("=" * 80) # Start training
    print(f"Start Training ({args.model}) for {args.epochs} epoch(s)...")

    # Training loop
    for epoch in range(args.epochs):
        # Train for one epoch
        avg_train = train_one_epoch(model, train_loader, criterion, optimizer, device, monitor, epoch, args)
        print(f"Epoch [{epoch + 1}/{args.epochs}], Train Loss: {avg_train:.4f}")

        val_metrics = validate(model, val_loader, criterion, device, args) # Validate model

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

    # Final model saving (using the last epoch's metrics)
    if 'val_metrics' in locals():
        final_path = output_mgr.save_model(model, args.epochs, val_metrics, is_best=False)
        print(f"SAVED: Final model saved: {os.path.basename(final_path)}")

    # Print summary
    summary = output_mgr.get_run_summary()
    print(f"\n--> Train Completed <--")
    print(f"Results saved to: {summary['run_dir']}")
    
    # 保存指标曲线图
    if 'monitor' in locals():
        from src.viz.visualizer import Visualizer
        visualizer = Visualizer()
        metrics_history = monitor.get_metrics_history()
        if metrics_history:
            curves_path = output_mgr.get_viz_path("training_curves.png")
            visualizer.plot_metrics_curves(metrics_history, curves_path)
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
