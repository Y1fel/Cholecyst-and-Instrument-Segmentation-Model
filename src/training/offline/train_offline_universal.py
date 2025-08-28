# src/training/train_offline_universal.py
"""
通用训练模板 - 集成监控、可视化、评估功能
基于train_offline_min改进，适用于各种模型的训练
"""

import os, argparse, yaml, torch
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

def parse_args():
    """参数配置 - 可根据不同模型需求调整"""
    p = argparse.ArgumentParser("Offline Universal Trainer")
    
    # 基础训练参数
    p.add_argument("--cfg", type=str, default=None, help="Optional YAML config")
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
    p.add_argument("--num_classes", type=int, default=2,
                   help="多类时>=2；--binary 生效时忽略此项。")
    
    # 模型插拔
    p.add_argument("--model", type=str, default="unet_min",
                   choices=["unet_min", "mobile_unet", "adaptive_unet"])
    
    # 兼容 OutputManager 的模型类型标记（用于 run 目录命名）
    p.add_argument("--model_type", type=str, default=None,
                   help="若不指定，将自动使用 --model 的值。")
    
    return p.parse_args()

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

        # temp: BCE
        loss = criterion(logits, masks)
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

    return running_loss / (len(loader.dataset) if hasattr(loader, 'dataset') else (total * args.batch_size))

# Validation
@torch.inference_mode()
def validate(model, loader, criterion, device):
    evaluator = Evaluator(device=device, threshold=0.5)
    # evaluator.evaluate：
    # - forward + BCE loss
    # - use Sigmoid>0.5 
    # - calculate IoU/Dice/Acc/Precision/Recall
    return evaluator.evaluate(model, loader, criterion)

# 
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # train monitor
    monitor = TrainMonitor(enable_gpu_monitor=args.enable_gpu_monitor)
    monitor.start_timing()

    # output manager
    model_tag  = args.model if args.model_type is None else args.model_type
    output_mgr = OutputManager(model_type=model_tag)
    output_mgr.save_config(vars(args))

    # Dataloader
    full_dataset = SegDatasetMin(args.data_root, dtype=args.split, img_size=args.img_size)
    # split ratio
    val_ratio  = 0.2
    val_size   = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    # create datasets with random split and seed
    seed = 42
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"Dataset: Train={len(train_ds)}, Val={len(val_ds)}")

    # Model
    num_classes = 2 if args.binary else args.num_classes
    # model = build_model(args.model, num_classes=num_classes, in_ch=3).to(device)

    # temp: Loss function
    if args.binary:
        # 二分类：模型输出1个通道，用于BCEWithLogitsLoss
        model = build_model(args.model, num_classes=1, in_ch=3).to(device)
        criterion = nn.BCEWithLogitsLoss()  # 二分类用BCE
    else:
        # 多分类：模型输出num_classes个通道，用于CrossEntropyLoss  
        model = build_model(args.model, num_classes=args.num_classes, in_ch=3).to(device)
        criterion = nn.CrossEntropyLoss()   # 多分类用CE

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val_loss = float("inf")

    print("=" * 80) # Start training
    print(f"Start Training ({args.model}) for {args.epochs} epoch(s)...")

    # Training loop
    for epoch in range(args.epochs):
        avg_train = train_one_epoch(model, train_loader, criterion, optimizer, device, monitor, epoch, args)
        print(f"Epoch [{epoch + 1}/{args.epochs}], Train Loss: {avg_train:.4f}")

        # val_metrics = evaluator.evaluate(model, val_loader, criterion)
        val_metrics = validate(model, val_loader, criterion, device)

        print(
            f"[Epoch {epoch+1}] "
            f"Val loss: {val_metrics['val_loss']:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | Dice: {val_metrics['dice']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f}"
        )

        combined_metrics = {"train_loss": avg_train}
        combined_metrics.update(val_metrics)
        output_mgr.save_metrics_csv(combined_metrics, epoch + 1)

        # save checkpoints
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            output_mgr.save_model(model, epoch + 1, val_metrics)
            print(f"Saved best model at epoch {epoch + 1} with loss {val_metrics['val_loss']:.4f}")

        # 🆕 添加 epoch 总结 (从 train_offline_min 移植)
        train_metrics = {"loss": avg_train}
        monitor.print_epoch_summary(epoch + 1, train_metrics, val_metrics)


   # 训练完成后的可视化 (在训练循环外)
    if args.save_viz:
        print("\n-- Loading Best Model for Visualization --")
        
        # 加载最佳模型
        best_model_path = output_mgr.get_best_checkpoint()  # 注意方法名
        if best_model_path and os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded best model from {best_model_path}")
        
        print("-- Saving Visualizations --")
        visualizer = Visualizer()
        viz_dir = output_mgr.get_vis_dir()
        
        visualizer.save_comparison_predictions(
            model, val_loader, viz_dir, max_samples=args.viz_samples, device=device
        )
        visualizer.save_basic_predictions(
            model, val_loader, viz_dir, max_samples=args.viz_samples, device=device
        )

    # Save final model
    output_mgr.save_model(model)

    # Print summary
    summary = output_mgr.get_run_summary()
    print(f"\n--> Train Completed <--")
    print(f"Results saved to: {summary['run_dir']}")
    print(f"Metrics: {val_metrics['val_loss']:.4f}, {val_metrics['iou']:.4f}, {val_metrics['dice']:.4f}, {val_metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()
