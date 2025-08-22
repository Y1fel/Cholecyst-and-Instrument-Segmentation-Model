"""
最小离线训练脚本
只训练1个epoch并保存模型，用于快速测试和原型验证
"""
import os, argparse, yaml, torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataio.datasets.seg_dataset_min import SegDatasetMin
from src.models.baseline.unet_min import UNetMin

from src.metrics.evaluator import Evaluator
from src.viz.visualizer import Visualizer
from src.common.output_manager import OutputManager

from src.common.train_monitor import TrainMonitor

# argument parsing
def parse_args():
    p = argparse.ArgumentParser("Minimal offline training")
    p.add_argument("--split", type=str, default="train")                                    # dataset split
    p.add_argument("--img_size", type=int, default=512)                                     # image size
    p.add_argument("--batch_size", type=int, default=6)                                     # batch size
    p.add_argument("--epochs", type=int, default=1)                                         # number of epochs
    p.add_argument("--lr", type=float, default=3e-4)                                        # learning rate
    p.add_argument("--num_classes", type=int, default=2)                                    # number of classes
    p.add_argument("--save_path", type=str, default="checkpoints/baseline_offline_min.pth") # model save path
    p.add_argument("--num_workers", type=int, default=0)                                    # number of workers
    p.add_argument("--data_root", type=str, required=True)                                  # dataset root directory
    p.add_argument("--model_type", type=str, default="baseline")                            # model type
    p.add_argument("--val_ratio", type=float, default=0.2)                                  # validation split ratio
    p.add_argument("--save_viz", action='store_true', help="Save visualizations")           # save visualizations
    p.add_argument("--monitor_interval", type=int, default=5, help="Progress update interval (batches)")
    p.add_argument("--enable_gpu_monitor", action='store_true', default=True, help="Enable GPU monitoring")

    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize training monitor
    monitor = TrainMonitor(enable_gpu_monitor=True)
    monitor.start_timing()

    #  build output directories
    output_mgr = OutputManager(model_type=args.model_type)
    output_mgr.save_config(vars(args))

    # dataset initialization
    full_dataset = SegDatasetMin(args.data_root, "", args.img_size)

    # 🆕 添加数据分析
    print("🔍 分析修复后的数据集...")
    if hasattr(full_dataset, 'analyze_mask_distribution'):
        full_dataset.analyze_mask_distribution(num_samples=20)
    
    val_size = int(len(full_dataset) * args.val_ratio)

    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # Model and optimizer
    out_ch = 1 if args.num_classes == 2 else args.num_classes
    model = UNetMin(in_ch=3, num_classes=out_ch, base=32).to(device)
    criterion = nn.BCEWithLogitsLoss() if args.num_classes == 2 else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # best checkpoint value
    best_checkpoints = float("inf")
    
    # train
    for epoch in range(args.epochs):
        # set model to training mode
        model.train()
        running = 0.0

        total_batches = len(train_loader)

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad(set_to_none = True)
            loss.backward()
            optimizer.step()

            running += loss.item() * images.size(0)

            if batch_idx % 5 == 0:
                current_avg_loss = running / ((batch_idx + 1) * args.batch_size)
                monitor.print_progress(
                    epoch + 1, args.epochs,
                    batch_idx + 1, total_batches,
                    {"loss": current_avg_loss},
                    refresh=True
                )
        
        avg_train_loss = running / len(train_dataset)
        print(f"[epoch {epoch+1} / {args.epochs}] Train loss={avg_train_loss:.4f}")

        # validation
        print("-- Validation Running --")
        evaluator = Evaluator(device=device)
        val_metrics = evaluator.evaluate(model, val_loader, criterion)

        print(
            f"Validation Metrics - Loss: {val_metrics['val_loss']:.4f}, "
            f"IoU: {val_metrics['iou']:.4f}, "
            f"Dice: {val_metrics['dice']:.4f}, "
            f"Accuracy: {val_metrics['accuracy']:.4f}"
        )

        # training metrics
        train_metrics = {"loss": avg_train_loss}
        monitor.print_epoch_summary(epoch + 1, train_metrics, val_metrics) # Print epoch summary by monitor

        combined_metrics = {"train_loss": avg_train_loss}
        combined_metrics.update(val_metrics)
        output_mgr.save_metrics_csv(combined_metrics, epoch + 1)

        # save checkpoints
        if val_metrics['val_loss'] < best_checkpoints:
            best_checkpoints = val_metrics['val_loss']
            output_mgr.save_model(model, epoch + 1, val_metrics)
            print(f"Saved best model at epoch {epoch + 1} with loss {val_metrics['val_loss']:.4f}")

    # Save visualizations results
    if args.save_viz:
        print("Loading Best Model for Visualization")

        # checkpoint with best performance
        best_model_path = output_mgr.get_best_model_path()
        if best_model_path and os.path.exists(best_model_path):
            print(f"Best model found: {best_model_path}")
            current_checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(current_checkpoint['model_state_dict'])
            print(f"Best model loaded: {best_model_path}")
        else:
            print(f"Best model file not found: {best_model_path}, using latest")

        print("-- Visualizations Saving --")
        visualizer = Visualizer()
        viz_dir = output_mgr.get_vis_dir()

        visualizer.save_comparison_predictions(
            model,
            val_loader,
            viz_dir,
            max_samples = 50,
            device = device
        )
        
        visualizer.save_basic_predictions(
            model,
            val_loader,
            viz_dir,
            max_samples = 50,
            device = device
        )

    # Print summary
    summary = output_mgr.get_run_summary()
    print(f"\n--> Train Completed <--")
    print(f"Results saved to: {summary['run_dir']}")
    print(f"Metrics: {val_metrics['val_loss']:.4f}, {val_metrics['iou']:.4f}, {val_metrics['dice']:.4f}, {val_metrics['accuracy']:.4f}")

    
if __name__ == "__main__":
    main()