"""
通用训练模板 - 集成监控、可视化、评估功能
基于train_offline_min改进，适用于各种模型的训练
"""
import os, argparse, yaml, torch
from torch import nn
from torch.utils.data import DataLoader

# 导入通用模块
from src.metrics.evaluator import Evaluator
from src.viz.visualizer import Visualizer
from src.common.output_manager import OutputManager
from src.common.train_monitor import TrainMonitor

# 示例模型导入 - 根据实际模型替换
from src.dataio.datasets.seg_dataset_min import SegDatasetMin
from src.models.baseline.unet_min import UNetMin

def parse_args():
    """参数配置 - 可根据不同模型需求调整"""
    p = argparse.ArgumentParser("Universal training template with monitoring.")
    
    # 基础训练参数
    p.add_argument("--cfg", type=str, default=None, help="Optional YAML config")
    p.add_argument("--data_root", type=str, required=True, help="Dataset root path")
    p.add_argument("--model_type", type=str, default="universal", help="Model type identifier")
    
    # 数据参数
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=0)
    
    # 训练参数
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_classes", type=int, default=2)
    
    # 监控和输出参数
    p.add_argument("--monitor_interval", type=int, default=10, help="Progress update interval (batches)")
    p.add_argument("--enable_gpu_monitor", action='store_true', default=True, help="Enable GPU monitoring")
    p.add_argument("--save_viz", action='store_true', help="Save visualizations")
    p.add_argument("--viz_samples", type=int, default=50, help="Number of visualization samples")
    
    # 调试和高级选项
    p.add_argument("--debug", action='store_true', help="Enable debug mode")
    p.add_argument("--save_best_only", action='store_true', default=True, help="Only save best checkpoints")
    
    return p.parse_args()

def setup_model_and_criterion(args, device):
    """
    模型和损失函数设置 - 根据不同模型修改此函数
    
    返回: model, criterion
    """
    # 示例：UNet模型设置
    out_ch = 1 if args.num_classes == 2 else args.num_classes
    model = UNetMin(in_ch=3, num_classes=out_ch, base=32).to(device)
    
    # 损失函数选择
    if args.num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    return model, criterion

def setup_data_loaders(args):
    """
    数据加载器设置 - 根据不同数据集修改此函数
    
    返回: train_loader, val_loader, dataset_info
    """
    # 示例：分割数据集设置
    full_dataset = SegDatasetMin(args.data_root, "", args.img_size)
    
    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
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
    
    dataset_info = {
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'total_size': len(full_dataset)
    }
    
    return train_loader, val_loader, dataset_info

def train_one_epoch(model, train_loader, criterion, optimizer, device, monitor, epoch, args):
    """
    训练一个epoch - 集成监控功能
    
    返回: 平均训练损失
    """
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # 实时监控显示
        if batch_idx % args.monitor_interval == 0:
            current_avg_loss = running_loss / ((batch_idx + 1) * args.batch_size)
            monitor.print_progress(
                epoch + 1, args.epochs,
                batch_idx + 1, total_batches,
                {"loss": current_avg_loss},
                refresh=True
            )
    
    avg_train_loss = running_loss / len(train_loader.dataset)
    return avg_train_loss

def validate_model(model, val_loader, criterion, device):
    """
    模型验证 - 使用集成的评估器
    
    返回: 验证指标字典
    """
    evaluator = Evaluator(device=device)
    val_metrics = evaluator.evaluate(model, val_loader, criterion)
    return val_metrics

def save_visualizations(model, val_loader, viz_dir, args, device):
    """
    保存可视化结果 - 使用集成的可视化器
    """
    visualizer = Visualizer()
    visualizer.save_basic_predictions(
        model,
        val_loader,
        viz_dir,
        max_samples=args.viz_samples,
        device=device
    )

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Starting training with device: {device}")
    print(f"Model type: {args.model_type}")
    
    # 初始化监控器
    monitor = TrainMonitor(enable_gpu_monitor=args.enable_gpu_monitor)
    monitor.start_timing()
    
    # 初始化输出管理器
    output_mgr = OutputManager(model_type=args.model_type)
    output_mgr.save_config(vars(args))
    
    # 设置数据加载器
    train_loader, val_loader, dataset_info = setup_data_loaders(args)
    print(f"📁 Dataset: Train={dataset_info['train_size']}, Val={dataset_info['val_size']}")
    
    # 设置模型和损失函数
    model, criterion = setup_model_and_criterion(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 训练状态跟踪
    best_val_loss = float("inf")
    
    print(f"🎯 Starting training for {args.epochs} epochs...")
    print("=" * 80)
    
    # 训练循环
    for epoch in range(args.epochs):
        # 训练阶段
        avg_train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, monitor, epoch, args
        )
        
        # 验证阶段
        print(f"\n🔍 Validation for epoch {epoch + 1}...")
        val_metrics = validate_model(model, val_loader, criterion, device)
        
        # 组合指标
        combined_metrics = {"train_loss": avg_train_loss}
        combined_metrics.update(val_metrics)
        
        # 保存指标到CSV
        output_mgr.save_metrics_csv(combined_metrics, epoch + 1)
        
        # 显示epoch总结
        train_metrics = {"loss": avg_train_loss}
        monitor.print_epoch_summary(epoch + 1, train_metrics, val_metrics)
        
        # 保存最佳模型
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            output_mgr.save_model(model, epoch + 1, val_metrics)
            print(f"✅ Saved best model at epoch {epoch + 1} with val_loss {val_metrics['val_loss']:.4f}")
        
        # 保存可视化结果（可选）
        if args.save_viz and (epoch + 1) % max(1, args.epochs // 3) == 0:
            print(f"🎨 Saving visualizations for epoch {epoch + 1}...")
            viz_dir = os.path.join(output_mgr.get_vis_dir(), f"epoch_{epoch + 1:03d}")
            save_visualizations(model, val_loader, viz_dir, args, device)
    
    # 训练完成总结
    summary = output_mgr.get_run_summary()
    print("\n" + "=" * 80)
    print("🎉 Training Completed!")
    print(f"📂 Results saved to: {summary['run_dir']}")
    print(f"📈 Best validation metrics:")
    print(f"   Loss: {best_val_loss:.4f}")
    print(f"⏱️  Total training time: {monitor.get_elapsed_time()}")
    
    # 最终可视化保存
    if args.save_viz:
        print("🎨 Saving final visualizations...")
        final_viz_dir = output_mgr.get_vis_dir()
        save_visualizations(model, val_loader, final_viz_dir, args, device)

if __name__ == "__main__":
    main()
