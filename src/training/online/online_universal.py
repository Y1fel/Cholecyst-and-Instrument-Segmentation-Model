# src/training/online/online_universal.py
"""
通用在线学习模板 - 集成监控、可视化、评估功能
基于adaptive_unet_main.py和train_offline_universal.py改进
移除模型保存功能，改为定期可视化
"""
import argparse
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 路径修正，确保src包可import
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

# 导入通用模块
from src.eval.evaluator import Evaluator
from src.viz.visualizer import Visualizer
from src.common.output_manager import OutputManager
from src.common.train_monitor import TrainMonitor
from src.common.pseudo_label_quality import quality_filter

# 模型导入
from src.models.model_zoo import build_model
from src.dataio.datasets.seg_dataset_min import SegDatasetMin

#导入工具类
from utils.class_frame_extractor import VideoFrameExtractor

# 帧选择器（从adaptive_unet_main.py移植）
class FrameSelector:
    """帧选择器，根据帧间相似度决定处理策略"""

    def __init__(self, threshold=0.9, adaptive=True, window_size=5):
        self.threshold = threshold
        self.adaptive = adaptive
        self.window_size = window_size
        self.last_frame = None
        self.similarity_history = deque(maxlen=10)

    def simplified_ssim(self, frame1, frame2, size=64):
        """简化版SSIM计算，提高计算速度"""
        # 先将帧下采样到较小尺寸以减少计算量
        frame1_small = torch.nn.functional.interpolate(
            frame1.unsqueeze(0), size=(size, size), mode='bilinear'
        ).squeeze(0)
        frame2_small = torch.nn.functional.interpolate(
            frame2.unsqueeze(0), size=(size, size), mode='bilinear'
        ).squeeze(0)

        # 计算均值和方差
        mu1 = frame1_small.mean()
        mu2 = frame2_small.mean()

        sigma1_sq = ((frame1_small - mu1) ** 2).mean()
        sigma2_sq = ((frame2_small - mu2) ** 2).mean()
        sigma12 = ((frame1_small - mu1) * (frame2_small - mu2)).mean()

        # 简化SSIM计算
        C1 = (0.01 * 1) ** 2
        C2 = (0.03 * 1) ** 2

        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean().item()

    def should_process_all(self, frames):
        """
        判断是否需要处理所有帧
        frames: (5, C, H, W) 的张量
        返回: True如果需要处理所有帧，False如果只需处理中间帧
        """
        if self.last_frame is None:
            self.last_frame = frames[2]  # 保存中间帧作为参考
            self.similarity_history.append(0.5)  # 初始值
            return True  # 第一组帧总是处理所有

        # 计算当前帧与上一参考帧的相似度
        current_similarity = self.simplified_ssim(self.last_frame, frames[2])
        self.similarity_history.append(current_similarity)

        # 自适应阈值计算
        if self.adaptive and len(self.similarity_history) > 5:
            # 计算相似度的均值和标准差
            mean_sim = np.mean(self.similarity_history)
            std_sim = np.std(self.similarity_history)
            # 自适应阈值：均值减去0.5倍标准差
            adaptive_threshold = max(0.7, min(0.95, float(mean_sim - 0.5 * std_sim)))
        else:
            adaptive_threshold = float(self.threshold)

        # 更新参考帧
        self.last_frame = frames[2]

        # 如果相似度低于阈值，说明有较大变化，需要处理所有帧
        return current_similarity < adaptive_threshold


# 经验回放缓冲区（从adaptive_unet_main.py移植）
class ExperienceReplayBuffer:
    """经验回放缓冲区，用于存储和采样历史数据"""

    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, target):
        """将数据添加到缓冲区"""
        self.buffer.append((state, target))

    def sample(self, batch_size):
        """从缓冲区随机采样一批数据"""
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def parse_args():
    """参数配置 - 结合离线训练和在线学习的需求"""
    p = argparse.ArgumentParser("Online Universal Trainer")

    # 基础训练参数
    p.add_argument("--cfg", type=str, default=None, help="Optional YAML config")
    #p.add_argument("--data_root", type=str, required=True, help="Dataset root path")
    p.add_argument("--video_root", type=str, default=None, help="Video root path")

    # 数据参数
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)

    # 训练参数
    p.add_argument("--online_steps", type=int, default=100, help="Number of online learning steps")
    p.add_argument("--lr", type=float, default=1e-3)

    # 监控和输出参数
    p.add_argument("--monitor_interval", type=int, default=10, help="Progress update interval (steps)")
    p.add_argument("--enable_gpu_monitor", action='store_true', default=True, help="Enable GPU monitoring")
    p.add_argument("--eval_interval", type=int, default=20, help="Evaluation interval (steps)")
    p.add_argument("--save_viz", action='store_true', help="Save visualizations")
    p.add_argument("--viz_interval", type=int, default=50, help="Visualization interval (steps)")
    p.add_argument("--viz_samples", type=int, default=20, help="Number of visualization samples")

    # 在线学习特定参数
    p.add_argument("--use_frame_selector", action='store_true', default=True, help="Use frame selector for efficiency")
    p.add_argument("--use_replay_buffer", action='store_true', default=True, help="Use experience replay buffer")
    p.add_argument("--replay_capacity", type=int, default=1000, help="Replay buffer capacity")
    p.add_argument("--frame_selector_threshold", type=float, default=0.85, help="Frame similarity threshold")

    # 任务定义
    p.add_argument("--binary", action="store_true",
                   help="二分类（胆囊+器械=前景=1）。若关闭则按多类训练。")
    p.add_argument("--num_classes", type=int, default=2,
                   help="多类时>=2；--binary 生效时忽略此项。")

    # 模型选择
    p.add_argument("--model", type=str, default="adaptive_unet",
                   choices=["unet_min", "mobile_unet", "adaptive_unet"])

    # 兼容 OutputManager 的模型类型标记
    p.add_argument("--model_type", type=str, default=None,
                   help="若不指定，将自动使用 --model 的值。")

    # 离线模型参数
    p.add_argument("--offline_model_path", type=str, help="Path to the offline model checkpoint")
    p.add_argument("--offline_model_name", type=str, help="Model name for the offline model")
    p.add_argument("--offline_num_classes", type=int, help="Number of classes for the offline model")

    return p.parse_args()


# ==== 离线模型伪标签推理工具 ==== #
def load_offline_model(model_path, model_name, num_classes, device):
    from src.models.model_zoo import build_model
    model = build_model(model_name, num_classes=num_classes, in_ch=3)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    return model


class OnlineLearner:
    """在线学习器，管理模型训练和缓冲区"""

    def __init__(self, model, device, args, output_mgr, monitor,
                 offline_model_path=None, offline_model_name=None, offline_num_classes=None):
        self.model = model.to(device)
        self.device = device
        self.args = args
        self.output_mgr = output_mgr
        self.monitor = monitor

        # 初始化帧选择器和回放缓冲区
        self.frame_selector = FrameSelector(
            threshold=args.frame_selector_threshold, adaptive=True
        ) if args.use_frame_selector else None

        self.buffer = ExperienceReplayBuffer(
            args.replay_capacity
        ) if args.use_replay_buffer else None

        # 损失函数和优化器
        if args.binary:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        # 只优化适配器参数（对于自适应模型）
        # adapter_params = []
        # for name, param in self.model.named_parameters():
        #     if 'adapter' in name or not args.use_frame_selector:
        #         adapter_params.append(param)
        # 处理adapter_params为空的情况
        adapter_params = [p for n, p in self.model.named_parameters() if 'adapter' in n]
        if len(adapter_params) == 0:
            print("Warning: no adapter params found — using all trainable params")
            adapter_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.Adam(adapter_params, lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # 评估器
        self.evaluator = Evaluator(device=device, threshold=0.5)

        # 训练状态
        self.loss_history = []
        self.metrics_history = []

        # ==== 离线模型加载（伪标签生成用）==== #
        self.offline_model = None
        if offline_model_path and offline_model_name and offline_num_classes:
            self.offline_model = load_offline_model(
                offline_model_path, offline_model_name, offline_num_classes, device)

    def process_single_frame(self, x):
        """处理单帧的辅助函数"""
        return self.model(x.unsqueeze(0)).squeeze(0)

    def adapt_step(self, inputs, targets):
        """执行一步在线适应"""
        self.model.train()
        self.optimizer.zero_grad()

        processed_outputs = []
        processed_targets = []
        frame_stats = {'all_frames': 0, 'single_frame': 0}

        # 处理每个样本的帧
        for i in range(inputs.size(0)):
            frames = inputs[i]  # (5, C, H, W)
            target_frames = targets[i]  # (5, C, H, W)

            # 检查是否需要处理所有帧
            if self.frame_selector is None or self.frame_selector.should_process_all(frames):
                frame_stats['all_frames'] += 1
                # 处理所有5帧
                for j in range(frames.size(0)):
                    output = self.process_single_frame(frames[j])
                    processed_outputs.append(output)
                    processed_targets.append(target_frames[j])
            else:
                frame_stats['single_frame'] += 1
                # 只处理中间帧（第3帧，索引2），然后复制结果
                middle_output = self.process_single_frame(frames[2])
                for j in range(frames.size(0)):
                    processed_outputs.append(middle_output)
                    processed_targets.append(target_frames[j])

        # 转换为张量
        outputs = torch.stack(processed_outputs, dim=0)
        targets = torch.stack(processed_targets, dim=0)

        # 从缓冲区采样历史数据
        if self.buffer is not None:
            replay_data = self.buffer.sample(min(8, len(self.buffer)))
            if replay_data is not None:
                replay_inputs, replay_targets = zip(*replay_data)
                replay_inputs = torch.stack(replay_inputs).to(self.device)
                replay_targets = torch.stack(replay_targets).to(self.device)

                # 合并新旧数据
                outputs = torch.cat([outputs, replay_inputs], dim=0)
                targets = torch.cat([targets, replay_targets], dim=0)

        # 计算损失
        loss = self.criterion(outputs, targets)

        # 反向传播和优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 将当前数据添加到缓冲区
        if self.buffer is not None:
            for i in range(outputs.size(0)):
                self.buffer.push(
                    outputs[i].detach().cpu(),
                    targets[i].detach().cpu()
                )

        return loss.item(), frame_stats

    @torch.no_grad()
    def evaluate(self, data_loader):
        """评估模型性能"""
        self.model.eval()
        return self.evaluator.evaluate(self.model, data_loader, self.criterion)

    def run_online_learning(self, train_loader, val_loader=None):
        """运行在线学习循环"""
        print("=" * 80)
        print(f"Starting Online Learning ({self.args.model}) for {self.args.online_steps} steps...")

        # 初始化可视化器
        visualizer = Visualizer()
        viz_dir = self.output_mgr.get_vis_dir()

        # 初始评估
        if val_loader is not None:
            initial_metrics = self.evaluate(val_loader)
            self.metrics_history.append(initial_metrics)
            print(f"Initial Metrics - Loss: {initial_metrics['val_loss']:.4f}, IoU: {initial_metrics['iou']:.4f}")

        # 在线学习循环
        for step, (images, _) in enumerate(train_loader):
            if step >= self.args.online_steps: break;
            images = images.to(self.device)
            # ==== 伪标签生成与训练 ====
            if self.offline_model is not None:
                # 先帧选择，再伪标签生成
                batch_size, frames, c, h, w = images.shape  # (B, 5, C, H, W)
                selected_indices = []
                for i in range(batch_size):
                    frame_group = images[i]  # (5, C, H, W)
                    if self.frame_selector is None or self.frame_selector.should_process_all(frame_group):
                        selected_indices.extend([(i, j) for j in range(frames)])
                    else:
                        selected_indices.append((i, 2))  # 只选中间帧
                if not selected_indices:
                    print(f"Step {step}: No frames selected for pseudo-labeling, skipping batch.")
                    continue
                # 收集选中的帧
                selected_imgs = torch.stack([
                    images[i, j] for (i, j) in selected_indices
                ]).to(self.device)
                with torch.no_grad():
                    logits = self.offline_model(selected_imgs)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    pseudo_labels = torch.argmax(probs, axis=1)
                    # 伪标签质控
                    mask = quality_filter(probs, pseudo_labels)
                    if not np.any(mask):
                        print(f"Step {step}: No pseudo labels passed quality control, skipping batch.")
                        continue
                    pseudo_labels = torch.from_numpy(pseudo_labels[mask]).to(self.device)
                    selected_imgs = selected_imgs[mask]
                # 用伪标签训练online模型
                self.model.train()
                self.optimizer.zero_grad()
                outputs = self.model(selected_imgs)
                if self.args.binary:
                    loss = self.criterion(outputs, pseudo_labels.float())
                else:
                    loss = self.criterion(outputs, pseudo_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.loss_history.append(loss.item())
                frame_stats = {'all_frames': 0, 'single_frame': 0} # 伪标签流程不统计帧选择
            else:
                loss = None
                frame_stats = None

            # 监控进度
            if step % self.args.monitor_interval == 0:
                if loss is not None:
                    self.monitor.print_progress(
                        step + 1, self.args.online_steps,
                        0, 0,
                        {"loss": loss},
                        refresh=True
                    )
                if self.frame_selector is not None and frame_stats is not None:
                    total = frame_stats['all_frames'] + frame_stats['single_frame']
                    if total > 0:
                        all_percent = 100 * frame_stats['all_frames'] / total
                        single_percent = 100 * frame_stats['single_frame'] / total
                        print(f"  Frame processing: All={all_percent:.1f}%, Single={single_percent:.1f}%")

            # 定期评估
            if val_loader is not None and step % self.args.eval_interval == 0:
                if loss is not None:
                    val_metrics = self.evaluate(val_loader)
                    self.metrics_history.append(val_metrics)
                    combined_metrics = {"step": step, "train_loss": loss}
                    combined_metrics.update(val_metrics)
                    self.output_mgr.save_metrics_csv(combined_metrics, step)
                    print(
                        f"[Step {step}] "
                        f"Val loss: {val_metrics['val_loss']:.4f} | "
                        f"IoU: {val_metrics['iou']:.4f} | Dice: {val_metrics['dice']:.4f}"
                    )

            # 定期可视化
            if self.args.save_viz and step % self.args.viz_interval == 0 and val_loader is not None:
                print(f"Generating visualizations at step {step}...")
                step_viz_dir = os.path.join(viz_dir, f"step_{step}")
                os.makedirs(step_viz_dir, exist_ok=True)

                visualizer.save_comparison_predictions(
                    self.model, val_loader, step_viz_dir,
                    max_samples=self.args.viz_samples, device=self.device
                )


        # 最终可视化
        if self.args.save_viz and val_loader is not None:
            print("Generating final visualizations...")
            final_viz_dir = os.path.join(viz_dir, "final")
            os.makedirs(final_viz_dir, exist_ok=True)

            visualizer.save_comparison_predictions(
                self.model, val_loader, final_viz_dir,
                max_samples=self.args.viz_samples, device=self.device
            )

        self.last_results = (self.loss_history, self.metrics_history)
        return self.last_results

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化监控器和输出管理器
    monitor = TrainMonitor(enable_gpu_monitor=args.enable_gpu_monitor)
    monitor.start_timing()

    model_tag = args.model if args.model_type is None else args.model_type
    output_mgr = OutputManager(model_type=model_tag, run_type="online")
    output_mgr.save_config(vars(args))

    # 创建模型
    num_classes = 1 if args.binary else args.num_classes
    model = build_model(args.model, num_classes=num_classes, in_ch=3)

    # 初始化在线学习器
    learner = OnlineLearner(
        model, device, args, output_mgr, monitor,
        offline_model_path=args.offline_model_path,
        offline_model_name=args.offline_model_name,
        offline_num_classes=args.offline_num_classes
    )
    def train_fn(batch_frames):
        """
        每当切好一批帧就触发，用 learner 进行在线训练。
        batch_frames: [str] (帧目录路径列表)
        """
        # 创建数据集
        dataset = SegDatasetMin(batch_frames, dtype=args.split, img_size=args.img_size)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # 创建验证集用于评估和可视化
        val_dataset = SegDatasetMin(batch_frames, dtype="val", img_size=args.img_size)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        # 在线学习入口
        learner.run_online_learning(data_loader, val_loader)

    # 启动切帧（异步+并行训练）
    extractor = VideoFrameExtractor(output_dir="") # 自定义输出路径
    extractor.extract(
        video_path=args.video_root,
        fps=2,
        start=10,
        end=60,
        size=(args.img_size, args.img_size),
        fmt="png",
        batch_size=5,
        mode=2,
        train_fn=train_fn  # 绑定训练回调
    )
    loss_history, metrics_history = learner.last_results
    # 打印总结
    summary = output_mgr.get_run_summary()
    final_metrics = metrics_history[-1] if metrics_history else {}
    print(f"\n--> Online Learning Completed <--")
    print(f"Results saved to: {summary['run_dir']}")
    if final_metrics:
        print(f"Final Metrics - Loss: {final_metrics['val_loss']:.4f}, IoU: {final_metrics['iou']:.4f}")


if __name__ == "__main__":
    main()

