import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import deque
import random
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.online.adaptive_unet import AdaptiveUNet


class FrameSelector:
    """帧选择器，根据帧间相似度决定处理策略"""

    def __init__(self, threshold=0.9, adaptive=True, window_size=5):
        self.threshold = threshold
        self.adaptive = adaptive
        self.window_size = window_size
        self.last_frame = None
        self.similarity_history = deque(maxlen=10)  # 存储最近10次的相似度值

    def simplified_ssim(self, frame1, frame2, size=64):
        """简化版SSIM计算，提高计算速度"""
        # 先将帧下采样到较小尺寸以减少计算量
        frame1_small = F.interpolate(frame1.unsqueeze(0), size=(size, size), mode='bilinear').squeeze(0)
        frame2_small = F.interpolate(frame2.unsqueeze(0), size=(size, size), mode='bilinear').squeeze(0)

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
            adaptive_threshold = max(0.7, min(0.95, mean_sim - 0.5 * std_sim))
        else:
            adaptive_threshold = self.threshold

        # 更新参考帧
        self.last_frame = frames[2]

        # 如果相似度低于阈值，说明有较大变化，需要处理所有帧
        return current_similarity < adaptive_threshold


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


class OnlineLearner:
    """在线学习器，管理模型训练和缓冲区"""

    def __init__(self, model, device, buffer_capacity=1000, learning_rate=0.001,
                 frame_selector=None):
        self.model = model.to(device)
        self.device = device
        self.buffer = ExperienceReplayBuffer(buffer_capacity)
        self.criterion = nn.BCELoss()
        self.frame_selector = frame_selector or FrameSelector()

        # 只优化适配器参数
        adapter_params = []
        for name, param in self.model.named_parameters():
            if 'adapter' in name:
                adapter_params.append(param)

        self.optimizer = optim.Adam(adapter_params, lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    def process_single_frame(self, x):
        """处理单帧的辅助函数"""
        return self.model(x.unsqueeze(0)).squeeze(0)

    def adapt(self, data_loader, steps=100, eval_loader=None):
        """在线适应新数据"""
        self.model.train()
        losses = []
        eval_scores = []
        frame_stats = {'all_frames': 0, 'single_frame': 0}

        for step in range(steps):
            try:
                # 获取新数据 (假设每个样本是5帧)
                inputs, targets = next(iter(data_loader))
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # 处理每个样本的帧
                processed_outputs = []
                processed_targets = []

                for i in range(inputs.size(0)):
                    frames = inputs[i]  # (5, C, H, W)
                    target_frames = targets[i]  # (5, C, H, W)

                    # 检查是否需要处理所有帧
                    if self.frame_selector.should_process_all(frames):
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
                replay_data = self.buffer.sample(min(8, len(self.buffer)))
                if replay_data is not None:
                    replay_inputs, replay_targets = zip(*replay_data)
                    replay_inputs = torch.stack(replay_inputs).to(self.device)
                    replay_targets = torch.stack(replay_targets).to(self.device)

                    # 合并新旧数据
                    outputs = torch.cat([outputs, replay_inputs], dim=0)
                    targets = torch.cat([targets, replay_targets], dim=0)

                # 前向传播
                self.optimizer.zero_grad()
                loss = self.criterion(outputs, targets)

                # 反向传播和优化
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                losses.append(loss.item())

                # 将当前数据添加到缓冲区
                for i in range(outputs.size(0)):
                    self.buffer.push(outputs[i].detach().cpu(), targets[i].detach().cpu())

                # 定期评估
                if step % 10 == 0 and eval_loader is not None:
                    eval_score = self.evaluate(eval_loader)
                    eval_scores.append(eval_score)
                    print(f"Step {step}: Loss={loss.item():.4f}, Eval Score={eval_score:.4f}")

                    # 打印帧处理统计
                    total = frame_stats['all_frames'] + frame_stats['single_frame']
                    if total > 0:
                        all_percent = 100 * frame_stats['all_frames'] / total
                        single_percent = 100 * frame_stats['single_frame'] / total
                        print(f"  Frame processing: All={all_percent:.1f}%, Single={single_percent:.1f}%")

            except StopIteration:
                # 数据加载器耗尽，重新创建
                data_loader = DataLoader(data_loader.dataset, batch_size=data_loader.batch_size, shuffle=True)

        return losses, eval_scores

    def evaluate(self, data_loader):
        """评估模型性能"""
        self.model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        self.model.train()
        return total_loss / total_samples

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# 模拟多帧数据集类
class MultiFrameDataset(Dataset):
    def __init__(self, size=100, img_size=(64, 64), num_frames=5):
        self.size = size
        self.img_size = img_size
        self.num_frames = num_frames

        # 生成基础图像和变化模式
        self.base_images = [torch.rand(1, *img_size) for _ in range(size)]
        self.variation_patterns = [torch.randn(num_frames, 1, *img_size) * 0.1 for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 生成连续帧序列
        base = self.base_images[idx]
        variation = self.variation_patterns[idx]

        # 创建帧序列
        frames = []
        masks = []

        for i in range(self.num_frames):
            # 图像帧 = 基础图像 + 变化模式
            frame = base + variation[i]
            frame = torch.clamp(frame, 0, 1)

            # 模拟掩码（这里简单处理）
            mask = torch.sigmoid(frame * 5 - 2.5)  # 模拟二值化掩码

            frames.append(frame)
            masks.append(mask)

        # 转换为张量 (num_frames, C, H, W)
        frames = torch.stack(frames, dim=0)
        masks = torch.stack(masks, dim=0)

        return frames, masks


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型
    model = AdaptiveUNet(in_channels=1, out_channels=1)

    # 初始化帧选择器（使用自适应阈值）
    frame_selector = FrameSelector(threshold=0.85, adaptive=True)

    # 初始化在线学习器
    learner = OnlineLearner(
        model,
        device,
        buffer_capacity=1000,
        learning_rate=0.001,
        frame_selector=frame_selector
    )

    # 模拟在线学习场景
    num_domains = 3
    results = {}

    for domain_id in range(num_domains):
        print(f"\n=== Adapting to Domain {domain_id + 1} ===")

        # 创建模拟多帧数据集
        train_dataset = MultiFrameDataset(size=50, num_frames=5)
        eval_dataset = MultiFrameDataset(size=20, num_frames=5)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False)

        # 在线适应
        losses, eval_scores = learner.adapt(
            train_loader,
            steps=30,
            eval_loader=eval_loader
        )

        # 保存结果
        results[domain_id] = {
            'losses': losses,
            'eval_scores': eval_scores
        }

        # 保存模型检查点
        checkpoint_path = f"checkpoints/domain_{domain_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        learner.save_model(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    # 绘制学习曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for domain_id, result in results.items():
        plt.plot(result['losses'], label=f'Domain {domain_id + 1}')
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for domain_id, result in results.items():
        plt.plot(result['eval_scores'], label=f'Domain {domain_id + 1}')
    plt.title('Evaluation Score')
    plt.xlabel('Steps (x10)')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('online_learning_curves.png')
    print("Learning curves saved to online_learning_curves.png")


if __name__ == "__main__":
    main()