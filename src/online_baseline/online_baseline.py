"""
# 基本用法：只处理前1000帧
python online_baseline.py --input video01.mp4 --output output_video.mp4 --max_frames 1000
建议把input的video放到本文件夹中，方便操作
output的视频也将被存在本文件夹中
"""
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque
import random
from mobile_unet import MobileUNet
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime

class VideoFrameDataset(Dataset):
    """视频帧数据集"""
    def __init__(self, frames, transform=None):
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.transform:
            frame = self.transform(frame)
        return frame

class FrameBuffer:
    """帧缓冲区，用于存储处理后的帧"""
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add_frame(self, frame):
        """添加帧到缓冲区"""
        self.buffer.append(frame)

    def get_frames(self):
        """获取所有帧"""
        return list(self.buffer)

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()

class OnlineLearner:
    """在线学习器，管理模型训练"""
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

        # 用于自监督学习的缓冲区
        self.replay_buffer = deque(maxlen=1000)

        # 记录训练指标
        self.loss_history = []
        self.miou_history = []

        # 初始化模型权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        print("模型权重已初始化")

    def calculate_miou(self, pred, target):
        """计算mIoU（平均交并比）"""
        # 将预测值转换为二值掩码
        pred_binary = (torch.sigmoid(pred) > 0.5).float()

        # 计算交集和并集
        intersection = (pred_binary * target).sum(dim=(1, 2, 3))
        union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection

        # 避免除以零
        iou = (intersection + 1e-6) / (union + 1e-6)

        # 返回平均IoU
        return iou.mean().item()

    def self_supervised_loss(self, pred1, pred2):
        """自监督损失函数 - 鼓励相邻帧的预测一致性"""
        return F.mse_loss(pred1, pred2)

    def update(self, frames):
        """使用帧序列更新模型"""
        if len(frames) < 2:
            return 0, 0  # 需要至少两帧进行自监督学习

        self.model.train()
        total_loss = 0
        total_miou = 0
        update_count = 0

        # 将帧添加到回放缓冲区
        for frame in frames:
            if len(self.replay_buffer) < self.replay_buffer.maxlen:
                self.replay_buffer.append(frame)

        # 从回放缓冲区采样进行训练
        if len(self.replay_buffer) >= 2:
            # 随机选择两帧进行自监督学习
            idx1, idx2 = random.sample(range(len(self.replay_buffer)), 2)
            frame1 = self.replay_buffer[idx1].unsqueeze(0).to(self.device)
            frame2 = self.replay_buffer[idx2].unsqueeze(0).to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            pred1 = self.model(frame1)
            pred2 = self.model(frame2)

            # 计算损失 - 结合分割损失和自监督一致性损失
            seg_loss1 = self.criterion(pred1, torch.sigmoid(pred1.detach()))
            seg_loss2 = self.criterion(pred2, torch.sigmoid(pred2.detach()))
            seg_loss = (seg_loss1 + seg_loss2) / 2

            consistency_loss = self.self_supervised_loss(pred1, pred2)
            loss = seg_loss + 0.1 * consistency_loss  # 加权组合

            # 计算mIoU (使用自生成的伪标签)
            with torch.no_grad():
                target1 = (torch.sigmoid(pred1.detach()) > 0.5).float()
                target2 = (torch.sigmoid(pred2.detach()) > 0.5).float()
                miou1 = self.calculate_miou(pred1, target1)
                miou2 = self.calculate_miou(pred2, target2)
                miou = (miou1 + miou2) / 2

            # 反向传播和优化
            loss.backward()
            self.optimizer.step()

            total_loss = loss.item()
            total_miou = miou
            update_count = 1

            # 记录指标
            self.loss_history.append(total_loss)
            self.miou_history.append(total_miou)

        return total_loss, total_miou

    def plot_metrics(self, save_path=None):
        """绘制训练指标曲线"""
        if not self.loss_history or not self.miou_history:
            print("没有足够的训练数据来绘制图表")
            return

        plt.figure(figsize=(12, 5))

        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history)
        plt.title('Training Loss')
        plt.xlabel('Update Steps')
        plt.ylabel('Loss')
        plt.grid(True)

        # 绘制mIoU曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.miou_history)
        plt.title('mIoU Score')
        plt.xlabel('Update Steps')
        plt.ylabel('mIoU')
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"指标图表已保存: {save_path}")
        else:
            plt.show()

def process_video(input_video_path, output_video_path, save_model=False, model_save_path=None, max_frames=None, max_steps=None):
    """处理视频的主函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化模型
    model = MobileUNet(n_channels=3, n_classes=1)  # 输入RGB图像，输出单通道掩码

    # 初始化在线学习器
    learner = OnlineLearner(model, device, learning_rate=0.0001)

    # 初始化帧缓冲区
    frame_buffer = FrameBuffer()

    # 打开视频文件
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {input_video_path}")
        return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 确保输入尺寸为512×512
    if width != 512 or height != 512:
        print(f"警告: 输入视频尺寸为 {width}×{height}，将调整为512×512")
        width, height = 512, 512

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("开始处理视频...")
    if max_frames:
        print(f"最多处理 {max_frames} 帧")
    if max_steps:
        print(f"最多进行 {max_steps} 次训练步骤")

    frame_count = 0
    step_count = 0
    batch_frames = []

    # 创建输出目录
    os.makedirs("output_metrics", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    while True:
        # 检查是否达到最大帧数限制
        if max_frames and frame_count >= max_frames:
            print(f"已达到最大帧数限制 ({max_frames} 帧)，停止处理")
            break

        ret, frame = cap.read()
        if not ret:
            break

        # 调整帧尺寸为512×512
        if frame.shape[0] != 512 or frame.shape[1] != 512:
            frame = cv2.resize(frame, (512, 512))

        # 转换帧格式 (BGR to RGB and HWC to CHW)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb.transpose(2, 0, 1)).float() / 255.0

        # 使用模型预测掩码
        with torch.no_grad():
            model.eval()
            mask = model(frame_tensor.unsqueeze(0).to(device))
            mask = torch.sigmoid(mask).squeeze().cpu().numpy()

        # 将掩码转换为二值图像
        mask_binary = (mask > 0.5).astype(np.uint8) * 255

        # 创建彩色掩码（红色）
        color_mask = np.zeros_like(frame)
        color_mask[:, :, 2] = mask_binary  # 红色通道

        # 将原帧与掩码叠加
        alpha = 0.5
        blended = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)

        # 将处理后的帧添加到输出队列
        frame_buffer.add_frame(blended)
        out.write(blended)

        # 收集帧用于在线学习
        batch_frames.append(frame_tensor)

        # 每处理10帧进行一次模型更新
        if len(batch_frames) >= 10:
            # 检查是否达到最大训练步数限制
            if max_steps and step_count >= max_steps:
                print(f"已达到最大训练步数限制 ({max_steps} 步)，停止训练")
                # 清空批次，但继续处理剩余帧
                batch_frames = []
            else:
                loss, miou = learner.update(batch_frames)
                if loss > 0:
                    step_count += 1
                    print(f"已处理 {frame_count} 帧, 训练步数: {step_count}, 损失: {loss:.4f}, mIoU: {miou:.4f}")
                batch_frames = []

        frame_count += 1

        # 每100帧保存一次指标图表
        if frame_count % 100 == 0 and frame_count > 0:
            chart_path = f"output_metrics/metrics_{timestamp}_frame_{frame_count}.png"
            learner.plot_metrics(chart_path)

        # 显示进度
        if frame_count % 50 == 0:
            print(f"已处理 {frame_count} 帧")

    # 释放资源
    cap.release()
    out.release()

    # 保存最终指标图表
    final_chart_path = f"output_metrics/metrics_{timestamp}_final.png"
    learner.plot_metrics(final_chart_path)

    # 保存更新后的模型（如果用户指定）
    if save_model and model_save_path:
        torch.save(model.state_dict(), model_save_path)
        print(f"模型已保存: {model_save_path}")
    elif save_model:
        # 如果没有指定保存路径，使用默认路径
        default_path = f"mobile_unet_model_{timestamp}.pth"
        torch.save(model.state_dict(), default_path)
        print(f"模型已保存: {default_path}")

    print(f"视频处理完成! 共处理 {frame_count} 帧，进行 {step_count} 次训练")
    print(f"输出视频已保存: {output_video_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='使用MobileUNet处理视频并进行在线学习')
    parser.add_argument('--input', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output', type=str, required=True, help='输出视频路径')
    parser.add_argument('--save_model', action='store_true', help='是否保存训练后的模型')
    parser.add_argument('--model_path', type=str, default=None, help='模型保存路径（可选）')
    parser.add_argument('--max_frames', type=int, default=None, help='最大处理帧数（可选）')
    parser.add_argument('--max_steps', type=int, default=None, help='最大训练步数（可选）')

    args = parser.parse_args()

    # 处理视频
    process_video(args.input, args.output, args.save_model, args.model_path, args.max_frames, args.max_steps)