# src/training/online/online_universal.py
"""
合并版：保留 student-teacher 增强（AMP、best save 等），并恢复 origin 中的 adapt_step / evaluate / frame-selector /
experience-replay / 定期评估与可视化逻辑。
"""
import math
import glob
import cv2
import argparse
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt

# 路径修正，确保src包可import
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

# 通用模块
from pathlib import Path
from utils.class_frame_to_video import VideoFrameMerger
from utils.class_frame_extractor import VideoFrameExtractor
from src.viz.visualizer import Visualizer
from src.common.output_manager import OutputManager
from src.common.train_monitor import TrainMonitor
from src.common.pseudo_label_quality import (
    quality_filter, denoise_pseudo_label, pixel_gate_mask, mask_quality_filter_with_pixel_mask
)
from src.common.ema_safety import EMASafetyManager

# 模型导入
from src.models.model_zoo import build_model
from src.dataio.datasets.seg_dataset_min import SegDatasetMin

# AMP
try:
    from torch.cuda.amp import autocast, GradScaler
except Exception:
    autocast = None
    GradScaler = None


def parse_args():
    p = argparse.ArgumentParser("Online Universal Trainer (merged)")

    # 基本和数据参数
    p.add_argument("--video_root", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)

    # 训练参数
    p.add_argument("--online_steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)

    # 监控/可视化
    p.add_argument("--monitor_interval", type=int, default=10)
    p.add_argument("--enable_gpu_monitor", action='store_true', default=True)
    p.add_argument("--eval_interval", type=int, default=20)
    p.add_argument("--save_viz", action='store_true', help="Save per-step visualizations")
    p.add_argument("--viz_interval", type=int, default=5)
    p.add_argument("--viz_samples", type=int, default=20)

    # 在线相关
    p.add_argument("--use_frame_selector", action='store_true', default=True)
    p.add_argument("--use_replay_buffer", action='store_true', default=True)
    p.add_argument("--replay_capacity", type=int, default=1000)
    p.add_argument("--frame_selector_threshold", type=float, default=0.85)
    p.add_argument("--max_train_samples_per_step", type=int, default=8)

    # 任务/模型
    p.add_argument("--binary", action="store_true")
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--model", type=str, default="adaptive_unet",
                   choices=["unet_min", "mobile_unet", "adaptive_unet"])
    p.add_argument("--model_type", type=str, default=None)

    # 离线 teacher
    p.add_argument("--offline_model_path", type=str, help="Path to offline/teacher checkpoint")
    p.add_argument("--offline_model_name", type=str, help="Teacher model name")
    p.add_argument("--offline_num_classes", type=int, default=3)

    # student 已有 checkpoint
    p.add_argument("--student_model_path", type=str, help="Path to student checkpoint (required)")
    p.add_argument("--student_model_name", type=str, default=None)
    p.add_argument("--student_num_classes", type=int, default=None)

    # 其他增强
    p.add_argument("--save_pred_every_batch", action='store_true', default=True)
    p.add_argument("--use_amp", action='store_true', default=False)

    # 重新添加缺失的调度控制参数 (来自版本1)
    p.add_argument("--update_mode", type=str, default="fixed",
                   choices=["all", "fixed", "triggered"])
    p.add_argument("--update_interval", type=int, default=5)
    p.add_argument("--update_layers", type=str, default="bn",
                   choices=["all", "bn", "lastN"])
    p.add_argument("--update_last_n", type=int, default=2)

    return p.parse_args()


@torch.no_grad()
def compute_val_metrics(model, data_iter, criterion, device, num_classes=3, binary=False):
    """
    简单、健壮的验证函数（替代外部 Evaluator 的 evaluate）：
    - data_iter: iterable that yields (images, labels) where images: [B, C, H, W], labels shaped according to `binary`
    - returns dict: {"val_loss": float, "iou": float}
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_iou = 0.0

    for images, labels in data_iter:
        # ensure tensors
        images = images.to(device)
        if labels is not None:
            labels = labels.to(device)

        # forward
        logits = model(images)  # [N, C, H, W] or [N,1,H,W]
        # loss (make sure labels dtype matches criterion)
        if binary:
            # BCEWithLogitsLoss expects float targets [N,1,H,W]
            if labels.ndim == 3:
                labels_for_loss = labels.unsqueeze(1).float()
            else:
                labels_for_loss = labels.float()
        else:
            # CrossEntropyLoss expects long targets [N, H, W]
            if labels.ndim == 4 and labels.size(1) == 1:
                labels_for_loss = labels.squeeze(1).long()
            else:
                labels_for_loss = labels.long()

        loss = criterion(logits, labels_for_loss)
        b = images.size(0)
        total_loss += float(loss.item()) * b
        total_samples += b

        # predictions -> compute IoU
        if binary:
            probs = torch.sigmoid(logits)  # [N,1,H,W]
            preds = (probs > 0.5).long().squeeze(1)  # [N,H,W]
            gts = labels_for_loss.squeeze(1).long()  # [N,H,W]
            # per-sample IoU for foreground (class 1)
            inter = ((preds == 1) & (gts == 1)).sum(dim=(1, 2)).float()
            union = ((preds == 1) | (gts == 1)).sum(dim=(1, 2)).float()
            iou_per_sample = (inter / (union + 1e-6)).cpu().numpy()
            # handle case union==0 -> define IoU = 1.0 (both empty); we clip below
            iou_per_sample = [float(i) if not math.isnan(i) else 1.0 for i in iou_per_sample]
            total_iou += float(sum(iou_per_sample))
        else:
            # multiclass
            # logits: [N,C,H,W] -> pred: [N,H,W]
            preds = torch.argmax(logits, dim=1)  # [N,H,W]
            gts = labels_for_loss  # [N,H,W]
            # compute mean IoU across classes for this batch (includes background/class 0)
            batch_ious = []
            for cls in range(num_classes):
                pred_c = (preds == cls)
                gt_c = (gts == cls)
                inter = (pred_c & gt_c).sum(dim=(1, 2)).float()
                union = (pred_c | gt_c).sum(dim=(1, 2)).float()
                # per-sample IoU for this class
                iou_c = inter / (union + 1e-6)
                # treat union==0 (no gt and no pred for this class) as IoU=1.0
                iou_c = torch.where(union == 0, torch.ones_like(iou_c), iou_c)
                batch_ious.append(iou_c)  # list length=num_classes, each [N]
            # stack -> [num_classes, N] -> mean over classes then over samples
            batch_ious = torch.stack(batch_ious, dim=0)  # [C, N]
            # mean over classes, then sum over samples
            mean_iou_per_sample = batch_ious.mean(dim=0)  # [N]
            total_iou += float(mean_iou_per_sample.sum().cpu().numpy())

    if total_samples == 0:
        return {"val_loss": float("nan"), "iou": float("nan")}

    avg_loss = total_loss / total_samples
    avg_iou = total_iou / total_samples
    return {"val_loss": float(avg_loss), "iou": float(avg_iou)}


def flatten_val_loader(val_loader, binary: bool = False):
    """
    把 val_loader 的输出从 [B, T, C, H, W] 展平为 [B*T, C, H, W]
    并根据 binary 标志把 labels 转为正确的 dtype/shape:
      - binary=True  -> labels float, shape [N,1,H,W]
      - binary=False -> labels long,  shape [N,H,W]
    """
    for images, labels in val_loader:
        # images: [B, T, C, H, W] 或 [B, C, H, W]
        if images.ndim == 5:  # [B, T, C, H, W]
            b, t, c, h, w = images.shape
            images = images.view(b * t, c, h, w)
            if labels is not None:
                # labels 原为 [B, T, H, W] 或 [B, T, 1, H, W]（你 dataset 返回的是 [T,1,H,W]）
                # 目标最终希望是 [B*T, H, W] (multiclass) or [B*T,1,H,W] (binary)
                # 先 squeeze 可能存在的 channel dim
                if labels.ndim == 4:  # [B, T, H, W] already
                    labels = labels.view(b * t, h, w)
                elif labels.ndim == 5:  # [B, T, 1, H, W]
                    labels = labels.view(b * t, 1, h, w)
                else:
                    # 其它情况尝试展平
                    labels = labels.view(b * t, *labels.shape[2:])

        else:
            # images is [B, C, H, W], labels maybe [B, H, W] or [B,1,H,W]
            if labels is not None:
                if labels.ndim == 4:  # [B, 1, H, W]
                    labels = labels.view(labels.size(0), 1, labels.size(2), labels.size(3))
                # else assume [B, H, W], leave as is

        # 根据 binary 标志决定 labels 的 dtype/shape
        if labels is not None:
            if binary:
                # BCEWithLogitsLoss: inputs float [N,1,H,W], targets float [N,1,H,W]
                # 如果 labels 是 [N,H,W] -> unsqueeze channel dim
                if labels.ndim == 3:  # [N, H, W]
                    labels = labels.unsqueeze(1)  # -> [N,1,H,W]
                # to float
                if labels.dtype != torch.float32 and labels.dtype != torch.float:
                    labels = labels.float()
            else:
                # CrossEntropyLoss: inputs [N,C,H,W], targets long [N,H,W]
                if labels.ndim == 4 and labels.size(1) == 1:
                    labels = labels.squeeze(1)  # [N,H,W]
                if labels.dtype != torch.long:
                    labels = labels.long()

        yield images, labels


# -----------------------
# Dataset (保持 online_universal.py 的 OnlineFrameDataset)
# -----------------------
class OnlineFrameDataset(Dataset):
    def __init__(self, frame_paths, img_size=512, transform=None, sequence_length=5):
        self.frame_paths = frame_paths
        self.img_size = img_size
        self.transform = transform
        self.sequence_length = sequence_length

    def __len__(self):
        return max(0, len(self.frame_paths) - self.sequence_length + 1)

    def __getitem__(self, index):
        frames = []
        for i in range(index, index + self.sequence_length):
            img_path = self.frame_paths[i]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            frames.append(torch.from_numpy(img).float())
        frames = torch.stack(frames, dim=0)
        dummy_mask = torch.zeros(1, self.img_size, self.img_size)
        targets = torch.stack([dummy_mask] * self.sequence_length, dim=0)
        return frames, targets


# -----------------------
# FrameSelector (来自版本1)
# -----------------------
class FrameSelector:
    def __init__(self, threshold=0.92, adaptive=True, window_size=5, freq=30):
        self.threshold = threshold
        self.adaptive = adaptive
        self.window_size = window_size
        self.last_frame = None
        self.similarity_history = deque(maxlen=10)
        self.frame_count = 0
        self.freq = freq  # 每freq帧判断一次
        self.last_result = True

    def simplified_ssim(self, frame1, frame2, size=64):
        frame1_small = torch.nn.functional.interpolate(frame1.unsqueeze(0), size=(size, size), mode='bilinear').squeeze(
            0)
        frame2_small = torch.nn.functional.interpolate(frame2.unsqueeze(0), size=(size, size), mode='bilinear').squeeze(
            0)
        mu1 = frame1_small.mean()
        mu2 = frame2_small.mean()
        sigma1_sq = ((frame1_small - mu1) ** 2).mean()
        sigma2_sq = ((frame2_small - mu2) ** 2).mean()
        sigma12 = ((frame1_small - mu1) * (frame2_small - mu2)).mean()
        C1 = (0.01 * 1) ** 2
        C2 = (0.03 * 1) ** 2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean().item()

    def should_process_all(self, frames):
        self.frame_count += 1
        if self.frame_count % self.freq != 0:
            return self.last_result
        if self.last_frame is None:
            self.last_frame = frames[2]
            self.similarity_history.append(0.5)
            self.last_result = True
            return True
        current_similarity = self.simplified_ssim(self.last_frame, frames[2])
        self.similarity_history.append(current_similarity)
        if self.adaptive and len(self.similarity_history) > 5:
            mean_sim = np.mean(self.similarity_history)
            std_sim = np.std(self.similarity_history)
            adaptive_threshold = max(0.7, min(0.98, float(mean_sim - 0.5 * std_sim)))
        else:
            adaptive_threshold = float(self.threshold)
        self.last_frame = frames[2]
        self.last_result = current_similarity < adaptive_threshold
        return self.last_result


# -----------------------
# Experience Replay Buffer (来自 origin，保留 push clone)
# -----------------------
class ExperienceReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, target):
        # 保存 CPU clone，避免引用共享
        self.buffer.append((state.detach().cpu().clone(), target.detach().cpu().clone()))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size or batch_size == 0:
            return None
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# -----------------------
# 模型加载工具（teacher / student）
# -----------------------
def load_offline_model(model_path, model_name, num_classes, device):
    model = build_model(model_name, num_classes=num_classes, in_ch=3)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
    else:
        state = checkpoint
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def load_student_model(model_path, model_name, num_classes, device):
    model = build_model(model_name, num_classes=num_classes, in_ch=3)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
    else:
        state = checkpoint
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad = True
    model.to(device)
    model.train()
    return model


# -----------------------
# 设置可训练层的工具函数 (来自版本1)
# -----------------------
def set_trainable_layers(model, mode="bn", last_n=2):
    """
    根据剂量控制方式设置模型参数的 requires_grad。
    mode: all/bn/lastN
    last_n: 只在lastN时有效
    """
    for name, param in model.named_parameters():
        param.requires_grad = False
    if mode == "all":
        for param in model.parameters():
            param.requires_grad = True
    elif mode == "bn":
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                for p in m.parameters():
                    p.requires_grad = True
    elif mode == "lastN":
        # 只更新最后N个参数组
        params = list(model.named_parameters())
        for name, param in params[-last_n:]:
            param.requires_grad = True
    else:
        raise ValueError(f"未知的update_layers模式: {mode}")


# -----------------------
# OnlineLearner: 合并 adapt_step + student/teacher 逻辑
# -----------------------
class OnlineLearner:
    def __init__(self, model, device, args, output_mgr, monitor,
                 offline_model_path=None, offline_model_name=None, offline_num_classes=None):
        self.model = model.to(device)
        self.device = device
        self.args = args
        self.output_mgr = output_mgr
        self.monitor = monitor

        # 损失
        self.criterion = nn.BCEWithLogitsLoss() if args.binary else nn.CrossEntropyLoss()

        # 优化器：训练可训练参数（如果模型里有 adapter，只调整 adapter 的话可修改这里）
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            for p in self.model.parameters():
                p.requires_grad = True
            trainable_params = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # teacher (offline) 加载（可选）
        self.offline_model = None
        if offline_model_path and offline_model_name and offline_num_classes:
            self.offline_model = load_offline_model(offline_model_path, offline_model_name, offline_num_classes, device)

        # evaluator / viz / histories
        self.visualizer = Visualizer()
        self.loss_history = []
        self.metrics_history = []

        # replay & frame selector 保持持久化实例（origin 行为）
        self.buffer = ExperienceReplayBuffer(capacity=args.replay_capacity) if args.use_replay_buffer else None
        self.frame_selector = FrameSelector(
            threshold=args.frame_selector_threshold) if args.use_frame_selector else None

        # global step 和 best model 保存（来自新版）
        self.global_step = 0
        self.best_loss = float('inf')
        try:
            self.best_model_path = os.path.join(self.output_mgr.get_run_dir(), "student_best.pth")
        except Exception:
            self.best_model_path = os.path.join(project_root, "student_best.pth")

        # EMA 安全管理（通用）
        self.ema_safety = EMASafetyManager(self.model, alpha=0.99, loss_window_size=10, grad_explode_thresh=10.0,
                                           cooldown_period=5)

    def _save_best_model_if_improved(self, loss_val):
        if loss_val < self.best_loss:
            self.best_loss = loss_val
            try:
                torch.save({'model_state_dict': self.model.state_dict(), 'loss': loss_val, 'step': self.global_step},
                           self.best_model_path)
                print(f"[INFO] Saved improved student model to {self.best_model_path} (loss={loss_val:.6f})")
            except Exception as e:
                print(f"[WARN] Failed to save best model: {e}")

    def adapt_step(self, inputs, targets):
        """
        从 origin 迁移过来的 adapt_step：基于 frame_selector 决定是否用所有帧或仅中间帧，
        与 experience replay 混合，计算 loss 并更新（针对 student）。
        inputs: tensor (B, seq_len, C, H, W)
        targets: tensor (B, seq_len, ...) - 这里只是占位（通常为 dummy）
        返回: loss_val, frame_stats
        """
        self.model.train()
        self.optimizer.zero_grad()

        processed_outputs = []
        processed_targets = []
        frame_stats = {'all_frames': 0, 'single_frame': 0}

        for i in range(inputs.size(0)):
            frames = inputs[i]  # (seq_len, C, H, W)
            tgs = targets[i]  # (seq_len, ...)
            if self.frame_selector is None or self.frame_selector.should_process_all(frames):
                frame_stats['all_frames'] += 1
                for j in range(frames.size(0)):
                    out = self.model(frames[j].unsqueeze(0)).squeeze(0)
                    processed_outputs.append(out)
                    processed_targets.append(tgs[j])
            else:
                frame_stats['single_frame'] += 1
                middle_out = self.model(frames[frames.size(0) // 2].unsqueeze(0)).squeeze(0)
                for j in range(frames.size(0)):
                    processed_outputs.append(middle_out)
                    processed_targets.append(tgs[j])

        if len(processed_outputs) == 0:
            return None, frame_stats

        outputs = torch.stack(processed_outputs, dim=0)
        ttargets = torch.stack(processed_targets, dim=0)

        # 从 buffer 采样并拼接（origin 行为）
        if self.buffer is not None:
            n_replay = min(8, len(self.buffer))
            replay_data = self.buffer.sample(n_replay) if n_replay > 0 else None
            if replay_data is not None:
                replay_inputs, replay_targets = zip(*replay_data)
                replay_inputs = torch.stack(replay_inputs).to(self.device)
                replay_targets = torch.stack(replay_targets).to(self.device)
                outputs = torch.cat([outputs.to(self.device), replay_inputs], dim=0)
                ttargets = torch.cat([ttargets.to(self.device), replay_targets], dim=0)
            else:
                outputs = outputs.to(self.device)
                ttargets = ttargets.to(self.device)
        else:
            outputs = outputs.to(self.device)
            ttargets = ttargets.to(self.device)

        # 计算损失并更新（注意 outputs shape 需要和 criterion 匹配）
        loss = self.criterion(outputs, ttargets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # push 当前数据到 buffer（origin 行为）
        if self.buffer is not None:
            for i in range(outputs.size(0)):
                self.buffer.push(outputs[i].detach().cpu(), ttargets[i].detach().cpu())

        return float(loss.item()), frame_stats

    def run_online_learning(self, train_loader, val_loader=None):
        """
        主循环：整合 student 无梯度预测 -> frame selection -> teacher 伪标签 -> 限样 + replay -> student 更新
        并在周期内做评估/可视化/保存 best model。
        """
        print("=" * 80)
        print(f"Starting Online Learning ({self.args.model}) for {self.args.online_steps} steps...")

        viz_dir = self.output_mgr.get_vis_dir()
        use_amp = self.args.use_amp and torch.cuda.is_available() and (autocast is not None)
        scaler = GradScaler() if use_amp and GradScaler is not None else None

        # --- 调度与剂量控制 --- (来自版本1)
        update_mode = self.args.update_mode
        update_interval = self.args.update_interval
        update_layers = self.args.update_layers
        update_last_n = self.args.update_last_n
        skip_count = 0
        update_count = 0
        bad_update_count = 0
        import time
        step_times = []

        for (batch_idx, (images, _)) in enumerate(train_loader):
            if self.global_step >= self.args.online_steps:
                break

            images = images.to(self.device, non_blocking=True)  # (B, seq_len, C, H, W)
            batch_size, seq_len, c, h, w = images.shape

            # 调度判断：是否执行本步更新 (来自版本1)
            do_update = False
            reason = ""
            if update_mode == "fixed":
                if self.global_step % update_interval == 0:
                    do_update = True
                    reason = f"fixed: step%{update_interval}=0"
                else:
                    do_update = False
                    reason = f"fixed: step%{update_interval}!=0"
            elif update_mode == "triggered":
                if len(self.loss_history) < 5:
                    do_update = True
                    reason = "triggered: warmup (<5 steps)"
                else:
                    mean_loss = np.mean(self.loss_history)
                    std_loss = np.std(self.loss_history)
                    if len(self.loss_history) > 0 and self.loss_history[-1] > mean_loss + 2 * std_loss:
                        do_update = True
                        reason = f"triggered: loss({self.loss_history[-1]:.4f})>mean+2std"
                    else:
                        do_update = False
                        reason = f"triggered: loss({self.loss_history[-1]:.4f})<=mean+2std"
            else:
                do_update = True
                reason = "no schedule"

            # 剂量控制：每步都设置可训练层 (来自版本1)
            set_trainable_layers(self.model, mode=update_layers, last_n=update_last_n)

            t0 = time.time()
            if not do_update:
                print(f"[Step {self.global_step}] 跳过参数更新，原因: {reason}")
                skip_count += 1
                self.global_step += 1
                continue
            else:
                print(f"[Step {self.global_step}] 执行参数更新，原因: {reason}，剂量控制: {update_layers}")
                update_count += 1

            # 1) student 无梯度预测整个 batch（保存预测用于可视化/对比）
            self.model.eval()
            with torch.no_grad():
                flat_imgs = images.view(batch_size * seq_len, c, h, w)
                student_logits = self.model(flat_imgs)
                if student_logits.shape[1] == 1:
                    probs = torch.sigmoid(student_logits)
                    student_preds = (probs > 0.5).float().squeeze(1)
                else:
                    student_preds = torch.argmax(student_logits, dim=1)

            if self.args.save_viz:
                batch_viz_dir = os.path.join(viz_dir, f"step_{self.global_step}_preds")
                os.makedirs(batch_viz_dir, exist_ok=True)
                for i in range(student_preds.size(0)):
                    pred_np = student_preds[i].cpu().numpy().astype(np.uint8)
                    img_np = flat_imgs[i].cpu().numpy().transpose(1, 2, 0)
                    if img_np.max() <= 1.0:
                        img_vis = (img_np * 255).astype(np.uint8)
                    else:
                        img_vis = img_np.astype(np.uint8)
                    overlay = self.visualizer.create_overlay_image(img_vis, pred_np)
                    save_path = os.path.join(batch_viz_dir, f"pred_{i:03d}.png")
                    plt.imsave(save_path, overlay)

            # 2) frame selector -> selected_indices
            selected_indices = []
            for i in range(batch_size):
                frame_group = images[i]
                if self.frame_selector is not None and self.frame_selector.should_process_all(frame_group):
                    selected_indices.extend([(i, j) for j in range(seq_len)])
                else:
                    selected_indices.append((i, seq_len // 2))
            selected_flat_indices = [i * seq_len + j for (i, j) in selected_indices]

            # 3) teacher 生成伪标签（若 teacher 不存在则跳过训练）
            if self.offline_model is None or len(selected_flat_indices) == 0:
                self.global_step += 1
                continue

            sel_pairs = [(idx // seq_len, idx % seq_len) for idx in selected_flat_indices]
            selected_imgs_for_teacher = torch.stack([images[i, j] for (i, j) in sel_pairs], dim=0).to(self.device)

            with torch.no_grad():
                teacher_logits = self.offline_model(selected_imgs_for_teacher)
                if teacher_logits.shape[1] == 1:
                    teacher_probs = torch.sigmoid(teacher_logits).cpu().numpy()
                    pseudo_labels_np = (teacher_probs > 0.5).astype(np.uint8)[:, 0, ...]
                else:
                    teacher_probs_soft = torch.softmax(teacher_logits, dim=1)
                    teacher_probs = teacher_probs_soft.cpu().numpy()
                    pseudo_labels_np = np.argmax(teacher_probs, axis=1)

            # 释放 teacher 临时 tensor
            try:
                del teacher_logits
                if 'teacher_probs_soft' in locals():
                    del teacher_probs_soft
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 4) 伪标签质控 (质量控制后处理参考版本2)
            if pseudo_labels_np.ndim == 2:  # [H,W]
                pseudo_labels_np = pseudo_labels_np[None, ...]  # -> [1,H,W]
            elif pseudo_labels_np.ndim == 3:  # [N,H,W]，正常情况
                pass
            elif pseudo_labels_np.ndim == 4:  # [N,1,H,W]
                pseudo_labels_np = pseudo_labels_np.squeeze(1)
            try:
                pseudo_labels_np = denoise_pseudo_label(
                    pseudo_labels_np[np.newaxis, ...] if pseudo_labels_np.ndim == 2 else pseudo_labels_np,
                    min_area=100, morph_op='open', morph_structure=np.ones((3, 3))
                )
                if pseudo_labels_np.shape[0] == 1:
                    pseudo_labels_np = pseudo_labels_np[0]
                pixel_masks = pixel_gate_mask(teacher_probs, pseudo_labels_np)
                mask_ok = mask_quality_filter_with_pixel_mask(teacher_probs, pseudo_labels_np, pixel_masks)

                # 确保 mask_ok 为 1D boolean 数组 (来自版本2)
                mask_ok = np.asarray(mask_ok).astype(bool).ravel()
                n_selected = len(selected_flat_indices)
                n_mask = mask_ok.shape[0]
                if n_mask != n_selected:
                    print(
                        f"[DEBUG] QC output length mismatch: n_selected={n_selected}, n_mask={n_mask}. Trimming/padding to match.")
                # 如果 mask 比选中帧多，截断；如果少，补 False
                if n_mask > n_selected:
                    mask_ok = mask_ok[:n_selected]
                elif n_mask < n_selected:
                    pad = np.zeros(n_selected - n_mask, dtype=bool)
                    mask_ok = np.concatenate([mask_ok, pad], axis=0)

                # 简化版的质控结果显示
                total_samples = len(mask_ok)
                passed_samples = np.sum(mask_ok)
                failed_samples = total_samples - passed_samples

                # 打印简单结果
                if passed_samples > 0:
                    print(f"[Step {self.global_step}] 伪标签质控: {passed_samples}/{total_samples} 样本通过")
                else:
                    print(f"[Step {self.global_step}] 伪标签质控: 0/{total_samples} 样本通过 (全部未通过)")

            except Exception as e:
                print(f"[Step {self.global_step}] 质控过程中发生异常: {e}")
                mask_ok = np.ones(len(pseudo_labels_np), dtype=bool)

            if not np.any(mask_ok):
                print(f"[Step {self.global_step}] 没有样本通过质控，跳过本步更新")
                self.global_step += 1
                continue

            kept_idx = np.where(mask_ok)[0]
            kept_global_flat_indices = [selected_flat_indices[i] for i in kept_idx]

            # 5) 准备训练数据（仅保留通过 QC 的样本）
            selected_imgs_train = selected_imgs_for_teacher[kept_idx].to(self.device)
            pseudo_targets_np = pseudo_labels_np[kept_idx]
            if self.args.binary:
                pseudo_targets = torch.from_numpy(pseudo_targets_np).float().unsqueeze(1).to(self.device)
            else:
                pseudo_targets = torch.from_numpy(pseudo_targets_np).long().to(self.device)

            # 6) 限制训练样本数并混合 replay（来自新版）
            max_samples = max(1, int(self.args.max_train_samples_per_step))
            replay_samples = []
            if self.buffer is not None and len(self.buffer) > 0:
                n_replay = min(max_samples // 2, len(self.buffer))
                replay_data = self.buffer.sample(n_replay) if n_replay > 0 else None
                if replay_data is not None:
                    replay_inputs, replay_targets = zip(*replay_data)
                    replay_inputs = torch.stack(replay_inputs).to(self.device)
                    replay_targets = torch.stack(replay_targets).to(self.device)
                    replay_samples = (replay_inputs, replay_targets)

            # 随机缩减 teacher 样本量
            if selected_imgs_train.size(0) > max_samples:
                perm = torch.randperm(selected_imgs_train.size(0), device=self.device)
                sel = perm[:max_samples]
                selected_imgs_train = selected_imgs_train[sel]
                pseudo_targets = pseudo_targets[sel]

            # 混合 replay（保证总数 <= max_samples）
            if replay_samples:
                r_in, r_tg = replay_samples
                needed = max_samples - selected_imgs_train.size(0)
                if needed > 0:
                    r_in = r_in[:needed]
                    r_tg = r_tg[:needed]
                    selected_imgs_train = torch.cat([selected_imgs_train, r_in], dim=0)
                    pseudo_targets = torch.cat([pseudo_targets, r_tg], dim=0)

            if selected_imgs_train.size(0) == 0:
                self.global_step += 1
                continue

            # 7) Student 更新（支持 AMP & EMA 安全机制）
            self.model.train()
            self.optimizer.zero_grad()

            if use_amp:
                with autocast():
                    outputs = self.model(selected_imgs_train)
                    loss_tensor = self.criterion(outputs,
                                                 pseudo_targets.float() if self.args.binary else pseudo_targets.long())
                scaler.scale(loss_tensor).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                skip_update = self.ema_safety.step(float(loss_tensor.item()), self.model)
                if skip_update:
                    print(f"[Step {self.global_step}] EMA/异常/冷却机制触发，跳过参数更新。")
                else:
                    scaler.step(self.optimizer)
                scaler.update()  # AMP训练后参考版本2
            else:
                outputs = self.model(selected_imgs_train)
                loss_tensor = self.criterion(outputs,
                                             pseudo_targets.float() if self.args.binary else pseudo_targets.long())
                loss_tensor.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                skip_update = self.ema_safety.step(float(loss_tensor.item()), self.model)
                if skip_update:
                    print(f"[Step {self.global_step}] EMA/异常/冷却机制触发，跳过参数更新。")
                else:
                    self.optimizer.step()

            self.loss_history.append(float(loss_tensor.item()))

            # 8) push 训练样本（和 targets）进 replay buffer（来自 origin）
            if self.buffer is not None:
                for idx in range(selected_imgs_train.size(0)):
                    t_in = selected_imgs_train[idx].detach().cpu().clone()
                    t_target = pseudo_targets[idx].detach().cpu().clone()
                    self.buffer.push(t_in, t_target)

            # 9) 保存 best model（实时）
            try:
                self._save_best_model_if_improved(float(loss_tensor.item()))
            except Exception as e:
                print(f"[WARN] _save_best_model_if_improved failed: {e}")

            # 10) 清理显存并记录 step
            try:
                del selected_imgs_train, pseudo_targets, outputs
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 11) 周期性监控 / 评估 / 可视化（origin 的 eval & viz 行为）
            if self.global_step % self.args.monitor_interval == 0:
                print(
                    f"[Step {self.global_step}] loss={self.loss_history[-1]:.6f} | buffer={len(self.buffer) if self.buffer is not None else 0}")

            if val_loader is not None and (self.global_step % self.args.eval_interval == 0):
                # 使用内部稳定的验证函数（不依赖外部 Evaluator 的内部实现）(调用验证参考版本2)
                flat_val_iterable = flatten_val_loader(val_loader, binary=self.args.binary)
                # 注意 num_classes：优先使用 student/args 中的设置
                num_classes = getattr(self.args, "student_num_classes", None) or getattr(self.args, "num_classes",
                                                                                         None) or 2
                val_metrics = compute_val_metrics(self.model, flat_val_iterable, self.criterion, device=self.device,
                                                  num_classes=int(num_classes), binary=bool(self.args.binary))

                self.metrics_history.append(val_metrics)
                combined_metrics = {"step": self.global_step, "train_loss": float(loss_tensor.item())}
                combined_metrics.update(val_metrics)
                try:
                    self.output_mgr.save_metrics_csv(combined_metrics, self.global_step)
                except Exception:
                    pass
                print(
                    f"[Step {self.global_step}] Val loss: {val_metrics.get('val_loss', 0):.4f} | IoU: {val_metrics.get('iou', 0):.4f}")

            if self.args.save_viz and (self.global_step % self.args.viz_interval == 0) and val_loader is not None:
                print(f"Generating visualizations at step {self.global_step}...")
                step_viz_dir = os.path.join(viz_dir, f"step_{self.global_step}")
                os.makedirs(step_viz_dir, exist_ok=True)
                try:
                    self.visualizer.save_comparison_predictions(self.model, val_loader, step_viz_dir,
                                                                max_samples=self.args.viz_samples, device=self.device)
                except Exception as e:
                    print(f"[WARN] visualizer failed: {e}")

            # 增加全局 step
            self.global_step += 1

        # 结束，返回历史记录
        self.last_results = (self.loss_history, self.metrics_history)
        return self.last_results


# -----------------------
# main
# -----------------------
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    monitor = TrainMonitor(enable_gpu_monitor=args.enable_gpu_monitor)
    monitor.start_timing()

    model_tag = args.model if args.model_type is None else args.model_type
    # OutputManager初始化参考版本2
    output_mgr = OutputManager(model_type=model_tag)
    output_mgr.save_config(vars(args))

    # student 必须提供 checkpoint（保持新版要求）
    if not args.student_model_path or not args.student_model_name or args.student_num_classes is None:
        raise ValueError(
            "student_model_path, student_model_name and student_num_classes must be provided to load existing student model.")

    print(f"Loading student model from {args.student_model_path} ...")
    student_model = load_student_model(args.student_model_path, args.student_model_name, args.student_num_classes,
                                       device)
    print("Student model loaded.")

    learner = OnlineLearner(student_model, device, args, output_mgr, monitor,
                            offline_model_path=args.offline_model_path,
                            offline_model_name=args.offline_model_name,
                            offline_num_classes=args.offline_num_classes)

    def train_fn(batch_dir_path):
        batch_frames = sorted(glob.glob(os.path.join(batch_dir_path, "*.png")))
        if len(batch_frames) < 5:
            print(f"Warning: Not enough frames in {batch_dir_path}, skipping batch")
            return
        print(f"Processing batch with {len(batch_frames)} frames: {batch_dir_path}")

        dataset = OnlineFrameDataset(batch_frames, img_size=args.img_size, sequence_length=5)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                 pin_memory=True)

        # 创建 val_loader（使用同一批次做简单验证，用于 viz / evaluate）
        val_dataset = OnlineFrameDataset(batch_frames, img_size=args.img_size, sequence_length=5)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        learner.run_online_learning(data_loader, val_loader)

    extractor = VideoFrameExtractor(output_dir="src/dataio/datasets")
    _, native_fps=extractor.extract(
        video_path=args.video_root,
        start=10,
        end=60,
        size=(args.img_size, args.img_size),
        fmt="png",
        batch_size=20,
        mode=2,
        train_fn=train_fn
    )

    loss_history, metrics_history = learner.last_results if hasattr(learner, 'last_results') else ([], [])
    summary = output_mgr.get_run_summary()
    final_metrics = metrics_history[-1] if metrics_history else {}
    print(f"--> Online Learning Completed <--")
    print(f"Results saved to: {summary.get('run_dir', 'N/A')}")
    if final_metrics:
        print(f"Final Metrics - Loss: {final_metrics.get('val_loss', 0):.4f}, IoU: {final_metrics.get('iou', 0):.4f}")

    # 合并 step_* 目录生成视频（保持新版行为）
    try:
        viz_dir = output_mgr.get_vis_dir()
        step_dirs = sorted([str(p) for p in Path(viz_dir).iterdir() if p.is_dir() and p.name.startswith("step_")])
        if step_dirs:
            video_out = os.path.join(summary.get("run_dir", "./outputs"), "pred_overlay.mp4")
            merger = VideoFrameMerger(frame_dirs=step_dirs, output_path=video_out, fps=int(native_fps), fourcc="mp4v",
                                      auto_batches=False)
            merger.merge()
        else:
            print(f"[WARN] No step_* directories found in {viz_dir}")
    except Exception as e:
        print(f"[WARN] Failed to merge video: {e}")


if __name__ == "__main__":
    main()