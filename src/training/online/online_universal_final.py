# src/training/online/online_universal_streaming_enhanced.py
"""
增强版流式训练：
1. Frame Selection机制：基于FPS缓冲帧，计算相似度差异选择训练帧
2. 更新控制策略：只更新BN层，支持triggered/fixed模式
3. 完整Replay机制：质量加权采样，经验衰减，多样性保证
4. 流式处理：生产者-消费者模式，实时视频处理
"""
import math
import matplotlib.pyplot as plt
import cv2
import argparse
import random
import threading
import queue
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import os

# 路径修正，确保src包可import
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

# 通用模块
from src.viz.visualizer import Visualizer
from src.common.output_manager import OutputManager
from src.common.train_monitor import TrainMonitor
from src.common.pseudo_label_quality import (
    denoise_pseudo_label, pixel_gate_mask, mask_quality_filter_with_pixel_mask
)
from src.common.ema_safety import EMASafetyManager
from src.common.pseudo_label_generator import PseudoLabelGenerator

# 模型导入
from src.models.model_zoo import build_model

# AMP
try:
    from torch.cuda.amp import autocast, GradScaler
except Exception:
    autocast = None
    GradScaler = None


def parse_args():
    p = argparse.ArgumentParser("Enhanced Streaming Online Trainer")

    # 基本和数据参数
    p.add_argument("--video_root", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)

    # 训练参数
    p.add_argument("--online_steps", type=int, default=-1, help=">0 则限制处理帧数，否则直到视频结束")
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

    # 更新控制策略参数（简化版）
    p.add_argument("--update_layers", type=str, default="bn",
                   choices=["all", "bn", "lastN"], help="只更新指定层类型")
    p.add_argument("--update_last_n", type=int, default=2, help="lastN模式时更新最后N层")

    # 流式与可视化参数
    p.add_argument("--queue_maxsize", type=int, default=128)
    p.add_argument("--producer_sleep", type=float, default=0.0)
    p.add_argument("--consumer_timeout", type=float, default=0.5)
    p.add_argument("--writer_fps", type=float, default=25.0)
    p.add_argument("--stream_output", action='store_true', default=True)
    p.add_argument("--viz_stride", type=int, default=1)

    # 新增：student预测去噪与伪标签保存
    p.add_argument("--denoise_student_pred", action='store_true', default=True, help="对student预测进行去噪处理")
    p.add_argument("--save_pseudo_labels", action='store_true', default=False, help="保存通过质控的伪标签到本地")
    p.add_argument("--pseudo_save_dir", type=str, default=None, help="伪标签保存目录")

    p.add_argument("--teacher_only", action='store_true', default=False,
                   help="仅使用teacher伪标签（默认使用混合标签）")

    return p.parse_args()


# -----------------------
# Frame Selection机制
# -----------------------
class EnhancedFrameSelector:
    def __init__(self, threshold=0.85, video_fps=25, adaptive=True):
        self.threshold = threshold
        self.video_fps = max(1, int(video_fps))
        self.adaptive = adaptive
        self.frame_buffer = []
        self.similarity_history = deque(maxlen=20)
        self.frame_count = 0

    def simplified_ssim(self, frame1, frame2, size=64):
        """计算两帧之间的SSIM相似度"""
        if isinstance(frame1, torch.Tensor):
            frame1 = frame1.cpu().numpy()
        if isinstance(frame2, torch.Tensor):
            frame2 = frame2.cpu().numpy()

        # 如果是彩色图像，转为灰度
        if len(frame1.shape) == 3 and frame1.shape[0] == 3:
            frame1 = np.mean(frame1, axis=0)
        if len(frame2.shape) == 3 and frame2.shape[0] == 3:
            frame2 = np.mean(frame2, axis=0)

        # 缩放到固定大小加速计算
        frame1_small = cv2.resize(frame1, (size, size), interpolation=cv2.INTER_LINEAR)
        frame2_small = cv2.resize(frame2, (size, size), interpolation=cv2.INTER_LINEAR)

        # SSIM计算
        mu1 = np.mean(frame1_small)
        mu2 = np.mean(frame2_small)
        sigma1_sq = np.var(frame1_small)
        sigma2_sq = np.var(frame2_small)
        sigma12 = np.cov(frame1_small.flatten(), frame2_small.flatten())[0, 1]

        C1 = (0.01) ** 2
        C2 = (0.03) ** 2

        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(ssim)

    def add_frame(self, frame_item):
        """添加帧到缓冲区"""
        self.frame_buffer.append(frame_item)
        self.frame_count += 1

        # 当缓冲区达到fps大小时，进行帧选择
        if len(self.frame_buffer) >= self.video_fps:
            selected_frames = self._select_frames()
            self.frame_buffer = []  # 清空缓冲区
            return selected_frames

        return []

    def _select_frames(self):
        """基于相似度差异选择训练帧"""
        if len(self.frame_buffer) < 2:
            return self.frame_buffer

        # 计算帧间平均相似度
        similarities = []
        for i in range(len(self.frame_buffer) - 1):
            sim = self.simplified_ssim(
                self.frame_buffer[i].tensor_chw,
                self.frame_buffer[i + 1].tensor_chw
            )
            similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 1.0
        self.similarity_history.append(avg_similarity)

        # 自适应阈值
        if self.adaptive and len(self.similarity_history) > 5:
            mean_sim = np.mean(self.similarity_history)
            std_sim = np.std(self.similarity_history)
            adaptive_threshold = max(0.7, min(0.98, mean_sim - 0.5 * std_sim))
        else:
            adaptive_threshold = self.threshold

        n_frames = len(self.frame_buffer)

        # 根据相似度差异选择帧
        if avg_similarity < adaptive_threshold:
            # 差异很大：随机选择n/2帧
            n_select = max(1, n_frames // 2)
            selected_indices = random.sample(range(n_frames), n_select)
            selected_frames = [self.frame_buffer[i] for i in selected_indices]
            selection_reason = f"high_diff(sim={avg_similarity:.3f}, selected={n_select}/{n_frames})"
        else:
            # 差异不大：选择首、中、尾3帧
            if n_frames >= 3:
                indices = [0, n_frames // 2, n_frames - 1]
            elif n_frames == 2:
                indices = [0, 1]
            else:
                indices = [0]
            selected_frames = [self.frame_buffer[i] for i in indices]
            selection_reason = f"low_diff(sim={avg_similarity:.3f}, selected={len(indices)}/{n_frames})"

        print(f"[FrameSelector] {selection_reason}")
        return selected_frames

    def get_remaining_frames(self):
        """获取缓冲区中剩余的帧（视频结束时调用）"""
        if self.frame_buffer:
            remaining = self._select_frames()
            self.frame_buffer = []
            return remaining
        return []


# -----------------------
# 增强版Experience Replay Buffer
# -----------------------
class EnhancedExperienceReplayBuffer:
    def __init__(self, capacity=1000, quality_weight=0.7, diversity_weight=0.3):
        self.capacity = capacity
        self.quality_weight = quality_weight
        self.diversity_weight = diversity_weight

        # 存储结构：[image, target, quality_score, timestamp, feature_hash]
        self.buffer = []
        self.quality_scores = []
        self.timestamps = []
        self.feature_hashes = []

        self.global_step = 0

    def _compute_feature_hash(self, image_tensor):
        """计算图像特征哈希，用于多样性评估"""
        # 简单的特征哈希：使用图像的均值和方差
        if isinstance(image_tensor, torch.Tensor):
            img = image_tensor.detach().cpu().numpy()
        else:
            img = image_tensor

        # 计算简单特征向量
        mean_vals = np.mean(img, axis=(1, 2)) if len(img.shape) == 3 else [np.mean(img)]
        var_vals = np.var(img, axis=(1, 2)) if len(img.shape) == 3 else [np.var(img)]

        # 组合成特征哈希
        feature_vec = np.concatenate([mean_vals, var_vals])
        return hash(tuple(feature_vec.round(4)))  # 四舍五入避免精度问题

    def _compute_quality_score(self, loss_val, teacher_confidence=None):
        """计算样本质量分数"""
        # 基于损失值计算质量（损失越低质量越高）
        loss_quality = 1.0 / (1.0 + loss_val)

        # 如果有teacher置信度，结合使用
        if teacher_confidence is not None:
            confidence_quality = teacher_confidence
            quality = 0.6 * loss_quality + 0.4 * confidence_quality
        else:
            quality = loss_quality

        return float(quality)

    def push(self, image, target, loss_val=None, teacher_confidence=None):
        """添加经验到replay buffer"""
        self.global_step += 1

        # 计算质量分数和特征哈希
        quality = self._compute_quality_score(loss_val or 1.0, teacher_confidence)
        feature_hash = self._compute_feature_hash(image)

        # 存储数据
        experience = (
            image.detach().cpu().clone(),
            target.detach().cpu().clone()
        )

        self.buffer.append(experience)
        self.quality_scores.append(quality)
        self.timestamps.append(self.global_step)
        self.feature_hashes.append(feature_hash)

        # 超出容量时移除最旧的低质量样本
        if len(self.buffer) > self.capacity:
            self._remove_least_valuable()

    def _remove_least_valuable(self):
        """移除最不有价值的样本（结合质量和时效性）"""
        if len(self.buffer) <= self.capacity:
            return

        # 计算每个样本的综合价值（质量 + 时效性衰减）
        current_step = self.global_step
        values = []

        for i, (quality, timestamp) in enumerate(zip(self.quality_scores, self.timestamps)):
            # 时效性衰减：越旧的样本权重越低
            age_factor = np.exp(-(current_step - timestamp) / 1000.0)
            value = quality * age_factor
            values.append(value)
        # 找到价值最低的样本并移除
        min_idx = np.argmin(values)

        self.buffer.pop(min_idx)
        self.quality_scores.pop(min_idx)
        self.timestamps.pop(min_idx)
        self.feature_hashes.pop(min_idx)

    def sample(self, batch_size, ensure_diversity=True):
        """智能采样：结合质量权重和多样性"""
        if batch_size <= 0 or len(self.buffer) == 0:
            return []

        actual_batch_size = min(batch_size, len(self.buffer))

        if not ensure_diversity or len(self.buffer) <= actual_batch_size:
            # 简单质量加权采样
            weights = np.array(self.quality_scores)
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)

            indices = np.random.choice(len(self.buffer), size=actual_batch_size,
                                       replace=False, p=weights)
        else:
            # 多样性保证采样
            indices = self._diversity_aware_sampling(actual_batch_size)

        return [self.buffer[i] for i in indices]

    def _diversity_aware_sampling(self, batch_size):
        """多样性感知采样"""
        selected_indices = []
        selected_hashes = set()
        candidates = list(range(len(self.buffer)))

        # 第一步：质量加权采样
        quality_weights = np.array(self.quality_scores)
        quality_weights = quality_weights / np.sum(quality_weights) if np.sum(quality_weights) > 0 else np.ones_like(
            quality_weights) / len(quality_weights)

        while len(selected_indices) < batch_size and candidates:
            # 根据质量权重采样
            weights = quality_weights[candidates]
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)

            idx_in_candidates = np.random.choice(len(candidates), p=weights)
            idx = candidates[idx_in_candidates]

            feature_hash = self.feature_hashes[idx]

            # 检查多样性
            if feature_hash not in selected_hashes or len(selected_indices) < batch_size // 2:
                selected_indices.append(idx)
                selected_hashes.add(feature_hash)

            candidates.remove(idx)

        return selected_indices

    def get_stats(self):
        """获取buffer统计信息"""
        if not self.buffer:
            return {"size": 0, "avg_quality": 0, "quality_std": 0}

        return {
            "size": len(self.buffer),
            "avg_quality": np.mean(self.quality_scores),
            "quality_std": np.std(self.quality_scores),
            "unique_features": len(set(self.feature_hashes))
        }

    def __len__(self):
        return len(self.buffer)


# -----------------------
# 设置可训练层的工具函数（只支持BN层）
# -----------------------
def set_trainable_layers(model, mode="bn", last_n=2):
    """设置模型的可训练层，重点支持BN层"""
    # 首先将所有参数设为不可训练
    for name, param in model.named_parameters():
        param.requires_grad = False

    if mode == "all":
        for param in model.parameters():
            param.requires_grad = True
    elif mode == "bn":
        # 只训练BatchNorm层
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm, nn.InstanceNorm2d)):
                for param in module.parameters():
                    param.requires_grad = True
    elif mode == "lastN":
        # 训练最后N个参数组
        params = list(model.named_parameters())
        for name, param in params[-last_n:]:
            param.requires_grad = True
    else:
        raise ValueError(f"未知的update_layers模式: {mode}")


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
# OnlineLearner（用于管理优化器/损失/EMA等）
# -----------------------
class OnlineLearner:
    def __init__(self, model, device, args, output_mgr, monitor,
                 offline_model_path=None, offline_model_name=None, offline_num_classes=None):
        self.model = model.to(device)
        self.device = device
        self.args = args
        self.output_mgr = output_mgr
        self.monitor = monitor
        self.global_frame_idx = 0
        self.pseudo_label_generator = PseudoLabelGenerator(
            initial_teacher_weight=0.8,
            min_teacher_weight=0.05
        )
        self.criterion = nn.BCEWithLogitsLoss() if args.binary else nn.CrossEntropyLoss()

        # 优化器
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            for p in self.model.parameters():
                p.requires_grad = True
            trainable_params = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

        # teacher（可选）
        self.offline_model = None
        if offline_model_path and offline_model_name and offline_num_classes:
            self.offline_model = load_offline_model(offline_model_path, offline_model_name, offline_num_classes, device)

        self.visualizer = Visualizer()
        self.loss_history = []
        self.metrics_history = []
        self.step_time_history = []
        self.miou_history = []
        self.dice_history = []
        self.pred_confidence_history = []
        self.pseudo_confidence_history = []
        self.pseudo_entropy_history = []
        self.loss_variance_history = []

        # 增强版replay buffer
        self.buffer = EnhancedExperienceReplayBuffer(
            capacity=args.replay_capacity,
            quality_weight=0.7,
            diversity_weight=0.3
        ) if args.use_replay_buffer else None

        self.global_step = 0
        self.best_loss = float('inf')
        try:
            self.best_model_path = os.path.join(self.output_mgr.get_run_dir(), "student_best.pth")
        except Exception:
            self.best_model_path = os.path.join(project_root, "student_best.pth")

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


# -----------------------
# 流式生产者-消费者实现
# -----------------------
class FrameItem:
    def __init__(self, index: int, rgb_np: np.ndarray, tensor_chw: torch.Tensor):
        self.index = index
        self.rgb_np = rgb_np  # HxWx3 RGB uint8
        self.tensor_chw = tensor_chw  # CxHxW float32 [0,1]


def _frame_producer(video_path: str, img_size: int, q: "queue.Queue", stop_event: "threading.Event", sleep_s: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    idx = 0
    try:
        while not stop_event.is_set():
            ret, bgr = cap.read()
            if not ret:
                break
            if sleep_s > 0:
                time.sleep(sleep_s)
            bgr = cv2.resize(bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            chw = rgb.astype(np.float32) / 255.0
            chw = np.transpose(chw, (2, 0, 1))
            tensor = torch.from_numpy(chw).float()
            q.put(FrameItem(index=idx, rgb_np=rgb, tensor_chw=tensor))
            idx += 1
    finally:
        try:
            q.put(None)  # 哨兵
        except Exception:
            pass
        cap.release()


# 修改后的streaming_online_loop函数，实现所有帧都用于视频生成
def streaming_online_loop(args, learner, video_fps=25):
    device = learner.device
    visualizer = learner.visualizer
    output_mgr = learner.output_mgr

    # 获取视频FPS
    try:
        _cap = cv2.VideoCapture(args.video_root)
        detected_fps = _cap.get(cv2.CAP_PROP_FPS)
        _cap.release()
        if detected_fps and detected_fps > 0:
            video_fps = int(detected_fps)
        print(f"[INFO] Video FPS detected: {video_fps}")
    except Exception:
        print(f"[INFO] Using default FPS: {video_fps}")

    # Frame Selector初始化（仅用于训练帧选择）
    frame_selector = EnhancedFrameSelector(
        threshold=args.frame_selector_threshold,
        video_fps=video_fps,
        adaptive=True
    ) if args.use_frame_selector else None

    # writer延后创建
    writer = None
    writer_fps = args.writer_fps
    src_fps = video_fps

    q = queue.Queue(maxsize=max(2, int(args.queue_maxsize)))
    stop_event = threading.Event()
    producer = threading.Thread(target=_frame_producer,
                                args=(args.video_root, int(args.img_size), q, stop_event, float(args.producer_sleep)),
                                daemon=True)
    producer.start()

    processed = 0
    sentinel_seen = False
    recent_step_ms = []
    all_frames_count = 0  # 记录所有帧的数量

    try:
        while True:
            step_t0 = time.time()
            try:
                item = q.get(timeout=float(args.consumer_timeout))
            except queue.Empty:
                if sentinel_seen:
                    # 处理frame selector中剩余的帧进行训练（如果有）
                    if frame_selector:
                        remaining_frames = frame_selector.get_remaining_frames()
                        if remaining_frames:
                            print(
                                f"[INFO] Processing {len(remaining_frames)} remaining frames from buffer for training")
                            # 对剩余帧进行训练处理
                            _process_training_frames(remaining_frames, learner, args, device)
                    break
                print(f"[INFO] Queue empty, waiting for new frames ... (size={q.qsize()})")
                continue

            if item is None:
                sentinel_seen = True
                continue

            all_frames_count += 1

            # === 步骤1：为每一帧生成预测结果并写入视频 ===
            learner.model.eval()
            with torch.no_grad():
                img_tensor = item.tensor_chw.unsqueeze(0).to(device)
                logits = learner.model(img_tensor)
                if logits.shape[1] == 1:
                    probs = torch.sigmoid(logits)
                    pred_for_video = (probs > 0.5).float().squeeze(1)[0]
                else:
                    pred_for_video = torch.argmax(logits, dim=1)[0]

            # 生成可视化结果并写入视频
            if args.stream_output and all_frames_count % max(1, args.viz_stride) == 0:
                img_vis = item.rgb_np
                pred_np = pred_for_video.detach().cpu().numpy().astype(np.uint8)
                overlay = visualizer.create_overlay_image(img_vis, pred_np)

                if writer is None:
                    run_dir = output_mgr.get_run_dir()
                    out_path = os.path.join(run_dir, "pred_overlay.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                    if args.writer_fps > 0:
                        _fps = float(args.writer_fps)
                    elif args.writer_fps == 0:
                        _fps = float(src_fps)
                    else:
                        _fps = float(src_fps)
                    writer = cv2.VideoWriter(out_path, fourcc, _fps, (args.img_size, args.img_size))
                    print(f"[INFO] VideoWriter opened at fps={_fps}")

                writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            # === 步骤2：Frame Selection处理（仅用于训练） ===
            selected_frames_for_training = []
            if frame_selector:
                selected_frames_for_training = frame_selector.add_frame(item)
                if not selected_frames_for_training:
                    # 缓冲区未满，继续收集
                    continue
            else:
                # 如果不使用frame selector，每帧都用于训练
                selected_frames_for_training = [item]

            # === 步骤3：训练模型（仅使用选中的帧） ===
            # 总是更新，因为Frame Selection已经选择了关键帧
            # 处理训练帧
            if selected_frames_for_training:
                training_result = _process_training_frames(selected_frames_for_training, learner, args, device)
                successful_updates = training_result['successful_updates']
                total_loss = training_result['total_loss']
                total_samples = training_result['total_samples']

                # 更新计数器和统计
                learner.global_frame_idx += len(selected_frames_for_training)
                learner.global_step += 1
                processed += len(selected_frames_for_training)

                step_ms = (time.time() - step_t0) * 1000.0
                recent_step_ms.append(step_ms)
                learner.step_time_history.append(step_ms)
                if len(recent_step_ms) > 50:
                    recent_step_ms.pop(0)

                # 输出处理状态
                avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
                buffer_stats = learner.buffer.get_stats() if learner.buffer else {"size": 0}

                print(
                    f"[Step {learner.global_step}] all_frames={all_frames_count} training_frames={len(selected_frames_for_training)} "
                    f"updates={successful_updates}/{len(selected_frames_for_training)} avg_loss={avg_loss:.6f} "
                    f"buffer_size={buffer_stats['size']} buffer_quality={buffer_stats.get('avg_quality', 0):.3f} "
                    f"qsize={q.qsize()} time_ms={step_ms:.1f}"
                )

                # 周期性输出缓冲区统计
                if learner.global_step % 20 == 0 and learner.buffer:
                    stats = learner.buffer.get_stats()
                    print(f"[Buffer Stats] 大小:{stats['size']}/{learner.buffer.capacity} "
                          f"平均质量:{stats['avg_quality']:.3f}±{stats['quality_std']:.3f} "
                          f"特征多样性:{stats['unique_features']}")

            # 检查步数限制
            if int(args.online_steps) > 0 and learner.global_step >= int(args.online_steps):
                print(f"[INFO] Reached online_steps limit ({args.online_steps}). Current global_step: {learner.global_step}. Stopping streaming loop.")
                break

    finally:
        stop_event.set()
        if writer is not None:
            try:
                writer.release()
                print(f"[INFO] Video saved with {all_frames_count} frames processed")
            except Exception:
                pass


def _process_training_frames(selected_frames, learner, args, device):
    """处理训练帧的独立函数"""
    # 设置可训练层（仅BN层）
    set_trainable_layers(learner.model, mode="bn")

    print(f"[Step {learner.global_step}] 处理{len(selected_frames)}个选中帧进行训练")

    # 批量处理选中的帧
    total_loss = 0.0
    total_samples = 0
    successful_updates = 0

    for frame_item in selected_frames:
        # Student 预测（无梯度）
        learner.model.eval()
        with torch.no_grad():
            img_tensor = frame_item.tensor_chw.unsqueeze(0).to(device)
            logits = learner.model(img_tensor)
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits)
                student_pred = (probs > 0.5).float().squeeze(1)[0]
            else:
                student_pred = torch.argmax(logits, dim=1)[0]

        # Student 预测去噪处理（可选）
        if args.denoise_student_pred:
            try:
                student_pred_np = student_pred.detach().cpu().numpy().astype(np.uint8)
                student_pred_np = student_pred_np[np.newaxis, ...]
                student_pred_denoised = denoise_pseudo_label(student_pred_np, min_area=100, morph_op='open',
                                                             morph_structure=np.ones((3, 3)))
                student_pred_denoised = student_pred_denoised[0]
                student_pred = torch.from_numpy(student_pred_denoised).to(device)
            except Exception as e:
                print(f"[Step {learner.global_step}] Student预测去噪异常: {e}")

        # 使用新的混合伪标签生成机制 OR 纯teacher伪标签
        pseudo_targets = None
        teacher_confidence = None
        student_confidence = None
        use_pseudo_labels = False

        if learner.offline_model is not None:
            learner.model.eval()
            with torch.no_grad():
                # 获取teacher logits
                teacher_logits = learner.offline_model(img_tensor)

                # ========== 条件判断：混合标签 or 纯teacher ==========
                if not args.teacher_only:
                    # 混合标签模式（原有逻辑）
                    student_logits = learner.model(img_tensor)

                    hybrid_labels, teacher_conf, student_conf, current_teacher_weight, debug_info = \
                        learner.pseudo_label_generator.generate_hybrid_labels(
                            teacher_logits, student_logits,
                            current_loss=learner.loss_history[-1] if learner.loss_history else None,
                            binary_task=args.binary
                        )

                    if not hasattr(learner, '_temp_pred_confs'):
                        learner._temp_pred_confs = []
                        learner._temp_pseudo_confs = []
                        learner._temp_entropies = []

                    learner._temp_pred_confs.append(student_conf)
                    learner._temp_pseudo_confs.append(teacher_conf)

                    teacher_probs = torch.softmax(teacher_logits, dim=1)
                    entropy = -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-10), dim=1).mean().item()
                    learner._temp_entropies.append(entropy)

                    use_pseudo_labels = learner.pseudo_label_generator.should_use_pseudo_labels(
                        teacher_conf, student_conf, min_confidence=0.9
                    )

                    if use_pseudo_labels:
                        if args.binary:
                            pseudo_targets = hybrid_labels.float().unsqueeze(1)
                        else:
                            pseudo_targets = hybrid_labels.long()

                        teacher_confidence = teacher_conf
                        student_confidence = student_conf

                        print(f"[Step {learner.global_step}] 混合伪标签: teacher={teacher_conf:.3f}, "
                              f"student={student_conf:.3f}, weight={current_teacher_weight:.3f}")
                    else:
                        print(f"[Step {learner.global_step}] 置信度不足，跳过")

                else:
                    # 纯Teacher模式（v4逻辑）
                    if teacher_logits.shape[1] == 1:
                        tprobs = torch.sigmoid(teacher_logits).cpu().numpy()
                        pseudo_np = (tprobs > 0.5).astype(np.uint8)[:, 0, ...]
                        teacher_probs_np = tprobs
                        teacher_confidence = float(np.mean(np.max([tprobs, 1 - tprobs], axis=0)))
                    else:
                        tsoft = torch.softmax(teacher_logits, dim=1)
                        tprobs = tsoft.cpu().numpy()
                        pseudo_np = np.argmax(tprobs, axis=1)
                        teacher_probs_np = tprobs
                        teacher_confidence = float(np.mean(np.max(tprobs, axis=1)))

                    try:
                        pseudo_np = denoise_pseudo_label(pseudo_np, min_area=100, morph_op='open',
                                                         morph_structure=np.ones((3, 3)))
                        pixel_masks = pixel_gate_mask(teacher_probs_np, pseudo_np)
                        mask_ok = mask_quality_filter_with_pixel_mask(teacher_probs_np, pseudo_np, pixel_masks)
                        mask_ok = np.asarray(mask_ok).astype(bool).ravel()
                        use_pseudo_labels = bool(mask_ok.any())
                    except Exception as e:
                        print(f"[Step {learner.global_step}] 质控异常: {e}")
                        mask_ok = np.array([True], dtype=bool)
                        use_pseudo_labels = True

                    if use_pseudo_labels:
                        keep_idx = np.where(mask_ok)[0]
                        pseudo_np_kept = pseudo_np[keep_idx]
                        if args.binary:
                            pseudo_targets = torch.from_numpy(pseudo_np_kept).float().unsqueeze(1).to(device)
                        else:
                            pseudo_targets = torch.from_numpy(pseudo_np_kept).long().to(device)

                        print(f"[Step {learner.global_step}] Teacher伪标签: conf={teacher_confidence:.3f}")
                    else:
                        print(f"[Step {learner.global_step}] Teacher质控未通过")

                # 可视化保存（两种模式共用）
                if learner.global_step % 50 == 0 and pseudo_targets is not None:
                    try:
                        save_dir = args.pseudo_save_dir
                        os.makedirs(save_dir, exist_ok=True)

                        img_vis = frame_item.rgb_np
                        cmap = plt.get_cmap("tab10", args.offline_num_classes)

                        pred_mask = pseudo_targets[0].detach().cpu().numpy()
                        if pred_mask.ndim == 3:
                            pred_mask = pred_mask[0]

                        colored_mask = cmap(pred_mask / max(1, args.offline_num_classes - 1))[:, :, :3]
                        colored_mask = (colored_mask * 255).astype(np.uint8)

                        alpha = 0.4
                        overlay = cv2.addWeighted(img_vis, 1 - alpha, colored_mask, alpha, 0)

                        mode_tag = "teacher" if args.teacher_only else "hybrid"
                        base_name = f"frame_{frame_item.index:06d}_{mode_tag}"
                        cv2.imwrite(os.path.join(save_dir, f"{base_name}_original.png"),
                                    cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(os.path.join(save_dir, f"{base_name}_pseudo.png"),
                                    cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(os.path.join(save_dir, f"{base_name}_overlay.png"),
                                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                        print(f"[INFO] Saved {mode_tag} label: {save_dir}/{base_name}_overlay.png")

                    except Exception as e:
                        print(f"[Step {learner.global_step}] 保存失败: {e}")

        # Student 模型更新
        if use_pseudo_labels and pseudo_targets is not None and pseudo_targets.shape[0] > 0:
            learner.model.train()
            learner.optimizer.zero_grad()

            use_amp = args.use_amp and torch.cuda.is_available() and (autocast is not None)

            try:
                if use_amp:
                    with autocast():
                        out = learner.model(img_tensor)
                        loss_tensor = learner.criterion(out,
                                                        pseudo_targets.float() if args.binary else pseudo_targets.long())
                    loss_tensor.backward()
                else:
                    out = learner.model(img_tensor)
                    loss_tensor = learner.criterion(out,
                                                    pseudo_targets.float() if args.binary else pseudo_targets.long())
                    loss_tensor.backward()

                torch.nn.utils.clip_grad_norm_(learner.model.parameters(), max_norm=1.0)

                # EMA安全机制检查
                skip = learner.ema_safety.step(float(loss_tensor.item()), learner.model)
                if not skip:
                    learner.optimizer.step()
                    successful_updates += 1

                    loss_val = float(loss_tensor.item())
                    total_loss += loss_val
                    total_samples += 1

                    learner.loss_history.append(loss_val)

                    # 计算当前batch的IoU
                    with torch.no_grad():
                        if args.binary:
                            pred_mask = (torch.sigmoid(out) > 0.5).long()
                        else:
                            pred_mask = torch.argmax(out, dim=1)

                        # 简单IoU计算
                        intersection = (pred_mask == pseudo_targets.long()).float().sum()
                        union = pred_mask.numel()
                        iou = (intersection / union).item() if union > 0 else 0

                        if not hasattr(learner, '_temp_ious'):
                            learner._temp_ious = []
                            learner._temp_dices = []
                        learner._temp_ious.append(iou)

                    # Dice Coefficient计算
                    pred_positive = (pred_mask == 1).float().sum()
                    target_positive = (pseudo_targets == 1).float().sum()
                    dice = (2.0 * intersection) / (pred_positive + target_positive + 1e-6)
                    learner._temp_dices.append(dice.item())

                    # 添加到增强版replay buffer
                    if learner.buffer is not None:
                        learner.buffer.push(
                            img_tensor[0].detach().cpu().clone(),
                            pseudo_targets[0].detach().cpu().clone(),
                            loss_val,
                            teacher_confidence
                        )

                    # 保存最佳模型
                    try:
                        learner._save_best_model_if_improved(loss_val)
                    except Exception as e:
                        print(f"[WARN] best-save failed: {e}")
                else:
                    print(f"[Step {learner.global_step}] Frame {frame_item.index}: EMA安全机制跳过更新")

            except Exception as e:
                print(f"[Step {learner.global_step}] Frame {frame_item.index}: 训练过程异常: {e}")
                continue

        # 使用replay buffer进行额外训练（每处理几帧进行一次）
        if learner.buffer is not None and len(learner.buffer) > 50 and learner.global_step % 10 == 0:
            try:
                replay_samples = learner.buffer.sample(min(4, args.max_train_samples_per_step))
                if replay_samples:
                    learner.model.train()
                    learner.optimizer.zero_grad()

                    replay_inputs, replay_targets = zip(*replay_samples)
                    replay_batch_inputs = torch.stack(replay_inputs).to(device)
                    replay_batch_targets = torch.stack(replay_targets).to(device)

                    if use_amp:
                        with autocast():
                            replay_outputs = learner.model(replay_batch_inputs)
                            replay_loss = learner.criterion(
                                replay_outputs,
                                replay_batch_targets.float() if args.binary else replay_batch_targets.long()
                            )
                        replay_loss.backward()
                    else:
                        replay_outputs = learner.model(replay_batch_inputs)
                        replay_loss = learner.criterion(
                            replay_outputs,
                            replay_batch_targets.float() if args.binary else replay_batch_targets.long()
                        )
                        replay_loss.backward()

                    torch.nn.utils.clip_grad_norm_(learner.model.parameters(), max_norm=1.0)

                    replay_skip = learner.ema_safety.step(float(replay_loss.item()), learner.model)
                    if not replay_skip:
                        learner.optimizer.step()
                        print(
                            f"[Step {learner.global_step}] Replay训练: loss={replay_loss.item():.6f}, samples={len(replay_samples)}")

            except Exception as e:
                print(f"[Step {learner.global_step}] Replay训练异常: {e}")

    if hasattr(learner, '_temp_dices') and learner._temp_dices:
        learner.dice_history.append(np.mean(learner._temp_dices))
        learner._temp_dices = []

    if hasattr(learner, '_temp_ious') and learner._temp_ious:
        learner.miou_history.append(np.mean(learner._temp_ious))
        learner._temp_ious = []

    if hasattr(learner, '_temp_pred_confs') and learner._temp_pred_confs:
        learner.pred_confidence_history.append(np.mean(learner._temp_pred_confs))
        learner._temp_pred_confs = []

    if hasattr(learner, '_temp_pseudo_confs') and learner._temp_pseudo_confs:
        learner.pseudo_confidence_history.append(np.mean(learner._temp_pseudo_confs))
        learner._temp_pseudo_confs = []

    if hasattr(learner, '_temp_entropies') and learner._temp_entropies:
        learner.pseudo_entropy_history.append(np.mean(learner._temp_entropies))
        learner._temp_entropies = []

    return {
        'successful_updates': successful_updates,
        'total_loss': total_loss,
        'total_samples': total_samples
    }

# -----------------------
# main
# -----------------------
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    monitor = TrainMonitor(enable_gpu_monitor=args.enable_gpu_monitor)
    monitor.start_timing()

    model_tag = args.model if args.model_type is None else args.model_type
    output_mgr = OutputManager(model_type=model_tag)
    output_mgr.save_config(vars(args))

    # student 必须提供 checkpoint
    if not args.student_model_path or not args.student_model_name or args.student_num_classes is None:
        raise ValueError(
            "student_model_path, student_model_name and student_num_classes must be provided to load existing student model.")

    print(f"Loading student model from {args.student_model_path} ...")
    student_model = load_student_model(args.student_model_path, args.student_model_name, args.student_num_classes,
                                       device)
    print("Student model loaded.")

    # 如果用户没有指定伪标签保存目录，则默认放到 run_dir 下
    if args.pseudo_save_dir is None:
        args.pseudo_save_dir = os.path.join(output_mgr.get_run_dir(), "pseudo_labels")
    os.makedirs(args.pseudo_save_dir, exist_ok=True)
    print(f"[INFO] Pseudo labels will be saved to: {args.pseudo_save_dir}")

    learner = OnlineLearner(student_model, device, args, output_mgr, monitor,
                            offline_model_path=args.offline_model_path,
                            offline_model_name=args.offline_model_name,
                            offline_num_classes=args.offline_num_classes)

    # 进入流式在线循环
    streaming_online_loop(args, learner)

    if learner.step_time_history:
        avg_time = np.mean(learner.step_time_history)
        print(f"Average step time: {avg_time:.2f} ms")

        # 生成时间可视化图
        plt.figure(figsize=(10, 6))
        plt.plot(learner.step_time_history, alpha=0.6, label='Step Time')
        plt.axhline(y=avg_time, color='r', linestyle='--', label=f'Average: {avg_time:.2f}ms')
        plt.xlabel('Step')
        plt.ylabel('Time (ms)')
        plt.title('Step Processing Time')
        plt.legend()
        plt.grid(True)
        time_plot_path = os.path.join(output_mgr.get_run_dir(), "step_time_visualization.png")
        plt.savefig(time_plot_path)
        plt.close()
        print(f"Step time visualization saved to: {time_plot_path}")

    if learner.miou_history:
        plt.figure(figsize=(10, 6))
        plt.plot(learner.miou_history, alpha=0.6, label='mIoU')
        plt.xlabel('Update Step')
        plt.ylabel('mIoU')
        plt.title('Mean Intersection over Union')
        plt.legend()
        plt.grid(True)
        miou_plot_path = os.path.join(output_mgr.get_run_dir(), "miou_visualization.png")
        plt.savefig(miou_plot_path)
        plt.close()
        print(f"mIoU visualization saved to: {miou_plot_path}")

    if learner.loss_history and len(learner.loss_history) > 1:
        loss_variance = np.var(learner.loss_history)
        loss_std = np.std(learner.loss_history)
        print(f"\n[Training Stability Metrics]")
        print(f"Loss Variance: {loss_variance:.6f}")
        print(f"Loss Std Dev: {loss_std:.6f}")

        # 计算滑动窗口方差（更能反映训练过程的稳定性变化）
        window_size = 20
        if len(learner.loss_history) >= window_size:
            rolling_variance = []
            for i in range(len(learner.loss_history) - window_size + 1):
                window = learner.loss_history[i:i + window_size]
                rolling_variance.append(np.var(window))
            learner.loss_variance_history = rolling_variance

        # 1. Dice Coefficient (单独保存)
        if learner.dice_history:
            plt.figure(figsize=(10, 6))
            steps = np.arange(len(learner.dice_history))
            plt.plot(steps, learner.dice_history, color='#2E86AB', linewidth=2, alpha=0.8)
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Dice Score', fontsize=12)
            plt.title('Dice Coefficient Over Training', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=np.mean(learner.dice_history), color='r', linestyle='--',
                        label=f'Mean: {np.mean(learner.dice_history):.3f}', linewidth=1.5)
            plt.legend(fontsize=10)
            plt.ylim([0, 1])
            plt.tight_layout()
            dice_plot_path = os.path.join(output_mgr.get_run_dir(), "dice_coefficient.png")
            plt.savefig(dice_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[Visualization] Dice Coefficient saved to: {dice_plot_path}")

        # 2. Prediction Confidence (单独保存)
        if learner.pred_confidence_history:
            plt.figure(figsize=(10, 6))
            pred_steps = np.arange(len(learner.pred_confidence_history))
            plt.plot(pred_steps, learner.pred_confidence_history, color='#A23B72', linewidth=2, alpha=0.8)
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Confidence', fontsize=12)
            plt.title('Student Prediction Confidence (Entropy-based)', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=np.mean(learner.pred_confidence_history), color='r', linestyle='--',
                        label=f'Mean: {np.mean(learner.pred_confidence_history):.3f}', linewidth=1.5)
            plt.legend(fontsize=10)
            plt.ylim([0, 1])

            # 添加说明文本
            conf_text = "Entropy-inverted confidence (lower entropy = higher confidence)"
            plt.text(0.5, 0.05, conf_text, transform=plt.gca().transAxes,
                     fontsize=9, ha='center', va='bottom', style='italic',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

            plt.tight_layout()
            pred_conf_plot_path = os.path.join(output_mgr.get_run_dir(), "prediction_confidence.png")
            plt.savefig(pred_conf_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[Visualization] Prediction Confidence saved to: {pred_conf_plot_path}")

        # 3. Pseudo-Label Confidence (单独保存，带双y轴)
        if learner.pseudo_confidence_history:
            fig, ax1 = plt.subplots(figsize=(10, 6))

            pseudo_steps = np.arange(len(learner.pseudo_confidence_history))
            ax1.plot(pseudo_steps, learner.pseudo_confidence_history, color='#F18F01',
                     linewidth=2, alpha=0.8, label='Teacher Confidence')
            ax1.set_xlabel('Training Step', fontsize=12)
            ax1.set_ylabel('Confidence', fontsize=12, color='#F18F01')
            ax1.set_title('Pseudo-Label Quality (Teacher Confidence)', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=np.mean(learner.pseudo_confidence_history), color='r', linestyle='--',
                        label=f'Mean: {np.mean(learner.pseudo_confidence_history):.3f}', linewidth=1.5)
            ax1.tick_params(axis='y', labelcolor='#F18F01')
            ax1.set_ylim([0, 1])
            ax1.legend(loc='upper left', fontsize=10)

            # 如果有熵数据，添加第二个y轴
            if learner.pseudo_entropy_history:
                ax2 = ax1.twinx()
                entropy_steps = np.arange(len(learner.pseudo_entropy_history))
                ax2.plot(entropy_steps, learner.pseudo_entropy_history, color='#06A77D',
                         linewidth=2, alpha=0.6, linestyle='--', label='Entropy (lower=better)')
                ax2.set_ylabel('Entropy', fontsize=12, color='#06A77D')
                ax2.tick_params(axis='y', labelcolor='#06A77D')
                ax2.legend(loc='upper right', fontsize=10)

            plt.tight_layout()
            pseudo_conf_plot_path = os.path.join(output_mgr.get_run_dir(), "pseudo_label_confidence.png")
            plt.savefig(pseudo_conf_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[Visualization] Pseudo-Label Confidence saved to: {pseudo_conf_plot_path}")

        # 4. Loss Variance (单独保存)
        if learner.loss_variance_history:
            plt.figure(figsize=(10, 6))
            var_steps = np.arange(len(learner.loss_variance_history))
            plt.plot(var_steps, learner.loss_variance_history, color='#C73E1D', linewidth=2, alpha=0.8)
            plt.xlabel('Window Position (Step)', fontsize=12)
            plt.ylabel('Variance', fontsize=12)
            plt.title('Loss Variance (Training Stability)', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=np.mean(learner.loss_variance_history), color='r', linestyle='--',
                        label=f'Mean: {np.mean(learner.loss_variance_history):.4f}', linewidth=1.5)
            plt.legend(fontsize=10)

            # 添加稳定性说明文本
            stability_text = "Lower variance = More stable training"
            plt.text(0.5, 0.95, stability_text, transform=plt.gca().transAxes,
                     fontsize=10, ha='center', va='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

            plt.tight_layout()
            loss_var_plot_path = os.path.join(output_mgr.get_run_dir(), "loss_variance.png")
            plt.savefig(loss_var_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[Visualization] Loss Variance saved to: {loss_var_plot_path}")
        elif learner.loss_history:
            # 如果没有rolling variance，保存整体variance说明
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'Overall Loss Variance:\n{np.var(learner.loss_history):.6f}\n\n'
                               f'Loss Std Dev:\n{np.std(learner.loss_history):.6f}',
                     transform=plt.gca().transAxes, fontsize=16, ha='center', va='center',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            plt.title('Loss Variance (Training Stability)', fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            loss_var_plot_path = os.path.join(output_mgr.get_run_dir(), "loss_variance.png")
            plt.savefig(loss_var_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[Visualization] Loss Variance saved to: {loss_var_plot_path}")

    loss_history, metrics_history = learner.last_results if hasattr(learner, 'last_results') else ([], [])
    summary = output_mgr.get_run_summary()
    final_metrics = metrics_history[-1] if metrics_history else {}
    print("--> Enhanced Online Learning Completed <--")
    print(f"Results saved to: {summary.get('run_dir', 'N/A')}")
    if final_metrics:
        print(f"Final Metrics - Loss: {final_metrics.get('val_loss', 0):.4f}, IoU: {final_metrics.get('iou', 0):.4f}")


if __name__ == "__main__":
    main()