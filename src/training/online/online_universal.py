# src/training/online/online_universal.py
"""
通用在线学习模板 - 集成监控、可视化、评估功能
改动：支持加载已有 student checkpoint，student 先预测输出（线上结果），
      使用 teacher (offline model) 生成伪标签并在线更新 student。
修复/增强：
 - 确保 teacher 的 no_grad 作用域不会影响 student 训练 forward
 - 在每个 batch 保存 student 的预测（包括将被用于训练的帧），并在训练前保存
 - 限制每次在线训练使用的伪标签数量，避免 OOM；支持可选 AMP 混合精度
 - 在 teacher 推理后尽快释放显存
 - 使用 global_step 避免覆盖可视化；实时保存 best student model
"""
import glob
import cv2
import argparse
import random
from collections import deque
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

# 路径修正，确保src包可import
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

# 导入通用模块
from pathlib import Path
from utils.class_frame_to_video import VideoFrameMerger
from src.eval.evaluator import Evaluator
from src.viz.visualizer import Visualizer
from src.common.output_manager import OutputManager
from src.common.train_monitor import TrainMonitor
from src.common.pseudo_label_quality import quality_filter, denoise_pseudo_label, pixel_gate_mask, mask_quality_filter_with_pixel_mask

# 模型导入
from src.models.model_zoo import build_model
from src.dataio.datasets.seg_dataset_min import SegDatasetMin

# 导入工具类
from utils.class_frame_extractor import VideoFrameExtractor

# AMP
try:
    from torch.cuda.amp import autocast, GradScaler
except Exception:
    autocast = None
    GradScaler = None


def parse_args():
    p = argparse.ArgumentParser("Online Universal Trainer")

    # 基础训练参数
    p.add_argument("--cfg", type=str, default=None, help="Optional YAML config")
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

    # 离线（teacher）模型参数（用于生成伪标签）
    p.add_argument("--offline_model_path", type=str, help="Path to the offline (teacher) model checkpoint")
    p.add_argument("--offline_model_name", type=str, help="Model name for the offline (teacher) model")
    p.add_argument("--offline_num_classes", type=int, help="Number of classes for the offline model")

    # 学生（student）模型参数（已有 checkpoint）
    p.add_argument("--student_model_path", type=str, help="Path to the trained student checkpoint (required)")
    p.add_argument("--student_model_name", type=str, help="Student model name (e.g. adaptive_unet)", default=None)
    p.add_argument("--student_num_classes", type=int, help="Number of classes for student model", default=None)

    # 可选增强
    p.add_argument("--save_pred_every_batch", action='store_true', default=True,
                   help="Save student predictions for frames used for training before training")
    p.add_argument("--max_train_samples_per_step", type=int, default=8,
                   help="最大每步训练样本数（包括 experience replay），以避免 OOM")
    p.add_argument("--use_amp", action='store_true', default=False, help="Use mixed precision (AMP) if available")

    return p.parse_args()


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


class FrameSelector:
    def __init__(self, threshold=0.9, adaptive=True, window_size=5):
        self.threshold = threshold
        self.adaptive = adaptive
        self.window_size = window_size
        self.last_frame = None
        self.similarity_history = deque(maxlen=10)

    def simplified_ssim(self, frame1, frame2, size=64):
        frame1_small = torch.nn.functional.interpolate(
            frame1.unsqueeze(0), size=(size, size), mode='bilinear'
        ).squeeze(0)
        frame2_small = torch.nn.functional.interpolate(
            frame2.unsqueeze(0), size=(size, size), mode='bilinear'
        ).squeeze(0)

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
        if self.last_frame is None:
            self.last_frame = frames[2]
            self.similarity_history.append(0.5)
            return True

        current_similarity = self.simplified_ssim(self.last_frame, frames[2])
        self.similarity_history.append(current_similarity)

        if self.adaptive and len(self.similarity_history) > 5:
            mean_sim = np.mean(self.similarity_history)
            std_sim = np.std(self.similarity_history)
            adaptive_threshold = max(0.7, min(0.95, float(mean_sim - 0.5 * std_sim)))
        else:
            adaptive_threshold = float(self.threshold)

        self.last_frame = frames[2]
        return current_similarity < adaptive_threshold


class ExperienceReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, target):
        self.buffer.append((state.detach().cpu().clone(), target.detach().cpu().clone()))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size or batch_size == 0:
            return None
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ==== 模型加载工具 ====
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

    # 确保参数可训练
    for p in model.parameters():
        p.requires_grad = True

    model.to(device)
    model.train()
    return model


class OnlineLearner:
    def __init__(self, model, device, args, output_mgr, monitor,
                 offline_model_path=None, offline_model_name=None, offline_num_classes=None):
        self.model = model.to(device)
        self.device = device
        self.args = args
        self.output_mgr = output_mgr
        self.monitor = monitor

        if args.binary:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            for p in self.model.parameters():
                p.requires_grad = True
            trainable_params = list(self.model.parameters())

        self.optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        self.offline_model = None
        if offline_model_path and offline_model_name and offline_num_classes:
            self.offline_model = load_offline_model(
                offline_model_path, offline_model_name, offline_num_classes, device)

        self.evaluator = Evaluator(device=device, threshold=0.5)
        self.loss_history = []
        self.metrics_history = []

        self.buffer = ExperienceReplayBuffer(capacity=args.replay_capacity) if args.use_replay_buffer else None
        # ---- NEW: global step & best model tracking ----
        self.global_step = 0
        self.best_loss = float('inf')
        try:
            self.best_model_path = os.path.join(self.output_mgr.get_run_dir(), "student_best.pth")
        except Exception:
            self.best_model_path = os.path.join(project_root, "student_best.pth")
        # === 新增EMA安全管理器 ===
        self.ema_safety = EMASafetyManager(self.model, alpha=0.99, loss_window_size=10, grad_explode_thresh=10.0, cooldown_period=5)

    def _save_best_model_if_improved(self, loss_val):
        if loss_val < self.best_loss:
            self.best_loss = loss_val
            try:
                torch.save({'model_state_dict': self.model.state_dict(), 'loss': loss_val, 'step': self.global_step},
                           self.best_model_path)
                print(f"[INFO] Saved improved student model to {self.best_model_path} (loss={loss_val:.6f})")
            except Exception as e:
                print(f"[WARN] Failed to save best model: {e}")

    def run_online_learning(self, train_loader, val_loader=None):
        print("=" * 80)
        print(f"Starting Online Learning ({self.args.model}) for {self.args.online_steps} steps...")

        visualizer = Visualizer()
        viz_dir = self.output_mgr.get_vis_dir()

        use_amp = self.args.use_amp and torch.cuda.is_available() and (autocast is not None)
        scaler = GradScaler() if use_amp and GradScaler is not None else None

        for step, (images, _) in enumerate(train_loader):
            if self.global_step >= self.args.online_steps:
                break

            images = images.to(self.device, non_blocking=True)
            batch_size, seq_len, c, h, w = images.shape

            # 1) student 预测整个 batch (no grad)
            self.model.eval()
            with torch.no_grad():
                flat_imgs = images.view(batch_size * seq_len, c, h, w)
                student_logits = self.model(flat_imgs)
                if student_logits.shape[1] == 1:
                    probs = torch.sigmoid(student_logits)
                    student_preds = (probs > 0.5).float().squeeze(1)
                else:
                    student_preds = torch.argmax(student_logits, dim=1)
            print("Unique preds:", torch.unique(student_preds))
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

                    overlay = visualizer.create_overlay_image(img_vis, pred_np)
                    save_path = os.path.join(batch_viz_dir, f"pred_{i:03d}.png")
                    plt.imsave(save_path, overlay)

            # 2) 选帧（frame selector）——得到 selected_indices 列表
            selected_indices = []
            for i in range(batch_size):
                frame_group = images[i]
                if self.args.use_frame_selector and FrameSelector(threshold=self.args.frame_selector_threshold).should_process_all(frame_group):
                    selected_indices.extend([(i, j) for j in range(seq_len)])
                else:
                    selected_indices.append((i, seq_len // 2))

            # map to flattened indices
            selected_flat_indices = [i * seq_len + j for (i, j) in selected_indices]


            # 3) teacher 对 selected frames 生成伪标签（只运行在选中的小集合上，减少显存占用）
            if self.offline_model is None or len(selected_flat_indices) == 0:
                # increment global step to avoid repeating the same step across calls
                self.global_step += 1
                continue

            # 构造 selected imgs 张量并送到 device
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

            # free teacher tensors from GPU ASAP
            try:
                del teacher_logits
                if 'teacher_probs_soft' in locals():
                    del teacher_probs_soft
            except Exception:
                pass
            torch.cuda.empty_cache()

            # === 新增伪标签质控（更细致） ===
            try:
                pseudo_labels_np = denoise_pseudo_label(
                    pseudo_labels_np[np.newaxis, ...] if pseudo_labels_np.ndim==2 else pseudo_labels_np,
                    min_area=100, morph_op='open', morph_structure=np.ones((3,3))
                )
                if pseudo_labels_np.shape[0] == 1:
                    pseudo_labels_np = pseudo_labels_np[0]
                pixel_masks = pixel_gate_mask(teacher_probs, pseudo_labels_np)
                mask_ok = mask_quality_filter_with_pixel_mask(teacher_probs, pseudo_labels_np, pixel_masks)
            except Exception:
                mask_ok = np.ones(len(pseudo_labels_np), dtype=bool)

            if not np.any(mask_ok):
                self.global_step += 1
                continue

            # 只保留通过 QC 的样本
            kept_idx = np.where(mask_ok)[0]
            kept_global_flat_indices = [selected_flat_indices[i] for i in kept_idx]

            # 准备训练用数据（在 GPU 上）
            selected_imgs_train = selected_imgs_for_teacher[kept_idx].to(self.device)
            pseudo_targets_np = pseudo_labels_np[kept_idx]
            if self.args.binary:
                pseudo_targets = torch.from_numpy(pseudo_targets_np).float().unsqueeze(1).to(self.device)
            else:
                pseudo_targets = torch.from_numpy(pseudo_targets_np).long().to(self.device)

            # 4) 限制训练样本数量（包含 replay），避免 OOM
            max_samples = max(1, int(self.args.max_train_samples_per_step))
            replay_samples = []
            if self.buffer is not None and len(self.buffer) > 0:
                # we will sample at most max_samples//2 from replay to keep mix
                n_replay = min(max_samples // 2, len(self.buffer))
                replay_data = self.buffer.sample(n_replay) if n_replay > 0 else None
                if replay_data is not None:
                    replay_inputs, replay_targets = zip(*replay_data)
                    replay_inputs = torch.stack(replay_inputs).to(self.device)
                    replay_targets = torch.stack(replay_targets).to(self.device)
                    replay_samples = (replay_inputs, replay_targets)

            # if too many teacher samples, randomly pick
            if selected_imgs_train.size(0) > max_samples:
                perm = torch.randperm(selected_imgs_train.size(0), device=self.device)
                sel = perm[:max_samples]
                selected_imgs_train = selected_imgs_train[sel]
                pseudo_targets = pseudo_targets[sel]

            # merge replay if any and keep total <= max_samples
            if replay_samples:
                r_in, r_tg = replay_samples
                needed = max_samples - selected_imgs_train.size(0)
                if needed > 0:
                    r_in = r_in[:needed]
                    r_tg = r_tg[:needed]
                    selected_imgs_train = torch.cat([selected_imgs_train, r_in], dim=0)
                    pseudo_targets = torch.cat([pseudo_targets, r_tg], dim=0)

            # 5) 用伪标签训练 student（支持 AMP）
            if selected_imgs_train.size(0) == 0:
                self.global_step += 1
                continue

            self.model.train()
            self.optimizer.zero_grad()

            if use_amp:
                with autocast():
                    outputs = self.model(selected_imgs_train)
                    if self.args.binary:
                        loss_tensor = self.criterion(outputs, pseudo_targets.float())
                    else:
                        loss_tensor = self.criterion(outputs, pseudo_targets.long())
                scaler.scale(loss_tensor).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # === EMA安全机制 ===
                skip_update = self.ema_safety.step(loss_tensor.item(), self.model)
                if skip_update:
                    print(f"[Step {self.global_step}] EMA/异常/冷却机制触发，跳过参数更新。")
                else:
                    scaler.step(self.optimizer)
                scaler.update()
            else:
                outputs = self.model(selected_imgs_train)
                if self.args.binary:
                    loss_tensor = self.criterion(outputs, pseudo_targets.float())
                else:
                    loss_tensor = self.criterion(outputs, pseudo_targets.long())
                loss_tensor.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # === EMA安全机制 ===
                skip_update = self.ema_safety.step(loss_tensor.item(), self.model)
                if skip_update:
                    print(f"[Step {self.global_step}] EMA/异常/冷却机制触发，跳过参数更新。")
                else:
                    self.optimizer.step()

            self.loss_history.append(float(loss_tensor.item()))

            # push to replay buffer
            if self.buffer is not None:
                for idx in range(selected_imgs_train.size(0)):
                    t_in = selected_imgs_train[idx].detach().cpu().clone()
                    t_target = pseudo_targets[idx].detach().cpu().clone()
                    self.buffer.push(t_in, t_target)

            # 保存最佳模型（实时）
            try:
                self._save_best_model_if_improved(float(loss_tensor.item()))
            except Exception as e:
                print(f"[WARN] _save_best_model_if_improved failed: {e}")

            # free some memory
            try:
                del selected_imgs_train, pseudo_targets, outputs
            except Exception:
                pass
            torch.cuda.empty_cache()

            # 增加全局 step，防止重复覆盖
            self.global_step += 1

        self.last_results = (self.loss_history, self.metrics_history)
        return self.last_results


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    monitor = TrainMonitor(enable_gpu_monitor=args.enable_gpu_monitor)
    monitor.start_timing()

    output_mgr = OutputManager(model_type=args.model)
    output_mgr.save_config(vars(args))

    if not args.student_model_path or not args.student_model_name or args.student_num_classes is None:
        raise ValueError("student_model_path, student_model_name and student_num_classes must be provided to load existing student model.")

    print(f"Loading student model from {args.student_model_path} ...")
    student_model = load_student_model(
        args.student_model_path,
        args.student_model_name,
        args.student_num_classes,
        device
    )
    print("Student model loaded.")

    learner = OnlineLearner(
        student_model, device, args, output_mgr, monitor,
        offline_model_path=args.offline_model_path,
        offline_model_name=args.offline_model_name,
        offline_num_classes=args.offline_num_classes
    )

    def train_fn(batch_dir_path):
        batch_frames = sorted(glob.glob(os.path.join(batch_dir_path, "*.png")))
        if len(batch_frames) < 5:
            print(f"Warning: Not enough frames in {batch_dir_path}, skipping batch")
            return

        print(f"Processing batch with {len(batch_frames)} frames: {batch_dir_path}")

        dataset = OnlineFrameDataset(batch_frames, img_size=args.img_size, sequence_length=5)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

        learner.run_online_learning(data_loader, None)

    extractor = VideoFrameExtractor(output_dir="src/dataio/datasets")
    extractor.extract(
        video_path=args.video_root,
        fps=2,
        start=10,
        end=60,
        size=(args.img_size, args.img_size),
        fmt="png",
        batch_size=5,
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

    # === 合并所有 step_* 目录下的预测帧为视频 ===
    try:
        viz_dir = output_mgr.get_vis_dir()
        step_dirs = sorted([str(p) for p in Path(viz_dir).iterdir()
                            if p.is_dir() and p.name.startswith("step_")])

        if step_dirs:
            video_out = os.path.join(summary.get("run_dir", "./outputs"), "pred_overlay.mp4")
            merger = VideoFrameMerger(
                frame_dirs=step_dirs,
                output_path=video_out,
                fps=2,              # 建议和 extractor.extract() 里的 fps 保持一致
                fourcc="mp4v",
                auto_batches=False  # 因为这里不是 batch_x，而是 step_x
            )
            merger.merge()
        else:
            print(f"[WARN] No step_* directories found in {viz_dir}")
    except Exception as e:
        print(f"[WARN] Failed to merge video: {e}")

if __name__ == "__main__":
    main()
