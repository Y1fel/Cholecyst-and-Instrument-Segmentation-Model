# src/training/online/online_universal_streaming.py
"""
流式版：生产者-消费者（切菜-炒菜）模式。
- 生产者: VideoCapture 连续读帧，放入有限队列（带背压）。
- 消费者: 从队列取帧，立即执行 online learning（teacher 伪标签->质控->student 更新）。
- 背压: 消费太快则等待新帧；生产太快则阻塞放入。
- 可视化: 边处理边生成叠加图并直接写视频（尽量流式）。
"""
import math
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

# 模型导入
from src.models.model_zoo import build_model

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
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)

    # 训练参数
    p.add_argument("--online_steps", type=int, default=-1, help=">0 则限制处理帧数，否则直到视频结束")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--update_every_n", type=int, default=200, help="每N帧至多更新一次，0/1表示每帧都可更新")

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

    # 重新添加缺失的调度控制参数
    p.add_argument("--update_mode", type=str, default="fixed",
                   choices=["all", "fixed", "triggered"])
    p.add_argument("--update_interval", type=int, default=5)
    p.add_argument("--update_layers", type=str, default="bn",
                   choices=["all", "bn", "lastN"])
    p.add_argument("--update_last_n", type=int, default=2)

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
    p.add_argument("--pseudo_save_dir", type=str, default="pseudo_labels", help="伪标签保存目录")

    return p.parse_args()


# -----------------------
# Experience Replay Buffer（仍被流式训练使用）
# -----------------------
class ExperienceReplayBuffer:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.buffer = []

    def push(self, image, target):
        self.buffer.append((image.detach().cpu().clone(), target.detach().cpu().clone()))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        if batch_size <= 0 or len(self.buffer) == 0:
            return []
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

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
# 设置可训练层的工具函数
# -----------------------
def set_trainable_layers(model, mode="bn", last_n=2):
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
        params = list(model.named_parameters())
        for name, param in params[-last_n:]:
            param.requires_grad = True
    else:
        raise ValueError(f"未知的update_layers模式: {mode}")


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

        self.criterion = nn.BCEWithLogitsLoss() if args.binary else nn.CrossEntropyLoss()

        # 优化器（剔除未用的scheduler）
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

        # replay（保留）
        self.buffer = ExperienceReplayBuffer(capacity=args.replay_capacity) if args.use_replay_buffer else None

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
            q.put(FrameItem(index=idx, rgb_np=rgb, tensor_chw=tensor))  # 满则阻塞，形成背压
            idx += 1
    finally:
        try:
            q.put(None)  # 哨兵
        except Exception:
            pass
        cap.release()


def streaming_online_loop(args, learner):
    device = learner.device
    visualizer = learner.visualizer
    output_mgr = learner.output_mgr

    # writer 延后到首次写帧时创建；writer_fps: 0=源FPS，<0=吞吐FPS，>0=显式FPS
    writer = None
    writer_fps = args.writer_fps
    src_fps = None
    if writer_fps <= 0:
        try:
            _cap = cv2.VideoCapture(args.video_root)
            src_fps = _cap.get(cv2.CAP_PROP_FPS)
            _cap.release()
            if not src_fps or src_fps <= 0:
                src_fps = 25.0
        except Exception:
            src_fps = 25.0

    q = queue.Queue(maxsize=max(2, int(args.queue_maxsize)))
    stop_event = threading.Event()
    producer = threading.Thread(target=_frame_producer,
                                args=(args.video_root, int(args.img_size), q, stop_event, float(args.producer_sleep)),
                                daemon=True)
    producer.start()

    processed = 0
    sentinel_seen = False
    recent_step_ms = []

    try:
        while True:
            step_t0 = time.time()
            try:
                item = q.get(timeout=float(args.consumer_timeout))
            except queue.Empty:
                if sentinel_seen:
                    break
                print(f"[INFO] Queue empty, waiting for new frames ... (size={q.qsize()})")
                continue

            if item is None:
                sentinel_seen = True
                if q.empty():
                    print("[INFO] Producer finished. Draining complete.")
                    break
                else:
                    continue

            # 调度：设置可训练层
            set_trainable_layers(learner.model, mode=args.update_layers, last_n=args.update_last_n)

            # Student 预测（无梯度）
            learner.model.eval()
            with torch.no_grad():
                img_tensor = item.tensor_chw.unsqueeze(0).to(device)
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
                    # 添加批维度 [H,W] -> [1,H,W]
                    student_pred_np = student_pred_np[np.newaxis, ...]
                    # 去噪处理
                    student_pred_denoised = denoise_pseudo_label(student_pred_np, min_area=100, morph_op='open',
                                                                 morph_structure=np.ones((3, 3)))
                    # 移除批维度 [1,H,W] -> [H,W]
                    student_pred_denoised = student_pred_denoised[0]
                    # 转回tensor
                    student_pred = torch.from_numpy(student_pred_denoised).to(device)
                except Exception as e:
                    print(f"[Step {learner.global_step}] Student预测去噪异常: {e}")
                    # 去噪失败时保持原预测

            # Teacher 伪标签（可选）+ 质控（保持批维度 [N,H,W]）
            pseudo_targets = None
            teacher_probs_np = None
            qc_pass = None
            loss_val = None
            did_update = False
            update_reason = "no-teacher"
            wrote_viz = False
            if learner.offline_model is not None:
                with torch.no_grad():
                    tlogits = learner.offline_model(img_tensor)
                    if tlogits.shape[1] == 1:
                        tprobs = torch.sigmoid(tlogits).cpu().numpy()  # [N,1,H,W]
                        pseudo_np = (tprobs > 0.5).astype(np.uint8)[:, 0, ...]  # [N,H,W]
                        teacher_probs_np = tprobs
                    else:
                        tsoft = torch.softmax(tlogits, dim=1)
                        tprobs = tsoft.cpu().numpy()  # [N,C,H,W]
                        pseudo_np = np.argmax(tprobs, axis=1)  # [N,H,W]
                        teacher_probs_np = tprobs

                try:
                    pseudo_np = denoise_pseudo_label(pseudo_np, min_area=100, morph_op='open',
                                                     morph_structure=np.ones((3, 3)))
                    pixel_masks = pixel_gate_mask(teacher_probs_np, pseudo_np)
                    mask_ok = mask_quality_filter_with_pixel_mask(teacher_probs_np, pseudo_np, pixel_masks)
                    mask_ok = np.asarray(mask_ok).astype(bool).ravel()
                    qc_pass = bool(mask_ok.any())
                except Exception as e:
                    print(f"[Step {learner.global_step}] 质控异常: {e}")
                    mask_ok = np.array([True], dtype=bool)
                    qc_pass = True

                if qc_pass:
                    keep_idx = np.where(mask_ok)[0]
                    pseudo_np_kept = pseudo_np[keep_idx]
                    if args.binary:
                        pseudo_targets = torch.from_numpy(pseudo_np_kept).float().unsqueeze(1).to(device)  # [K,1,H,W]
                    else:
                        pseudo_targets = torch.from_numpy(pseudo_np_kept).long().to(device)  # [K,H,W]
                    update_reason = "qc-pass"

                    # 保存通过质控的伪标签到本地（可选）
                    # 如果你想要更高级的可视化（比如显示teacher和student预测的对比），可以这样：
                    if args.save_pseudo_labels:
                        try:
                            pseudo_save_dir = os.path.join(output_mgr.get_run_dir(), args.pseudo_save_dir)
                            os.makedirs(pseudo_save_dir, exist_ok=True)

                            for idx, pseudo_mask in enumerate(pseudo_np_kept):
                                # 创建对比图：原图 + teacher伪标签 + student预测
                                fig_height = item.rgb_np.shape[0]
                                fig_width = item.rgb_np.shape[1] * 3  # 三张图并排

                                comparison_img = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)

                                # 左侧：原图
                                comparison_img[:, :args.img_size] = item.rgb_np

                                # 中间：teacher伪标签叠加
                                teacher_overlay = visualizer.create_overlay_image(item.rgb_np, pseudo_mask)
                                comparison_img[:, args.img_size:args.img_size * 2] = teacher_overlay

                                # 右侧：student预测叠加
                                student_pred_np = student_pred.detach().cpu().numpy().astype(np.uint8)
                                student_overlay = visualizer.create_overlay_image(item.rgb_np, student_pred_np)
                                comparison_img[:, args.img_size * 2:] = student_overlay

                                # 保存对比图
                                comparison_path = os.path.join(pseudo_save_dir,
                                                               f"comparison_{learner.global_frame_idx:06d}_{idx:02d}.png")
                                comparison_bgr = cv2.cvtColor(comparison_img, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(comparison_path, comparison_bgr)

                        except Exception as e:
                            print(f"[Step {learner.global_step}] 保存伪标签对比图异常: {e}")
                else:
                    update_reason = "qc-fail"

            # Student 更新(受门控控制)
            may_update = (int(args.update_every_n) <= 1) or (learner.global_frame_idx % int(args.update_every_n) == 0)
            if not may_update and pseudo_targets is not None and pseudo_targets.shape[0] > 0:
                update_reason = f"gated(update_every_n={int(args.update_every_n)})"
            if pseudo_targets is not None and pseudo_targets.shape[0] > 0 and may_update:
                learner.model.train()
                learner.optimizer.zero_grad()
                use_amp = args.use_amp and torch.cuda.is_available() and (autocast is not None)
                skip = False
            if use_amp:
                with autocast():
                    out = learner.model(img_tensor)
                    loss_tensor = learner.criterion(out,
                                                    pseudo_targets.float() if args.binary else pseudo_targets.long())
                loss_tensor.backward()
                torch.nn.utils.clip_grad_norm_(learner.model.parameters(), max_norm=1.0)
                skip = learner.ema_safety.step(float(loss_tensor.item()), learner.model)
                if not skip:
                    learner.optimizer.step()
            else:
                out = learner.model(img_tensor)
                loss_tensor = learner.criterion(out, pseudo_targets.float() if args.binary else pseudo_targets.long())
                loss_tensor.backward()
                torch.nn.utils.clip_grad_norm_(learner.model.parameters(), max_norm=1.0)
                skip = learner.ema_safety.step(float(loss_tensor.item()), learner.model)
                if not skip:
                    learner.optimizer.step()
                loss_val = float(loss_tensor.item())
                if not skip:
                    did_update = True
                    update_reason = update_reason
                else:
                    did_update = False
                    update_reason = "ema-safety-skip"
                learner.loss_history.append(loss_val)
                try:
                    learner._save_best_model_if_improved(loss_val)
                except Exception as e:
                    print(f"[WARN] best-save failed: {e}")
                if learner.buffer is not None:
                    learner.buffer.push(img_tensor[0].detach().cpu().clone(),
                                        (pseudo_targets[0].detach().cpu().clone()))

            # 可视化叠加并流式写出（首次写时创建 writer）
            if learner.global_frame_idx % max(1, args.viz_stride) == 0:
                img_vis = item.rgb_np
                pred_np = student_pred.detach().cpu().numpy().astype(np.uint8)
                overlay = visualizer.create_overlay_image(img_vis, pred_np)
                if args.stream_output:
                    if writer is None:
                        run_dir = output_mgr.get_run_dir()
                        out_path = os.path.join(run_dir, "pred_overlay.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        # 选择 FPS
                        if args.writer_fps > 0:
                            _fps = float(args.writer_fps)
                        elif args.writer_fps == 0:
                            _fps = float(src_fps)
                        else:
                            # writer_fps<0: 用吞吐估算，先暂用src_fps，后续打印说明
                            _fps = float(src_fps)
                        writer = cv2.VideoWriter(out_path, fourcc, _fps, (args.img_size, args.img_size))
                        print(f"[INFO] VideoWriter opened at fps={_fps}")
                    writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    wrote_viz = True

            learner.global_frame_idx += 1
            learner.global_step += 1
            processed += 1

            step_ms = (time.time() - step_t0) * 1000.0
            recent_step_ms.append(step_ms)
            if len(recent_step_ms) > 50:
                recent_step_ms.pop(0)

            # 如果选择按吞吐估算FPS（writer_fps<0），在前20帧后打印建议FPS与预计时长
            if args.stream_output and args.writer_fps < 0 and processed == 20:
                avg_ms = sum(recent_step_ms) / len(recent_step_ms)
                est_fps = max(1.0, 1000.0 / avg_ms)
                print(
                    f"[INFO] Estimated processing FPS ~ {est_fps:.2f}. You can set --writer_fps {est_fps:.2f} to lengthen video.")

                print(
                    f"[Step {learner.global_step}] frame={item.index} teacher={'on' if learner.offline_model is not None else 'off'} "
                    f"qc_pass={qc_pass if qc_pass is not None else 'NA'} update={'yes' if did_update else 'no'} "
                    f"reason={update_reason} loss={(f'{loss_val:.6f}' if loss_val is not None else 'NA')} "
                    f"viz={'yes' if wrote_viz else 'no'} qsize={q.qsize()} buffer={(len(learner.buffer) if learner.buffer is not None else 0)} "
                    f"time_ms={step_ms:.1f}")

            if int(args.online_steps) > 0 and processed >= int(args.online_steps):
                print("[INFO] Reached online_steps limit. Stopping streaming loop.")
                break

    finally:
        stop_event.set()
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass


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

    # 进入流式在线循环
    streaming_online_loop(args, learner)

    loss_history, metrics_history = learner.last_results if hasattr(learner, 'last_results') else ([], [])
    summary = output_mgr.get_run_summary()
    final_metrics = metrics_history[-1] if metrics_history else {}
    print("--> Online Learning Completed <--")
    print(f"Results saved to: {summary.get('run_dir', 'N/A')}")
    if final_metrics:
        print(f"Final Metrics - Loss: {final_metrics.get('val_loss', 0):.4f}, IoU: {final_metrics.get('iou', 0):.4f}")


if __name__ == "__main__":
    main()