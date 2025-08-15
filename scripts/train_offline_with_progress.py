# scripts/train_offline_with_progress.py
# 单行刷新版最小训练脚本（仅替换显示方式；其余保持不变）

import os, argparse, time, shutil, subprocess
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataio.datasets.seg_dataset_min import SegDatasetMin
from src.models.baseline.unet_min import UNetMin

import csv
from datetime import datetime


# ---------- GPU 监控 ----------
def _nvidia_smi_available():
    return shutil.which("nvidia-smi") is not None

def _gpu_stats_via_pynvml():
    try:
        import pynvml as N
        N.nvmlInit()
        h = N.nvmlDeviceGetHandleByIndex(0)
        util = N.nvmlDeviceGetUtilizationRates(h).gpu
        mem = N.nvmlDeviceGetMemoryInfo(h)
        used_mb = mem.used // (1024 * 1024)
        total_mb = mem.total // (1024 * 1024)
        N.nvmlShutdown()
        return util, used_mb, total_mb
    except Exception:
        return None

def _gpu_stats_via_smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True
        ).strip().splitlines()
        if not out: return None
        util, used, total = out[0].split(",")
        return int(util.strip()), int(used.strip()), int(total.strip())
    except Exception:
        return None

def get_gpu_stats():
    s = _gpu_stats_via_pynvml()
    if s is not None: return s
    if _nvidia_smi_available(): return _gpu_stats_via_smi()
    return None

# ---------- 参数 ----------
def parse_args():
    p = argparse.ArgumentParser("可视化版最小离线训练（单行刷新）")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--save_path", type=str, default="checkpoints/baseline_offline_min.pth")
    p.add_argument("--num_workers", type=int, default=0)  # Windows 建议 0
    return p.parse_args()

# ---------- 小工具 ----------
def fmt_time(sec: float) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h: return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

# ---------- 主逻辑（单行刷新） ----------
def main():
    # 保持loss
    os.makedirs("logs", exist_ok=True)
    log_csv = os.path.join("logs", "loss_log.csv")
    write_header = not os.path.exists(log_csv)

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    ds = SegDatasetMin(args.data_root, "", args.img_size)  # 递归模式
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=(device=="cuda"))

    out_ch = 1 if args.num_classes == 2 else args.num_classes
    model = UNetMin(in_ch=3, num_classes=out_ch, base=32).to(device)
    criterion = nn.BCEWithLogitsLoss() if args.num_classes == 2 else nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_batches = len(dl)
    total_steps = args.epochs * total_batches
    step = 0
    t0 = time.time()

    for ep in range(args.epochs):
        model.train()
        running = 0.0
        for bi, (imgs, msks) in enumerate(dl):
            imgs = imgs.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, msks)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running += loss.item() * imgs.size(0)
            step += 1

            # —— 追加写入 batch 级别日志到 CSV ——
            try:
                with open(log_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(["ts", "epoch", "batch", "step", "loss"])  # header
                        write_header = False
                    w.writerow([
                        datetime.now().isoformat(timespec="seconds"),
                        ep + 1,
                        bi + 1,
                        step,
                        float(loss.item())
                    ])
            except Exception:
                # 日志写入失败不中断训练（可根据需要改成 logger.warning）
                pass

            # —— 单行刷新输出 —— #
            elapsed = time.time() - t0
            it_per_s = step / max(elapsed, 1e-6)
            remain = (total_steps - step) / max(it_per_s, 1e-6)

            gpu_str = ""
            if device == "cuda":
                s = get_gpu_stats()
                if s is not None:
                    util, used, total = s
                    gpu_str = f" | GPU {util:>3}% {used}/{total}MB"

            line = (f"\rEpoch {ep+1}/{args.epochs}  "
                    f"[{bi+1:>4d}/{total_batches:<4d}]  "
                    f"loss={loss.item():.4f}  it/s={it_per_s:.2f}  "
                    f"ETA={fmt_time(remain)}{gpu_str}")
            print(line, end="", flush=True)

        # epoch 结束，换行并打印平均损失
        epoch_avg = running / len(ds)
        print(f"\n[epoch {ep+1}] avg_loss={epoch_avg:.4f}")

        # —— 追加写入 epoch 平均日志 ——
        try:
            with open(log_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    ep + 1,
                    "epoch_avg",
                    step,
                    float(epoch_avg)
                ])
        except Exception:
            pass

    torch.save({"state_dict": model.state_dict()}, args.save_path)
    print("Saved:", args.save_path)

if __name__ == "__main__":
    main()
