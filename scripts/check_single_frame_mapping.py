#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-frame WS→Train mapping self-check (3-class / 6-class).

- Loads RGB + WS(gray), applies a WS→Train mapping.
- Supports mapping from constants.py OR from YAML/JSON/CLI.
- Has scheme-aware palettes: 3class_org and 6class.
- Optionally applies an auto FOV mask for visualization.

Usage (6-class, 优先从 constants 读映射)：
  python check_single_frame_mapping.py \
    --image path/to/img.png --ws path/to/ws.png \
    --scheme 6class --use_constants --save_dir out_6c

Usage (从文件读映射，覆盖默认外部值为255忽略)：
  python check_single_frame_mapping.py \
    --image img.png --ws ws.png \
    --mapping ws2train.yaml --default_for_others 255 \
    --scheme 6class --save_dir out_fromfile
"""

import argparse, os, json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

# ---------------- optional YAML ----------------
try:
    import yaml
except Exception:
    yaml = None

# ---------------- try import constants ----------------
HAS_CONST = False
try:
    import src.common.constants as C  # 期望里面定义 6-class 的调色板与映射
    HAS_CONST = True
except Exception:
    HAS_CONST = False


# ================= Palettes (by scheme) =================
# 3-class: 0=bg, 1=instrument, 2=target_organ, 255=ignore
PALETTE_3 = {
    0: (0, 0, 0),
    1: (0, 200, 0),        # instrument - green
    2: (220, 0, 200),      # target organ - magenta
    255: (160, 160, 160),  # ignore
}

# 6-class: 1:liver, 2:fat, 3:gi_tract, 4:instrument, 5:gallbladder, 255:ignore
# 颜色与你展示图一致：liver(绿), fat(粉), gi(黄), inst(青), gall(红), ignore(灰)
PALETTE_6 = {
    0:   (0,   0,   0),     # 背景若出现，画黑（训练里通常不用0）
    1:   (50, 255,  50),    # liver - green
    2:   (255,  50, 255),   # fat - magenta/pink
    3:   (255, 255,  50),   # gi_tract - yellow
    4:   (50, 255, 255),    # instrument - cyan
    5:   (255,  50,  50),   # gallbladder - red
    255: (160, 160, 160),   # ignore - gray
}

def get_palette(scheme: str):
    if scheme == "6class":
        return PALETTE_6
    return PALETTE_3


# ================= Default WS→Train mapping =================
# 适配 Kaggle WS 灰度值：11/12/13/21/22/31/32/50/255
# 若从 constants.py 读取失败，则回退到这里
WS2TRAIN_6CLASS_FALLBACK = {
    11: 255,   # unknown → ignore
    12: 2,     # fat → class 2
    13: 3,     # gi_tract → class 3
    21: 1,     # liver → class 1
    22: 5,     # gallbladder → class 5
    31: 4,     # instrument → class 4
    32: 4,     # instrument → class 4
    50: 4,     # instrument → class 4
    255: 255   # ignore → ignore
}

# 3-class 示例（如果你还需要 3-class 调试）
WS2TRAIN_3CLASS_FALLBACK = {
    # 背景 / 非器械组织 → 0
    11: 0, 12: 0, 13: 0, 21: 0, 255: 255,
    # 目标器官（示例：gallbladder=22） → 2
    22: 2,
    # 器械 → 1
    31: 1, 32: 1, 50: 1,
}


# ================= I/O helpers =================
def load_img_rgb(path):
    return Image.open(path).convert('RGB')

def load_mask_gray(path):
    return np.array(Image.open(path).convert('L'), dtype=np.uint16)


# ================= FOV helper =================
def compute_fov_mask_from_rgb_np(img_rgb_uint8: np.ndarray, thr=None) -> np.ndarray:
    g = (0.299*img_rgb_uint8[...,0] + 0.587*img_rgb_uint8[...,1] + 0.114*img_rgb_uint8[...,2]).astype(np.float32)
    if thr is None:
        p1 = np.percentile(g, 1)
        thr = max(5.0, min(20.0, p1 + 2.0))
    in_fov = g > thr
    in_fov = ndi.binary_closing(in_fov, structure=np.ones((5,5), bool))
    lab, num = ndi.label(in_fov)
    if num > 0:
        sizes = np.bincount(lab.ravel())[1:]
        if sizes.size > 0:
            keep = 1 + int(np.argmax(sizes))
            in_fov = (lab == keep)
    if (1.0 - in_fov.mean()) < 0.05:  # 外圈比例<5%则视作全视野
        in_fov[:] = True
    return in_fov


# ================= mapping loader =================
def load_mapping(args):
    """
    优先级：
      1) --use_constants 且 constants.py 存在对应映射
      2) --mapping YAML/JSON 文件
      3) 命令行 --map k:v
      4) 退回内置 FALLBACK
    """
    mapping = {}
    default_for_others = args.default_for_others

    # 1) constants.py
    if args.use_constants and HAS_CONST:
        if args.scheme == "6class" and hasattr(C, "WS_TO_TRAIN_6CLASS"):
            mapping.update({int(k): int(v) for k, v in C.WS_TO_TRAIN_6CLASS.items()})
            if getattr(C, "DEFAULT_FOR_OTHERS_6CLASS", None) is not None:
                default_for_others = int(C.DEFAULT_FOR_OTHERS_6CLASS)
        elif args.scheme != "6class" and hasattr(C, "WS_TO_TRAIN_3CLASS_ORG"):
            mapping.update({int(k): int(v) for k, v in C.WS_TO_TRAIN_3CLASS_ORG.items()})
            if getattr(C, "DEFAULT_FOR_OTHERS_3CLASS", None) is not None:
                default_for_others = int(C.DEFAULT_FOR_OTHERS_3CLASS)

    # 2) 文件
    if not mapping and args.mapping:
        p = Path(args.mapping)
        text = Path(p).read_text(encoding='utf-8')
        data = yaml.safe_load(text) if p.suffix.lower() in ('.yaml', '.yml') else json.loads(text)
        mapping.update({int(k): int(v) for k, v in data.get('ws_to_train', {}).items()})
        if default_for_others is None and data.get("default_for_others", None) is not None:
            default_for_others = int(data["default_for_others"])

    # 3) CLI kv
    if args.map:
        for kv in args.map:
            k, v = kv.split(':')
            mapping[int(k)] = int(v)

    # 4) fallback
    if not mapping:
        mapping = WS2TRAIN_6CLASS_FALLBACK if args.scheme == "6class" else WS2TRAIN_3CLASS_FALLBACK

    # 默认外部值
    if default_for_others is None:
        default_for_others = 255 if args.scheme == "6class" else 0

    return mapping, int(default_for_others)


# ================= viz helpers =================
def rgb_from_labels(lbl, palette):
    h, w = lbl.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    known = set(palette.keys())
    for k, color in palette.items():
        rgb[lbl == k] = color
    rgb[np.isin(lbl, list(known), invert=True)] = (255, 255, 255)
    return rgb

def overlay(img_rgb, mask_rgb, alpha=0.5):
    base = np.array(img_rgb, dtype=np.float32)
    over = np.array(mask_rgb, dtype=np.float32)
    out = (1 - alpha) * base + alpha * over
    return np.clip(out, 0, 255).astype(np.uint8)

def summarize(arr, name, topn=20):
    vals, cnts = np.unique(arr, return_counts=True)
    order = np.argsort(-cnts)
    total = arr.size
    lines = [f'# Unique values in {name}: {len(vals)}']
    for i in order[:topn]:
        lines.append(f'  - {int(vals[i]):>4d}: {int(cnts[i])} ({cnts[i]/total:.2%})')
    return '\n'.join(lines), dict(zip([int(v) for v in vals], [int(c) for c in cnts]))


# ================= main =================
def main():
    ap = argparse.ArgumentParser()
    

    ap.add_argument('--image', required=True)
    ap.add_argument('--ws', required=True)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--scheme', default='6class', choices=['6class', '3class_org'])
    ap.add_argument('--classification_scheme', default=None, choices=['6class', '3class_org'], 
                    help='Alternative name for --scheme (for compatibility)')
    ap.add_argument('--use_constants', action='store_true', help='优先使用 constants.py 的 WS→Train 映射与默认值')
    ap.add_argument('--mapping', default=None, help='YAML/JSON 自定义映射文件（可选）')
    ap.add_argument('--map', nargs='*', help='附加/覆盖 k:v 形式（可选）')
    ap.add_argument('--default_for_others', type=int, default=None)
    ap.add_argument('--alpha', type=float, default=0.5)

    ap.add_argument('--apply_fov', action='store_true')
    ap.add_argument('--fov_thr', type=float, default=None)
    ap.add_argument('--fov_fill', type=int, default=0)  # 可选填充 0/255

    args = ap.parse_args()
    # argparse 后 - 优先使用 classification_scheme，如果没有则使用 scheme
    scheme = args.classification_scheme or args.scheme
    os.makedirs(args.save_dir, exist_ok=True)

    img = load_img_rgb(args.image)
    ws  = load_mask_gray(args.ws)

    mapping, dfo = load_mapping(args)
    palette = get_palette(scheme)

    # ---- stats
    sum_ws_txt, ws_hist = summarize(ws, 'WS(mask)')
    print(sum_ws_txt)

    # ---- apply mapping
    mapped = np.full(ws.shape, fill_value=dfo, dtype=np.uint16)
    for k, v in mapping.items():
        mapped[ws == k] = v

    sum_map_txt, map_hist = summarize(mapped, 'Mapped(train labels)')
    print(sum_map_txt)

    # ---- diagnostics
    ws_keys = set(ws_hist.keys())
    unmapped_ws = sorted([v for v in ws_keys if v not in mapping])
    ignore_frac = (map_hist.get(255, 0) / ws.size) if ws.size else 0.0

    # ---- render fig1
    fig1 = plt.figure(figsize=(14, 4))
    ax1 = fig1.add_subplot(1,4,1); ax1.imshow(img); ax1.set_title('Original'); ax1.axis('off')
    ax2 = fig1.add_subplot(1,4,2); im2=ax2.imshow(ws, cmap='gray'); ax2.set_title('WS (grayscale)'); ax2.axis('off'); plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fov = None
    mapped_vis = mapped.copy()
    if args.apply_fov:
        img_np = np.array(img, dtype=np.uint8)
        fov = compute_fov_mask_from_rgb_np(img_np, thr=args.fov_thr)
        mapped_vis[~fov] = int(args.fov_fill)

    ax3 = fig1.add_subplot(1,4,3)
    ax3.imshow(rgb_from_labels(mapped_vis.astype(np.uint16), palette)); ax3.set_title('Mapped labels'); ax3.axis('off')

    # 未映射处（以红色在原图标注）
    unm = np.zeros((*ws.shape, 3), np.uint8)
    for v in unmapped_ws: unm[ws == v] = (255, 0, 0)
    ax4 = fig1.add_subplot(1,4,4); ax4.imshow(overlay(img, unm, alpha=0.5)); ax4.set_title('Unmapped WS (red)'); ax4.axis('off')
    fig1.tight_layout(); fig1.savefig(os.path.join(args.save_dir, 'fig1_overview.png'), dpi=120); plt.close(fig1)

    # ---- fig2 overlay
    over = overlay(img, rgb_from_labels(mapped_vis.astype(np.uint16), palette), alpha=args.alpha)
    plt.figure(figsize=(8,6)); plt.imshow(over); plt.title('Overlay: Original + Mapped'); plt.axis('off')
    plt.savefig(os.path.join(args.save_dir, 'fig2_overlay.png'), dpi=120); plt.close()

    # ---- optional fov viz
    if fov is not None:
        edge = (ndi.binary_dilation(~fov, structure=np.ones((3,3))) & fov)
        edge_rgb = np.zeros((*fov.shape,3), np.uint8); edge_rgb[edge] = (255,0,0)
        plt.figure(figsize=(10,4))
        ax1 = plt.subplot(1,2,1); ax1.imshow(fov, cmap='gray'); ax1.set_title('FOV mask'); ax1.axis('off')
        ax2 = plt.subplot(1,2,2); ax2.imshow(overlay(img, edge_rgb, 0.6)); ax2.set_title('FOV boundary'); ax2.axis('off')
        plt.tight_layout(); plt.savefig(os.path.join(args.save_dir, 'fig3_fov.png'), dpi=120); plt.close()

    # ---- summary
    hints = []
    if args.scheme == "6class":
        must = {11,12,13,21,22,31,32,50,255}
        missing = sorted([v for v in must if v not in mapping])
        if missing: hints.append(f"- 6class 映射缺少关键 WS 值：{missing}。")
    if unmapped_ws:
        hints.append(f"- 存在未映射的 WS 灰度值：{unmapped_ws[:30]}，它们将被 default_for_others={dfo} 处理。")
    if ignore_frac > 0.3 and dfo == 255:
        hints.append(f"- ignore(255) 占比 {ignore_frac:.1%} 偏高，确认 WS→Train 是否缺项。")

    summary = [
        f"Scheme: {args.scheme}",
        f"Image: {args.image}",
        f"WS: {args.ws}",
        f"Use constants: {bool(args.use_constants and HAS_CONST)}",
        f"Default for others: {dfo}",
        "",
        sum_ws_txt, "", sum_map_txt, "",
        f"Unmapped WS values: {unmapped_ws[:50]}",
        f"Ignore(255) fraction: {ignore_frac:.2%}",
        "",
        "Hints:",
        *hints
    ]
    Path(args.save_dir, 'summary.txt').write_text('\n'.join(summary), encoding='utf-8')
    print('\n'.join(summary))


if __name__ == '__main__':
    main()
