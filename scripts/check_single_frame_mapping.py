
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-frame WS→Train mapping self-check.

What it does
------------
- Loads an RGB image and its WS (watershed) grayscale mask.
- Prints WS unique values & counts.
- Applies a WS→Train mapping (from YAML/JSON or CLI) with a configurable default_for_others.
- Prints mapped unique values & counts; reports fraction of ignore(255).
- Highlights *unmapped* WS values (i.e., values not present in the mapping keys).
- Saves diagnostic figures:
    fig1: [Original | WS(gray) | Mapped label color | Unmapped overlay]
    fig2: [Original with mapped overlay (alpha)]
- Writes a short summary.txt with key stats & heuristics.

Usage
-----
python check_single_frame_mapping.py --image path/to/img.png --ws path/to/ws.png         --mapping path/to/mapping.yaml --save_dir outputs/check_001

Mapping file (YAML/JSON) schema
-------------------------------
ws_to_train: { <ws_gray_value>: <train_class_id>, ... }
default_for_others: 0 or 255 (recommended 0 for robustness in 3-class setting)

Notes
-----
- In 3-class setting (0=background,1=instrument,2=target_organ), leaving many WS
  values unmapped will send them to default_for_others. If default=255, those pixels
  don't contribute loss (ignore), causing GT to appear as "holes" or "skirting" around
  the true organ in visualization. Prefer mapping most non-instrument tissue to 0.
"""

import argparse, os, json, sys, math
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

try:
    import yaml  # optional, only needed for YAML mappings
except Exception:
    yaml = None

import numpy as np
from scipy import ndimage as ndi

def compute_fov_mask_from_rgb_np(img_rgb_uint8: np.ndarray, thr=None) -> np.ndarray:
    """逐帧自适应估计内镜视野(FOV)。True=视野内，False=外圈黑环。"""
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
    # 保护：如果外圈比例很小，当作“全视野”
    if (1.0 - in_fov.mean()) < 0.05:
        in_fov[:] = True
    return in_fov

# ---------- palettes (BGR-like but written as RGB) ----------
# Adjust as needed for your project
PALETTE = {
    0: (0, 0, 0),          # background -> black
    1: (0, 200, 0),        # instrument -> green
    2: (220, 0, 200),      # target organ -> magenta
    255: (160, 160, 160),  # ignore -> gray
}

def load_img_rgb(path):
    return Image.open(path).convert('RGB')

def load_mask_gray(path):
    # Assumes 8-bit WS grayscale mask (values like 0,5,9,10,21,22,...)
    arr = np.array(Image.open(path).convert('L'), dtype=np.uint16)
    return arr

def load_mapping(path, cli_override=None, default_for_others=None):
    if path is None and cli_override is None:
        raise ValueError('Provide --mapping YAML/JSON or --map kv pairs.')
    mapping = {}
    df = None
    if path is not None:
        p = Path(path)
        with open(p, 'r', encoding='utf-8') as f:
            text = f.read()
        if p.suffix.lower() in ('.yaml', '.yml'):
            if yaml is None:
                raise RuntimeError('PyYAML not installed. Use JSON mapping or CLI --map.')
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        mapping.update({int(k): int(v) for k, v in data.get('ws_to_train', {}).items()})
        df = data.get('default_for_others', None)
    if cli_override:
        # cli_override format: ["10:2","5:1","9:1"]
        for kv in cli_override:
            k, v = kv.split(':')
            mapping[int(k)] = int(v)
    if default_for_others is not None:
        df = int(default_for_others)
    if df is None:
        df = 255  # fallback default; can be overridden via file or CLI
    return mapping, df

def apply_mapping(ws, mapping, default_for_others=255):
    mapped = np.full(ws.shape, fill_value=default_for_others, dtype=np.uint16)
    for k, v in mapping.items():
        mapped[ws == k] = v
    return mapped

def rgb_from_labels(lbl, palette=PALETTE):
    h, w = lbl.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # Assign colors for known keys; unknowns -> white for visibility
    known_keys = set(palette.keys())
    for k, color in palette.items():
        rgb[lbl == k] = color
    rgb[np.isin(lbl, list(known_keys), invert=True)] = (255, 255, 255)
    return rgb

def overlay(img_rgb, mask_rgb, alpha=0.5):
    base = np.array(img_rgb, dtype=np.float32)
    over = np.array(mask_rgb, dtype=np.float32)
    out = (1 - alpha) * base + alpha * over
    return np.clip(out, 0, 255).astype(np.uint8)

def summarize(arr, name, topn=20):
    vals, cnts = np.unique(arr, return_counts=True)
    order = np.argsort(-cnts)
    lines = [f'# Unique values in {name}: {len(vals)}']
    total = arr.size
    for i in order[:topn]:
        lines.append(f'  - {int(vals[i]):>4d}: {int(cnts[i])} ({cnts[i]/total:.2%})')
    return '\n'.join(lines), dict(zip([int(v) for v in vals], [int(c) for c in cnts]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply_fov', action='store_true', help='Apply FOV mask after mapping')
    ap.add_argument('--fov_thr', type=float, default=None, help='Optional fixed threshold for FOV (5~20). If None, auto')
    ap.add_argument('--fov_fill', type=int, default=0, help='Fill value outside FOV (0 for background, or 255 for ignore)')

    ap.add_argument('--image', required=True, help='RGB image path')
    ap.add_argument('--ws', required=True, help='WS grayscale mask path')
    ap.add_argument('--mapping', default=None, help='YAML/JSON mapping file')
    ap.add_argument('--map', nargs='*', help='Override/add k:v pairs (e.g., 10:2 5:1 9:1)')
    ap.add_argument('--default_for_others', type=int, default=None, help='Override default_for_others (0 or 255)')
    ap.add_argument('--save_dir', required=True, help='Output folder')
    ap.add_argument('--alpha', type=float, default=0.5, help='Overlay alpha')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    img = load_img_rgb(args.image)
    ws = load_mask_gray(args.ws)

    mapping, dfo = load_mapping(args.mapping, args.map, args.default_for_others)

    # Stats before mapping
    sum_ws_txt, ws_hist = summarize(ws, 'WS(mask)')
    print(sum_ws_txt)

    mapped = apply_mapping(ws, mapping, default_for_others=dfo)
    sum_map_txt, map_hist = summarize(mapped, 'Mapped(train labels)')
    print(sum_map_txt)

    # Diagnostics
    ws_keys = set(ws_hist.keys())
    map_keys = set(mapping.keys())
    unmapped_ws_vals = sorted([int(v) for v in (ws_keys - map_keys)])
    ignore_frac = (map_hist.get(255, 0) / ws.size) if ws.size > 0 else 0.0
    target_frac = (map_hist.get(2, 0) / (ws.size - map_hist.get(255, 0))) if (ws.size - map_hist.get(255, 0)) > 0 else 0.0

    # Build unmapped mask for visualization
    unmapped_mask = np.zeros_like(ws, dtype=bool)
    for v in unmapped_ws_vals:
        unmapped_mask[ws == v] = True

    # Render figures
    # fig1: Original | WS(gray) | Mapped color | Unmapped overlay
    fig1 = plt.figure(figsize=(14, 4))
    ax1 = fig1.add_subplot(1, 4, 1); ax1.imshow(img); ax1.set_title('Original'); ax1.axis('off')
    ax2 = fig1.add_subplot(1, 4, 2); im2 = ax2.imshow(ws, cmap='gray'); ax2.set_title('WS (grayscale)'); ax2.axis('off'); plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    mapped_rgb = rgb_from_labels(mapped.astype(np.uint16))

    fov = None
    if args.apply_fov:
        img_np = np.array(img, dtype=np.uint8)  # img 是 PIL Image; 若是别的变量名，取加载后的 RGB
        fov = compute_fov_mask_from_rgb_np(img_np, thr=args.fov_thr)
        mapped = mapped.copy()
        mapped[~fov] = int(args.fov_fill)
        # 重新着色
        mapped_rgb = rgb_from_labels(mapped.astype(np.uint16))

    ax3 = fig1.add_subplot(1, 4, 3); ax3.imshow(mapped_rgb); ax3.set_title('Mapped labels'); ax3.axis('off')
    # Unmapped overlay: red on original
    red_overlay = np.zeros((*ws.shape, 3), dtype=np.uint8); red_overlay[unmapped_mask] = (255, 0, 0)
    ax4 = fig1.add_subplot(1, 4, 4); ax4.imshow(overlay(img, red_overlay, alpha=0.5)); ax4.set_title('Unmapped WS (red)'); ax4.axis('off')
    fig1.tight_layout()

    if fov is not None:
        fig3 = plt.figure(figsize=(10,4))
        ax1 = fig3.add_subplot(1,2,1); ax1.imshow(fov, cmap='gray'); ax1.set_title('FOV mask (white=in FOV)'); ax1.axis('off')
        # 叠加边界可视化
        edge = (ndi.binary_dilation(~fov, structure=np.ones((3,3))) & fov)
        edge_rgb = np.zeros((*fov.shape,3), np.uint8); edge_rgb[edge] = (255,0,0)
        ax2 = fig3.add_subplot(1,2,2); ax2.imshow(overlay(img, edge_rgb, alpha=0.6)); ax2.set_title('FOV boundary on image'); ax2.axis('off')
        fig3.tight_layout()
        fig3.savefig(os.path.join(args.save_dir, 'fig3_fov.png'), dpi=120); plt.close(fig3)

    fig1_path = os.path.join(args.save_dir, 'fig1_overview.png')
    fig1.savefig(fig1_path, dpi=120)
    plt.close(fig1)

    # fig2: Original with mapped overlay
    over = overlay(img, mapped_rgb, alpha=args.alpha)
    fig2 = plt.figure(figsize=(8, 6))
    ax = fig2.add_subplot(1,1,1); ax.imshow(over); ax.set_title('Overlay: Original + Mapped'); ax.axis('off')
    fig2_path = os.path.join(args.save_dir, 'fig2_overlay.png')
    fig2.savefig(fig2_path, dpi=120)
    plt.close(fig2)

    # Summary heuristics
    hints = []
    if ignore_frac > 0.2:
        hints.append(f'- High ignore(255) fraction: {ignore_frac:.1%}. Consider mapping more WS values to background(0) instead of 255.')
    if 2 not in map_hist:
        hints.append('- No target_organ(2) after mapping. Check if WS value for gallbladder (e.g., 10) is included.')
    if len(unmapped_ws_vals) > 0:
        hints.append(f'- Unmapped WS gray values present: {unmapped_ws_vals[:20]} (showing up to 20). These go to default_for_others={dfo}.')
    if dfo == 255 and (len(unmapped_ws_vals) > 0):
        hints.append('- default_for_others=255 + unmapped WS leads to large gray regions in GT and weak supervision. Prefer default_for_others=0 for 3-class.')

    summary = [
        f'Image: {args.image}',
        f'WS mask: {args.ws}',
        f'Mapping file: {args.mapping}',
        f'Mapping keys: {sorted(list(map_keys))[:30]}',
        f'Default for others: {dfo}',
        '',
        sum_ws_txt,
        '',
        sum_map_txt,
        '',
        f'Ignore(255) fraction: {ignore_frac:.2%}',
        f'Target(2) fraction among non-ignore: {target_frac:.2%}',
        f'Unmapped WS values: {unmapped_ws_vals[:50]}',
        '',
        'Hints:',
        *hints,
        '',
        f'Saved: {fig1_path}',
        f'Saved: {fig2_path}',
    ]
    with open(os.path.join(args.save_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))

    print('\n'.join(summary))

if __name__ == '__main__':
    main()
