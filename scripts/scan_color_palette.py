#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_color_palette_color.py
---------------------------
用于 *color mask* 的调色板一致性检查（不是 WS 灰度）。
默认用“首张图的唯一颜色集合”作为参考调色板（reference palette），
之后每张图的颜色集合都必须是这个参考集合的子集；否则记为异常。

也支持从 YAML 文件提供参考调色板（见 --ref_yaml）。

用法示例
--------
# 方式1：自动从首张图学习参考调色板（推荐）
python scan_color_palette_color.py \
  --root /path/to/data/seg8k \
  --glob "**/*_endo_color_mask.png" \
  --csv_out palette_color_anomalies.csv

# 方式2：从 YAML 指定参考调色板（RGB 三元组列表）
python scan_color_palette_color.py \
  --root /path/to/data/seg8k \
  --glob "**/*_endo_color_mask.png" \
  --csv_out palette_color_anomalies.csv \
  --ref_yaml color_palette_ref.yaml

YAML 例子（color_palette_ref.yaml）
-----------------------------------
palette:
  - [127,127,127]
  - [169,255,184]
  - [170,255,0]
  - [186,183,75]
  - [210,140,140]
  - [231,70,156]
  - [255,114,114]
  - [255,160,165]
  - [255,255,255]
"""

import argparse, csv, json
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
from PIL import Image

try:
    import yaml
except Exception:
    yaml = None

RGB = Tuple[int, int, int]

def unique_colors(p: Path) -> List[RGB]:
    """读取 color mask（PIL->RGB），返回唯一 RGB 列表。"""
    arr = np.array(Image.open(p).convert("RGB"))
    u = np.unique(arr.reshape(-1, 3), axis=0)
    return [tuple(map(int, x)) for x in u.tolist()]

def load_ref_palette(ref_yaml: str) -> Set[RGB]:
    """从 YAML 读入参考调色板。"""
    if yaml is None:
        raise RuntimeError("PyYAML 未安装，无法读取 --ref_yaml。请 `pip install pyyaml` 或使用自动学习模式。")
    data = yaml.safe_load(Path(ref_yaml).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "palette" not in data:
        raise ValueError("ref_yaml 缺少 'palette' 键")
    pal = data["palette"]
    ref = set()
    for it in pal:
        if not (isinstance(it, (list, tuple)) and len(it) == 3):
            raise ValueError("palette 中每个颜色需为长度为3的数组，例如 [127,127,127]")
        r, g, b = map(int, it)
        ref.add((r, g, b))
    return ref

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="根目录（递归扫描）")
    ap.add_argument("--glob", default="**/*.png", help="相对 root 的通配模式")
    ap.add_argument("--csv_out", default="palette_color_anomalies.csv", help="输出 CSV")
    ap.add_argument("--limit", type=int, default=0, help="仅扫描前 N 个文件")
    ap.add_argument("--ref_yaml", type=str, default="", help="参考调色板 YAML（若不提供则自动用首张图学习）")
    args = ap.parse_args()

    root = Path(args.root)
    files = list(root.glob(args.glob))
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    if not files:
        print("[scan_color_palette_color] 没有匹配到文件。")
        return

    # 参考调色板
    if args.ref_yaml:
        ref_palette = load_ref_palette(args.ref_yaml)
        print(f"[ref] 从 YAML 载入参考调色板，颜色数={len(ref_palette)}")
    else:
        ref_palette = set(unique_colors(files[0]))
        print(f"[ref] 自动学习参考调色板，样本文件={files[0]}")
        print(f"[ref] 参考颜色（{len(ref_palette)} 个）: {sorted(ref_palette)}")

    # 扫描
    rows = []
    num_bad = 0
    for i, p in enumerate(files, 1):
        colors = set(unique_colors(p))
        # 子集判断：文件颜色必须都属于参考集合
        extra = colors - ref_palette          # 不在参考集合内的颜色
        missing = ref_palette - colors        # 参考集合里，这张图没出现（仅供信息，不算错误）
        is_bad = len(extra) > 0
        if is_bad:
            num_bad += 1
        rows.append({
            "filepath": str(p),
            "num_unique": len(colors),
            "extra_color_count": len(extra),
            "extra_colors_sample": json.dumps(sorted(list(extra))[:10]),
            "missing_ref_colors": len(missing),  # 仅信息
        })
        if i % 200 == 0:
            print(f"  processed {i}/{len(files)}")

    # 写 CSV
    csv_path = Path(args.csv_out)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filepath","num_unique","extra_color_count","extra_colors_sample","missing_ref_colors"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[scan_color_palette_color] scanned={len(files)}, with_bad={num_bad}. CSV -> {csv_path}")

if __name__ == "__main__":
    main()
