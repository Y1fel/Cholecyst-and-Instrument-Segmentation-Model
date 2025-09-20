# scripts/export_teacher_split.py
"""
导出Teacher训练时使用的训练/验证集划分
用于确保KD实验使用与Teacher完全相同的验证集
"""
import os
import yaml
import torch
from src.dataio.datasets.seg_dataset_min import SegDatasetMin  # 修正类名
from src.common.constants import compose_mapping

# Teacher训练时的精确参数 - 请确保与Teacher训练时完全一致
DATA_ROOT = "data/seg8k"
IMG_SIZE = 384
VAL_RATIO = 0.25
SEED = 42
APPLY_FOV = True
CLASSIFICATION_SCHEME = "3class_org"  # 与Teacher配置一致

print("🔍 Extracting Teacher's original train/val split...")
print(f"Parameters: img_size={IMG_SIZE}, val_ratio={VAL_RATIO}, seed={SEED}")
print(f"FOV mask: {APPLY_FOV}, classification: {CLASSIFICATION_SCHEME}")

# === 重要：与Teacher完全一致的数据集配置 ===
class_id_map = compose_mapping(
    classification_scheme=CLASSIFICATION_SCHEME,
    custom_mapping=None,
    target_classes=None
)

dataset_config = {
    "class_id_map": class_id_map,
    "return_multiclass": True,  # 多分类模式
    "apply_fov_mask": APPLY_FOV
}

# 构造与Teacher相同的数据集
full_dataset = SegDatasetMin(
    DATA_ROOT, 
    dtype="train", 
    img_size=IMG_SIZE,
    **dataset_config
)

print(f"Dataset loaded: {len(full_dataset)} samples")

# 使用Teacher的原始random_split逻辑（帧级划分）
N = len(full_dataset)
val_size = int(N * VAL_RATIO)
train_size = N - val_size
g = torch.Generator().manual_seed(SEED)
train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=g)

# 提取训练/验证的图像路径
def get_image_path(idx):
    return full_dataset.pairs[idx][0]  # (img_path, mask_path)

train_paths = [get_image_path(i) for i in train_ds.indices]
val_paths = [get_image_path(i) for i in val_ds.indices]

# 保存Teacher的分割信息
os.makedirs("splits", exist_ok=True)
teacher_split = {
    "train": train_paths,
    "val": val_paths,
    "metadata": {
        "total_samples": N,
        "train_samples": len(train_paths),
        "val_samples": len(val_paths),
        "actual_val_ratio": len(val_paths) / N,
        "seed": SEED,
        "img_size": IMG_SIZE,
        "apply_fov_mask": APPLY_FOV,
        "classification_scheme": CLASSIFICATION_SCHEME,
        "split_method": "frame_random",
        "source": "teacher_training",
        "export_date": "2025-09-19"
    }
}

output_file = "splits/teacher_frame_split.yaml"
with open(output_file, "w") as f:
    yaml.safe_dump(teacher_split, f, sort_keys=False)

print(f"✅ Teacher split exported:")
print(f"   Train: {len(train_paths)} samples")  
print(f"   Val: {len(val_paths)} samples")
print(f"   Ratio: {len(val_paths)/N:.3f}")
print(f"   Saved to: {output_file}")
print(f"\n📋 To use this split in experiments, add to config:")
print(f"   split_strategy: from_file")
print(f"   split_file: {output_file}")