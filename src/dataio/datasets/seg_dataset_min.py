"""
最小分割数据集类
支持标准的 images/ 和 masks/ 目录结构
"""

import os, glob, cv2, numpy as np, torch
from torch.utils.data import Dataset

# 
class SegDatasetMin(Dataset):
    def __init__(self, data_root: str, dtype: str = "train", img_size: int = 512):
        self.data_root = os.path.abspath(data_root)
        self.dtype = dtype if dtype is not None else ""
        self.img_size = int(img_size)

        ## get dataset path (for normal mask & img paths)
        # img_dir = os.path.join(self.data_root, self.dtype, "images")
        # msk_dir = os.path.join(self.data_root, self.dtype, "masks")
        # self.img_Paths = sorted(glob.glob(os.path.join(img_dir, "*")))

        # assert len(self.img_Paths) > 0, f"No images found in {img_dir}"

        # get dataset specific for seg8k (for current baseline, and offline in later process)
        # 
        pattern_img = [
            "**/*_endo.png", 
            "**/*_endo.jpg", 
            "**/*_endo.jpeg",
            "**/*_endo_color.png", 
            "**/*_endo_color.jpg"
        ]

        # 
        cand_imgs = []

        #
        for each in pattern_img:
            cand_imgs.extend(
                glob.glob(
                    os.path.join(self.data_root, each),
                    recursive = True
                )
            )
        cand_imgs = [each for each in cand_imgs if "mask" not in os.path.basename(each).lower()]

        #
        pairs = []

        for img_path in sorted(set(cand_imgs)):
            # match mask priority: color_mask > mask > watershed_mask
            stem, ext = os.path.splitext(img_path) # remove extension
            candidates = [
                stem.replace("_endo",       "_endo_mask") + ".png",       # 1. mask (最准确)
                stem + "_mask.png",                                       # 
                stem.replace("_endo", "_endo_watershed_mask") + ".png",   # 2. watershed mask
                stem.replace("_endo_color", "_endo_color_mask") + ".png", # 3. color mask
                stem.replace("_endo",       "_endo_color_mask") + ".png", #   for imgs end with *_endo.png 
            ]

            mask_path = next((
                c for c in candidates if os.path.exists(c)
            ), None)

            if mask_path is not None:
                pairs.append((img_path, mask_path))

        assert len(pairs) > 0, (
            f"No image-mask pairs found under {self.data_root}.\n"
            "Example expected: .../frame_80_endo.png  +  .../frame_80_endo_color_mask.png")
        
        self.pairs = pairs # [(img_path, mask_path), ...]

    def __len__(self):
       return len(self.pairs)

    def __getitem__(self, index):
        img_path, mask_path = self.pairs[index]

        #  read images from BGR to RGB
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # read mask, 1 channel, binary classification
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) # to gray scale

        mask = ((mask == 12) | (mask == 13) | (mask == 21) | (mask == 22)).astype(np.uint8)  # 胆囊+器械

        # resize to target size (same size for both img and mask)
        img = cv2.resize(img,
                         (self.img_size, self.img_size),
                         interpolation = cv2.INTER_LINEAR)
        mask= cv2.resize(mask,
                         (self.img_size, self.img_size),
                         interpolation = cv2.INTER_NEAREST)
        
        # align with unet
        img = img.astype(np.float32) / 255.0 
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        mask = mask[np.newaxis, :, :].astype(np.float32)  # add channel dimension

        if index < 10:
            unique_vals = np.unique(mask)
            foreground_ratio = mask.mean()
            print(f"Sample {index}: Mask unique values: {unique_vals}, Foreground ratio: {foreground_ratio:.3f}")

        return torch.from_numpy(img), torch.from_numpy(mask)

    def analyze_mask_distribution(self, num_samples=50):
            """分析mask分布情况"""
            print("🔍 分析前景分布...")
            
            foreground_ratios = []
            for i in range(min(num_samples, len(self.pairs))):
                _, mask_path = self.pairs[i]
                
                # 读取原始mask
                raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if raw_mask.ndim == 3:
                    raw_mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY)
                
                # 应用胆囊+器械策略
                processed_mask = ((raw_mask == 12) | (raw_mask == 13) | (raw_mask == 21) | (raw_mask == 22)).astype(np.uint8)
                
                foreground_ratio = processed_mask.mean()
                foreground_ratios.append(foreground_ratio)
                
                if i < 5:  # 打印前5个样本的详细信息
                    unique_raw = np.unique(raw_mask)
                    print(f"  Sample {i}: Raw values {unique_raw} -> Foreground ratio {foreground_ratio:.3f}")
            
            avg_ratio = np.mean(foreground_ratios)
            print(f"📊 平均前景比例: {avg_ratio:.3f}")
            print(f"📊 前景比例范围: {np.min(foreground_ratios):.3f} - {np.max(foreground_ratios):.3f}")
            
            return foreground_ratios