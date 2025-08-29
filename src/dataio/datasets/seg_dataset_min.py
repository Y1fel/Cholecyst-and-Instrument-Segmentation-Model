"""
最小分割数据集类
支持标准的 images/ 和 masks/ 目录结构
"""

import os, glob, cv2, numpy as np, torch
from torch.utils.data import Dataset
from src.common.constants import DATASET_CONFIG

# 
class SegDatasetMin(Dataset):
    def __init__(self, data_root: str, dtype: str = "train", img_size: int = 512,
            return_multiclass: bool = False,
            class_id_map: dict | None = None,
            ignore_index: int = 255
        ):

        self.data_root = os.path.abspath(data_root)
        self.dtype = dtype if dtype is not None else ""
        self.img_size = int(img_size)

        self.return_multiclass = bool(return_multiclass)
        self.ignore_index = int(ignore_index)

        self.class_id_map = class_id_map or DATASET_CONFIG["SEG8K_CLASS_MAPPING"]

        # get dataset specific for seg8k (for current baseline, and offline in later process)
        # 
        pattern_img = DATASET_CONFIG["SEG8K_IMAGE_PATTERNS"]

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
            # match mask priority: mask > watershed_mask > color_mask
            stem, _ = os.path.splitext(img_path) # remove extension

            if self.return_multiclass:
                # 多分类：优先精确mask
                candidates = self._get_multiclass_mask_candidates(stem)
            else:
                # 二分类：可以使用近似mask
                candidates = self._get_binary_mask_candidates(stem)

            mask_path = next((
                c for c in candidates if os.path.exists(c)
            ), None)

            if mask_path is not None:
                pairs.append((img_path, mask_path))

        assert len(pairs) > 0, (
            f"No image-mask pairs found under {self.data_root}.\n"
            "Example expected: .../frame_80_endo.png  +  .../frame_80_endo_color_mask.png")
        
        self.pairs = pairs # [(img_path, mask_path), ...]
        
        # 数据集总结输出
        mask_types = {}
        for _, mask_path in self.pairs[:20]:  # 检查前20个样本的类型分布
            if mask_path.endswith('_endo_watershed_mask.png'):
                mask_types['watershed'] = mask_types.get('watershed', 0) + 1
            elif mask_path.endswith('_precise_gt.png'):
                mask_types['precise_gt'] = mask_types.get('precise_gt', 0) + 1
            else:
                mask_types['original'] = mask_types.get('original', 0) + 1
        
        print(f"📁 数据集加载完成: {len(self.pairs)} 对图像-标签")
        if self.return_multiclass:
            print(f"🎯 多分类模式: Watershed区域→训练类别映射")
        else:
            print(f"🎯 二分类模式: 胆囊+器械 vs 背景")
        
        dominant_type = max(mask_types, key=mask_types.get) if mask_types else 'unknown'
        print(f"🏷️  主要标签类型: {dominant_type} ({mask_types.get(dominant_type, 0)}/{min(20, len(self.pairs))} 样本)")
        print(f"📏 图像尺寸: {self.img_size}x{self.img_size}")
        print("-" * 50)

    def __len__(self):
       return len(self.pairs)

    def __getitem__(self, index):
        img_path, mask_path = self.pairs[index]

        #  read images
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) # from BGR to RGB
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
        img = cv2.resize(img,
                         (self.img_size, self.img_size),
                         interpolation = cv2.INTER_LINEAR) # resize to target size (same size for both img and mask)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW

        # read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        # 处理精确GT文件（单通道）和原始mask文件（可能是3通道）
        if mask.ndim == 3:
            # 检查是否是精确GT（应该是单通道但被保存为3通道）
            if os.path.basename(mask_path).endswith('_precise_gt.png'):
                # 精确GT：取第一个通道（所有通道应该相同）
                mask = mask[:, :, 0]
                if index < 3:  # 只显示前3个样本的详细信息
                    print(f"[PRECISE_GT] 使用精确GT: {os.path.basename(mask_path)}")
            else:
                # 原始mask：转为灰度图
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                # 移除冗余的ORIGINAL输出，因为大部分都是watershed mask
        else:
            if index < 3:  # 只显示前3个样本的详细信息
                print(f"[SINGLE_CH] 使用单通道mask: {os.path.basename(mask_path)}")

        if self.return_multiclass:
            # 多类：根据 class_id_map 做 ID 映射，未知值 -> ignore_index
            m = np.full_like(mask, fill_value=self.ignore_index, dtype=np.uint8)

            # 检查是否使用watershed mask
            if os.path.basename(mask_path).endswith('_endo_watershed_mask.png'):
                # Watershed处理：重新编号区域ID
                unique_regions = np.unique(mask)
                region_mapping = {}
                
                # 背景始终为0
                region_mapping[0] = 0
                
                # 其他区域重新编号为1, 2, 3...
                class_id = 1
                for region_id in unique_regions:
                    if region_id != 0 and region_id != 255 and class_id < 10:  # 跳过背景和ignore_index
                        region_mapping[region_id] = class_id
                        class_id += 1
                
                # 应用映射
                for original_id, new_id in region_mapping.items():
                    m[mask == original_id] = new_id
                
                # 保持ignore_index不变
                m[mask == 255] = self.ignore_index
                
                # 只显示前5个样本的详细信息，之后只显示简化版本
                if index < 5:
                    print(f"  [WATERSHED] 区域重新编号: {len(unique_regions)}个原始区域→{class_id-1}个训练类别")
                elif index == 5:
                    print(f"  [WATERSHED] 后续样本区域编号将静默处理...")
            # 检查是否使用精确GT  
            elif os.path.basename(mask_path).endswith('_precise_gt.png'):
                # 精确GT：直接使用，不需要重新映射
                from src.common.constants import DATASET_CONFIG
                if "PRECISE_GT_CLASS_MAPPING" in DATASET_CONFIG:
                    precise_mapping = DATASET_CONFIG["PRECISE_GT_CLASS_MAPPING"]
                    for mask_id, train_name in precise_mapping.items():
                        m[mask == mask_id] = train_name
                    if index < 3:
                        print(f"  [PRECISE_GT] 直接使用精确类别映射")
                else:
                    # 如果没有精确映射，使用原始映射
                    for mask_id, train_name in self.class_id_map.items():
                        m[mask == mask_id] = train_name
                    if index < 3:
                        print(f"  [PRECISE_GT] 使用原始类别映射")
            else:
                # 原始mask：使用原有映射逻辑
                for mask_id, train_name in self.class_id_map.items():
                    m[mask == mask_id] = train_name
                if index < 3:
                    print(f"  [ORIGINAL] 使用原始类别映射")
            
            # 最近邻缩放，保持离散标签不被污染
            m           = cv2.resize(m, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask_tensor = torch.from_numpy(m).long()        # [H, W] for cross entropy loss
            img_tensor  = torch.from_numpy(img).float()     # [3, H, W] for input to model

            # sanity check - 只显示前5个样本的详细信息
            if index < 5:
                unique = np.unique(m[m != self.ignore_index])
                # 检测mask类型
                if os.path.basename(mask_path).endswith('_endo_watershed_mask.png'):
                    mask_type = "WATERSHED"
                elif os.path.basename(mask_path).endswith('_precise_gt.png'):
                    mask_type = "PRECISE_GT"  
                else:
                    mask_type = "ORIGINAL"
                print(f"[MC] Sample {index} ({mask_type}): classes={unique.tolist()} (ignore {self.ignore_index})")
            elif index == 5:
                print(f"[MC] 数据加载正常，后续样本将静默处理...")
            return img_tensor, mask_tensor
        else:
            # binary
            fg = ((mask == 12) | (mask == 13) | (mask == 21) | (mask == 22)).astype(np.uint8)  # 胆囊+器械
            fg = cv2.resize(fg, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

            mask_tensor = torch.from_numpy(fg)[None, ...].float()
            img_tensor  = torch.from_numpy(img)

            if index < 5:
                foreground_ratio = float(fg.mean())
                print(f"[BIN] Sample {index}: Foreground ratio={foreground_ratio:.3f}")
            elif index == 5:
                print(f"[BIN] 二分类数据加载正常，后续样本将静默处理...")
            return img_tensor, mask_tensor

    def _get_multiclass_mask_candidates(self, stem):
        """多分类任务的mask候选列表（watershed优先）"""
        return [
            stem.replace("_endo", "_endo_watershed_mask") + ".png", # 最优：watershed区域分割
            stem.replace("_endo", "_precise_gt") + ".png",          # 次选：精确GT（如果有）
            stem.replace("_endo", "_endo_mask") + ".png",           # 备选：原始mask
        ]

    def _get_binary_mask_candidates(self, stem):
        """二分类任务的mask候选列表（可用近似）"""
        return [
            stem.replace("_endo", "_endo_mask") + ".png",           # 精确
            stem.replace("_endo", "_endo_watershed_mask") + ".png", # 近似
            stem.replace("_endo_color", "_endo_color_mask") + ".png", # 粗糙但可用
            stem.replace("_endo", "_endo_color_mask") + ".png",
        ]

    def analyze_mask_distribution(self, num_samples=50):
            """分析mask分布情况"""
            print("🔍 分析前景分布（看二分类合并后的前景占比）...")
            # print("Analyzing foreground distribution...")
            
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
            print(f"平均前景比例: {avg_ratio:.3f}")
            print(f"前景比例范围: {np.min(foreground_ratios):.3f} - {np.max(foreground_ratios):.3f}")
            
            return foreground_ratios