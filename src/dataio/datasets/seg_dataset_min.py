"""
最小分割数据集类
支持标准的 images/ 和 masks/ 目录结构
"""

import os, glob, cv2, numpy as np, torch
from torch.utils.data import Dataset
from src.common.constants import DATASET_CONFIG
from src.common.constants import generate_class_mapping, CLASSIFICATION_SCHEMES, WATERSHED_TO_BASE_CLASS

# helper to get valid endoscope ellipse region
def _ellipse_inside_mask(h, w, margin=2):
    """返回一个布尔mask：True 表示在椭圆视野内，False 表示黑边区域。
    margin: 轻微收缩，避免边缘锯齿的误判。
    """
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    # 视野大致是内接椭圆（假设4:3/1:1都能容忍）
    ry, rx = (h / 2.0 - margin), (w / 2.0 - margin)
    norm = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    return norm <= 1.0

# 
class SegDatasetMin(Dataset):
    def __init__(self, data_root: str, dtype: str = "train", img_size: int = 512,
            return_multiclass: bool    = False,
            class_id_map: dict | None  = None,
            ignore_index: int          = 255,
            classification_scheme: str = None, # abonded
            custom_mapping: dict       = None, # abonded
            target_classes: list       = None, # abonded
        ):

        self.data_root = os.path.abspath(data_root)
        self.dtype = dtype if dtype is not None else ""
        self.img_size = int(img_size)

        self.return_multiclass = bool(return_multiclass)
        self.ignore_index = int(ignore_index)
        # self.target_classes = target_classes  # 添加缺失的属性

        # 优先使用新的class_id_map，fallback到旧逻辑
        if class_id_map is not None:
            # 新方式：直接使用传入的完整映射
            self.class_id_map = class_id_map
            self.num_classes = len(set(class_id_map.values()) - {self.ignore_index})
            self.classification_scheme = "direct_mapping"
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]  # 生成默认类别名
            print(f"[DATASET] Using direct class_id_map with {self.num_classes} classes")
        else:
            # 旧方式：兼容旧的分类方案（待废弃）
            if classification_scheme is None:
                classification_scheme = "3class" if self.return_multiclass else "binary"
            
            self.mapping, self.num_classes, self.class_names = generate_class_mapping(
                scheme_name=classification_scheme, 
                custom_mapping=custom_mapping, 
                target_classes=target_classes
            )
            self.class_id_map = self.mapping
            self.classification_scheme = classification_scheme
            print(f"[DATASET] Using legacy mapping scheme: {classification_scheme}")

        # 设置二分类标志
        self.is_binary = (self.classification_scheme == "binary")

        # get dataset specific for seg8k (for current baseline, and offline in later process)
        pattern_img = DATASET_CONFIG["SEG8K_IMAGE_PATTERNS"]
        cand_imgs = []

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
        
        print(f"-- Dataset loaded: {len(self.pairs)} image-mask pairs")
        print(f"-- Classification scheme: {self.classification_scheme} ({self.num_classes} classes)")
        print(f"-- Target classes: {self.class_names}")
        if not self.is_binary:
            print(f"-- Multi-class mode: Watershed regions → training class mapping")
            print(f"   Semantic mapping to {self.num_classes} classes")
        else:
            print(f"-- Binary mode: target + instrument vs background")
            print(f"    {self.class_names[1]} vs {self.class_names[0]}")

        dominant_type = max(mask_types, key=mask_types.get) if mask_types else 'unknown'
        print(f"-- Dominant mask type: {dominant_type} ({mask_types.get(dominant_type, 0)}/{min(20, len(self.pairs))} samples)")
        print(f"-- Image size: {self.img_size}x{self.img_size}")
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

        # # 检查是否为watershed mask
        # is_watershed = os.path.basename(mask_path).endswith('_endo_watershed_mask.png')
        
        # # 新的简化watershed处理逻辑
        # if is_watershed:
        #     # 导入必要的映射
        #     from src.common.constants import WATERSHED_TO_BASE, BASE_TO_TRAIN_3C
            
        #     # 创建label_map（字符串标签到数字ID的映射）
        #     label_map = {
        #         'background': 0, 'instrument': 1, 'target': 2, 'liver': 3, 
        #         'abdominal_wall': 4, 'ignore': 255
        #     }
            
        #     # 1) 灰度 -> 基础语义字符串
        #     base_sem = np.full_like(mask, fill_value=255, dtype=np.uint8)
        #     for g, label in WATERSHED_TO_BASE.items():
        #         base_sem[mask == g] = label_map[label]  # 例如临时映射到小整数/或先存字符串数组

        #     # 2) 基础语义 -> 训练类ID (3类/6类)
        #     train_mask = np.full_like(mask, fill_value=255, dtype=np.uint8)
        #     for label, cid in BASE_TO_TRAIN_3C.items():
        #         if cid == 255: continue
        #         train_mask[(base_sem == label_map[label])] = cid

        #     # 仅真正无效区域保持255；椭圆"内圈背景"强制为0
        #     # 你的数据是黑色圆外背景：把圆外=255，圆内非语义=0
        #     if hasattr(self, 'force_ellipse_background_zero') and self.force_ellipse_background_zero:
        #         # rr, cc = get_valid_ellipse(mask.shape)  # 你已有/可快速实现
        #         # train_mask[rr, cc] = np.where(train_mask[rr, cc]==255, 0, train_mask[rr, cc])
        #         pass  # 暂时注释掉，等待get_valid_ellipse函数实现

        #     mask = train_mask
            
        #     # 直接处理并返回，跳过后续复杂逻辑
        #     mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        #     mask_tensor = torch.from_numpy(mask).long()        # [H, W] for cross entropy loss
        #     img_tensor  = torch.from_numpy(img).float()       # [3, H, W] for input to model
        #     return img_tensor, mask_tensor

        # --- replace: unified two-stage mapping ---
        H, W = mask.shape[:2]
        inside = _ellipse_inside_mask(H, W)

        # 1) 灰度 → 基础语义 base_id
        #    对于 watershed：用 constants 里的 WATERSHED_TO_BASE_CLASS
        #    对于非-watershed：直接把原 mask 当 base_id（兼容已有精确GT或原始mask）
        from src.common.constants import WATERSHED_TO_BASE_CLASS, CLASSIFICATION_SCHEMES

        filename = os.path.basename(mask_path)
        is_watershed = filename.endswith("_endo_watershed_mask.png")

        if is_watershed:
            base = np.full_like(mask, 255, dtype=np.uint8)
            for g, base_id in WATERSHED_TO_BASE_CLASS.items():
                base[mask == g] = base_id
        else:
            # 已是语义 id（精确GT/原始mask）
            base = mask.copy()

        # 2) 基础语义 base_id → 训练类别 train_id
        scheme = self.classification_scheme
        if scheme == "direct_mapping":
            # 兼容：如果外部直接传了 class_id_map，就用 self.class_id_map
            base2train = self.class_id_map
        else:
            # 用预设方案（3/6/12 类）
            base2train = CLASSIFICATION_SCHEMES[scheme]["mapping"]

        train = np.full_like(base, self.ignore_index, dtype=np.uint8)

        # 椭圆外一律 ignore（255）
        train[~inside] = self.ignore_index

        # 椭圆内：已映射到具体类别的像素按映射赋值；未知 base_id → 设为背景 0
        for b, t in base2train.items():
            train[(inside) & (base == b)] = t

        # 把仍未命中的“椭圆内像素”安全落地到背景 0（避免全 255 的情况）
        train[(inside) & (train == self.ignore_index)] = 0

        # 最后缩放 & 转 tensor
        train = cv2.resize(train, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(train).long()
        img_tensor  = torch.from_numpy(img).float()
        return img_tensor, mask_tensor


        # if not self.is_binary:
        #     # 多类：实现两段式映射，根据 class_id_map 做 ID 映射，未知值 -> ignore_index
        #     m = np.full_like(mask, fill_value=self.ignore_index, dtype=np.uint8)

        #     # 查是否为Watershed mask，进行灰度值→基础语义ID映射
        #     if os.path.basename(mask_path).endswith('_endo_watershed_mask.png'):
        #         # 第一段映射：Watershed灰度值 → 基础语义ID
        #         from src.common.constants import WATERSHED_TO_BASE_CLASS
                
        #         # 创建基础语义mask
        #         base_semantic_mask = np.full_like(mask, 255, dtype=np.uint8)
                
        #         # 应用Watershed映射
        #         for watershed_gray, base_id in WATERSHED_TO_BASE_CLASS.items():
        #             base_semantic_mask[mask == watershed_gray] = base_id
                
        #         # 调试输出（前50个batch）
        #         # if index < 5:
        #         #     watershed_unique = np.unique(mask)
        #         #     base_unique = np.unique(base_semantic_mask[base_semantic_mask != 255])
        #         #     print(f"[WATERSHED {index}] 灰度值 {watershed_unique} → 基础语义ID {base_unique}")
                
        #         # 使用基础语义mask作为后续映射的输入
        #         mask_for_mapping = base_semantic_mask
        #     else:
        #         # 非Watershed mask直接使用原始mask
        #         mask_for_mapping = mask

        #     # 步骤2：基础语义ID → 训练类ID（第二段映射）
        #     # 对于Watershed数据，需要从基础语义ID映射到目标类别
        #     if os.path.basename(mask_path).endswith('_endo_watershed_mask.png'):
        #         # Watershed数据：使用基础语义ID进行映射
        #         from src.common.constants import CLASSIFICATION_SCHEMES
        #         if self.classification_scheme == "direct_mapping":
        #             # 使用3class_org的语义映射
        #             semantic_mapping = CLASSIFICATION_SCHEMES["3class_org"]["mapping"]
        #             default_value = CLASSIFICATION_SCHEMES["3class_org"]["default_for_others"]
                    
        #             for semantic_id, target_class in semantic_mapping.items():
        #                 m[mask_for_mapping == semantic_id] = target_class
                    
        #             # 处理未映射的语义ID
        #             unique_semantic_ids = np.unique(mask_for_mapping)
        #             for semantic_id in unique_semantic_ids:
        #                 if semantic_id != 255 and semantic_id not in semantic_mapping:
        #                     m[mask_for_mapping == semantic_id] = default_value
        #                     # if index < 3:
        #                     #     print(f"[MAPPING {index}] 语义ID {semantic_id} → ignore_index({default_value})")
        #                 # elif semantic_id in semantic_mapping:
        #                 #     if index < 3:
        #                 #         print(f"[MAPPING {index}] 语义ID {semantic_id} → 目标类别{semantic_mapping[semantic_id]}")
        #         else:
        #             # 旧的class_id_map方式（应该废弃）
        #             for base_id, train_class in self.class_id_map.items():
        #                 m[mask_for_mapping == base_id] = train_class
        #     else:
        #         # 非Watershed数据：直接使用class_id_map
        #         for base_id, train_class in self.class_id_map.items():
        #             m[mask_for_mapping == base_id] = train_class

        #     # 保持ignore_index不变
        #     m[mask == 255] = self.ignore_index

        #     # 调试输出：类别分布（前5个batch）
        #     # if index < 5:
        #     #     unique_values = np.unique(m)
        #     #     print(f"[MAPPING {index}] 最终训练类别: {unique_values}")
        #     #     
        #     #     # 计算各类像素占比（排除255）
        #     #     valid_mask = m[m != 255]
        #     #     if len(valid_mask) > 0:
        #     #         unique, counts = np.unique(valid_mask, return_counts=True)
        #     #         percentages = counts / len(valid_mask) * 100
        #     #         class_dist = {int(cls): f"{pct:.1f}%" for cls, pct in zip(unique, percentages)}
        #     #         print(f"[DISTRIBUTION {index}] 类别分布: {class_dist}")
        #     #     else:
        #     #         print(f"[DISTRIBUTION {index}] 警告：没有有效像素！")
            
        #     # 最近邻缩放，保持离散标签不被污染
        #     m           = cv2.resize(m, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        #     mask_tensor = torch.from_numpy(m).long()        # [H, W] for cross entropy loss
        #     img_tensor  = torch.from_numpy(img).float()     # [3, H, W] for input to model

        #     return img_tensor, mask_tensor
        # else:
        #     # binary
        #     # 器械：Class 5 (Grasper) + Class 9 (L-hook) 
        #     # 胆囊：Class 10 (Gallbladder)
        #     if self.classification_scheme == "binary":
        #         # 
        #         binary_mapping = CLASSIFICATION_SCHEMES["binary"]["mapping"]
        #         fg_mask        = np.zeros_like(mask, dtype=np.uint8)
        #         # 
        #         for original_id, target_class in binary_mapping.items():
        #             if target_class == 1:  # foreground
        #                 fg_mask[mask == original_id] = 1
        #         fg = fg_mask
        #     else:
        #         fg = ((mask == 5) | (mask == 9) | (mask == 10)).astype(np.uint8)
                
        #     fg = cv2.resize(fg, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        #     mask_tensor = torch.from_numpy(fg)[None, ...].float()
        #     img_tensor  = torch.from_numpy(img)

        #     return img_tensor, mask_tensor

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
            print("-- Analyzing foreground distribution (foreground ratio after binary merging)...")
            # print("Analyzing foreground distribution...")
            
            foreground_ratios = []
            for i in range(min(num_samples, len(self.pairs))):
                _, mask_path = self.pairs[i]
                
                # 读取原始mask
                raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if raw_mask.ndim == 3:
                    raw_mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY)
                
                # 应用胆囊+器械策略
                processed_mask = ((raw_mask == 5) | (raw_mask == 9) | (raw_mask == 10)).astype(np.uint8)

                foreground_ratio = processed_mask.mean()
                foreground_ratios.append(foreground_ratio)
                
                if i < 5:  # 打印前5个样本的详细信息
                    unique_raw = np.unique(raw_mask)
                    print(f"  Sample {i}: Raw values {unique_raw} -> Foreground ratio {foreground_ratio:.3f}")
            
            avg_ratio = np.mean(foreground_ratios)
            print(f"Average foreground ratio: {avg_ratio:.3f}")
            print(f"Foreground ratio range: {np.min(foreground_ratios):.3f} - {np.max(foreground_ratios):.3f}")

            return foreground_ratios
    