"""
最小分割数据集类
支持标准的 images/ 和 masks/ 目录结构
"""
import scipy.ndimage as ndi
import os, glob, cv2, numpy as np, torch
from torch.utils.data import Dataset
from src.common.constants import DATASET_CONFIG
from src.common.constants import generate_class_mapping, CLASSIFICATION_SCHEMES, WATERSHED_TO_BASE_CLASS

# helper to get valid endoscope ellipse region
# def _ellipse_inside_mask(h, w, margin=2):
#     """返回一个布尔mask：True 表示在椭圆视野内，False 表示黑边区域。
#     margin: 轻微收缩，避免边缘锯齿的误判。
#     """
#     yy, xx = np.ogrid[:h, :w]
#     cy, cx = h / 2.0, w / 2.0
#     # 视野大致是内接椭圆（假设4:3/1:1都能容忍）
#     ry, rx = (h / 2.0 - margin), (w / 2.0 - margin)
#     norm = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
#     return norm <= 1.0

def compute_fov_mask_from_rgb(rgb_uint8: np.ndarray, thr=None) -> np.ndarray:
    """
    从已resize到目标尺寸的 RGB 图，鲁棒地估计内镜视野(FOV)。
    返回 bool 数组：True=视野内, False=视野外黑环。
    """
    # 灰度 + 自适应阈值
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)

    # 经验：黑环通常非常接近 0。用分位数自适应一个阈值，
    # 如果图里几乎没有黑环，会得到很低的比例，后面有保护。
    if thr is None:
        p1 = np.percentile(gray, 1)
        thr = max(5, min(20, p1 + 2))  # 5~20 之间的自适应阈值

    # 先把“非黑环”找出来
    in_fov = gray > thr

    # 形态学清理 & 取最大连通域（避免角落/器械阴影误判）
    in_fov = ndi.binary_closing(in_fov, structure=np.ones((5, 5), bool))
    labeled, num = ndi.label(in_fov)
    if num > 0:
        sizes = ndi.sum(np.ones_like(gray), labeled, index=np.arange(1, num + 1))
        keep = (labeled == (1 + np.argmax(sizes)))
        in_fov = keep
    else:
        in_fov = np.ones_like(gray, dtype=bool)

    # 保护：如果“外圈像素比例”< 5%，直接认为全视野，避免过度屏蔽
    border_ratio = 1.0 - in_fov.mean()
    if border_ratio < 0.05:
        in_fov[:] = True

    return in_fov

# 
class SegDatasetMin(Dataset):
    def __init__(self, data_root: str, dtype: str = "train", img_size: int = 512,
            return_multiclass: bool    = False,
            class_id_map: dict | None  = None,
            ignore_index: int          = 255,
            classification_scheme: str = None, # abonded
            custom_mapping: dict       = None, # abonded
            target_classes: list       = None, # abonded
            apply_fov_mask: bool       = False,
        ):

        self.data_root = os.path.abspath(data_root)
        self.dtype = dtype if dtype is not None else ""
        self.img_size = int(img_size)

        self.return_multiclass = bool(return_multiclass)
        self.ignore_index = int(ignore_index)
        # self.target_classes = target_classes  # 添加缺失的属性
        self.apply_fov_mask = bool(apply_fov_mask)

        # 优先使用新的class_id_map，fallback到旧逻辑
        if class_id_map is not None:
             # 1) 直接用传进来的“最终映射”（已经是 WS→TRAIN 的 0/1/2/255）
            self.class_id_map = class_id_map
            self.ws2train = class_id_map  # ✅ 关键：不要再调用 compose_mapping 二次计算

            # 2) 正确计算 num_classes（忽略 255）
            valid_vals = {v for v in class_id_map.values() if v != self.ignore_index}
            self.num_classes = len(valid_vals)
            # self.num_classes = len(set(class_id_map.values()) - {self.ignore_index})
            # self.classification_scheme = "direct_mapping"

            # 3) 标个名字避免混淆（别再叫 direct_mapping）
            self.classification_scheme = "custom_direct"

            # 4) 生成可读的类名
            self.class_names = [f"class_{i}" for i in sorted(valid_vals)]
            print(f"[DATASET] Using direct class_id_map with {self.num_classes} classes")

            # from src.common.constants import compose_mapping
            # self.ws2train = compose_mapping(
            #     classification_scheme=self.classification_scheme  # 例如 "3class_org"
            # )

            # 安全网：检查映射值是不是 0..K-1 或 255
            allowed = set(range(self.num_classes)) | {self.ignore_index}
            vals = set(self.ws2train.values())
            if not vals.issubset(allowed):
                raise ValueError(
                    f"[Mapping Error] ws2train values {sorted(vals)} "
                    f"not subset of {sorted(allowed)} — did you pass WS→BASE instead of WS→TRAIN?"
                )

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

        # FOV
        img_rgb = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # 这里保存 uint8 的副本用于 FOV
        rgb_uint8_resized = img_rgb.copy()

        #  read images
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) # from BGR to RGB
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
        img = cv2.resize(img,
                         (self.img_size, self.img_size),
                         interpolation = cv2.INTER_LINEAR) # resize to target size (same size for both img and mask)
        
        # 归一化到 0~1
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


        # === 多类分支：标准两段式映射（唯一来源） ===
        if not self.is_binary:
            # 读取到的 mask 是 WS 灰度图：11/12/13/21/22/31/32/50/255/...
            ws = mask

            # 先全部置为 ignore，再按映射覆写（O(N) 向量化）
            m = np.full_like(ws, fill_value=self.ignore_index, dtype=np.uint8)

            # 逐项写入（最稳妥，且易于审计）
            for ws_val, train_id in self.ws2train.items():
                m[ws == ws_val] = train_id

            # （可选）安全网：如果还存在“未在映射表中但又不是 255 的灰度”，统统设为 ignore
            unknown = (m == self.ignore_index) & (ws != 255)
            if np.any(unknown):
                # 这里保持 ignore，不要自动归 0，避免把未知类当背景污染监督
                pass

            # 统计/调试：仅首批样本打印一次
            if index < 3:
                uniques, counts = np.unique(m, return_counts=True)
                print(f"[MAP DEBUG] train ids in sample#{index}: {dict(zip(uniques.tolist(), counts.tolist()))}")

            # 最近邻缩放，保持离散标签
            m = cv2.resize(m, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

            if self.apply_fov_mask:
                fov = compute_fov_mask_from_rgb(rgb_uint8_resized)  # True=视野内
                # 选择一种：训练推荐置背景0；评测/可视化若想忽略可用 255
                m[~fov] = 0   # 或者 m[~fov] = self.ignore_index
                
            mask_tensor = torch.from_numpy(m).long()
            img_tensor  = torch.from_numpy(img).float()
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
    