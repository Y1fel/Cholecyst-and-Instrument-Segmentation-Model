# src/common/constants.py

# class ID
CLASSES = {
    0: "background",
    1: "instrument",
    2: "target_organ",
    3: "liver",
    4: "fat",
    5: "gallbladder"
}
IGNORE_INDEX = 255

# Color palette - 极高对比度颜色方案，确保在任何医学图像上都清晰可见
PALETTE = {
    0: (0, 0, 0),        # background: 纯黑
    1: (0, 255, 0),      # instrument: 纯绿色
    2: (255, 0, 255),    # target_organ: 纯紫色
    3: (255, 255, 0),    # liver: 纯黄色
    4: (0, 255, 255),    # fat: 纯青色
    5: (255, 0, 0),      # gallbladder: 纯红色
    6: (255, 165, 0),    # 区域6：橙色
    7: (128, 0, 255),    # 区域7：紫蓝
    8: (255, 255, 255),  # 区域8：纯白
    9: (255, 192, 203),  # 区域9：粉色
    10: (0, 128, 255),   # 区域10：蓝色
    11: (255, 128, 0),   # 区域11：深橙
    12: (128, 255, 0),   # 区域12：黄绿
    13: (255, 0, 128),   # 区域13：粉红
    14: (0, 255, 128),   # 区域14：青绿
    15: (128, 0, 128),   # 区域15：紫色
    16: (255, 255, 128), # 区域16：浅黄
    17: (128, 255, 255), # 区域17：浅青
}

# 用于Ground Truth显示的高对比度颜色
GT_PALETTE = {
    0: (50, 50, 50),     # background: 深灰
    1: (50, 255, 50),    # instrument: 亮绿色
    2: (255, 50, 255),   # target_organ: 亮紫色  
    3: (255, 255, 50),   # liver: 亮黄色
    4: (50, 255, 255),   # fat: 亮青色
    5: (255, 50, 50),    # gallbladder: 亮红色
    6: (255, 165, 50),   # 区域6：亮橙
    7: (128, 50, 255),   # 区域7：亮紫蓝
    8: (255, 255, 200),  # 区域8：浅白
    9: (255, 150, 203),  # 区域9：亮粉
    10: (50, 128, 255),  # 区域10：亮蓝
    11: (255, 128, 50),  # 区域11：亮深橙
    12: (128, 255, 50),  # 区域12：亮黄绿
    13: (255, 50, 128),  # 区域13：亮粉红
    14: (50, 255, 128),  # 区域14：亮青绿
    15: (128, 50, 128),  # 区域15：亮紫
    16: (255, 255, 100), # 区域16：亮浅黄
    17: (100, 255, 255), # 区域17：亮浅青
}

# === New: palette selector per scheme ===
def get_palette_for_scheme(scheme: str):
    """
    Return a palette dict {train_class_id: (R,G,B)} that matches the
    class index order of the chosen scheme.
    """
    scheme = (scheme or "").lower()

    if scheme == "3class_org":
        # [background(0), instrument(1), target_organ(2)]
        return {
            0: (50, 50, 50),     # background (dark gray)
            1: (50, 255, 255),   # instrument (CYAN) -> 与可视化一致
            2: (255, 50, 255),   # target organ (MAGENTA)
        }

    if scheme == "6class":
        # [background(0), liver(1), fat(2), gi_tract(3), instrument(4), gallbladder(5)]
        return {
            0: (50, 50, 50),     # background
            1: (50, 255, 50),    # liver (GREEN)
            2: (255, 50, 255),   # fat (MAGENTA)
            3: (255, 255, 50),   # GI tract (YELLOW)
            4: (50, 255, 255),   # instrument (CYAN)  <- 关键：器械用青色
            5: (255, 50, 50),    # gallbladder (RED)
        }

    if scheme == "detailed":
        # 13 类直接复用 GT_PALETTE 的编号
        return GT_PALETTE

    # fallback
    return GT_PALETTE


# 专用于区域分离的超高对比度颜色
REGION_SEPARATION_COLORS = {
    0: (0, 0, 0),           # 背景：纯黑
    1: (0, 255, 0),         # 区域1：纯绿
    2: (255, 0, 255),       # 区域2：纯紫
    3: (255, 255, 0),       # 区域3：纯黄
    4: (0, 255, 255),       # 区域4：纯青
    5: (255, 0, 0),         # 区域5：纯红
    6: (255, 165, 0),       # 区域6：橙色
    7: (128, 0, 255),       # 区域7：紫蓝
    8: (255, 255, 255),     # 区域8：纯白
    9: (255, 192, 203),     # 区域9：粉色
}

# Watershed灰度值 → 基础语义ID (0..12) 映射表
# 这是解决标签对齐问题的关键映射
# WATERSHED_TO_BASE_CLASS = {
#     # 忽略/背景
#     255: 255,   # 明确忽略
#     0:   0,     # 内圈黑边或背景（若你希望不计入训练，可改成 255）

#     # 典型“背景”灰度
#     11: 0, 12: 0, 13: 0, 50: 0,

#     # 目标器官（本阶段合并为“胆囊”代表类 10；到 6/12 类时再细分）
#     21: 10,
#     22: 10,

#     # 器械两子类（统一会在 3 类方案里映到 instrument=1）
#     31: 5,      # grasper
#     32: 9,      # L-hook

#     # 其余较少出现的灰度，先并入“背景/解剖组织”以避免误导 3 类训练
#     23: 0,
#     24: 0,
#     25: 0,
# }


# Watershed灰度值 → 基础语义ID (0..12)
# 同时兼容：
#   - 官方灰度（5,17,18,19,33,34,35,36,37,49,50,51,80）
#   - 两位数别名（11,12,13,21,22,23,24,25,31,32,50）
WATERSHED_TO_BASE_CLASS = {
    # ignore / 黑边
    255: 255,
    0:   0,

    # —— 官方灰度 → 基础类ID ——
    80:  0,  # #505050 background
    17:  1,  # #111111 abdominal_wall
    33:  2,  # #212121 liver
    19:  3,  # #131313 gastrointestinal_tract
    18:  4,  # #121212 fat
    49:  5,  # #313131 grasper
    35:  6,  # #232323 connective_tissue
    36:  7,  # #242424 blood
    37:  8,  # #252525 cystic_duct
    50:  9,  # #323232 l_hook_electrocautery
    34: 10,  # #222222 gallbladder
    51: 11,  # #333333 hepatic_vein
    5:  12,  # #050505 liver_ligament

    # —— 两位数别名 → 基础类ID（你的数据里实际出现的）——
    11: 1,   # 代表 #111111 -> abdominal_wall
    12: 4,   # #121212 -> fat
    13: 3,   # #131313 -> gastrointestinal_tract
    21: 2,   # #212121 -> liver
    22: 10,  # #222222 -> gallbladder  ← 胆囊（关键！）
    23: 6,   # #232323 -> connective_tissue
    24: 7,   # #242424 -> blood
    25: 8,   # #252525 -> cystic_duct
    31: 5,   # #313131 -> grasper
    32: 9,   # #323232 -> l_hook_electrocautery
    # （50 已在官方灰度段映射为 9，无需重复）
}


# WATERSHED_TO_BASE = {   # 依据 seg8k 版式，按你的统计完善
#   11: 'background', 12: 'instrument', 13: 'instrument',
#   21: 'target',     22: 'target',      31: 'liver',
#   50: 'abdominal_wall', 255: 'ignore'
# }
# BASE_TO_TRAIN_3C = {      # 3类任务映射
#   'background': 0, 'liver': 0, 'abdominal_wall': 0,
#   'instrument': 1, 'target': 2, 'ignore': 255
# }

# 基础13类定义（对应Kaggle数据集）
BASE_CLASSES = {
    0: "background",
    1: "abdominal_wall", 
    2: "liver",
    3: "gastrointestinal_tract",
    4: "fat",
    5: "grasper",
    6: "connective_tissue", 
    7: "blood",
    8: "cystic_duct",
    9: "l_hook_electrocautery",
    10: "gallbladder",  # 当前目标器官
    11: "hepatic_vein",
    12: "liver_ligament",
    # 预留扩展槽位
    13: "target_organ_2",  # 为其他器官预留
    14: "target_organ_3",
    15: "instrument_3",    # 为其他器械预留
    16: "instrument_4"
}

# 分类方案定义
CLASSIFICATION_SCHEMES = {
    "binary": {
        "num_classes": 2,
        "target_classes": ["background", "foreground"],
        "mapping": {0: 0, 5: 1, 9: 1, 10: 1},  # 器械+目标器官=前景
        "default_for_others": 0,
        "description": "前景(器械+目标器官) vs 背景"
    },
    
    "3class_org": {
        "num_classes": 3,
        "target_classes": ["background", "instrument", "target_organ"],
        "mapping": {
            0: 0,   # 背景 → 背景
            5: 1,   # Grasper → 器械
            9: 1,   # L-hook → 器械  
            10: 2   # Gallbladder → 目标器官
        },
        "default_for_others": 0,  # 关键：未命中的都设为ignore，不是背景！
        "description": "背景 vs 器械 vs 目标器官（语义对齐版本）"
    },
    
    "3class_balanced": {
        "num_classes": 3,
        "target_classes": ["background", "surgical_instrument", "anatomical_structures"], 
        "mapping": {
            0: 0,   # background -> background
            1: 2,   # abdominal_wall -> anatomical_structures
            2: 2,   # liver -> anatomical_structures
            3: 2,   # gastrointestinal_tract -> anatomical_structures
            4: 2,   # fat -> anatomical_structures
            5: 1,   # grasper -> surgical_instrument
            6: 2,   # connective_tissue -> anatomical_structures
            7: 2,   # blood -> anatomical_structures
            8: 2,   # cystic_duct -> anatomical_structures
            9: 1,   # l_hook_electrocautery -> surgical_instrument
            10: 2,  # gallbladder -> anatomical_structures
            11: 2,  # hepatic_vein -> anatomical_structures
            12: 2   # liver_ligament -> anatomical_structures
        },
        "default_for_others": 0,
        "description": "背景 vs 器械 vs 解剖结构（平衡版本）"
    },
    
    "5class": {
        "num_classes": 5,
        "target_classes": ["background", "tissue", "liver", "surgical_instrument", "target_organ"],
        "mapping": {0: 0, 1: 1, 2: 2, 3: 1, 4: 1, 5: 3, 6: 1, 7: 1, 8: 2, 9: 3, 10: 4, 11: 2, 12: 2},
        "default_for_others": 0,
        "description": "背景、组织、肝脏、器械、目标器官"
    },

    "6class": {
        "num_classes": 6,
        "target_classes": ["background", "liver", "fat", "gi_tract", "instrument", "gallbladder"],
        "mapping": {
            0: 0,   # 背景
            2: 1,   # 肝脏
            4: 2,   # 脂肪
            3: 3,   # 消化道
            5: 4,   # Grasper → 器械
            9: 4,   # L-hook → 器械（合并到同一个器械类）
            10: 5   # 胆囊
        },
        "default_for_others": 255,
        "description": "6类精细分割：背景-肝脏-脂肪-消化道-器械-胆囊"
    },
    
    "detailed": {
        "num_classes": 13,
        "target_classes": list(BASE_CLASSES.values())[:13],
        "mapping": {i: i for i in range(13)},
        "default_for_others": 255,  # ignore_index
        "description": "完整13类分割（Kaggle标准）"
    }
}

# 数据集配置
DATASET_CONFIG = {
    # seg8k数据集的图像模式
    "SEG8K_IMAGE_PATTERNS": [
        "**/*_endo.png", 
        "**/*_endo.jpg", 
        "**/*_endo.jpeg",
        "**/*_endo_color.png", 
        "**/*_endo_color.jpg"
    ],
    
    # mask候选模式
    "BINARY_MASK_PATTERNS": [
        "_endo_mask.png",
        "_endo_watershed_mask.png", 
        "_endo_color_mask.png",
        "_endo_color_mask.png"  # 对应_endo结尾的图片
    ],
    
    "MULTICLASS_MASK_PATTERNS": [
        "_endo_mask.png",
        "_endo_watershed_mask.png"
    ],
    
    "SEG8K_CLASS_MAPPING": CLASSIFICATION_SCHEMES["3class_org"]["mapping"],  # use 3-class as default
    
    # 精确GT类别映射（已经是正确的类别ID）
    "PRECISE_GT_CLASS_MAPPING": {
        0: 0,   # background -> background
        1: 1,   # instrument -> instrument
        2: 2    # target_organ -> target_organ
    }
}

def generate_class_mapping(
    scheme_name = "3class",
    custom_mapping = None,
    target_classes = None
):
    """
    动态生成类别映射
    Args:
        scheme_name: 预设方案名
        custom_mapping: 自定义映射字典
        target_classes: 指定类别列表
    Returns:
        mapping: 类别映射字典
        num_classes: 类别数量
        class_names: 类别名称列表
    """
    if custom_mapping:
        # use custom mapping
        mapping = custom_mapping
        num_classes = len(set(mapping.values()))
        class_names = [f"class_{i}" for i in range(num_classes)]
        return mapping, num_classes, class_names
    
    if target_classes:
        # use target classes
        mapping = generate_class_mapping(target_classes=target_classes)
        num_classes = len(target_classes)
        class_names = target_classes
        return mapping, num_classes, class_names
    
    if scheme_name not in CLASSIFICATION_SCHEMES:
        raise ValueError(f"Unknown scheme: {scheme_name}. Available: {list(CLASSIFICATION_SCHEMES.keys())}")
    # Use the preset scheme
    scheme = CLASSIFICATION_SCHEMES[scheme_name]
    return scheme["mapping"], scheme["num_classes"], scheme["target_classes"]

def compose_mapping(classification_scheme=None, custom_mapping=None, target_classes=None):
    """
    组合映射函数：生成从watershed灰度值到目标类别ID的完整映射
    
    Args:
        classification_scheme: 预定义的分类方案名称
        custom_mapping: 自定义映射字典  
        target_classes: 目标类别列表
    
    Returns:
        dict: {watershed_gray_value: target_class_id}
    """
    final_mapping = {}
    
    if custom_mapping is not None:
        # 使用自定义映射
        return custom_mapping
    
    if classification_scheme and classification_scheme in CLASSIFICATION_SCHEMES:
        scheme = CLASSIFICATION_SCHEMES[classification_scheme]
        semantic_to_target = scheme['mapping']
        default_value = scheme.get('default_for_others', 255)
        
        # 两阶段映射：watershed → semantic → target
        for watershed_gray, semantic_id in WATERSHED_TO_BASE_CLASS.items():
            if semantic_id in semantic_to_target:
                final_mapping[watershed_gray] = semantic_to_target[semantic_id]
            else:
                final_mapping[watershed_gray] = default_value
        
        return final_mapping
    
    # 默认情况：直接使用watershed到语义的映射
    return WATERSHED_TO_BASE_CLASS

def generate_selective_mapping(target_classes):
    """根据指定类别生成映射"""
    # 预留接口，暂不实现
    pass

# 验证接口（预留）
def validate_classification_config(scheme_name, mapping, dataset_sample=None):
    """验证分类配置的有效性"""
    # 预留验证接口
    pass

def validate_class_distribution(dataset, mapping):
    """验证类别分布"""
    # 预留验证接口  
    pass

