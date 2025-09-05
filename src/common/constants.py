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
WATERSHED_TO_BASE_CLASS = {
    50: 0,   # 背景 (watershed中的灰度值50对应背景)
    11: 1,   # Abdominal Wall  
    21: 2,   # Liver
    13: 3,   # GI Tract
    12: 4,   # Fat
    31: 5,   # Instrument: Grasper
    23: 6,   # Connective Tissue (预留)
    24: 7,   # Blood (预留)
    25: 8,   # Cystic Duct (预留)
    32: 9,   # Instrument: L-hook
    22: 10,  # Gallbladder
    33: 11,  # Hepatic Vein (预留)
    5:  12,  # Liver Ligament (预留)
    255: 255 # ignore index保持不变
}

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
        "default_for_others": 255,  # 关键：未命中的都设为ignore，不是背景！
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
    
    # 类别ID映射 详细版无语义损失
    # "SEG8K_CLASS_MAPPING": {
    #     0: 0,   # Black Background -> background
    #     1: 1,   # Abdominal Wall -> abdominal_wall  
    #     2: 2,   # Liver -> liver
    #     3: 3,   # Gastrointestinal Tract -> gastrointestinal
    #     4: 4,   # Fat -> fat
    #     5: 5,   # Grasper -> instrument (器械1)
    #     6: 6,   # Connective Tissue -> connective_tissue
    #     7: 7,   # Blood -> blood
    #     8: 8,   # Cystic Duct -> cystic_duct
    #     9: 5,   # L-hook Electrocautery -> instrument (器械2，合并到类别5)
    #     10: 10, # Gallbladder -> gallbladder (胆囊)
    #     11: 11, # Hepatic Vein -> hepatic_vein
    #     12: 12, # Liver Ligament -> liver_ligament
    # },
    "SEG8K_CLASS_MAPPING": CLASSIFICATION_SCHEMES["3class_org"]["mapping"],  # use 3-class as default
    # {
    #     0: 0,   # Background
    #     1: 1,   # Tissue (Abdominal Wall)
    #     2: 2,   # Liver  
    #     3: 1,   # Gastrointestinal -> Tissue
    #     4: 1,   # Fat -> Tissue
    #     5: 3,   # Grasper -> Instrument
    #     6: 1,   # Connective Tissue -> Tissue
    #     7: 1,   # Blood -> Tissue  
    #     8: 2,   # Cystic Duct -> Liver
    #     9: 3,   # L-hook Electrocautery -> Instrument
    #     10: 4,  # Gallbladder -> Gallbladder
    #     11: 2,  # Hepatic Vein -> Liver
    #     12: 2,  # Liver Ligament -> Liver
    # },
    
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

