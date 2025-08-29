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
    
    # 类别ID映射
    "SEG8K_CLASS_MAPPING": {
        0: 0,   # background
        12: 1,  # instrument_1  
        13: 1,  # instrument_2
        21: 2,  # target_organ_1
        22: 2   # target_organ_2
    },
    
    # 精确GT类别映射（已经是正确的类别ID）
    "PRECISE_GT_CLASS_MAPPING": {
        0: 0,   # background -> background
        1: 1,   # instrument -> instrument
        2: 2    # target_organ -> target_organ
    }
}
