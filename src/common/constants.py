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

# Color palette
PALETTE = {
    0: (120, 120, 120),  # background: 深灰
    1: (  0, 180, 180),  # instrument: 青色
    2: (220,  70, 200),  # target_organ: 玫红
    3: (255, 140,   0),  # liver: 橙
    4: (230, 220,  90),  # fat: 黄
    5: (  0,   0, 255),  # gallbladder: 蓝
}
