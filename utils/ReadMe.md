# Utils 工具类说明文档

本目录包含项目中的实用工具类，主要用于知识蒸馏和视频帧提取功能。

## 📁 文件结构

```
utils/
├── ReadMe.md                    # 本说明文档
├── class_distillation.py        # 知识蒸馏相关类
└── class_frame_extractor.py     # 视频帧提取工具类
```

## 🔧 类文件详细说明

### 1. class_distillation.py - 知识蒸馏模块

#### 概述
该文件包含用于知识蒸馏的两个核心类：`DistillationLoss` 和 `FeatureExtractor`，用于实现教师模型向学生模型的知识迁移。

#### DistillationLoss 类

**功能**：实现知识蒸馏损失函数，结合任务损失、蒸馏损失和特征蒸馏损失。

**主要参数**：
- `temperature` (float): 蒸馏温度，控制知识迁移的"软度"（默认4.0）
- `alpha` (float): 蒸馏损失权重（默认0.7）
- `beta` (float): 任务损失权重（默认0.3）
- `feature_weight` (float): 特征蒸馏权重（默认0.1）

**损失组成**：
1. **任务损失** (BCE Loss): 学生模型输出与真实标签之间的损失
2. **蒸馏损失** (KL Divergence): 学生模型与教师模型输出之间的损失
3. **特征蒸馏损失** (MSE Loss): 中间特征层之间的损失

**使用示例**：
```python
from utils.class_distillation import DistillationLoss

# 创建损失函数
criterion = DistillationLoss(
    temperature=4.0,
    alpha=0.7,
    beta=0.3,
    feature_weight=0.1
)

# 计算损失
losses = criterion(
    student_outputs=student_pred,
    teacher_outputs=teacher_pred,
    targets=ground_truth,
    student_features=student_features,
    teacher_features=teacher_features
)

print(f"总损失: {losses['total_loss']}")
print(f"任务损失: {losses['task_loss']}")
print(f"蒸馏损失: {losses['distill_loss']}")
print(f"特征损失: {losses['feature_loss']}")
```

#### FeatureExtractor 类

**功能**：使用PyTorch钩子函数提取模型的中间特征，用于特征蒸馏。

**主要方法**：
- `__init__(model, layer_names)`: 初始化特征提取器
- `get_features()`: 获取提取的特征列表
- `clear_features()`: 清除特征缓存
- `remove_hooks()`: 移除钩子函数，防止内存泄漏

**使用示例**：
```python
from utils.class_distillation import FeatureExtractor

# 定义要提取的层名称
layer_names = ['encoder1.1', 'encoder2.1', 'encoder3.1']

# 创建特征提取器
extractor = FeatureExtractor(model, layer_names)

# 前向传播（自动提取特征）
output = model(input_data)

# 获取特征
features = extractor.get_features()

# 清理资源
extractor.remove_hooks()
```

### 2. class_frame_extractor.py - 视频帧提取模块

#### 概述
该文件包含 `VideoFrameExtractor` 类，用于从视频文件中提取帧图像，支持多种提取模式和参数配置。

#### VideoFrameExtractor 类

**功能**：从视频文件中提取帧图像，支持FFmpeg和OpenCV两种提取模式。

**主要特性**：
- 支持多种视频格式（mp4, mov, avi, mkv等）
- 支持FFmpeg和OpenCV两种提取引擎
- 支持时间范围提取（start/end）
- 支持帧率控制（fps）
- 支持图像尺寸调整
- 支持批量处理（OpenCV模式）
- 支持多种输出格式（jpg, png等）

**主要参数**：
- `output_dir` (str): 输出目录路径
- `use_ffmpeg` (bool): 是否优先使用FFmpeg（默认True）

**extract方法参数**：
- `video_path` (str): 视频文件路径
- `fps` (float): 提取帧率（默认2.0）
- `every_n` (int): 每隔N帧提取一帧
- `start` (float): 开始时间（秒）
- `end` (float): 结束时间（秒）
- `size` (tuple): 输出图像尺寸 (width, height)
- `fmt` (str): 输出格式（默认"png"）
- `jpg_quality` (int): JPEG质量（1-100，默认95）
- `batch_size` (int): 批量大小（OpenCV模式，默认30）
- `mode` (int): 提取模式（1=FFmpeg, 2=OpenCV，默认2）

**使用示例**：

```python
from utils.class_frame_extractor import VideoFrameExtractor

# 创建提取器
extractor = VideoFrameExtractor(output_dir="dataset_frames")

# 基本提取
frames = extractor.extract(
    video_path="video.mp4",
    fps=2,              # 每秒提取2帧
    size=(512, 512),    # 输出尺寸
    fmt="jpg",          # 输出格式
    jpg_quality=90      # JPEG质量
)

# 时间范围提取
frames = extractor.extract(
    video_path="video.mp4",
    fps=1,              # 每秒1帧
    start=10,           # 从第10秒开始
    end=60,             # 到第60秒结束
    size=(256, 256),    # 调整尺寸
    mode=2              # 使用OpenCV模式
)

# 批量处理（OpenCV模式）
frames = extractor.extract(
    video_path="video.mp4",
    fps=5,              # 每秒5帧
    batch_size=50,      # 每50帧一个批次
    mode=2              # OpenCV模式支持批量
)

print(f"共提取 {len(frames)} 帧")
print(f"前5张图片路径: {frames[:5]}")
```

#### 提取模式说明

**FFmpeg模式 (mode=1)**：
- 优点：速度快，内存占用少
- 缺点：不支持批量处理
- 适用：大视频文件，快速提取

**OpenCV模式 (mode=2)**：
- 优点：支持批量处理，更灵活
- 缺点：速度较慢，内存占用大
- 适用：需要批量处理的场景

#### 输出结构

**FFmpeg模式输出**：
```
output_dir/
└── video_name/
    ├── video_name_000001.jpg
    ├── video_name_000002.jpg
    └── ...
```

**OpenCV模式输出**：
```
output_dir/
└── video_name/
    ├── batch_1/
    │   ├── video_name_000001.jpg
    │   └── ...
    ├── batch_2/
    │   ├── video_name_000051.jpg
    │   └── ...
    └── ...
```

## 🚀 快速开始

### 知识蒸馏使用

```python
# 1. 导入模块
from utils.class_distillation import DistillationLoss, FeatureExtractor

# 2. 创建损失函数
criterion = DistillationLoss(temperature=4.0, alpha=0.7, beta=0.3)

# 3. 创建特征提取器
extractor = FeatureExtractor(model, ['layer1', 'layer2'])

# 4. 训练循环
for batch in dataloader:
    # 前向传播
    student_output = student_model(batch)
    teacher_output = teacher_model(batch)
    
    # 获取特征
    features = extractor.get_features()
    
    # 计算损失
    loss = criterion(student_output, teacher_output, targets, features)
    
    # 反向传播
    loss['total_loss'].backward()
```

### 视频帧提取使用

```python
# 1. 导入模块
from utils.class_frame_extractor import VideoFrameExtractor

# 2. 创建提取器
extractor = VideoFrameExtractor("frames_output")

# 3. 提取帧
frames = extractor.extract(
    "video.mp4",
    fps=2,
    start=0,
    end=30,
    size=(512, 512)
)

# 4. 处理结果
print(f"提取了 {len(frames)} 帧")
```

## 📝 注意事项

### 知识蒸馏
1. 确保教师模型已经预训练并冻结参数
2. 特征层名称需要与模型结构匹配
3. 及时清理钩子函数避免内存泄漏
4. 根据任务调整损失权重参数

### 视频帧提取
1. FFmpeg模式需要系统安装FFmpeg
2. OpenCV模式支持批量处理但速度较慢
3. 大视频文件建议使用FFmpeg模式
4. 注意输出目录的磁盘空间

## 🔧 依赖要求

- PyTorch >= 1.7.0
- OpenCV >= 4.0.0
- FFmpeg (可选，用于快速提取)
- NumPy
- Pathlib

## 📄 许可证

本工具类遵循项目主许可证。
