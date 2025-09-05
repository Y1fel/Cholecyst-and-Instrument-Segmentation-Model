# Utils 工具类说明文档

本目录包含了项目中使用的工具类，主要用于知识蒸馏和视频帧提取功能。

## 文件结构

```
utils/
├── class_distillation.py    # 知识蒸馏和伪标签损失函数
├── class_frame_extractor.py # 视频帧提取器
├── class_frame_to_video.py  # 视频帧合并器
└── README.md               # 本说明文档
```

## 1. 知识蒸馏模块 (class_distillation.py)

### 功能概述
实现了用于模型知识蒸馏和伪标签训练的损失函数，支持教师-学生模型的知识传递，包括任务损失、蒸馏损失、特征蒸馏损失和伪标签损失。

### 主要类

#### DistillationLoss
知识蒸馏损失函数类，支持二分类和多分类任务。

**初始化参数：**
- `num_classes` (int): 类别数，默认为2（二分类）
- `temperature` (float): 蒸馏温度，默认为4.0
- `alpha` (float): 蒸馏损失权重，默认为0.7
- `beta` (float): 任务损失权重，默认为0.3
- `feature_weight` (float): 特征蒸馏损失权重，默认为0.1

**主要方法：**

##### `forward(student_outputs, teacher_outputs, targets, student_features=None, teacher_features=None)`
计算蒸馏总损失

**参数：**
- `student_outputs` (torch.Tensor): 学生模型输出
- `teacher_outputs` (torch.Tensor): 教师模型输出
- `targets` (torch.Tensor): 真实标签
- `student_features` (Optional[List[torch.Tensor]]): 学生模型中间特征
- `teacher_features` (Optional[List[torch.Tensor]]): 教师模型中间特征

**返回值：**
- `Dict[str, torch.Tensor]`: 包含以下键的字典
  - `total_loss`: 总损失
  - `task_loss`: 任务损失
  - `distill_loss`: 蒸馏损失
  - `feature_loss`: 特征蒸馏损失

### 使用示例

```python
import torch
from utils.class_distillation import DistillationLoss

# 创建蒸馏损失函数
distill_loss = DistillationLoss(
    num_classes=2,
    temperature=4.0,
    alpha=0.7,
    beta=0.3,
    feature_weight=0.1
)

# 模拟模型输出
student_outputs = torch.randn(32, 1)  # 学生模型输出
teacher_outputs = torch.randn(32, 1)  # 教师模型输出
targets = torch.randint(0, 2, (32,))  # 真实标签

# 计算损失
loss_dict = distill_loss(student_outputs, teacher_outputs, targets)
print(f"总损失: {loss_dict['total_loss']}")
print(f"任务损失: {loss_dict['task_loss']}")
print(f"蒸馏损失: {loss_dict['distill_loss']}")
print(f"特征损失: {loss_dict['feature_loss']}")
```

#### PseudoLabelLoss
伪标签损失函数类，支持硬伪标签和软伪标签两种模式。

**初始化参数：**
- `num_classes` (int): 类别数，默认为2（二分类）
- `use_soft` (bool): 是否使用软伪标签，默认为False（硬标签）
- `confidence_threshold` (float): 伪标签置信度阈值，默认为0.7

**主要方法：**

##### `forward(student_outputs, teacher_outputs)`
计算伪标签损失

**参数：**
- `student_outputs` (torch.Tensor): 学生模型输出
- `teacher_outputs` (torch.Tensor): 教师模型输出（用于生成伪标签）

**返回值：**
- `Dict[str, torch.Tensor]`: 包含以下键的字典
  - `pseudo_loss`: 伪标签损失

**使用示例：**

```python
import torch
from utils.class_distillation import PseudoLabelLoss

# 创建伪标签损失函数
pseudo_loss_fn = PseudoLabelLoss(
    num_classes=2,
    use_soft=False,              # 使用硬伪标签
    confidence_threshold=0.7     # 置信度阈值
)

# 模拟模型输出
student_outputs = torch.randn(32, 1)  # 学生模型输出
teacher_outputs = torch.randn(32, 1)  # 教师模型输出

# 计算伪标签损失
loss_dict = pseudo_loss_fn(student_outputs, teacher_outputs)
print(f"伪标签损失: {loss_dict['pseudo_loss']}")

# 软伪标签示例
soft_pseudo_loss_fn = PseudoLabelLoss(
    num_classes=8,               # 多分类
    use_soft=True,               # 使用软伪标签
    confidence_threshold=0.5
)

# 多分类伪标签损失
student_outputs_multi = torch.randn(32, 8)  # 8分类
teacher_outputs_multi = torch.randn(32, 8)
loss_dict_multi = soft_pseudo_loss_fn(student_outputs_multi, teacher_outputs_multi)
```

### 训练使用指南

#### 1. 基本训练流程

在模型训练中，DistillationLoss通常与教师模型和学生模型配合使用。以下是完整的训练示例：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.class_distillation import DistillationLoss

# 假设你已经有了教师模型和学生模型
class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 教师模型定义
        pass
    
    def forward(self, x):
        # 返回logits和中间特征
        return logits, features

class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 学生模型定义（通常更小）
        pass
    
    def forward(self, x):
        # 返回logits和中间特征
        return logits, features

# 初始化模型
teacher_model = TeacherModel()
student_model = StudentModel()

# 设置教师模型为评估模式（不更新参数）
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False

# 创建蒸馏损失函数
distill_loss_fn = DistillationLoss(
    num_classes=2,
    temperature=4.0,
    alpha=0.7,      # 蒸馏损失权重
    beta=0.3,       # 任务损失权重
    feature_weight=0.1  # 特征蒸馏权重
)

# 优化器（只优化学生模型）
optimizer = optim.Adam(student_model.parameters(), lr=0.001)

# 训练循环
def train_epoch(student_model, teacher_model, dataloader, optimizer, distill_loss_fn):
    student_model.train()
    
    total_loss = 0
    for batch_idx, (data, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # 前向传播
        with torch.no_grad():
            teacher_outputs, teacher_features = teacher_model(data)
        
        student_outputs, student_features = student_model(data)
        
        # 计算蒸馏损失
        loss_dict = distill_loss_fn(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            targets=targets,
            student_features=student_features,
            teacher_features=teacher_features
        )
        
        # 反向传播
        loss_dict['total_loss'].backward()
        optimizer.step()
        
        total_loss += loss_dict['total_loss'].item()
        
        # 打印训练信息
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, '
                  f'Total Loss: {loss_dict["total_loss"]:.4f}, '
                  f'Task Loss: {loss_dict["task_loss"]:.4f}, '
                  f'Distill Loss: {loss_dict["distill_loss"]:.4f}, '
                  f'Feature Loss: {loss_dict["feature_loss"]:.4f}')
    
    return total_loss / len(dataloader)
```

#### 2. 多分类任务训练

对于多分类任务（如分割任务），使用方式类似：

```python
# 多分类蒸馏损失
distill_loss_fn = DistillationLoss(
    num_classes=8,  # 8个类别（如胆囊分割中的不同器官）
    temperature=3.0,
    alpha=0.6,
    beta=0.4,
    feature_weight=0.05
)

# 训练循环（多分类）
def train_multiclass_distillation(student_model, teacher_model, dataloader, optimizer, distill_loss_fn):
    student_model.train()
    
    for data, targets in dataloader:
        optimizer.zero_grad()
        
        # 教师模型前向传播（不计算梯度）
        with torch.no_grad():
            teacher_outputs, teacher_features = teacher_model(data)
        
        # 学生模型前向传播
        student_outputs, student_features = student_model(data)
        
        # 计算损失
        loss_dict = distill_loss_fn(
            student_outputs=student_outputs,      # [B, C, H, W]
            teacher_outputs=teacher_outputs,      # [B, C, H, W]
            targets=targets,                      # [B, H, W]
            student_features=student_features,
            teacher_features=teacher_features
        )
        
        # 反向传播
        loss_dict['total_loss'].backward()
        optimizer.step()
        
        return loss_dict
```

#### 3. 渐进式蒸馏训练

在实际应用中，可以采用渐进式蒸馏策略：

```python
class ProgressiveDistillation:
    def __init__(self, student_model, teacher_model):
        self.student_model = student_model
        self.teacher_model = teacher_model
        
        # 不同阶段的蒸馏参数
        self.distill_configs = [
            # 阶段1：高温度，重任务损失
            {'temperature': 8.0, 'alpha': 0.3, 'beta': 0.7, 'feature_weight': 0.0},
            # 阶段2：中等温度，平衡损失
            {'temperature': 4.0, 'alpha': 0.5, 'beta': 0.5, 'feature_weight': 0.05},
            # 阶段3：低温度，重蒸馏损失
            {'temperature': 2.0, 'alpha': 0.7, 'beta': 0.3, 'feature_weight': 0.1},
        ]
    
    def train_stage(self, stage, dataloader, epochs):
        config = self.distill_configs[stage]
        distill_loss_fn = DistillationLoss(
            num_classes=2,
            **config
        )
        
        optimizer = optim.Adam(self.student_model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            avg_loss = train_epoch(
                self.student_model, 
                self.teacher_model, 
                dataloader, 
                optimizer, 
                distill_loss_fn
            )
            print(f'Stage {stage+1}, Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}')

# 使用渐进式蒸馏
progressive_trainer = ProgressiveDistillation(student_model, teacher_model)

# 分阶段训练
for stage in range(3):
    print(f"开始第 {stage+1} 阶段训练...")
    progressive_trainer.train_stage(stage, train_dataloader, epochs=10)
```

#### 4. 训练技巧和最佳实践

##### 温度调度
```python
# 动态调整温度
def get_temperature(epoch, total_epochs, initial_temp=8.0, final_temp=2.0):
    return initial_temp * (final_temp / initial_temp) ** (epoch / total_epochs)

# 在训练循环中使用
for epoch in range(total_epochs):
    current_temp = get_temperature(epoch, total_epochs)
    distill_loss_fn = DistillationLoss(
        num_classes=2,
        temperature=current_temp,
        alpha=0.7,
        beta=0.3,
        feature_weight=0.1
    )
    # ... 训练代码
```

##### 损失权重调度
```python
# 动态调整损失权重
def get_loss_weights(epoch, total_epochs):
    # 早期更注重任务损失，后期更注重蒸馏损失
    alpha = min(0.9, 0.3 + 0.6 * epoch / total_epochs)
    beta = 1.0 - alpha
    return alpha, beta

# 使用示例
for epoch in range(total_epochs):
    alpha, beta = get_loss_weights(epoch, total_epochs)
    distill_loss_fn = DistillationLoss(
        num_classes=2,
        temperature=4.0,
        alpha=alpha,
        beta=beta,
        feature_weight=0.1
    )
```

#### 5. 验证和监控

```python
def validate_distillation(student_model, teacher_model, val_dataloader, distill_loss_fn):
    student_model.eval()
    
    total_loss = 0
    total_task_loss = 0
    total_distill_loss = 0
    total_feature_loss = 0
    
    with torch.no_grad():
        for data, targets in val_dataloader:
            # 教师模型输出
            teacher_outputs, teacher_features = teacher_model(data)
            
            # 学生模型输出
            student_outputs, student_features = student_model(data)
            
            # 计算损失
            loss_dict = distill_loss_fn(
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
                targets=targets,
                student_features=student_features,
                teacher_features=teacher_features
            )
            
            total_loss += loss_dict['total_loss'].item()
            total_task_loss += loss_dict['task_loss'].item()
            total_distill_loss += loss_dict['distill_loss'].item()
            total_feature_loss += loss_dict['feature_loss'].item()
    
    num_batches = len(val_dataloader)
    return {
        'val_total_loss': total_loss / num_batches,
        'val_task_loss': total_task_loss / num_batches,
        'val_distill_loss': total_distill_loss / num_batches,
        'val_feature_loss': total_feature_loss / num_batches
    }
```

## 2. 视频帧提取模块 (class_frame_extractor.py)

### 功能概述
提供视频帧提取功能，支持使用FFmpeg或OpenCV两种方式提取视频帧，支持批量处理和多种输出格式。

### 主要类

#### VideoFrameExtractor
视频帧提取器类，支持从视频文件中提取指定帧。

**初始化参数：**
- `output_dir` (str): 输出目录，默认为"frames_out"
- `use_ffmpeg` (bool): 是否优先使用FFmpeg，默认为True

**主要方法：**

##### `extract(video_path, fps=2.0, every_n=None, start=None, end=None, size=None, fmt="png", jpg_quality=95, batch_size=30, mode=2)`
从视频中提取帧

**参数：**
- `video_path` (str): 视频文件路径
- `fps` (float): 提取帧率，默认为2.0
- `every_n` (Optional[int]): 每隔n帧提取一次
- `start` (Optional[float]): 开始时间（秒）
- `end` (Optional[float]): 结束时间（秒）
- `size` (Optional[Tuple[int, int]]): 输出图像尺寸
- `fmt` (str): 输出格式，默认为"png"
- `jpg_quality` (int): JPEG质量（1-100），默认为95
- `batch_size` (int): 批处理大小，默认为30
- `mode` (int): 提取模式，1=FFmpeg，2=OpenCV

**返回值：**
- `list[str]`: 提取的帧文件路径列表

### 支持的视频格式
- MP4 (.mp4)
- MOV (.mov)
- AVI (.avi)
- MKV (.mkv)
- M4V (.m4v)
- WMV (.wmv)
- MPG (.mpg)
- MPEG (.mpeg)

### 使用示例

#### 基本用法
```python
from utils.class_frame_extractor import VideoFrameExtractor

# 创建提取器
extractor = VideoFrameExtractor(output_dir="dataset_frames")

# 提取视频帧
frames = extractor.extract(
    "path/to/video.mp4",
    fps=2,              # 每秒提取2帧
    start=10,           # 从第10秒开始
    end=60,             # 到第60秒结束
    size=(512, 512),    # 输出尺寸512x512
    fmt="jpg",          # 输出格式为JPEG
    jpg_quality=90,     # JPEG质量90%
    batch_size=30,      # 每30帧一个批次
    mode=2              # 使用OpenCV模式
)

print(f"共提取 {len(frames)} 帧")
```

#### 使用FFmpeg模式
```python
# 使用FFmpeg模式（需要系统安装FFmpeg）
frames = extractor.extract(
    "path/to/video.mp4",
    fps=1.0,            # 每秒1帧
    size=(1024, 768),   # 输出尺寸
    fmt="png",          # PNG格式
    mode=1              # FFmpeg模式
)
```

#### 按帧间隔提取
```python
# 每隔10帧提取一次
frames = extractor.extract(
    "path/to/video.mp4",
    every_n=10,         # 每隔10帧
    fmt="jpg",
    mode=2
)
```

### 输出目录结构

使用OpenCV模式时，输出目录结构如下：
```
dataset_frames/
└── video_name/
    ├── batch_1/
    │   ├── video_name_000000.jpg
    │   ├── video_name_000001.jpg
    │   └── ...
    ├── batch_2/
    │   ├── video_name_000030.jpg
    │   └── ...
    └── ...
```

使用FFmpeg模式时：
```
dataset_frames/
└── video_name/
    ├── video_name_000001.png
    ├── video_name_000002.png
    └── ...
```

## 3. 视频帧合并模块 (class_frame_to_video.py)

### 功能概述
提供将提取的视频帧重新合并成视频的功能，支持批量处理、多种编码格式和详细的日志记录。

### 主要类

#### VideoFrameMerger
视频帧合并器类，将多个帧文件合并成视频文件。

**初始化参数：**
- `frame_dirs` (Union[str, List[str]]): 帧目录，可以是单个目录或目录列表
- `output_path` (str): 输出视频路径
- `fps` (int): 视频帧率，默认为25
- `size` (tuple): 输出视频大小 (width, height)，默认为None（使用第一帧大小）
- `fourcc` (str): 编码格式，默认为"mp4v"
- `auto_batches` (bool): 是否自动识别batch_x子目录，默认为True
- `log_path` (str): 日志文件路径，默认为"succeed_frames.txt"
- `save_failed_log` (bool): 是否保存失败帧日志，默认为True

**主要方法：**

##### `merge()`
将帧文件合并成视频

**功能：**
- 自动发现并处理batch_x子目录
- 按文件名排序帧序列
- 自动调整帧尺寸
- 生成详细的处理日志
- 记录失败的帧文件

**使用示例：**

#### 基本用法
```python
from utils.class_frame_to_video import VideoFrameMerger

# 创建合并器
merger = VideoFrameMerger(
    frame_dirs="dataset_frames/video_name",  # 帧目录
    output_path="output_video.mp4",          # 输出视频路径
    fps=25,                                  # 帧率
    size=(512, 512),                         # 输出尺寸
    fourcc="mp4v"                            # 编码格式
)

# 合并视频
merger.merge()
```

#### 处理多个目录
```python
# 处理多个帧目录
merger = VideoFrameMerger(
    frame_dirs=["frames1", "frames2", "frames3"],  # 多个目录
    output_path="combined_video.mp4",
    fps=30,
    auto_batches=True  # 自动识别batch_x子目录
)

merger.merge()
```

#### 自定义编码格式
```python
# 使用不同的编码格式
merger = VideoFrameMerger(
    frame_dirs="frames",
    output_path="output.avi",
    fps=24,
    fourcc="XVID",        # XVID编码
    log_path="custom_log.txt",
    save_failed_log=True
)

merger.merge()
```

#### 自动尺寸调整
```python
# 使用第一帧的尺寸
merger = VideoFrameMerger(
    frame_dirs="frames",
    output_path="auto_size_video.mp4",
    fps=25,
    size=None,  # 自动使用第一帧尺寸
    fourcc="mp4v"
)

merger.merge()
```

### 支持的编码格式
- **mp4v**: MPEG-4编码，兼容性好
- **XVID**: Xvid编码，适合AVI格式
- **MJPG**: Motion JPEG编码，质量高但文件大
- **H264**: H.264编码，压缩效率高

### 输出文件
合并过程会生成以下文件：
- **视频文件**: 合并后的视频
- **成功日志**: 记录成功处理的帧文件路径
- **失败日志**: 记录无法读取的帧文件（如果启用）

### 目录结构支持
支持以下目录结构：
```
frames/
├── batch_1/
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
├── batch_2/
│   ├── frame_000031.jpg
│   └── ...
└── ...
```

## 依赖要求

### class_distillation.py
- torch
- torch.nn
- torch.nn.functional

### class_frame_extractor.py
- opencv-python (cv2)
- pathlib
- subprocess
- shutil
- re
- sympy

### class_frame_to_video.py
- opencv-python (cv2)
- pathlib

## 注意事项

1. **知识蒸馏模块**：
   - 确保教师模型和学生模型的输出维度匹配
   - 特征蒸馏需要中间层特征，确保特征维度兼容
   - 温度参数影响蒸馏效果，建议在3-5之间调整
   - **训练注意事项**：
     - 教师模型必须设置为`eval()`模式，并冻结参数（`requires_grad=False`）
     - 教师模型前向传播时使用`torch.no_grad()`避免计算梯度
     - 建议使用渐进式蒸馏策略，从高温度开始逐渐降低
     - 监控各项损失的变化，确保蒸馏损失和任务损失的平衡
     - 对于分割任务，确保输出张量维度为`[B, C, H, W]`格式
     - 特征蒸馏会增加显存占用，根据GPU内存调整batch_size

2. **视频帧提取模块**：
   - FFmpeg模式需要系统安装FFmpeg
   - OpenCV模式支持批量处理，适合大视频文件
   - 输出目录会自动创建
   - 文件名会自动处理特殊字符

3. **视频帧合并模块**：
   - 确保帧文件按正确顺序命名（建议使用数字序号）
   - 支持的图像格式：JPG、PNG
   - 自动处理batch_x子目录结构
   - 合并前确保有足够的磁盘空间
   - 不同编码格式的兼容性可能因系统而异

## 错误处理

1. **视频帧提取模块**：
   - 视频文件无法打开时会抛出RuntimeError
   - FFmpeg命令执行失败时会抛出subprocess.CalledProcessError
   - 确保输出目录有写入权限

2. **视频帧合并模块**：
   - 没有找到任何帧文件时会抛出RuntimeError
   - 帧文件无法读取时会记录到失败日志
   - 确保输出路径有写入权限

## 性能建议

1. **视频帧提取模块**：
   - 对于大视频文件，建议使用OpenCV模式并设置合适的batch_size
   - 根据存储空间和提取速度需求选择合适的图像格式和质量

2. **视频帧合并模块**：
   - 对于大量帧文件，建议使用高效的编码格式（如H264）
   - 合并前检查帧文件完整性，避免处理损坏的文件
   - 使用SSD存储可以提高合并速度

3. **知识蒸馏训练**：
   - **温度策略**：建议先用较高的temperature值（6-8）进行预热，然后逐渐降低到2-4
   - **损失权重调度**：训练初期更注重任务损失（beta=0.7），后期更注重蒸馏损失（alpha=0.7）
   - **特征蒸馏优化**：
     - 特征蒸馏会增加计算开销，可根据需要调整feature_weight（0.05-0.1）
     - 对于大模型，可以只使用部分中间层进行特征蒸馏
     - 使用梯度累积来减少显存占用
   - **训练策略**：
     - 使用渐进式蒸馏，分阶段调整参数
     - 监控各项损失的变化趋势，避免蒸馏损失过大导致性能下降
     - 对于分割任务，建议使用较小的batch_size（8-16）以节省显存
   - **模型选择**：
     - 确保学生模型有足够的容量学习教师模型的知识
     - 教师模型和学生模型的架构差异不宜过大

