# Project Structure

## 当前实现状态
✅ = 已实现  🚧 = 部分实现  📋 = 计划中

```
Cholecyst-and-Instrument-Segmentation-Model/
├─ README.md                          # ✅ 项目简介
├─ LICENSE                            # ✅ 许可证声明
├─ PROJECT_STRUCTURE.md               # ✅ 项目结构文档
├─ requirements.txt                   # 📋 Python依赖包列表
├─ pyproject.toml                     # 📋 可选：Python打包配置
├─ .gitignore                         # ✅ Git忽略文件列表
│
├─ configs/                           # ✅ YAML配置文件，统一管理参数（Single Source of Truth）
│  ├─ universal_template_config.yaml # ✅ 通用训练模板配置
│  ├─ datasets/                       # 🚧 数据集配置目录
│  │  ├─ cholec80.yaml               # 📋 Cholec80数据集路径与参数
│  │  └─ endovis.yaml                # 📋 EndoVis数据集路径与参数
│  ├─ offline/                        # 🚧 离线阶段（重模型训练）参数目录
│  │  ├─ baseline_min.yaml           # 📋 极简配置（可选，不想配也可用命令行）
│  │  ├─ unetpp_deeplabv3.yaml       # 📋 U-Net++和DeepLabV3配置
│  │  └─ hrnet.yaml                  # 📋 HRNet配置
│  ├─ online/                         # 🚧 在线阶段（轻量模型与自适应）参数目录
│  │  ├─ default_online.yaml         # 📋 默认在线推理参数（阈值、滑窗大小等）
│  │  ├─ gating.yaml                 # 📋 多信号权重配置
│  │  ├─ conservative_update.yaml     # 📋 BN-only / Decoder-lite / Adapter-only配置
│  │  ├─ buffer_trio.yaml            # 📋 缓冲机制（迟滞、冷却、回滚）
│  │  └─ escalation.yaml             # 📋 安全模式切换规则
│  └─ ablations/                      # 🚧 消融实验配置目录
│     ├─ offline_only.yaml           # 📋 仅离线模型
│     ├─ online_naive.yaml           # 📋 简单在线更新
│     ├─ no_entropy_gate.yaml        # 📋 无熵门控
│     ├─ no_buffer_trio.yaml         # 📋 无缓冲机制
│     └─ bn_only.yaml                # 📋 仅BN更新
│
├─ data/                              # ✅ 数据集元信息或软链接（原数据不放入仓库）
│  └─ README.md                      # 📋 数据挂载说明
│
├─ checkpoints/                       # ✅ 模型权重（离线重模型、蒸馏模型、EMA快照）
│
├─ logs/                              # ✅ 日志（tensorboard、JSON、事件追踪）
│
├─ outputs/                           # ✅ 推理结果、可视化、表格
│
├─ docs/                              # ✅ 文档与图表
│  ├─ UNIVERSAL_TEMPLATE_GUIDE.md    # ✅ 通用训练模板使用指南
│  ├─ diagrams/                      # ✅ Mermaid流程图目录
│  ├─ method_notes.md                # 📋 方法记录、调参笔记
│  └─ experiments_template.md        # 📋 实验报告模板
│
├─ scripts/                           # ✅ 命令行脚本（可复现运行）
│  ├─ seg8k_fetch.py                 # ✅ 数据集下载脚本
│  ├─ demo_multiclass_color.py       # ✅ 多类分割可视化演示脚本
│  ├─ train_monitor.py               # ✅ 训练监控模块（已移至src/common/）
│  ├─ train_offline.sh               # 📋 离线训练入口
│  ├─ distill.sh                     # 📋 重→轻蒸馏
│  ├─ export_onnx.sh                 # 📋 导出ONNX模型（TRT加速）
│  ├─ run_online.sh                  # 📋 在线推理入口
│  ├─ ablation_suite.sh              # 📋 批量运行消融实验
│  └─ eval_offline.sh                # 📋 离线模型验证
│
├─ notebooks/                         # ✅ Jupyter实验笔记
│  ├─ 01_data_checks.ipynb           # 📋 数据检查
│  ├─ 02_entropy_threshold_sweep.ipynb # 📋 熵阈值调优
│  └─ 03_runtime_budget.ipynb        # 📋 运行时资源评估
│
├─ docker/                            # ✅ Docker环境
│  ├─ environment.yml                # ✅ Conda环境配置
│  └─ Dockerfile                     # 📋 容器构建文件
│
├─ tests/                             # ✅ 核心功能单元测试目录
│  ├─ test_entropy.py                # 📋 熵计算测试
│  ├─ test_gating.py                 # 📋 门控逻辑测试
│  ├─ test_hysteresis.py             # 📋 迟滞机制测试
│  ├─ test_rollback.py               # 📋 回滚机制测试
│  └─ test_pseudo_labels.py          # 📋 伪标签生成测试
│
├─ src/                               # ✅ 核心源代码
   ├─ common/                        # ✅ 公共工具
   │  ├─ constants.py                # ✅ 全局常量定义（CLASSES、PALETTE、IGNORE_INDEX）
   │  ├─ output_manager.py           # ✅ 输出目录管理和结果保存
   │  └─ train_monitor.py            # ✅ 训练监控模块（单行刷新、GPU监控、进度条、ETA预估）
   ├─ dataio/                        # ✅ 数据加载与预处理
   │  └─ datasets/                   # ✅ 数据集类
   │     └─ seg_dataset_min.py       # ✅ 最小数据集类（images/、masks/）
   ├─ models/                        # ✅ 模型结构
   │  ├─ model_zoo.py                # ✅ 模型动物园（统一模型构建和管理）
   │  ├─ baseline/                   # ✅ 基线模型
   │  │  └─ unet_min.py              # ✅ 最小 U-Net
   │  ├─ offline/                    # ✅ 离线重模型目录（预留）
   │  └─ online/                     # ✅ 在线/轻量模型
   │     ├─ mobile_unet.py           # ✅ MobileNet风格的轻量U-Net（深度可分离卷积）
   │     └─ adaptive_unet.py         # ✅ 自适应U-Net（带适配器层，支持在线学习）
   ├─ training/                      # ✅ 训练流程
   │  ├─ offline/                    # ✅ 离线训练模块
   │  │  ├─ train_offline_min.py     # ✅ 最小离线训练（1个epoch测试）+ 🔄 可集成监控
   │  │  └─ train_offline_universal.py # ✅ 通用离线训练模板（集成监控、可视化、评估）
   │  └─ online/                     # ✅ 在线训练模块
   │     └─ adaptive_unet_main.py    # ✅ 在线自适应学习主程序（帧选择、经验回放、在线训练）
   ├─ eval/                          # ✅ 评估模块
   │  └─ evaluator.py                # ✅ 分割指标评估器（IoU、Dice、Accuracy等）
   ├─ online/                        # 📋 在线推理与自适应（预留目录）
   └─ viz/                          # ✅ 可视化工具
      ├─ colorize.py                 # ✅ 颜色映射和可视化函数（id_to_color、make_triplet、overlay）
      └─ visualizer.py               # ✅ 可视化器（预测结果、叠加图像）
```

## 新增功能模块说明

### 🆕 通用训练模板 (`src/training/offline/train_offline_universal.py`)
- **功能**: 集成监控、可视化、评估的通用训练框架
- **特点**: 
  - 实时进度监控（单行刷新）
  - GPU使用率和内存监控
  - 自动模型保存和指标记录
  - 可视化结果自动生成
  - 易于适配不同模型架构
  - 支持二分类和多分类分割任务
  - 模型插拔机制（通过model_zoo统一管理）

### 🆕 训练监控模块 (`src/common/train_monitor.py`)
- **功能**: 提供训练过程的实时监控和状态显示
- **特点**:
  - 单行刷新进度显示
  - CPU/GPU资源监控
  - 训练时间统计和ETA剩余时间预估
  - Epoch总结报告
  - 智能进度算法（基于已完成batch和epoch计算剩余时间）
- **集成**: 可直接集成到现有训练脚本中，包括`train_offline_min.py`

### 🆕 输出管理器 (`src/common/output_manager.py`)
- **功能**: 统一管理训练输出、模型保存和结果组织
- **特点**:
  - 自动创建时间戳目录
  - CSV指标记录
  - 模型检查点管理
  - 配置文件保存

### 🆕 评估器 (`src/eval/evaluator.py`)
- **功能**: 提供分割任务的标准评估指标
- **支持指标**: IoU、Dice、Accuracy、Precision、Recall
- **特点**: 支持二分类和多分类分割
- **预留接口**: 自动检测任务类型、通用评估、高级指标计算

### 🆕 可视化器 (`src/viz/visualizer.py`)
- **功能**: 生成训练和验证结果的可视化
- **输出**: 预测掩码、叠加图像、对比图
- **扩展接口**: 预留多面板可视化、指标曲线等功能

### 🆕 颜色映射模块 (`src/viz/colorize.py`)
- **功能**: 提供分割掩码的颜色映射和可视化函数
- **特点**:
  - `id_to_color()`: 将分割ID转换为彩色图像
  - `make_triplet()`: 创建原图-真值-预测的三联对比图
  - `overlay()`: 在原图上叠加半透明分割结果
- **应用**: 用于演示脚本和训练可视化

### 🆕 改进训练配置 (`improved_training_config.py`)
- **功能**: 提供改进的训练策略和配置建议
- **特点**:
  - 加权BCE损失函数处理类别不平衡
  - 优化的超参数配置（学习率、batch size、epochs）
  - Cosine学习率调度和warmup策略
  - 数据增强配置建议
- **应用**: 指导训练参数调优和性能提升

### 🆕 模型动物园 (`src/models/model_zoo.py`)
- **功能**: 统一的模型构建和管理中心
- **特点**:
  - 中心化模型注册和实例化
  - 支持二分类和多分类分割任务自动适配
  - 模型参数验证和错误处理
  - Fallback机制确保模型可用性
  - 集成调试日志用于问题诊断
- **支持模型**: UNetMin, MobileUNet, AdaptiveUNet
- **应用**: 训练脚本中的模型插拔和切换

### 🆕 模型注册表 (`src/common/model_registry.py`)
- **功能**: 模型名称到类的映射管理
- **特点**:
  - 去耦合的模型注册机制
  - 便于扩展新模型类型
  - 统一的模型访问接口
- **集成**: 与model_zoo.py协同工作
- **功能**: 统一管理项目全局常量和配置
- **内容**:
  - `CLASSES`: 类别ID到名称的映射（background, instrument, target_organ等）
  - `PALETTE`: 类别ID到RGB颜色的映射
  - `IGNORE_INDEX`: 忽略标签的索引值
- **优势**: 中心化配置，便于维护和修改

### 🆕 MobileUNet模型 (`src/models/online/mobile_unet.py`)
- **功能**: 轻量级分割模型，适用于在线推理
- **特点**:
  - 采用深度可分离卷积（DepthwiseSeparableConv）
  - 大幅减少参数量和计算量
  - 保持U-Net架构的跳跃连接设计
- **应用**: 移动设备部署、实时推理场景

### 🆕 自适应UNet模型 (`src/models/online/adaptive_unet.py`)
- **功能**: 支持在线学习的自适应分割模型
- **特点**:
  - 内置ConvAdapter适配器层，支持快速适应新数据分布
  - 可选择性冻结编码器/解码器，只训练适配器参数
  - 保持原有U-Net主干结构的稳定性
- **优势**: 在线学习时计算开销小，适应速度快

### 🆕 在线自适应学习系统 (`src/online/adaptive_unet_main.py`)
- **功能**: 完整的在线学习框架，支持实时数据适应
- **核心组件**:
  - **FrameSelector**: 智能帧选择器，基于SSIM相似度决定处理策略
  - **ExperienceReplayBuffer**: 经验回放缓冲区，防止灾难性遗忘
  - **OnlineLearner**: 在线学习管理器，协调模型训练和数据管理
- **特点**:
  - 自适应阈值调整
  - 帧间相似度分析（高相似度时仅处理中间帧）
  - 历史数据回放机制
  - 梯度裁剪和学习率调度
- **应用**: 手术视频实时分割、动态场景适应

### 🆕 多类分割演示脚本 (`scripts/demo_multiclass_color.py`)
- **功能**: 展示多类分割模型的推理和可视化效果
- **特点**:
  - 支持加载预训练模型进行推理
  - 可选择提供真值标签生成三联对比图
  - 输出彩色可视化结果
- **用法**: 用于模型效果演示和结果验证

## 使用指南

### 快速开始
```bash
# 激活环境
conda activate tti-seg-py310

# 使用通用模板训练
python src/training/offline/train_offline_universal.py \
  --data_root "data/seg8k" \
  --model unet_min \
  --epochs 5 \
  --save_viz

# 使用配置文件
python src/training/offline/train_offline_universal.py \
  --cfg configs/universal_template_config.yaml \
  --data_root "data/seg8k"
```

### 输出结构
```
outputs/
└── your_model_20250822_143052/
    ├── config.json                 # 训练配置
    ├── metrics.csv                 # 训练指标记录
    ├── checkpoints/                # 模型检查点
    └── visualizations/             # 可视化结果
```

### 单行完整命令（train_offline_min.py为例）
python src/training/offline/train_offline_min.py --data_root "data/seg8k" --model_type "baseline_monitored" --img_size 512 --batch_size 6 --epochs 3 --lr 3e-4 --num_classes 2 --val_ratio 0.2 --save_viz --monitor_interval 5 --enable_gpu_monitor --num_workers 4

### 二分类/多分类通用训练命令
```bash
# 二分类训练
python src/training/offline/train_offline_universal.py \
  --data_root "data/seg8k" \
  --binary \
  --model unet_min \
  --epochs 1 \
  --batch_size 2 \
  --img_size 256

# 多分类训练
python src/training/offline/train_offline_universal.py \
  --data_root "data/seg8k" \
  --model adaptive_unet \
  --num_classes 3 \
  --epochs 5 \
  --batch_size 4 \
  --img_size 512
```

## 🔧 监控模块集成指南

### train_offline_min.py 集成步骤

#### 1. 导入监控模块
```python
from src.common.train_monitor import TrainMonitor
```

#### 2. 初始化监控器（在main函数开始）
```python
monitor = TrainMonitor(enable_gpu_monitor=True)
monitor.start_timing()
```

#### 3. 添加命令行参数
```python
p.add_argument("--monitor_interval", type=int, default=5, help="Progress update interval")
p.add_argument("--enable_gpu_monitor", action='store_true', default=True, help="Enable GPU monitoring")
```

#### 4. 训练循环中添加实时监控
```python
for batch_idx, (images, masks) in enumerate(train_loader):
    # ... 训练代码 ...
    if batch_idx % args.monitor_interval == 0:
        monitor.print_progress(epoch+1, args.epochs, batch_idx+1, len(train_loader), {"loss": current_loss})
```

#### 5. Epoch结束添加总结
```python
monitor.print_epoch_summary(epoch+1, {"loss": avg_train_loss}, val_metrics)
```

### 预期监控输出
```
Epoch 1/20 [66/1078] | loss: 0.4183 | CPU: 7.1% | GPU: 99% (7971/8188MB) | Time: 00:21:02 | ETA: 06:23:45
Epoch 1 Summary:
  Train - loss: 0.2456
  Val   - val_loss: 0.2234 | iou: 0.7234 | dice: 0.8156 | accuracy: 0.9123
--------------------------------------------------------------------------------
```

## 🔧 模型动物园使用指南

### 模型注册和使用
```python
from src.models.model_zoo import build_model

# 构建二分类模型
model = build_model(
    model_name="unet_min",
    in_channels=3,
    out_channels=1,  # 二分类
    binary=True
)

# 构建多分类模型
model = build_model(
    model_name="adaptive_unet", 
    in_channels=3,
    out_channels=3,  # 背景 + 器械 + 目标器官
    binary=False
)
```

### 支持的模型类型
- `unet_min`: 基础U-Net模型（src/models/baseline/unet_min.py）
- `mobile_unet`: 轻量级移动端U-Net（src/models/online/mobile_unet.py）
- `adaptive_unet`: 自适应U-Net（src/models/online/adaptive_unet.py）

### Fallback机制
当指定模型不可用时，自动回退到UNetMin基础模型，确保训练继续进行。

## 开发路线图

### 近期目标 (已完成)
- ✅ 基础训练框架
- ✅ 监控和可视化模块（包含ETA时间预估）
- ✅ 通用训练模板（支持二分类/多分类切换）
- ✅ 评估指标模块
- ✅ 实时训练进度监控（时间、资源、ETA）
- ✅ 分割可视化工具（颜色映射、对比图生成）
- ✅ 全局常量和配置管理系统
- ✅ 多类分割演示脚本
- ✅ 轻量级MobileUNet模型（深度可分离卷积）
- ✅ 自适应UNet模型（适配器机制）
- ✅ 在线学习和自适应系统（帧选择、经验回放、实时适应）
- ✅ 模型动物园系统（统一模型管理、插拔机制、Fallback保护）
- ✅ 训练模块重组（offline/online分离）

### 中期目标 (计划中)
- 📋 模型蒸馏功能
- 📋 多数据集支持
- 📋 消融实验套件

### 长期目标 (未来)
- 📋 实时推理优化
- 📋 模型压缩和加速
- 📋 云端部署支持
