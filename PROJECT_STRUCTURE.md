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
└─ src/                               # ✅ 核心源代码
   ├─ common/                        # ✅ 公共工具
   │  ├─ output_manager.py           # ✅ 输出目录管理和结果保存
   │  └─ train_monitor.py            # ✅ 训练监控模块（单行刷新、GPU监控、进度条）
   ├─ dataio/                        # ✅ 数据加载与预处理
   │  └─ datasets/                   # ✅ 数据集类
   │     └─ seg_dataset_min.py       # ✅ 最小数据集类（images/、masks/）
   ├─ models/                        # ✅ 模型结构
   │  └─ baseline/                   # ✅ 基线模型
   │     └─ unet_min.py              # ✅ 最小 U-Net
   ├─ training/                      # ✅ 训练流程
   │  ├─ train_offline_min.py        # ✅ 最小离线训练（1个epoch测试）+ 🔄 可集成监控
   │  └─ train_universal_template.py # ✅ 通用训练模板（集成监控、可视化、评估）
   ├─ online/                        # 🚧 在线推理与自适应（预留目录）
   ├─ metrics/                       # ✅ 评价指标
   │  └─ evaluator.py                # ✅ 分割指标评估器（IoU、Dice、Accuracy等）
   └─ viz/                          # ✅ 可视化工具
      └─ visualizer.py               # ✅ 可视化器（预测结果、叠加图像）
```

## 新增功能模块说明

### 🆕 通用训练模板 (`src/training/train_universal_template.py`)
- **功能**: 集成监控、可视化、评估的通用训练框架
- **特点**: 
  - 实时进度监控（单行刷新）
  - GPU使用率和内存监控
  - 自动模型保存和指标记录
  - 可视化结果自动生成
  - 易于适配不同模型架构

### 🆕 训练监控模块 (`src/common/train_monitor.py`)
- **功能**: 提供训练过程的实时监控和状态显示
- **特点**:
  - 单行刷新进度显示
  - CPU/GPU资源监控
  - 训练时间统计
  - Epoch总结报告
- **集成**: 可直接集成到现有训练脚本中，包括`train_offline_min.py`

### 🆕 输出管理器 (`src/common/output_manager.py`)
- **功能**: 统一管理训练输出、模型保存和结果组织
- **特点**:
  - 自动创建时间戳目录
  - CSV指标记录
  - 模型检查点管理
  - 配置文件保存

### 🆕 评估器 (`src/metrics/evaluator.py`)
- **功能**: 提供分割任务的标准评估指标
- **支持指标**: IoU、Dice、Accuracy、Precision、Recall
- **特点**: 支持二分类和多分类分割

### 🆕 可视化器 (`src/viz/visualizer.py`)
- **功能**: 生成训练和验证结果的可视化
- **输出**: 预测掩码、叠加图像、对比图
- **扩展接口**: 预留多面板可视化、指标曲线等功能

## 使用指南

### 快速开始
```bash
# 激活环境
conda activate tti-seg-py310

# 使用通用模板训练
python src/training/train_universal_template.py \
  --data_root "data/seg8k" \
  --model_type "my_model" \
  --epochs 5 \
  --save_viz

# 使用配置文件
python src/training/train_universal_template.py \
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
python src/training/train_offline_min.py --data_root "data/seg8k" --model_type "baseline_monitored" --img_size 512 --batch_size 6 --epochs 3 --lr 3e-4 --num_classes 2 --val_ratio 0.2 --save_viz --monitor_interval 5 --enable_gpu_monitor --num_workers 4

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
Epoch 1/3 [10/67] | loss: 0.3245 | CPU: 25.3% | GPU: 78% (1024/8192MB) | Time: 00:02:45
Epoch 1 Summary:
  Train - loss: 0.2456
  Val   - val_loss: 0.2234 | iou: 0.7234 | dice: 0.8156 | accuracy: 0.9123
```

## 开发路线图

### 近期目标 (已完成)
- ✅ 基础训练框架
- ✅ 监控和可视化模块
- ✅ 通用训练模板
- ✅ 评估指标模块

### 中期目标 (计划中)
- 📋 在线推理和自适应模块
- 📋 模型蒸馏功能
- 📋 多数据集支持
- 📋 消融实验套件

### 长期目标 (未来)
- 📋 实时推理优化
- 📋 模型压缩和加速
- 📋 云端部署支持
