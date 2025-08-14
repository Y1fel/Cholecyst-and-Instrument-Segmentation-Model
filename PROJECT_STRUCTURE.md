# Project Structure

```
Cholecyst-and-Instrument-Segmentation-Model/
├─ README.md                          # 项目简介
├─ LICENSE                            # 许可证声明
├─ requirements.txt                   # Python依赖包列表
├─ pyproject.toml                     # 可选：Python打包配置
├─ .gitignore                         # Git忽略文件列表
│
├─ configs/                           # YAML配置文件，统一管理参数（Single Source of Truth）
│  ├─ datasets/                       # 数据集配置
│  │  ├─ cholec80.yaml               # Cholec80数据集路径与参数
│  │  └─ endovis.yaml                # EndoVis数据集路径与参数
│  ├─ offline/                        # 离线阶段（重模型训练）参数
│  │  ├─ baseline_min.yaml           # 极简配置（可选，不想配也可用命令行）
│  │  ├─ unetpp_deeplabv3.yaml       # U-Net++和DeepLabV3配置
│  │  └─ hrnet.yaml                  # HRNet配置
│  ├─ online/                         # 在线阶段（轻量模型与自适应）参数
│  │  ├─ default_online.yaml         # 默认在线推理参数（阈值、滑窗大小等）
│  │  ├─ gating.yaml                 # 多信号权重配置
│  │  ├─ conservative_update.yaml     # BN-only / Decoder-lite / Adapter-only配置
│  │  ├─ buffer_trio.yaml            # 缓冲机制（迟滞、冷却、回滚）
│  │  └─ escalation.yaml             # 安全模式切换规则
│  └─ ablations/                      # 消融实验配置
│     ├─ offline_only.yaml           # 仅离线模型
│     ├─ online_naive.yaml           # 简单在线更新
│     ├─ no_entropy_gate.yaml        # 无熵门控
│     ├─ no_buffer_trio.yaml         # 无缓冲机制
│     └─ bn_only.yaml                # 仅BN更新
│
├─ data/                              # 数据集元信息或软链接（原数据不放入仓库）
│  └─ README.md                      # 数据挂载说明
│
├─ checkpoints/                       # 模型权重（离线重模型、蒸馏模型、EMA快照）
│
├─ logs/                              # 日志（tensorboard、JSON、事件追踪）
│
├─ outputs/                           # 推理结果、可视化、表格
│
├─ docs/                              # 文档与图表
│  ├─ diagrams/                      # Mermaid流程图
│  ├─ method_notes.md                # 方法记录、调参笔记
│  └─ experiments_template.md        # 实验报告模板
│
├─ scripts/                           # 命令行脚本（可复现运行）
│  ├─ train_offline.sh               # 离线训练入口
│  ├─ distill.sh                     # 重→轻蒸馏
│  ├─ export_onnx.sh                 # 导出ONNX模型（TRT加速）
│  ├─ run_online.sh                  # 在线推理入口
│  ├─ ablation_suite.sh              # 批量运行消融实验
│  └─ eval_offline.sh                # 离线模型验证
│
├─ notebooks/                         # Jupyter实验笔记
│  ├─ 01_data_checks.ipynb           # 数据检查
│  ├─ 02_entropy_threshold_sweep.ipynb # 熵阈值调优
│  └─ 03_runtime_budget.ipynb        # 运行时资源评估
│
├─ docker/                            # Docker环境
│  ├─ Dockerfile                     # 容器构建文件
│  └─ environment.yml                # Conda环境配置
│
├─ tests/                             # 核心功能单元测试
│  ├─ test_entropy.py                # 熵计算测试
│  ├─ test_gating.py                 # 门控逻辑测试
│  ├─ test_hysteresis.py             # 迟滞机制测试
│  ├─ test_rollback.py               # 回滚机制测试
│  └─ test_pseudo_labels.py          # 伪标签生成测试
│
└─ src/                               # 核心源代码
   ├─ common/                        # 公共工具
   ├─ dataio/                        # 数据加载与预处理
   │  └─ datasets/                   # 数据集类
   │     └─ seg_dataset_min.py       # 最小数据集类（images/、masks/）
   ├─ models/                        # 模型结构
   │  └─ baseline/                   # 基线模型
   │     └─ unet_min.py              # 最小 U-Net
   ├─ training/                      # 训练流程
   │  └─ train_offline_min.py        # 只训练1个epoch并保存
   ├─ online/                        # 在线推理与自适应
   ├─ metrics/                       # 评价指标
   └─ viz/                          # 可视化工具
```
