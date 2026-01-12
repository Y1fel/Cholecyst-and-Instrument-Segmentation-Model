# Offline Training and Knowledge Distillation Pipeline

This document summarizes the end-to-end offline training workflow implemented in `src/training/offline/train_offline_universal.py`. This script serves as the unified entry point for standard training and knowledge distillation experiments on medical segmentation datasets.

## 1. Overview

The design allows the same entry point to support:

- Teacher training (e.g. UNet++ on 6 classes)
- Student KD training (e.g. Adaptive UNet distilled from UNet++)
- Binary, 3-class, and 6-class regimes (as well as custom mappings)
- Video-aware splits that prevent leakage between training and validation videos
- Resume runs, including optional creation of a brand-new run directory (`--resume_new_run`)

## 2. Data Ingestion (`SegDatasetMin`)

Offline training consumes segmentation data through `src/dataio/datasets/seg_dataset_min.py`. Key features:

### 2.1 Pair discovery

```text
images:  **/*_endo.(png|jpg|jpeg)    (filters out filenames containing 'mask')
masks:   *_endo_watershed_mask.png  > *_precise_gt.png > *_endo_mask.png
```

The dataset stores `(img_path, mask_path)` tuples in `self.pairs` and prints a 20-sample summary (mask types, target scheme, class names).

### 2.2 Label mapping

- Accepts a precomputed mapping (`class_id_map`) passed from the training script.
- Supports legacy flows (`classification_scheme` + `custom_mapping`) as fallback.
- Tracks `self.ws2train` so that runtime access never re-applies `compose_mapping`.
- Validates mapping values to ensure they belong to `{0,...,K-1} ∪ {255}` (255 is the ignore index).

### 2.3 FOV masking

`compute_fov_mask_from_rgb` erodes the black border by thresholding, morphological closing, and selecting the largest connected component. It flips to 'all in view' if the border ratio falls below 5%. When `apply_fov_mask=True`, pixels outside the FOV are set to 0 (background) or can be redirected to 255 if desired.

### 2.4 Sample loading

For each pair: load RGB image, resize to the requested `img_size`, and normalise to `[0,1]`. Masks are read as grayscale, mapped through `ws2train`, and optionally FOV-masked. Multi-class outputs remain `torch.long`; binary outputs are cast to `float` tensors to match `BCEWithLogitsLoss`.

### 2.5 Health checks

A startup probe iterates over up to 200 samples, counts class frequencies after mapping, and raises if every pixel collapsed to 255. Statistics are printed for debugging mismatches (`[HEALTH CHECK]`).

### 2.6 Data augmentation

The system supports runtime data augmentation for training datasets (disabled for validation):

**Random Horizontal Flipping**: Applied with probability `flip_prob` (default: 0.5)
$$P(\text{flip}) = \text{flip\_prob}$$

**Random Rotation**: Rotates both image and mask by random angle within `±rotation_degree`
$$\theta \sim \mathcal{U}(-\text{rotation\_degree}, +\text{rotation\_degree})$$

**FOV-aware Processing**: When `apply_fov_mask=True`, the field-of-view mask is computed using adaptive thresholding:

$$\text{threshold} = \max(5, \min(20, P_1 + 2))$$

where $P_1$ is the 1st percentile of grayscale intensity. Pixels outside the FOV (black borders) are masked to preserve anatomical context while removing endoscopic artifacts.

## 3. Label Taxonomy and Mapping

### 3.1 Watershed → base semantics

`src/common/constants.py` defines `WATERSHED_TO_BASE_CLASS`, a dictionary mapping SEG8K watershed gray values (e.g. 11, 12, 21, 22, …, 255) to a 13-class base taxonomy. For example:

- 21 → `liver`
- 22 → `gallbladder`
- 31 → `grasper`
- 32 → `l_hook_electrocautery`
- 255 → ignore

### 3.2 Base semantics → training regime

`CLASSIFICATION_SCHEMES` describes higher-level schemes (`binary`, `3class_org`, `3class_balanced`, `5class`, `6class`, `detailed`). Each scheme stores:

```python
{
    "num_classes": K,
    "target_classes": [...],
    "mapping": { base_id: target_id },
    "default_for_others": 255 or 0,
}
```

For example, the 6-class mapping folds abdominal wall, blood, connective tissue, hepatic vein, and liver ligament into background (0), while mapping `grasper`/`l_hook` to a shared instrument class (4) and `gallbladder` to class 5.

### 3.3 Compose mapping

`compose_mapping` composes the watershed and classification mappings:

$$\varphi(w) = \begin{cases}
\text{normalized}(\text{custom\_mapping}[w]) & \text{if custom provided}, \\
\psi(\text{WATERSHED\_TO\_BASE\_CLASS}[w]) & \text{if } w \neq 255, \\
255 & \text{otherwise}
\end{cases}$$

where $\psi$ is the scheme-specific mapping from base semantics to training labels. Custom dictionaries are normalised to integer keys/values, avoiding JSON string keys.

## 4. Configuration and Argument Handling

CLI options are defined in `parse_args()`. YAML configs (e.g. `configs/offline/6classes/teacher_training_improved_6class_configV2.yaml`) are loaded with `load_config` and merged by `merge_config_with_args`. The merge logic honours command-line precedence: every `--flag` present in `sys.argv` is remembered, and YAML values are applied only when the flag was absent.

Typical launch command:

```bash
python src/training/offline/train_offline_universal.py   --config configs/offline/6classes/teacher_training_improved_6class_configV2.yaml   --data_root data/seg8k
```

Resume jobs add `--resume outputs/<run>/checkpoints`, with optional `--resume_new_run` for a fresh destination and overrides such as `--resume_lr` or `--resume_epochs`.

## 5. Data Splitting Strategies

The training script offers three mutually-exclusive split paths:

1. **Video-aware (`video_aware_train_val_split`)**: group frames by `video\d+` id, shuffle videos deterministically, and allocate entire videos to train or validation to prevent leakage. A YAML summary of the split (file lists + metadata) is written under `<run>/splits/`.
2. **Frame random (`frame_random_split`)**: fall back to frame-level random sampling using `torch.utils.data.random_split`.
3. **From file (`load_split_from_file`)**: load explicit train/val lists from YAML, reconcile with the dataset, and warn about missing entries.

## 6. Loss Functions

The `create_advanced_loss_function` builder selects among several criteria:

### 6.1 Basic Loss Functions

- **Cross-Entropy (CE)** with optional class weights: $$L_{CE} = -\sum_{i=1}^{N} \sum_{c=1}^{C} w_c \cdot y_{i,c} \log(p_{i,c})$$
- **CombinedLoss** (CE + Dice). Dice uses $$\mathrm{Dice}(p,t) = \frac{2|p \cap t| + \epsilon}{|p| + |t| + \epsilon}$$, while the total loss is $$L = (1 - \lambda) L_{CE} + \lambda L_{Dice}$$
- **FocalLoss**: $$L = \alpha (1 - p_{t})^{\gamma} L_{CE}$$, mitigating class imbalance.

### 6.2 Advanced Loss Functions

**LabelSmoothingCrossEntropy**: Replaces one-hot targets with smoothed distributions to improve calibration:
$$\tilde{y}_k = \begin{cases}
1 - \alpha + \frac{\alpha}{K-1} & \text{if } k = y \\
\frac{\alpha}{K-1} & \text{otherwise}
\end{cases}$$

where $\alpha$ is the smoothing factor and $K$ is the number of classes.

**DiceLoss (standalone)**: For pure shape-based optimization:
$$L_{Dice} = 1 - \frac{2 \sum_{i} p_i t_i + \epsilon}{\sum_{i} p_i + \sum_{i} t_i + \epsilon}$$

### 6.3 Automatic Class Weights

The system supports automatic class weight computation to handle imbalanced datasets:

$$w_c = \frac{N_{total}}{K \cdot N_c}$$

where $N_{total}$ is total pixels, $K$ is number of classes, and $N_c$ is pixels for class $c$.

Two implementations are available:
- **Legacy weights** (`compute_class_weights`): Fast sampling-based estimation
- **Advanced weights** (`compute_auto_class_weights`): More precise calculation with configurable sampling ratio

### 6.4 Knowledge Distillation Mathematics

For KD runs, `DistillationLoss` combines:

$$L_{total} = \alpha\, T^{2} \operatorname{KL}(\sigma(z_s/T) \parallel \sigma(z_t/T)) + \beta\, L_{task} + \gamma\, L_{feature}$$

where `z_s`/`z_t` are student/teacher logits, `T` is the temperature, `α`, `β`, and `γ` correspond to `distill_alpha`, `distill_beta`, and `feature_weight`, and `L_{task}` is BCE or CE depending on class count.

The KL divergence term uses temperature scaling to soften the probability distributions:

$$\sigma(z_i/T) = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

The complete distillation loss formulation includes:

- **Knowledge Transfer**: $$L_{KL} = \operatorname{KL}(\sigma(z_s/T) \parallel \sigma(z_t/T)) = \sum_i \sigma(z_{t,i}/T) \log\frac{\sigma(z_{t,i}/T)}{\sigma(z_{s,i}/T)}$$
- **Task Performance**: $$L_{task} = \text{CE}(\sigma(z_s), y_{true})$$ or $$L_{task} = \text{BCE}(\sigma(z_s), y_{true})$$
- **Feature Alignment** (optional): $$L_{feature} = \|f_s - f_t\|_2^2$$ where $f_s, f_t$ are intermediate features

The temperature factor $T^2$ in the KL term compensates for the gradient scaling effect of temperature division.

## 7. Training Loop (`train_one_epoch`)

- Sets the model to train mode; for KD, keeps the teacher in evaluation mode and freezes its parameters.
- Moves mini-batches to device with `non_blocking=True`.
- In KD mode, asserts that teacher and student logits share the same class dimension and passes them (plus ground-truth masks) through `DistillationLoss`.
- In standard mode, forwards the batch and computes the chosen criterion.
- Accumulates running totals for logging and prints progress every `monitor_interval`.

The validation pass (`validate`) switches to inference mode, applies the same criterion (except KD always uses CE/BCE), and collects metrics through `Evaluator`. For multi-class validation, metrics include per-class IoU, Dice, accuracy, plus mean aggregates and optional background exclusion.

### 7.1 Metrics formulas

Given class-specific true positives (TP), false positives (FP), and false negatives (FN):

- IoU: $$\mathrm{IoU} = \frac{TP}{TP + FP + FN + \epsilon}$$
- Dice: $$\mathrm{Dice} = \frac{2TP}{2TP + FP + FN + \epsilon}$$
- Accuracy: $$\frac{TP + TN}{TP + FP + FN + TN + \epsilon}$$

Background exclusion simply removes class id 0 from the per-class arrays before averaging.

### 7.2 Early stopping

`early_stopping_metric` chooses between loss minimisation and mIoU maximisation. A patience counter is reset whenever the monitored metric improves; otherwise, training halts after `patience` stale epochs.

### 7.3 Optimizer formulas

The training system supports multiple optimizers with their mathematical formulations:

**Adam Optimizer:**
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**AdamW (with weight decay):**
$$\theta_{t+1} = \theta_t - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

**SGD with Momentum:**
$$v_t = \mu v_{t-1} + g_t$$
$$\theta_{t+1} = \theta_t - \alpha v_t$$

### 7.4 Learning rate scheduler formulas

**Cosine Annealing:**
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{T_{cur}}{T_{max}}\pi))$$

**Step Scheduler:**
$$\eta_t = \eta_0 \times \gamma^{\lfloor \frac{t}{step\_size} \rfloor}$$

**Plateau Scheduler:**
$$\eta_{t+1} = \begin{cases} 
\eta_t \times factor & \text{if no improvement for } patience \text{ epochs} \\
\eta_t & \text{otherwise}
\end{cases}$$

**Cosine Warmup (Sequential Scheduler):**
For the first $T_{warmup}$ epochs (linear warmup):
$$\eta_t = \eta_0 \times (0.1 + 0.9 \times \frac{t}{T_{warmup}})$$

For remaining epochs (cosine annealing):
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t - T_{warmup}}{T_{max} - T_{warmup}}\pi))$$

where $T_{warmup} = \max(2, 0.1 \times T_{total})$ provides stable training for complex multi-class scenarios.

## 8. Knowledge Distillation Extras

When `enable_distillation=True`, additional logic runs:

1. **Teacher loading** via `build_model` and `torch.load`, with shape checks and debug statistics.
2. **Optional evidence package** (`generate_kd_evidence_package`): evaluates both teacher and student, generates CSV summaries, calibration plots, reliability diagrams, and four-panel analyses using `DistillationVisualizer`. Outputs accumulate under `<run>/visualizations/`.
3. **Pseudo label utilities** are available in `utils/class_distillation.py` (`PseudoLabelLoss`), although the current training script focuses on direct KD.

## 9. Output Management

`OutputManager` abstracts run directories:

- Directory layout: `<output_dir>/<model_type>_<timestamp>/` with subfolders `checkpoints/` and `visualizations/`.
- `save_config` writes `config.json` (original merged arguments).
- `save_metrics_csv` appends rows to `metrics.csv`, enabling long-term tracking even across resumes.
- `save_model` stores checkpoints (optionally with suffixes such as `student` or `teacher_reference`).

### 9.1 Hybrid evaluation strategy

`save_checkpoint_with_hybrid_evaluation` implements a two-stage decision mechanism:

$$\text{Save Model} = \begin{cases}
\text{True} & \text{if } \Delta L < -\text{loss\_threshold} \\
\text{True} & \text{if } \Delta L \geq -\text{loss\_degradation\_threshold} \text{ and } \Delta \text{mIoU} > 0 \\
\text{False} & \text{otherwise}
\end{cases}$$

where:
- $\Delta L = L_{current} - L_{best}$ (loss change)
- $\Delta \text{mIoU} = \text{mIoU}_{current} - \text{mIoU}_{best}$ (mIoU change)
- `loss_threshold` = minimum loss improvement required (default: 0.02)
- `loss_degradation_threshold` = maximum acceptable loss degradation (default: 0.05)

This strategy prioritizes loss reduction but allows mIoU improvements when loss doesn't degrade significantly.

- For best models, `best_model_info.json` records epoch, metrics, and path.
- KD helpers expose dedicated directories for experiment summaries and calibration studies.

`ResumeManager` complements this by locating the latest or best checkpoint inside a `checkpoints/` folder, loading stored metrics, and retrieving the original config for reference.

## 10. Resume Semantics

- Pass `--resume outputs/<run>/checkpoints` to reload the latest checkpoint.
- Add `--resume_from_best` to start from `*_best.pth`.
- Use `--resume_new_run` to fork the training into a fresh directory while keeping the original run intact.
- Metrics CSV entries, visualisations, and hybrid checkpoint logic continue seamlessly when resuming in-place.

Because optimizer and scheduler states are not persisted, the first resumed epoch may show a brief loss spike before stabilising; this is expected.

## 11. Offline Model Zoo

`src/models/model_zoo.py` exposes a simple factory:

- `unet_min`: lightweight baseline.
- `unet_plus_plus`: nested UNet++ with deep supervision (default `base=16`).
- `deeplabv3_plus`: atrous encoder-decoder (ResNet-50 backbone).
- `hrnet`: high-resolution network (HRNet-W18).
- `adaptive_unet` and `mobile_unet`: also usable offline, especially for KD student runs.

The builder enforces stage compatibility: if a non-offline model is requested for the offline stage, it falls back to `unet_min`.

## 12. Evaluation and Visualisation

- `Evaluator.evaluate` handles binary metrics; `evaluate_multiclass` constructs a confusion matrix to derive IoU/Dice/Acc (with optional class exclusion).
- `Visualizer` (in `src/viz/visualizer.py`) renders comparison grids and basic predictions onto the validation data. KD runs add teacher-vs-student overlays, knowledge transfer analyses, and summary reports.

## 13. Configuration Templates

Offline configs live under `configs/offline/`. Notable files:

- `6classes/teacher_training_improved_6class_configV2.yaml`: teacher UNet++ recipe using cosine warmup, video-aware split, CE+Dice.
- `6classes/kd_student_config_6class.yaml`: Adaptive UNet student distilled from the above teacher.
- Additional templates exist for 3-class and baseline runs.

All configs rely on the same CLI entry point. Example KD run (resume from checkpoint):

```bash
python src/training/offline/train_offline_universal.py   --config configs/offline/6classes/kd_student_config_6class.yaml   --data_root data/seg8k   --resume outputs/distill_unet_plus_plus_to_adaptive_unet_20251005_042739/checkpoints
```

## 14. Offline Model Architectures

The offline training pipeline primarily utilizes two sophisticated deep learning architectures for medical image segmentation, with UNet++ being the primary choice for production training and DeepLabV3+ available for advanced experiments.

### 14.1 UNet++ (Nested U-Net) - Primary Architecture

UNet++ serves as the backbone architecture for offline training, particularly excelling in teacher model training and knowledge distillation scenarios. This implementation is located in `src/models/offline/unet_plus_plus.py`.

#### 14.1.1 Architecture Overview

UNet++ enhances the traditional U-Net by introducing nested dense skip connections, creating a grid of decoder nodes at multiple semantic levels. The architecture can be represented as:

$$X^{i,j} = \text{ConvBlock}(\text{Cat}([X^{i,0}, X^{i,1}, ..., X^{i,j-1}, \text{Up}(X^{i+1,j-1})]))$$

where:
- $X^{i,j}$ represents a decoder node at resolution level $i$ and semantic level $j$
- $i \in \{0,1,2,3,4\}$ corresponds to spatial resolutions $\{1, 1/2, 1/4, 1/8, 1/16\}$
- $j \in \{0,1,2,3,4\}$ represents the semantic aggregation level
- $\text{Cat}$ denotes channel-wise concatenation
- $\text{Up}$ is bilinear upsampling

#### 14.1.2 Key Components

**ConvBlock Module**: Each convolutional block follows the pattern:
$$\text{ConvBlock}(x) = \text{Dropout}(\text{ReLU}(\text{BN}(\text{Conv}(\text{ReLU}(\text{BN}(\text{Conv}(x)))))))$$

with optional dropout regularization for improved generalization.

**Dense Skip Connections**: The nested structure ensures that each decoder node receives feature aggregations from:
- All horizontal connections at the same resolution level
- One vertical connection from the corresponding lower-resolution node

**Multi-Scale Feature Fusion**: The architecture enables feature reuse across multiple semantic scales, mathematically expressed as:

$$\text{Output} = \begin{cases}
[\text{Out}_1, \text{Out}_2, \text{Out}_3, \text{Out}_4] & \text{if deep supervision} \\
\text{Out}_4 = \text{Conv}_{1×1}(X^{0,4}) & \text{otherwise}
\end{cases}$$

#### 14.1.3 Implementation Details

- **Base Channels**: Default `base_ch=32`, with channel progression `[32, 64, 128, 256, 512]`
- **Deep Supervision**: Optional multi-scale output for enhanced gradient flow during training
- **Upsampling Strategy**: Bilinear interpolation with `align_corners=False` for consistent behavior
- **Memory Optimization**: Efficient concatenation strategy to manage GPU memory usage

#### 14.1.4 Training Configuration

The UNet++ is typically configured with:
- **Input Resolution**: 384×384 for optimal performance-memory trade-off
- **Batch Size**: 6 for 6-class scenarios, 8-12 for binary tasks
- **Optimizer**: AdamW with weight decay 0.0005
- **Scheduler**: Cosine annealing with linear warmup (10% of total epochs)
- **Loss Function**: Combined CE+Dice loss with automatic class weights

### 14.2 DeepLabV3+ - Advanced Alternative

DeepLabV3+ provides an alternative encoder-decoder architecture based on atrous (dilated) convolutions and is available in `src/models/offline/deeplabv3_plus.py`. While not extensively tested in production, it offers theoretical advantages for certain scenarios.

#### 14.2.1 Architecture Components

**ResNet Backbone**: Modified ResNet-50/101 with controlled output stride:
$$\text{Output Stride} = \begin{cases}
8 & \text{for high-resolution feature extraction} \\
16 & \text{for balanced speed-accuracy trade-off}
\end{cases}$$

**Atrous Spatial Pyramid Pooling (ASPP)**: Multi-scale feature extraction using parallel atrous convolutions:

$$\text{ASPP}(x) = \text{Concat}[\text{Conv}_{1×1}(x), \text{AConv}_{3×3}^{r_1}(x), \text{AConv}_{3×3}^{r_2}(x), \text{AConv}_{3×3}^{r_3}(x), \text{GlobalPool}(x)]$$

where $\text{AConv}_{3×3}^{r_i}$ denotes 3×3 atrous convolution with dilation rate $r_i$.

**Decoder Module**: Combines high-level ASPP features with low-level backbone features:

$$\text{Decoder} = \text{Refine}(\text{Concat}[\text{Upsample}(\text{ASPP}), \text{LowLevelProj}(\text{C2})])$$

#### 14.2.2 Architectural Advantages

- **Multi-Scale Context**: ASPP captures features at multiple receptive field scales
- **Semantic Boundary Preservation**: Low-level feature integration maintains fine-grained details
- **Computational Efficiency**: Controlled output stride balances accuracy and speed

#### 14.2.3 Configuration Parameters

- **Atrous Rates**: `(6, 12, 18)` for OS=16, `(12, 24, 36)` for OS=8
- **ASPP Output Channels**: 256 (standard configuration)
- **Low-Level Projection**: 48 channels for efficient feature fusion
- **Backbone**: ResNet-50 with Bottleneck blocks

### 14.3 Model Selection Strategy

The offline training system employs a strategic model selection approach:

#### 14.3.1 Primary Choice: UNet++
- **Teacher Training**: UNet++ with `base_ch=32` for 6-class scenarios
- **Knowledge Distillation**: Teacher UNet++ → Student Adaptive UNet
- **Production Deployment**: Validated performance across multiple medical datasets

#### 14.3.2 Alternative: DeepLabV3+
- **Research Experiments**: Available for comparative studies
- **High-Resolution Requirements**: Better suited for fine-grained segmentation tasks
- **Future Integration**: Planned for comprehensive evaluation in upcoming releases

#### 14.3.3 Selection Criteria

$$\text{Model Choice} = \begin{cases}
\text{UNet++} & \text{if proven performance required} \\
\text{DeepLabV3+} & \text{if research/experimental scenario} \\
\text{UNet Min} & \text{if lightweight deployment needed}
\end{cases}$$

The model zoo automatically handles architecture compatibility and provides fallback mechanisms to ensure robust training across different model configurations.

---

This markdown captures the offline path end-to-end, aligning terminology with the repository so that diagrams, equations, and textual descriptions can be lifted directly into your final report or paper.