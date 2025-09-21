# Project Directory Structure and Functionality Table

| Parent Path | Sub Path | Main Function | Content Description | Key Files/Components |
|-------------|----------|---------------|-------------------|---------------------|
| **Root** | | Project Root | Core project files and documentation | README.md, LICENSE, .gitignore |
| **src** | | Core Source Code | Training, models, data processing, evaluation, visualization | Main implementation logic |
| **src** | **training** | Training System | Offline training, online learning, resume management | train_offline_universal.py, online systems |
| **src** | **training/offline** | Offline Training | Universal training scripts, knowledge distillation integration | Multi-model support, video-aware splitting |
| **src** | **training/online** | Online Learning | Real-time adaptation, video processing | Adaptive learning, frame selection algorithms |
| **src** | **models** | Model Architecture | Baseline models, online models, model factory | model_zoo.py unified management |
| **src** | **models/baseline** | Baseline Models | Standard model implementations | UNet-Min, basic architectures |
| **src** | **models/offline** | Offline Models | Heavy models for offline training | Teacher models, complex architectures |
| **src** | **models/online** | Online Models | Lightweight models for real-time inference | Mobile-UNet, Adaptive-UNet |
| **src** | **dataio** | Data Processing | Dataset classes, preprocessing pipeline | seg_dataset_min.py core dataset |
| **src** | **dataio/datasets** | Dataset Classes | Core segmentation dataset implementations | Multi-class mapping, FOV processing |
| **src** | **eval** | Evaluation System | Metrics calculation, performance assessment | Multi-metric support (IoU/Dice/Acc) |
| **src** | **viz** | Visualization Tools | Training monitoring, result display | Real-time visualization, coloring tools |
| **src** | **common** | Common Components | Constants definition, monitoring, management tools | Project infrastructure components |
| **configs** | | Configuration Management | YAML config files, experiment parameters | Unified parameter management system |
| **configs** | **offline** | Offline Training Configs | Configuration files for offline training | Training parameter templates |
| **configs** | **online** | Online Learning Configs | Configuration files for online learning | Real-time adaptation parameters |
| **configs** | **experiments** | Experiment Configurations | KD experiments, baseline comparison configs | Research experiment specific configs |
| **scripts** | | Tool Scripts | Data validation, experiment running, analysis tools | Independent functionality script collection |
| **utils** | | Utility Modules | Knowledge distillation, video processing, loss functions | Reusable tool components |
| **data** | | Data Storage | Dataset files, cache paths | Seg8K/CholecT80 datasets |
| **data** | **seg8k** | Seg8K Dataset | Segmentation dataset storage | Image and mask files |
| **data** | **cholec80** | CholecT80 Dataset | Video dataset storage | Surgical video data |
| **outputs** | | Training Outputs | Model checkpoints, logs, visualization results | Experiment result storage |
| **checkpoints** | | Model Weights Storage | Pre-trained models, best model saves (created during training) | Model checkpoint management |
| **logs** | | Log Files Storage | Training logs, error records (created during training) | System runtime records |
| **splits** | | Data Splitting | Video-aware splitting results | Train/validation data partitioning |

## Core Functional Module Details

### 🏋️ Training System (`src/training/`)
- **Offline Training**: Complete deep learning training pipeline with knowledge distillation support
- **Online Learning**: Real-time model adaptation with frame selection and experience replay
- **Resume Management**: Training interruption recovery and checkpoint integrity verification

### 🦁 Model System (`src/models/`)  
- **Model Factory**: Unified model construction and management interface
- **Architecture Support**: UNet, UNet++, Adaptive UNet and other architectures
- **Layered Design**: Baseline/offline/online model categorized management

### 📊 Data System (`src/dataio/`)
- **Dataset Classes**: Core segmentation dataset supporting multiple mapping schemes
- **Preprocessing**: FOV masking, data augmentation, health checking
- **Format Support**: Watershed region remapping, multi-class adaptation

### 📏 Evaluation System (`src/eval/`)
- **Multi-Metrics**: IoU, Dice, Accuracy, Precision, Recall
- **Hybrid Strategy**: Dual Loss and mIoU determination mechanism
- **Adaptive**: Automatic task type detection and evaluation adjustment

### 🎨 Visualization System (`src/viz/`)
- **Real-time Monitoring**: Training process GPU/CPU/Loss real-time display
- **Result Display**: Segmentation result coloring, comparison image generation
- **Distillation Visualization**: Knowledge distillation specific visualization tools

### ⚙️ Configuration System (`configs/`)
- **Layered Management**: Offline/online/experiment configuration categorization
- **Parameter Unification**: YAML format unified management of all parameters
- **Experiment Support**: Dedicated experiment configurations for research comparison

### 🔧 Tool Scripts (`scripts/`)
- **Data Validation**: WS value scanning, mapping checking, palette validation
- **Experiment Automation**: KD experiment runner, metrics visualization
- **Analysis Tools**: Parameter statistics, mechanism validation, result comparison

### 🛠️ Utility Modules (`utils/`)
- **Knowledge Distillation**: Complete Teacher-Student distillation framework
- **Video Processing**: Frame extraction, video synthesis tools
- **Loss Functions**: Composite loss function library

## Data Flow Diagram

### Offline Training Data Flow
```
Dataset → src/dataio → src/training/offline → src/models → src/eval → outputs/
   ↓           ↓               ↓                ↓           ↓          ↓
data/    Preprocessing    Batch Training      Model       Metrics    Result
   ↓        Enhancement    Knowledge          Inference   Calculation Storage
   ↓           ↓          Distillation         ↓           ↓          ↓
Config     Health Check   Visualization   Architecture  Hybrid    Checkpoint
Files                                     Management   Evaluation Management
```

### Online Learning Data Flow
```
Video    →  Frame      →  src/training/online  →  Real-time  →  Adaptation
Stream      Extraction     ↓                      Inference     Results
   ↓           ↓           Frame Selection            ↓            ↓
Live        Buffer         ↓                    Pseudo-label   Experience
Input       Management    Quality Control       Generation     Replay
   ↓           ↓           ↓                        ↓            ↓
Teacher  →  Student   →  Online Update    →   Model Update → Performance
Model       Model        Pipeline              Weights        Monitoring
```

### Integrated Data Flow Components
```
📊 Data Sources:
   ├── Static Dataset (data/seg8k, data/cholec80)
   ├── Live Video Streams (real-time input)
   └── Configuration Files (configs/)

🔄 Processing Pipeline:
   ├── Preprocessing (src/dataio/datasets/)
   ├── Training Systems (src/training/offline & online)
   ├── Model Management (src/models/model_zoo.py)
   └── Evaluation & Visualization (src/eval, src/viz)

💾 Output Management:
   ├── Model Checkpoints (outputs/*/checkpoints/)
   ├── Training Logs (logs/)
   ├── Visualization Results (outputs/*/visualizations/)
   └── Experiment Reports (outputs/*/reports/)
```

## Project Distinctive Features

### 🎯 **Unified Interface Design**
- All models managed through model_zoo
- Configuration files enable parameter standardization
- Evaluation system automatically adapts to task types

### 🔄 **Complete Lifecycle**
- Full pipeline from data preprocessing to model deployment
- Integrated training, validation, testing, and visualization
- Experiment management and result tracking system

### 🛡️ **Quality Assurance System**
- Data health checks prevent training issues
- Multiple validations ensure experiment reliability
- Comprehensive error handling and recovery mechanisms

### ⚡ **Efficient Development Experience**
- Real-time training monitoring and progress display
- Rich visualization and analysis tools
- Modular design for easy extension and maintenance

## Key Implementation Highlights

### **Video-Aware Data Splitting**
- Prevents video-level data leakage
- Intelligent video grouping and frame allocation
- Supports multiple splitting strategies

### **Knowledge Distillation Framework**
- Complete Teacher-Student architecture
- Feature-level and output-level distillation
- Temperature scaling and weight balancing

### **Advanced Loss Functions**
- CE+Dice combined loss
- Focal Loss integration
- Label smoothing techniques
- Automatic class weight calculation

### **Hybrid Evaluation Strategy**
- Dual Loss and mIoU determination
- Early stopping threshold control
- Training stability assurance

### **Comprehensive Health Check System**
- Label distribution comprehensive validation
- Mapping chain integrity verification
- Palette consistency validation
- Single-frame mapping visual inspection