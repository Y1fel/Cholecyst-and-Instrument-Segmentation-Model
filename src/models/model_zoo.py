# src/models/model_zoo.py

from typing import Literal

# Baseline model import
try:
    from .baseline.unet_min import UNetMin
except ImportError:
    UNetMin = None

# Online models import  
try:
    from .online.mobile_unet import MobileUNet
    from .online.adaptive_unet import AdaptiveUNet
except ImportError:
    MobileUNet = None
    AdaptiveUNet = None

# Offline models import
try:
    from .offline.unet_plus_plus import UNetPlusPlus
except ImportError:
    UNetPlusPlus = None

try:
    from .offline.deeplabv3_plus import DeepLabV3Plus
except ImportError:
    DeepLabV3Plus = None

try:
    from .offline.hrnet import HRNet
except ImportError:
    HRNet = None

# 模型分类定义
OFFLINE_MODELS = ['unet_min', 'unet_plus_plus', 'deeplabv3_plus', 'hrnet', 'adaptive_unet']  # 添加adaptive_unet支持offline训练
ONLINE_MODELS = ['mobile_unet', 'adaptive_unet']  # adaptive_unet同时支持两种阶段
ALL_MODELS = OFFLINE_MODELS + ONLINE_MODELS

def build_model(
        model_name: Literal['unet_min', 'unet_plus_plus', 'deeplabv3_plus', 'hrnet', 'mobile_unet', 'adaptive_unet'],
        num_classes: int,
        in_ch: int = 3,
        base: int = 32,
        stage: Literal['offline', 'online', 'auto'] = 'auto'
    ):
    """
    构建模型
    Args:
        stage: 'offline'只允许离线模型, 'online'只允许在线模型, 'auto'自动判断
    """

    out_ch = int(num_classes)
    print(f"-- Building model: {model_name} (stage={stage}, in_ch={in_ch}, num_classes={num_classes}) --")

    # 阶段检查
    if stage == 'offline' and model_name not in OFFLINE_MODELS:
        print(f"--! WARNING !--  {model_name} is not an offline model, falling back to unet_min")
        model_name = 'unet_min'
    elif stage == 'online' and model_name not in ONLINE_MODELS:
        print(f"--! WARNING !--  {model_name} is not an online model, falling back to mobile_unet")
        model_name = 'mobile_unet'

    # if model_name == 'unet_min':
    #     print(f"[CHECKED] Using UNetMin with {out_ch} output channels")
    #     return UNetMin(in_ch = in_ch, num_classes = out_ch, base = base)
    
    # Baseline model
    # UNetMin
    if model_name == 'unet_min':
        print(f"[CHECKED] Using UNetMin with {out_ch} output channels")
        return UNetMin(in_ch = in_ch, num_classes = out_ch, base = base)
    
    # Offline models
    # UNet++
    if model_name == 'unet_plus_plus':
        if UNetPlusPlus is not None:
            print(f"[CHECKED] Using UNet++ with {out_ch} output channels")
            return UNetPlusPlus(in_ch=in_ch, num_classes=out_ch, base_ch=base)
        else:
            print(f"[FAILED] UNet++ not available, falling back to UNetMin")
            return UNetMin(in_ch=in_ch, num_classes=out_ch, base=base)
    
    # DeepLabV3+
    if model_name == 'deeplabv3_plus':
        if DeepLabV3Plus is not None:
            print(f"[CHECKED] Using DeepLabV3+ with {out_ch} output channels")
            return DeepLabV3Plus(in_channels=in_ch, num_classes=out_ch)
        else:
            print(f"[FAILED] DeepLabV3+ not available, falling back to UNetMin")
            return UNetMin(in_ch=in_ch, num_classes=out_ch, base=base)
    
    # HRNet
    if model_name == 'hrnet':
        if HRNet is not None:
            print(f"[CHECKED] Using HRNet with {out_ch} output channels")
            return HRNet(in_channels=in_ch, num_classes=out_ch)
        else:
            print(f"[FAILED] HRNet not available, falling back to UNetMin")
            return UNetMin(in_ch=in_ch, num_classes=out_ch, base=base)
    
    # Online models
    # MobileUNet
    if model_name == 'mobile_unet' and MobileUNet is not None:
        print(f"[CHECKED] Using MobileUNet with {out_ch} output channels")
        return MobileUNet(n_channels = in_ch, n_classes = out_ch)
    elif model_name == 'mobile_unet':
        print(f"[FAILED] MobileUNet not available, falling back to UNetMin")
        return UNetMin(in_ch = in_ch, num_classes = out_ch, base = base)

    # AdaptiveUNet
    if model_name == 'adaptive_unet' and AdaptiveUNet is not None:
        print(f"[CHECKED] Using AdaptiveUNet with {out_ch} output channels")
        return AdaptiveUNet(in_channels = in_ch, out_channels = out_ch, init_features = base)
    elif model_name == 'adaptive_unet':
        print(f"[FAILED] AdaptiveUNet not available, falling back to UNetMin")
        return UNetMin(in_ch = in_ch, num_classes = out_ch, base = base)

    
    raise ValueError(f"Unknown model name: {model_name}")

def get_model_config(model_name: str) -> dict:
    # gain model's default config
    config = {
        # Offline models
        'unet_min': {
            'description': 'Basic UNet model',
            'params': {'base': 32},
            'complexity': 'low',
        },
        'unet_plus_plus': {
            'description': 'UNet++ model with nested architecture',
            'params': {'deep_supervision': True, 'base': 16},
            'complexity': 'medium-high to high',
        },
        'deeplabv3_plus': {
            'description': 'DeepLabV3+ model with atrous convolutions and encoder-decoder structure',
            'params': {'backbone': 'resnet50', 'output_stride': 16},
            'complexity': 'high',
        },
        'hrnet': {
            'description': 'High-Resolution Network (HRNet) model',
            'params': {'width': 18, 'blocks': [1, 4, 4, 4]},
            'complexity': 'high',
        },
        # Online models
        'mobile_unet': {
            'description': 'Lightweight MobileNet-based UNet',
            'params': {'base': 32},
            'complexity': 'low',
        },
        'adaptive_unet': {
            'description': 'Adaptive UNet with dynamic features',
            'params': {'init_features': 32},
            'complexity': 'medium',
        },
    }
    return config.get(model_name, {})

def list_available_models(stage: str = 'all') -> dict:
    # list available models
    model_status = {}

    models_to_check = {
        'offline': [('unet_min', UNetMin), ('unet_plus_plus', UNetPlusPlus), ('deeplabv3_plus', DeepLabV3Plus), ('hrnet', HRNet)],
        'online': [('mobile_unet', MobileUNet), ('adaptive_unet', AdaptiveUNet)],
    }

    for stage_name, models in models_to_check.items():
        if stage in ['all', stage_name]:
            model_status[stage_name] = {}
            for model_name, model_class in models:
                model_status[stage_name][model_name] = {
                    'available': model_class is not None,
                    'config': get_model_config(model_name)
                }

    return model_status

def print_model_summary():
    print("=+" * 50)
    print("Model Zoo Summary:")
    print("=+" * 50)

    status = list_available_models()

    for stage, models in status.items():
        print(f"\n-- {stage.upper()} MODELS --")
        for model_name, model_info in models.items():
            status_icon = "[CHECKED]" if model_info['available'] else "[FAILED]"
            complexity = model_info['config'].get('complexity', 'N/A')
            description = model_info['config'].get('description', 'No description')
            print(f"   {status_icon} {model_name:15} | {complexity:6} | {description}")
    print("\n" + "=" * 60)