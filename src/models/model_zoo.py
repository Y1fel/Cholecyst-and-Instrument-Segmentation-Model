# src/models/model_zoo.py

from typing import Literal

# Offline模型导入
try:
    from .baseline.unet_min import UNetMin
except ImportError:
    UNetMin = None

# Online模型导入  
try:
    from .online.mobile_unet import MobileUNet
    from .online.adaptive_unet import AdaptiveUNet
except ImportError:
    MobileUNet = None
    AdaptiveUNet = None

# 模型分类定义
OFFLINE_MODELS = ['unet_min']
ONLINE_MODELS = ['mobile_unet', 'adaptive_unet']
ALL_MODELS = OFFLINE_MODELS + ONLINE_MODELS

def build_model(
        model_name: Literal['unet_min', 'mobile_unet', 'adaptive_unet'],
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
        print(f"⚠️  {model_name} is not an offline model, falling back to unet_min")
        model_name = 'unet_min'
    elif stage == 'online' and model_name not in ONLINE_MODELS:
        print(f"⚠️  {model_name} is not an online model, falling back to mobile_unet")
        model_name = 'mobile_unet'

    if model_name == 'unet_min':
        print(f"✅ Using UNetMin with {out_ch} output channels")
        return UNetMin(in_ch = in_ch, num_classes = out_ch, base = base)
    
    
    if model_name == 'mobile_unet' and MobileUNet is not None:
        print(f"✅ Using MobileUNet with {out_ch} output channels")
        return MobileUNet(n_channels = in_ch, n_classes = out_ch)
    elif model_name == 'mobile_unet':
        print(f"❌ MobileUNet not available, falling back to UNetMin")
        return UNetMin(in_ch = in_ch, num_classes = out_ch, base = base)


    if model_name == 'adaptive_unet' and AdaptiveUNet is not None:
        print(f"✅ Using AdaptiveUNet with {out_ch} output channels")
        return AdaptiveUNet(in_channels = in_ch, out_channels = out_ch, init_features = base)
    elif model_name == 'adaptive_unet':
        print(f"❌ AdaptiveUNet not available, falling back to UNetMin")
        return UNetMin(in_ch = in_ch, num_classes = out_ch, base = base)

    
    raise ValueError(f"Unknown model name: {model_name}")
