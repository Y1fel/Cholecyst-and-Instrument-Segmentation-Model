# src/models/model_zoo.py

from typing import Literal
from .baseline.unet_min import UNetMin

try:
    from .online.mobile_unet import MobileUNet
except Exception:
    MobileUNet = None

try:
    from .online.adaptive_unet import AdaptiveUNet
except Exception:
    AdaptiveUNet = None


def build_model(
        model_name: Literal['unet_min', 'mobile_unet', 'adaptive_unet'],
        num_classes: int,
        in_ch: int = 3,
        base: int = 32,
    ):

    out_ch = int(num_classes)
    print(f"🏗️  Building model: {model_name} (in_ch={in_ch}, num_classes={num_classes})")

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
