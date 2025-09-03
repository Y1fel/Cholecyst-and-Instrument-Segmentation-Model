import os
import torch
from torch.utils.data import DataLoader
from src.models.model_zoo import build_model
from src.dataio.datasets.seg_dataset_min import SegDatasetMin
from src.common.pseudo_label_quality import quality_filter
import numpy as np
import argparse

# 伪标签生成脚本

def generate_pseudo_labels(model_path, data_root, output_path, model_name='unet_min', num_classes=10, img_size=512, batch_size=8, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model = build_model(model_name, num_classes=num_classes, in_ch=3, stage="offline")
    model = model.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载数据集
    dataset = SegDatasetMin(data_root, dtype="test", img_size=img_size, return_multiclass=(num_classes>2))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds = []
    all_probs = []
    all_names = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            # 质控筛选
            mask = quality_filter(probs, preds)
            preds = preds[mask]
            all_preds.append(preds)
            all_probs.append(probs[mask])
            # 假设dataset返回文件名或索引
            if hasattr(dataset, 'get_names'):
                names = dataset.get_names()
                all_names.extend(np.array(names)[mask])

    if all_preds:
        all_preds = np.concatenate(all_preds, axis=0)
        np.save(output_path, all_preds)
        print(f"Pseudo labels saved to {output_path}")
    else:
        print("No pseudo labels passed quality control.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate pseudo labels using trained model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to best model .pth file')
    parser.add_argument('--data_root', type=str, required=True, help='Root path of target dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Output npy file for pseudo labels')
    parser.add_argument('--model_name', type=str, default='unet_min')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    generate_pseudo_labels(args.model_path, args.data_root, args.output_path, args.model_name, args.num_classes, args.img_size, args.batch_size)
