"""
最小离线训练脚本
只训练1个epoch并保存模型，用于快速测试和原型验证
"""
import os, argparse, yaml, torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataio.datasets.seg_dataset_min import SegDatasetMin
from src.models.baseline.unet_min import UNetMin

#参数配置
def parse_args():
    p = argparse.ArgumentParser("Minimal offline training (1 epoch, save weights).")
    p.add_argument("--cfg", type=str, default=None, help="Optional YAML config")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=6)
    #训练次数
    p.add_argument("--epochs", type=int, default=1)
    #学习率
    p.add_argument("--lr", type=float, default=3e-4)
    #分类数
    p.add_argument("--num_classes", type=int, default=2)
    #权重输出
    p.add_argument("--save_path", type=str, default="checkpoints/baseline_offline_min.pth")
    #并行数量
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()

def load_cfg(args):
    if args.cfg is None:
        return vars(args)
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v
    return cfg

def main():
    args = parse_args()
    cfg = load_cfg(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(cfg["save_path"]), exist_ok=True)

    #初始化
    train_dataset = SegDatasetMin(cfg["data_root"], cfg["split"], cfg["img_size"])
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True)

    out_ch = 1 if cfg["num_classes"] == 2 else cfg["num_classes"]
    model = UNetMin(in_ch=3, num_classes=out_ch, base=32).to(device)
    criterion = nn.BCEWithLogitsLoss() if cfg["num_classes"] == 2 else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    #训练
    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item() * images.size(0)
        print(f"[epoch {epoch+1}] loss={running/len(train_dataset):.4f}")

    torch.save({"state_dict": model.state_dict()}, cfg["save_path"])
    print("Saved:", cfg["save_path"])

if __name__ == "__main__":
    main()