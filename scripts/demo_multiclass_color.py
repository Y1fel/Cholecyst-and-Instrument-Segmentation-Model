# scripts/demo_multiclass_color.py
# $env:PYTHONPATH="F:\Documents\Courses\CIS\Cholecyst-and-Instrument-Segmentation-Model"
import argparse, os, cv2, torch
import numpy as np
from src.models.baseline.unet_min import UNetMin
from src.viz.colorize import make_triplet, overlay
from src.common.constants import IGNORE_INDEX

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise FileNotFoundError(path)
    return img

def preprocess(img_bgr, size=512):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(img_rgb).permute(2,0,1).float() / 255.0
    x = x.unsqueeze(0)  # BxCxHxW
    return x

@torch.no_grad()
def infer(model, x):
    model.eval()
    logits = model(x)                  # BxKxHxW
    pred_id = logits.softmax(1).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred_id

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="原图路径")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--num_classes", type=int, default=5)
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--gt", type=str, default="")  # 可选：提供GT用于三联图
    ap.add_argument("--out", type=str, default="viz_triplet.png")
    args = ap.parse_args()

    img_bgr = load_image(args.img)
    x = preprocess(img_bgr, args.size)

    model = UNetMin(in_ch=3, num_classes=args.num_classes, base=32)
    if args.ckpt and os.path.isfile(args.ckpt):
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state["state_dict"] if "state_dict" in state else state)

    pred_id = infer(model, x)

    if args.gt and os.path.isfile(args.gt):
        gt = cv2.imread(args.gt, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (pred_id.shape[1], pred_id.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        gt = np.full_like(pred_id, IGNORE_INDEX)  # 没有GT也能跑

    triplet = make_triplet(img_bgr, gt, pred_id)
    cv2.imwrite(args.out, triplet)

if __name__ == "__main__":
    main()
