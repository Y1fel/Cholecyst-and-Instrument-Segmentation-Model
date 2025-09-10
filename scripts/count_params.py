# count_params.py
import torch
from src.models.model_zoo import build_model  # 修正函数名

def count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params(Total)={total/1e6:.2f}M, Trainable={trainable/1e6:.2f}M")

print("Building teacher model...")
teacher = build_model(model_name="unet_plus_plus", stage="offline", in_ch=3, num_classes=3)
print("Teacher model:", end=" ")
count(teacher)

print("\nBuilding student model...")
student = build_model(model_name="adaptive_unet", stage="online", in_ch=3, num_classes=3)
print("Student model:", end=" ")
count(student)
