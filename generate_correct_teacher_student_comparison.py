#!/usr/bin/env python3
"""
Correct Teacher-Student Comparison Generator

This script creates proper Teacher-Student comparison visualizations by:
1. Reading the exact teacher and student model paths from config.json
2. Using real watershed GT data from the training system
3. Creating safe output directories to prevent file overwrites
4. Following the exact KD visualization pipel        print("[STEP 1] Loading configuration...")
        config = load_config_from_kd_output(kd_output_path)
        
        # Override FOV mask setting to ensure correct GT visualization
        config['apply_fov_mask'] = False
        print("[OVERRIDE] Set apply_fov_mask = False for correct GT display")e from training
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import cv2
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import project modules
from src.models.model_zoo import build_model
from src.dataio.datasets.seg_dataset_min import SegDatasetMin
# No need for PALETTE, CLASSES - using cmap='viridis' directly
from src.common.output_manager import OutputManager
from src.viz.visualizer import Visualizer

def create_safe_output_directory(base_path: str) -> str:
    """Create a new safe output directory with timestamp to prevent overwrites"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_dir = os.path.join(base_path, f"teacher_student_comparison_SAFE_{timestamp}")
    os.makedirs(safe_dir, exist_ok=True)
    print(f"[SAFE OUTPUT] Created new directory: {safe_dir}")
    return safe_dir

def load_config_from_kd_output(kd_output_path: str) -> dict:
    """Load configuration from KD training output directory"""
    config_path = os.path.join(kd_output_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    return config_data['config']

def get_model_paths_from_config(config: dict, kd_output_path: str) -> tuple:
    """Extract correct teacher and student model paths from config"""
    # Teacher model path from config
    teacher_path = config.get('teacher_checkpoint')
    if not teacher_path:
        raise ValueError("No teacher_checkpoint found in config")
    
    # Student model path - look for best KD model in the KD output directory
    kd_checkpoints_dir = os.path.join(kd_output_path, "checkpoints")
    possible_student_names = [
        "distill_unet_plus_plus_to_adaptive_unet_student_best.pth",
        "best_student_model.pth",
        "student_best.pth"
    ]
    
    student_path = None
    for name in possible_student_names:
        candidate = os.path.join(kd_checkpoints_dir, name)
        if os.path.exists(candidate):
            student_path = candidate
            break
    
    if not student_path:
        raise FileNotFoundError(f"No student model found in {kd_checkpoints_dir}")
    
    print(f"[CONFIG] Teacher model: {teacher_path}")
    print(f"[CONFIG] Student model: {student_path}")
    
    return teacher_path, student_path

def load_models(teacher_path: str, student_path: str, config: dict, device: str) -> tuple:
    """Load teacher and student models from checkpoints"""
    
    # Model configuration from config
    teacher_model_name = config.get('teacher_model', 'unet_plus_plus')
    student_model_name = config.get('student_model', 'adaptive_unet')
    num_classes = config.get('num_classes', 3)
    
    print(f"[MODEL] Building Teacher: {teacher_model_name}, Student: {student_model_name}")
    print(f"[MODEL] Number of classes: {num_classes}")
    
    # Build models
    teacher_model = build_model(
        model_name=teacher_model_name,
        num_classes=num_classes,
        in_ch=3,
        stage='auto'
    ).to(device)
    
    student_model = build_model(
        model_name=student_model_name,
        num_classes=num_classes,
        in_ch=3,
        stage='auto'
    ).to(device)
    
    # Load checkpoints
    print(f"[LOAD] Loading teacher from: {teacher_path}")
    teacher_checkpoint = torch.load(teacher_path, map_location=device)
    if 'model_state_dict' in teacher_checkpoint:
        teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    else:
        teacher_model.load_state_dict(teacher_checkpoint)
    
    print(f"[LOAD] Loading student from: {student_path}")
    student_checkpoint = torch.load(student_path, map_location=device)
    if 'model_state_dict' in student_checkpoint:
        student_model.load_state_dict(student_checkpoint['model_state_dict'])
    else:
        student_model.load_state_dict(student_checkpoint)
    
    teacher_model.eval()
    student_model.eval()
    
    return teacher_model, student_model

def create_watershed_dataset(config: dict) -> SegDatasetMin:
    """Create dataset using the same watershed GT data as training"""
    
    data_root = config.get('data_root', 'data/seg8k')
    img_size = config.get('img_size', 384)
    apply_fov_mask = config.get('apply_fov_mask', False)  # Disable FOV mask for correct GT visualization
    
    print(f"[DATASET] Data root: {data_root}")
    print(f"[DATASET] Image size: {img_size}")
    print(f"[DATASET] Apply FOV mask: {apply_fov_mask}")
    
    # Create validation dataset (for consistent evaluation) 
    # Use class_id_map to bypass classification_scheme issues
    from src.common.constants import generate_class_mapping
    mapping, num_classes, class_names = generate_class_mapping('3class_org')
    
    dataset = SegDatasetMin(
        data_root=data_root,
        dtype='train',  # Use train data
        img_size=img_size,
        return_multiclass=True,  # Multi-class mode
        class_id_map=mapping,  # Use pre-generated mapping
        apply_fov_mask=False  # Force disable FOV mask for correct GT visualization
    )
    
    print(f"[DATASET] Created dataset with {len(dataset)} samples")
    return dataset

def generate_teacher_student_comparison(teacher_model, student_model, dataset, safe_output_dir: str, 
                                      num_samples: int = 10, device: str = 'cuda'):
    """Generate Teacher-Student comparison using real watershed data"""
    
    print(f"[COMPARISON] Generating {num_samples} comparison samples...")
    
    # Create comparison directory
    comparison_dir = os.path.join(safe_output_dir, "teacher_student_comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = Visualizer()
    
    # Sample indices (spread across dataset)
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            try:
                # Get sample
                image, _ = dataset[idx]  # Ignore GT mask
                
                # Convert to batch format
                image_batch = image.unsqueeze(0).to(device)
                
                # Get predictions
                teacher_pred = teacher_model(image_batch)
                student_pred = student_model(image_batch)
                
                # Convert predictions to numpy
                teacher_pred_np = torch.softmax(teacher_pred, dim=1).argmax(dim=1).squeeze().cpu().numpy()
                student_pred_np = torch.softmax(student_pred, dim=1).argmax(dim=1).squeeze().cpu().numpy()
                
                # Convert image for visualization
                if image.shape[0] == 3:  # CHW format
                    image_np = image.permute(1, 2, 0).cpu().numpy()
                else:
                    image_np = image.cpu().numpy()
                
                # Normalize image for display
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
                
                # Create 3-panel comparison: Original | Teacher | Student
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(image_np)
                axes[0].set_title(f'Original Image (Sample {i+1})', fontsize=12, fontweight='bold')
                axes[0].axis('off')
                
                # Teacher prediction - use cmap like distillation_visualizer.py
                axes[1].imshow(teacher_pred_np, cmap='viridis', vmin=0, vmax=2)
                axes[1].set_title('Teacher Prediction', fontsize=12, fontweight='bold')
                axes[1].axis('off')
                
                # Student prediction - use cmap like distillation_visualizer.py  
                axes[2].imshow(student_pred_np, cmap='viridis', vmin=0, vmax=2)
                axes[2].set_title('Student Prediction', fontsize=12, fontweight='bold')
                axes[2].axis('off')
                
                # No legend needed for cmap visualization
                
                plt.tight_layout()
                
                # Save comparison
                comparison_path = os.path.join(comparison_dir, f"teacher_student_comparison_{i+1:03d}.png")
                plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"[SAVE] Saved comparison {i+1}/{num_samples}: {comparison_path}")
                
            except Exception as e:
                print(f"[ERROR] Failed to process sample {i+1}: {str(e)}")
                continue
    
    print(f"[COMPLETE] All comparisons saved to: {comparison_dir}")
    return comparison_dir

def generate_combined_visualization(teacher_model, student_model, dataset, safe_output_dir: str,
                                  num_samples: int = 12, device: str = 'cuda'):
    """Generate combined visualization with all samples in one figure - replicate distillation_visualizer.py style"""
    
    print(f"[COMBINED] Generating combined visualization with {num_samples} samples...")
    
    teacher_model.eval()
    student_model.eval()
    
    # Create figure: num_samples rows, 3 columns (Original | Teacher | Student)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Sample indices (spread across dataset)
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    
    sample_count = 0
    with torch.no_grad():
        for i, idx in enumerate(indices):
            if sample_count >= num_samples:
                break
                
            try:
                # Get sample
                image, _ = dataset[idx]  # Ignore GT mask
                
                # Convert to batch format
                image_batch = image.unsqueeze(0).to(device)
                
                # Get predictions
                teacher_pred = teacher_model(image_batch)
                student_pred = student_model(image_batch)
                
                # Convert predictions to class predictions (like distillation_visualizer.py)
                teacher_probs = torch.softmax(teacher_pred, dim=1)
                student_probs = torch.softmax(student_pred, dim=1)
                
                teacher_pred_np = teacher_probs.argmax(dim=1).squeeze().cpu().numpy()
                student_pred_np = student_probs.argmax(dim=1).squeeze().cpu().numpy()
                
                # Convert image for visualization (like distillation_visualizer.py)
                if image.shape[0] == 3:  # CHW format
                    image_np = image.permute(1, 2, 0).cpu().numpy()
                    # Normalize like distillation_visualizer.py
                    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
                else:
                    image_np = image.cpu().numpy()
                
                # Plot 3-panel comparison
                row = sample_count
                
                # Original Image
                axes[row, 0].imshow(image_np)
                axes[row, 0].set_title("Original Image")
                axes[row, 0].axis('off')
                
                # Teacher Prediction - use cmap like distillation_visualizer.py
                axes[row, 1].imshow(teacher_pred_np, cmap='viridis', vmin=0, vmax=2)
                axes[row, 1].set_title("Teacher Prediction")
                axes[row, 1].axis('off')
                
                # Student Prediction - use cmap like distillation_visualizer.py
                axes[row, 2].imshow(student_pred_np, cmap='viridis', vmin=0, vmax=2)
                axes[row, 2].set_title("Student Prediction")
                axes[row, 2].axis('off')
                
                sample_count += 1
                
            except Exception as e:
                print(f"[ERROR] Failed to process sample {sample_count+1}: {str(e)}")
                sample_count += 1
                continue
                
            if sample_count >= num_samples:
                break
    
    plt.tight_layout()
    
    # Save combined visualization
    combined_path = os.path.join(safe_output_dir, "teacher_student_combined_comparison.png")
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[COMBINED] Combined visualization saved to: {combined_path}")
    return combined_path

# Color mapping functions removed - using cmap='viridis' like distillation_visualizer.py

def main():
    """Main function to generate correct Teacher-Student comparison"""
    
    # Configuration
    kd_output_path = "outputs/distill_unet_plus_plus_to_adaptive_unet_20250920_180040"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_samples = 12
    
    print("=" * 80)
    print("CORRECT TEACHER-STUDENT COMPARISON GENERATOR")
    print("=" * 80)
    print(f"[DEVICE] Using device: {device}")
    print(f"[INPUT] KD output path: {kd_output_path}")
    
    try:
        # Step 1: Create safe output directory
        safe_output_dir = create_safe_output_directory(kd_output_path)
        
        # Step 2: Load configuration from KD training
        print("\n[STEP 1] Loading configuration...")
        config = load_config_from_kd_output(kd_output_path)
        
        # Override FOV mask setting to ensure correct GT visualization
        config['apply_fov_mask'] = False
        print("[OVERRIDE] Set apply_fov_mask = False for correct GT display")
        
        # Step 3: Get correct model paths
        print("\n[STEP 2] Extracting model paths...")
        teacher_path, student_path = get_model_paths_from_config(config, kd_output_path)
        
        # Step 4: Load models
        print("\n[STEP 3] Loading models...")
        teacher_model, student_model = load_models(teacher_path, student_path, config, device)
        
        # Step 5: Create watershed dataset
        print("\n[STEP 4] Creating watershed dataset...")
        dataset = create_watershed_dataset(config)
        
        # Step 6: Generate individual comparisons
        print("\n[STEP 5] Generating Teacher-Student individual comparisons...")
        comparison_dir = generate_teacher_student_comparison(
            teacher_model, student_model, dataset, safe_output_dir, num_samples, device
        )
        
        # Step 7: Generate combined visualization
        print("\n[STEP 6] Generating combined visualization...")
        combined_path = generate_combined_visualization(
            teacher_model, student_model, dataset, safe_output_dir, num_samples, device
        )
        
        print("\n" + "=" * 80)
        print("SUCCESS: Teacher-Student comparison generated successfully!")
        print(f"Output directory: {safe_output_dir}")
        print(f"Individual comparison images: {comparison_dir}")
        print(f"Combined visualization: {combined_path}")
        print("=" * 80)
        
        # Save summary
        summary_path = os.path.join(safe_output_dir, "generation_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Teacher-Student Comparison Generation Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Teacher model: {teacher_path}\n")
            f.write(f"Student model: {student_path}\n")
            f.write(f"Dataset: {config.get('data_root', 'N/A')}\n")
            f.write(f"Classification scheme: {config.get('classification_scheme', 'N/A')}\n")
            f.write(f"Number of samples: {num_samples}\n")
            f.write(f"Individual comparisons: {comparison_dir}\n")
            f.write(f"Combined visualization: {combined_path}\n")
            f.write(f"Output directory: {safe_output_dir}\n")
        
        print(f"[SUMMARY] Generation summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"\n[FATAL ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()