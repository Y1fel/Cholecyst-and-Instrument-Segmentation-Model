import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import pandas as pd
import cv2
from torch.utils.data import DataLoader

class DistillationVisualizer:
    """知识蒸馏专用可视化器"""
    
    def __init__(self, save_dir: str, device: str = "cuda"):
        self.save_dir = save_dir
        self.device = device
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建子目录
        self.distill_viz_dir = os.path.join(save_dir, "distillation_analysis")
        os.makedirs(self.distill_viz_dir, exist_ok=True)
    
    def visualize_prediction_comparison(self, 
                                      teacher_model,
                                      student_model,
                                      val_loader: DataLoader,
                                      num_samples: int = 6,
                                      save_name: str = "teacher_student_comparison.png"):
        """可视化Teacher和Student的预测对比"""
        print(f"Generating Teacher-Student prediction comparison...")
        
        teacher_model.eval()
        student_model.eval()
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        sample_count = 0
        with torch.no_grad():
            for images, masks in val_loader:
                if sample_count >= num_samples:
                    break
                    
                images = images.to(self.device)
                
                # Teacher和Student预测
                teacher_outputs = teacher_model(images)
                student_outputs = student_model(images)
                
                # 检查Teacher输出是否有效
                if torch.all(teacher_outputs == 0) or torch.isnan(teacher_outputs).any():
                    print(f"⚠️ WARNING: Teacher model outputs are invalid (all zeros or NaN)")
                    print(f"Teacher output stats: min={teacher_outputs.min():.4f}, max={teacher_outputs.max():.4f}")
                
                # 检查Student输出是否有效
                if torch.all(student_outputs == 0) or torch.isnan(student_outputs).any():
                    print(f"⚠️ WARNING: Student model outputs are invalid (all zeros or NaN)")
                    print(f"Student output stats: min={student_outputs.min():.4f}, max={student_outputs.max():.4f}")
                
                # 转换为预测类别
                teacher_probs = torch.softmax(teacher_outputs, dim=1)
                student_probs = torch.softmax(student_outputs, dim=1)
                
                teacher_preds = teacher_probs.argmax(dim=1)
                student_preds = student_probs.argmax(dim=1)
                
                # 调试输出（前几个batch）
                if sample_count == 0:
                    print(f"[DEBUG] Teacher probs range: {teacher_probs.min():.4f} - {teacher_probs.max():.4f}")
                    print(f"[DEBUG] Student probs range: {student_probs.min():.4f} - {student_probs.max():.4f}")
                    print(f"[DEBUG] Teacher pred unique: {torch.unique(teacher_preds)}")
                    print(f"[DEBUG] Student pred unique: {torch.unique(student_preds)}")
                
                for i in range(min(images.shape[0], num_samples - sample_count)):
                    # 原图
                    img = images[i].cpu().permute(1, 2, 0).numpy()
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # 防止除零
                    
                    # Ground Truth
                    gt = masks[i].cpu().numpy()
                    
                    # Teacher预测
                    teacher_pred = teacher_preds[i].cpu().numpy()
                    
                    # Student预测
                    student_pred = student_preds[i].cpu().numpy()
                    
                    row = sample_count + i
                    
                    # 绘制
                    axes[row, 0].imshow(img)
                    axes[row, 0].set_title("Original Image")
                    axes[row, 0].axis('off')
                    
                    axes[row, 1].imshow(gt, cmap='viridis', vmin=0, vmax=2)
                    axes[row, 1].set_title("Ground Truth")
                    axes[row, 1].axis('off')
                    
                    axes[row, 2].imshow(teacher_pred, cmap='viridis', vmin=0, vmax=2)
                    axes[row, 2].set_title("Teacher Prediction")
                    axes[row, 2].axis('off')
                    
                    axes[row, 3].imshow(student_pred, cmap='viridis', vmin=0, vmax=2)
                    axes[row, 3].set_title("Student Prediction")
                    axes[row, 3].axis('off')
                
                sample_count += images.shape[0]
                if sample_count >= num_samples:
                    break
        
        plt.tight_layout()
        save_path = os.path.join(self.distill_viz_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Teacher-Student comparison saved to: {save_path}")
    
    def visualize_knowledge_transfer(self,
                                   teacher_model,
                                   student_model, 
                                   val_loader: DataLoader,
                                   temperature: float = 4.0,
                                   max_samples: int = 1000,  # 限制最大样本数
                                   save_name: str = "knowledge_transfer_analysis.png"):
        """可视化知识传递过程"""
        print(f"Processing Knowledge Transfer Analysis (max {max_samples} samples)...")
        
        teacher_model.eval()
        student_model.eval()
        
        # 收集软标签和预测的差异数据
        kl_divergences = []
        prediction_agreements = []
        confidence_differences = []
        
        total_batches = len(val_loader)
        processed_samples = 0
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(val_loader):
                # 检查是否已收集足够样本
                if processed_samples >= max_samples:
                    break
                    
                images = images.to(self.device)
                batch_size = images.shape[0]
                
                # 计算ETA
                if batch_idx > 0:
                    elapsed = time.time() - start_time
                    samples_per_sec = processed_samples / elapsed
                    remaining_samples = min(max_samples - processed_samples, 
                                          (total_batches - batch_idx) * batch_size)
                    eta_seconds = remaining_samples / max(samples_per_sec, 1)
                    eta_str = f"{int(eta_seconds//60):02d}:{int(eta_seconds%60):02d}"
                else:
                    eta_str = "calculating..."
                
                # 单行进度显示
                progress = min(processed_samples / max_samples * 100, 100)
                print(f"\rKnowledge Transfer Analysis [{batch_idx+1}/{total_batches}] "
                      f"| Samples: {processed_samples}/{max_samples} ({progress:.1f}%) "
                      f"| ETA: {eta_str}", end="", flush=True)
                
                teacher_outputs = teacher_model(images)
                student_outputs = student_model(images)
                
                # 计算软标签
                teacher_soft = torch.softmax(teacher_outputs / temperature, dim=1)
                student_soft = torch.softmax(student_outputs / temperature, dim=1)
                
                # KL散度（批量计算，减少内存占用）
                kl_div = torch.sum(teacher_soft * torch.log(teacher_soft / (student_soft + 1e-8)), dim=1)
                # 随机采样减少内存占用
                sample_mask = torch.randperm(kl_div.numel())[:min(1000, kl_div.numel())]
                kl_divergences.extend(kl_div.flatten()[sample_mask].cpu().numpy())
                
                # 预测一致性（采样）
                teacher_preds = teacher_outputs.argmax(dim=1)
                student_preds = student_outputs.argmax(dim=1)
                agreement = (teacher_preds == student_preds).float()
                prediction_agreements.extend(agreement.flatten()[sample_mask].cpu().numpy())
                
                # 置信度差异（采样）
                teacher_conf = torch.max(teacher_soft, dim=1)[0]
                student_conf = torch.max(student_soft, dim=1)[0]
                conf_diff = torch.abs(teacher_conf - student_conf)
                confidence_differences.extend(conf_diff.flatten()[sample_mask].cpu().numpy())
                
                processed_samples += batch_size
                
                # 定期清理GPU缓存
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        print()  # 换行
        print(f"Knowledge transfer analysis completed, processed {len(kl_divergences)} sample points")
        
        # 绘制分析图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # KL散度分布
        axes[0, 0].hist(kl_divergences, bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title("KL Divergence Distribution")
        axes[0, 0].set_xlabel("KL Divergence")
        axes[0, 0].set_ylabel("Frequency")
        
        # 预测一致性
        agreement_rate = np.mean(prediction_agreements)
        axes[0, 1].bar(['Agree', 'Disagree'], 
                      [agreement_rate, 1-agreement_rate],
                      color=['green', 'red'], alpha=0.7)
        axes[0, 1].set_title(f"Teacher-Student Agreement: {agreement_rate:.3f}")
        axes[0, 1].set_ylabel("Proportion")
        
        # 置信度差异
        axes[1, 0].hist(confidence_differences, bins=50, alpha=0.7, color='orange')
        axes[1, 0].set_title("Confidence Difference Distribution")
        axes[1, 0].set_xlabel("Confidence Difference")
        axes[1, 0].set_ylabel("Frequency")
        
        # KL散度 vs 预测一致性的关系
        # 将数据分bin来显示关系
        kl_binned = np.array(kl_divergences)
        agreement_binned = np.array(prediction_agreements)
        
        # 按KL散度排序并分组
        sorted_indices = np.argsort(kl_binned)
        bin_size = len(sorted_indices) // 10
        
        bin_centers = []
        bin_agreements = []
        
        for i in range(10):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < 9 else len(sorted_indices)
            bin_indices = sorted_indices[start_idx:end_idx]
            
            bin_centers.append(np.mean(kl_binned[bin_indices]))
            bin_agreements.append(np.mean(agreement_binned[bin_indices]))
        
        axes[1, 1].plot(bin_centers, bin_agreements, 'o-', color='purple')
        axes[1, 1].set_title("KL Divergence vs Agreement")
        axes[1, 1].set_xlabel("Mean KL Divergence")
        axes[1, 1].set_ylabel("Agreement Rate")
        
        plt.tight_layout()
        save_path = os.path.join(self.distill_viz_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Knowledge transfer analysis saved to: {save_path}")
        
        # 返回统计数据
        return {
            'mean_kl_divergence': np.mean(kl_divergences),
            'agreement_rate': agreement_rate,
            'mean_confidence_diff': np.mean(confidence_differences)
        }
    
    def create_distillation_summary_report(self,
                                         metrics_history: Dict[str, List],
                                         distillation_stats: Dict,
                                         teacher_model_name: str,
                                         student_model_name: str,
                                         save_name: str = "distillation_summary_report.png"):
        """创建蒸馏过程总结报告"""
        print(f"Generating distillation summary report...")
        
        fig = plt.figure(figsize=(16, 12))
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 训练损失曲线
        ax1 = fig.add_subplot(gs[0, :2])
        epochs = range(1, len(metrics_history['total_loss']) + 1)
        ax1.plot(epochs, metrics_history['total_loss'], 'b-', label='Total Loss', linewidth=2)
        ax1.plot(epochs, metrics_history['task_loss'], 'r--', label='Task Loss', linewidth=2)
        ax1.plot(epochs, metrics_history['distill_loss'], 'g:', label='Distillation Loss', linewidth=2)
        ax1.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 验证指标曲线
        ax2 = fig.add_subplot(gs[1, :2])
        if 'val_loss' in metrics_history:
            ax2.plot(epochs, metrics_history['val_loss'], 'purple', label='Val Loss', linewidth=2)
        if 'miou' in metrics_history:
            ax2_twin = ax2.twinx()
            ax2_twin.plot(epochs, metrics_history['miou'], 'orange', label='mIoU', linewidth=2)
            ax2_twin.set_ylabel('mIoU', color='orange')
            ax2_twin.legend(loc='upper right')
        
        ax2.set_title('Validation Metrics', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss', color='purple')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. 蒸馏统计信息
        ax3 = fig.add_subplot(gs[0, 2])
        stats_text = f"""
Distillation Statistics:

Teacher: {teacher_model_name}
Student: {student_model_name}

Mean KL Divergence: {distillation_stats.get('mean_kl_divergence', 0):.4f}
Agreement Rate: {distillation_stats.get('agreement_rate', 0):.3f}
Mean Confidence Diff: {distillation_stats.get('mean_confidence_diff', 0):.4f}

Final Val Loss: {metrics_history.get('val_loss', [0])[-1]:.4f}
Final mIoU: {metrics_history.get('miou', [0])[-1]:.4f}
        """
        ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
        ax3.set_title('Distillation Stats', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # 4. 损失组成饼图（最后一个epoch）
        ax4 = fig.add_subplot(gs[1, 2])
        if len(metrics_history['total_loss']) > 0:
            last_total = metrics_history['total_loss'][-1]
            last_task = metrics_history['task_loss'][-1]
            last_distill = metrics_history['distill_loss'][-1]
            
            # 计算百分比
            task_pct = (last_task / last_total) * 100
            distill_pct = (last_distill / last_total) * 100
            
            labels = ['Task Loss', 'Distillation Loss']
            sizes = [task_pct, distill_pct]
            colors = ['#ff9999', '#66b3ff']
            
            ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Final Loss Composition', fontsize=14, fontweight='bold')
        
        # 5. 训练进程表格
        ax5 = fig.add_subplot(gs[2, :])
        
        # 创建训练进程表格数据
        table_data = []
        for i, epoch in enumerate(epochs):
            row = [
                f"Epoch {epoch}",
                f"{metrics_history['total_loss'][i]:.4f}",
                f"{metrics_history['task_loss'][i]:.4f}",
                f"{metrics_history['distill_loss'][i]:.4f}",
                f"{metrics_history.get('val_loss', [0]*len(epochs))[i]:.4f}",
                f"{metrics_history.get('miou', [0]*len(epochs))[i]:.4f}"
            ]
            table_data.append(row)
        
        columns = ['Epoch', 'Total Loss', 'Task Loss', 'Distill Loss', 'Val Loss', 'mIoU']
        
        # 创建表格
        table = ax5.table(cellText=table_data, colLabels=columns, 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # 设置表格样式
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax5.set_title('Training Progress Summary', fontsize=14, fontweight='bold')
        ax5.axis('off')
        
        # 保存报告
        save_path = os.path.join(self.distill_viz_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Distillation summary report saved to: {save_path}")
        
        return save_path
    
    def save_distillation_metrics_table(self,
                                      metrics_history: Dict[str, List],
                                      distillation_stats: Dict,
                                      save_name: str = "distillation_metrics.csv"):
        """保存蒸馏指标表格"""
        print(f"Saving distillation metrics table...")
        
        # 创建DataFrame
        df_data = {
            'epoch': range(1, len(metrics_history['total_loss']) + 1),
            'total_loss': metrics_history['total_loss'],
            'task_loss': metrics_history['task_loss'],
            'distillation_loss': metrics_history['distill_loss']
        }
        
        # 添加验证指标（如果存在）
        if 'val_loss' in metrics_history:
            df_data['val_loss'] = metrics_history['val_loss']
        if 'miou' in metrics_history:
            df_data['miou'] = metrics_history['miou']
        if 'mdice' in metrics_history:
            df_data['mdice'] = metrics_history['mdice']
        if 'macc' in metrics_history:
            df_data['macc'] = metrics_history['macc']
        
        df = pd.DataFrame(df_data)
        
        # 保存CSV
        save_path = os.path.join(self.distill_viz_dir, save_name)
        df.to_csv(save_path, index=False)
        print(f"Distillation metrics table saved to: {save_path}")
        
        # 创建总结统计表格
        summary_data = {
            'Metric': [
                'Mean KL Divergence',
                'Teacher-Student Agreement Rate', 
                'Mean Confidence Difference',
                'Final Total Loss',
                'Final Task Loss',
                'Final Distillation Loss',
                'Final Validation Loss',
                'Final mIoU'
            ],
            'Value': [
                f"{distillation_stats.get('mean_kl_divergence', 0):.6f}",
                f"{distillation_stats.get('agreement_rate', 0):.4f}",
                f"{distillation_stats.get('mean_confidence_diff', 0):.6f}",
                f"{metrics_history['total_loss'][-1]:.6f}",
                f"{metrics_history['task_loss'][-1]:.6f}",
                f"{metrics_history['distill_loss'][-1]:.6f}",
                f"{metrics_history.get('val_loss', [0])[-1]:.6f}",
                f"{metrics_history.get('miou', [0])[-1]:.6f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.distill_viz_dir, "distillation_summary_stats.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Distillation summary stats saved to: {summary_path}")
        
        return save_path, summary_path
