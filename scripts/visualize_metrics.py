#!/usr/bin/env python3
"""
独立的模型训练结果可视化工具
功能：
1. 绘制完整的训练曲线（所有epochs）
2. 检测并标记resume点（训练中断恢复）
3. 显示配置信息和最佳模型性能
4. 保存所有可视化结果到指定目录

使用方法:
    # 自动检测resume点
    python scripts/visualize_metrics.py --output_dir outputs/unet_plus_plus_20250911_230321
    
    # 手动指定resume点（推荐）
    python scripts/visualize_metrics.py --output_dir outputs/unet_plus_plus_20250911_230321 --resume_epochs 4,11
    
    # 对于你的具体例子：3次epoch时中断，10epoch时中断，最终到20epoch
    python scripts/visualize_metrics.py --output_dir outputs/unet_plus_plus_20250911_230321 --resume_epochs 4,11
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
import shutil

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

class MetricsVisualizer:
    """训练结果可视化器"""
    
    def __init__(self, output_dir, resume_epochs=None):
        """
        初始化可视化器
        
        Args:
            output_dir: 模型输出目录路径
            resume_epochs: 手动指定的resume epochs列表，例如 [3, 7, 12]
        """
        self.output_dir = Path(output_dir)
        self.resume_epochs = resume_epochs or []
        
        # 检查路径是否存在
        if not self.output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {output_dir}")
        
        # 创建新的可视化目录
        date_str = datetime.now().strftime("%Y%m%d")
        self.vis_dir = self.output_dir / f"new_vis_date{date_str}"
        self.vis_dir.mkdir(exist_ok=True)
        
        print(f"✅ 可视化结果将保存到: {self.vis_dir}")
        
        # 加载数据
        self.load_data()
    
    def load_data(self):
        """加载所有需要的数据文件"""
        print("📊 加载训练数据...")
        
        # 加载metrics数据
        metrics_file = self.output_dir / "metrics.csv"
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
        
        self.metrics_df = pd.read_csv(metrics_file)
        self.metrics_df['timestamp'] = pd.to_datetime(self.metrics_df['timestamp'])
        
        print(f"   - 加载 {len(self.metrics_df)} 个epoch的训练数据")
        
        # 加载配置信息
        config_file = self.output_dir / "config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            print("   - 加载配置信息")
        else:
            self.config = {}
            print("   - ⚠️ 配置文件未找到")
        
        # 加载最佳模型信息
        best_model_file = self.output_dir / "best_model_info.json"
        if best_model_file.exists():
            with open(best_model_file, 'r', encoding='utf-8') as f:
                self.best_model_info = json.load(f)
            print("   - 加载最佳模型信息")
        else:
            self.best_model_info = {}
            print("   - ⚠️ 最佳模型信息未找到")
    
    def detect_resume_points(self):
        """
        自动检测训练中断恢复点
        通过分析时间戳间隔来识别可能的resume点
        """
        if len(self.resume_epochs) > 0:
            print(f"📍 使用手动指定的resume点: {self.resume_epochs}")
            # 验证手动指定的resume点是否有效
            valid_resume_points = []
            for resume_epoch in self.resume_epochs:
                if resume_epoch in self.metrics_df['epoch'].values:
                    valid_resume_points.append(resume_epoch)
                    print(f"   ✅ Resume点 Epoch {resume_epoch} 有效")
                else:
                    print(f"   ❌ Resume点 Epoch {resume_epoch} 无效（不存在于数据中）")
            return valid_resume_points
        
        print("🔍 自动检测训练中断恢复点...")
        
        # 计算相邻epoch之间的时间间隔
        time_diffs = self.metrics_df['timestamp'].diff()
        
        # 计算正常训练时间的统计信息（排除第一个NaN值和异常值）
        normal_diffs = time_diffs[1:].copy()
        
        # 使用IQR方法识别异常时间间隔
        q1 = normal_diffs.quantile(0.25)
        q3 = normal_diffs.quantile(0.75)
        iqr = q3 - q1
        median_time = normal_diffs.median()
        
        # 设置阈值：超过中位数的5倍或者Q3 + 3*IQR的较小值
        threshold = min(median_time * 5, q3 + 3 * iqr)
        
        print(f"   - 正常训练时间中位数: {median_time.total_seconds()/3600:.2f} 小时")
        print(f"   - 异常时间间隔阈值: {threshold.total_seconds()/3600:.2f} 小时")
        
        resume_points = []
        
        for i, diff in enumerate(time_diffs):
            if pd.notna(diff) and diff > threshold:
                resume_epoch = self.metrics_df.iloc[i]['epoch']
                resume_points.append(resume_epoch)
                time_gap = diff.total_seconds() / 3600  # 转换为小时
                prev_epoch = self.metrics_df.iloc[i-1]['epoch'] if i > 0 else 0
                print(f"   - 检测到resume点: Epoch {prev_epoch} → {resume_epoch} (间隔 {time_gap:.1f} 小时)")
        
        if not resume_points:
            print("   - 未检测到训练中断")
        
        return resume_points
    
    def parse_array_column(self, series):
        """解析字符串形式的数组列"""
        def parse_array(s):
            if isinstance(s, str):
                # 移除方括号并分割
                s = s.strip('[]')
                return [float(x.strip()) for x in s.split(',')]
            return s
        
        return series.apply(parse_array)
    
    def plot_training_curves(self):
        """绘制完整的训练曲线"""
        print("📈 绘制训练曲线...")
        
        resume_points = self.detect_resume_points()
        
        # 解析数组类型的列
        iou_per_class = self.parse_array_column(self.metrics_df['iou_per_class'])
        dice_per_class = self.parse_array_column(self.metrics_df['dice_per_class'])
        acc_per_class = self.parse_array_column(self.metrics_df['acc_per_class'])
        
        # 创建子图
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle(f'Training Metrics - {self.output_dir.name}', fontsize=16, fontweight='bold')
        
        # 1. Loss曲线
        ax = axes[0, 0]
        ax.plot(self.metrics_df['epoch'], self.metrics_df['train_loss'], 
               'b-', marker='o', linewidth=2, markersize=4, label='Train Loss')
        ax.plot(self.metrics_df['epoch'], self.metrics_df['val_loss'], 
               'r-', marker='s', linewidth=2, markersize=4, label='Val Loss')
        
        # 标记resume点
        for resume_epoch in resume_points:
            ax.axvline(x=resume_epoch, color='orange', linestyle='--', alpha=0.7, linewidth=2)
            ax.text(resume_epoch, ax.get_ylim()[1]*0.9, f'Resume\nEpoch {resume_epoch}', 
                   ha='center', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.7))
        
        ax.set_title('Training & Validation Loss', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. IoU曲线
        ax = axes[0, 1]
        ax.plot(self.metrics_df['epoch'], self.metrics_df['miou'], 
               'g-', marker='d', linewidth=2, markersize=4, label='Mean IoU')
        
        for resume_epoch in resume_points:
            ax.axvline(x=resume_epoch, color='orange', linestyle='--', alpha=0.7, linewidth=2)
        
        ax.set_title('Mean Intersection over Union (mIoU)', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mIoU')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Dice系数曲线
        ax = axes[1, 0]
        ax.plot(self.metrics_df['epoch'], self.metrics_df['mdice'], 
               'purple', marker='^', linewidth=2, markersize=4, label='Mean Dice')
        
        for resume_epoch in resume_points:
            ax.axvline(x=resume_epoch, color='orange', linestyle='--', alpha=0.7, linewidth=2)
        
        ax.set_title('Mean Dice Coefficient', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dice')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Accuracy曲线
        ax = axes[1, 1]
        ax.plot(self.metrics_df['epoch'], self.metrics_df['macc'], 
               'brown', marker='v', linewidth=2, markersize=4, label='Mean Accuracy')
        
        for resume_epoch in resume_points:
            ax.axvline(x=resume_epoch, color='orange', linestyle='--', alpha=0.7, linewidth=2)
        
        ax.set_title('Mean Accuracy', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 每类IoU曲线
        ax = axes[2, 0]
        class_names = ['Class 0', 'Class 1', 'Class 2']
        colors = ['red', 'green', 'blue']
        
        for i in range(3):
            class_ious = [iou_list[i] for iou_list in iou_per_class]
            ax.plot(self.metrics_df['epoch'], class_ious, 
                   color=colors[i], marker='o', linewidth=2, markersize=3, 
                   label=f'{class_names[i]} IoU')
        
        for resume_epoch in resume_points:
            ax.axvline(x=resume_epoch, color='orange', linestyle='--', alpha=0.7, linewidth=2)
        
        ax.set_title('Per-Class IoU', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('IoU')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. 每类Dice曲线
        ax = axes[2, 1]
        for i in range(3):
            class_dices = [dice_list[i] for dice_list in dice_per_class]
            ax.plot(self.metrics_df['epoch'], class_dices, 
                   color=colors[i], marker='s', linewidth=2, markersize=3, 
                   label=f'{class_names[i]} Dice')
        
        for resume_epoch in resume_points:
            ax.axvline(x=resume_epoch, color='orange', linestyle='--', alpha=0.7, linewidth=2)
        
        ax.set_title('Per-Class Dice Coefficient', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dice')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        curve_file = self.vis_dir / "training_curves_complete.png"
        plt.savefig(curve_file, dpi=300, bbox_inches='tight')
        print(f"   - 训练曲线保存到: {curve_file}")
        
        plt.show()
        
        return resume_points
    
    def plot_resume_analysis(self, resume_points):
        """绘制resume点分析图"""
        if not resume_points:
            print("📊 无resume点，跳过resume分析")
            return
        
        print("📊 绘制resume点分析...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Resume Points Analysis - {len(resume_points)} Resumes Detected', 
                    fontsize=14, fontweight='bold')
        
        # 1. 时间轴分析
        ax = axes[0, 0]
        timestamps = self.metrics_df['timestamp']
        epochs = self.metrics_df['epoch']
        
        ax.plot(timestamps, epochs, 'b-', marker='o', markersize=4, alpha=0.7)
        
        for resume_epoch in resume_points:
            resume_time = self.metrics_df[self.metrics_df['epoch'] == resume_epoch]['timestamp'].iloc[0]
            ax.axvline(x=resume_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
            ax.plot(resume_time, resume_epoch, 'ro', markersize=8)
        
        ax.set_title('Training Timeline with Resume Points')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Epoch')
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴时间显示
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. 训练间隔分析
        ax = axes[0, 1]
        time_diffs = self.metrics_df['timestamp'].diff()
        time_diffs_hours = time_diffs.dt.total_seconds() / 3600
        
        ax.bar(self.metrics_df['epoch'][1:], time_diffs_hours[1:], 
               color='lightblue', alpha=0.7, edgecolor='navy')
        
        for resume_epoch in resume_points:
            if resume_epoch > 1:
                idx = self.metrics_df[self.metrics_df['epoch'] == resume_epoch].index[0]
                if idx > 0:
                    ax.bar(resume_epoch, time_diffs_hours.iloc[idx], 
                          color='red', alpha=0.8, edgecolor='darkred')
        
        ax.set_title('Time Intervals Between Epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Hours')
        ax.grid(True, alpha=0.3)
        
        # 3. Resume前后性能对比
        ax = axes[1, 0]
        metrics_to_compare = ['val_loss', 'miou', 'mdice']
        x_pos = np.arange(len(metrics_to_compare))
        
        before_resume_avg = []
        after_resume_avg = []
        
        for metric in metrics_to_compare:
            before_vals = []
            after_vals = []
            
            for resume_epoch in resume_points:
                # Resume前的值（如果存在）
                before_idx = self.metrics_df[self.metrics_df['epoch'] == resume_epoch - 1].index
                if len(before_idx) > 0:
                    before_vals.append(self.metrics_df.loc[before_idx[0], metric])
                
                # Resume后的值
                after_idx = self.metrics_df[self.metrics_df['epoch'] == resume_epoch].index
                if len(after_idx) > 0:
                    after_vals.append(self.metrics_df.loc[after_idx[0], metric])
            
            before_resume_avg.append(np.mean(before_vals) if before_vals else 0)
            after_resume_avg.append(np.mean(after_vals) if after_vals else 0)
        
        width = 0.35
        ax.bar(x_pos - width/2, before_resume_avg, width, label='Before Resume', 
               color='lightcoral', alpha=0.8)
        ax.bar(x_pos + width/2, after_resume_avg, width, label='After Resume', 
               color='lightgreen', alpha=0.8)
        
        ax.set_title('Performance Before vs After Resume')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics_to_compare)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Resume统计信息
        ax = axes[1, 1]
        ax.axis('off')
        
        # 计算统计信息
        total_training_time = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 3600
        
        # 计算纯训练时间（排除中断时间）
        pure_training_time = 0
        for i in range(1, len(time_diffs_hours)):
            if not pd.isna(time_diffs_hours.iloc[i]):
                # 如果这个间隔不是resume间隔（即正常训练时间）
                epoch = self.metrics_df.iloc[i]['epoch']
                if epoch not in resume_points:
                    pure_training_time += time_diffs_hours.iloc[i]
        
        avg_epoch_time = pure_training_time / (len(epochs) - len(resume_points)) if len(epochs) > len(resume_points) else 0
        
        # 计算训练段落
        training_segments = []
        if resume_points:
            # 添加第一段
            start_epoch = 1
            for resume_epoch in resume_points:
                end_epoch = resume_epoch - 1
                if end_epoch >= start_epoch:
                    training_segments.append((start_epoch, end_epoch))
                start_epoch = resume_epoch
            
            # 添加最后一段
            if start_epoch <= epochs.iloc[-1]:
                training_segments.append((start_epoch, epochs.iloc[-1]))
        else:
            training_segments = [(1, epochs.iloc[-1])]
        
        resume_info = [
            f"Total Resumes: {len(resume_points)}",
            f"Resume Epochs: {', '.join(map(str, resume_points))}",
            f"Total Wall Time: {total_training_time:.1f} hours",
            f"Pure Training Time: {pure_training_time:.1f} hours",
            f"Average Epoch Time: {avg_epoch_time:.2f} hours",
            f"Training Segments: {len(training_segments)}",
        ]
        
        # 添加训练段落信息
        for i, (start, end) in enumerate(training_segments[:3]):  # 只显示前3个段落
            resume_info.append(f"  Segment {i+1}: Epoch {start}-{end}")
        
        if len(training_segments) > 3:
            resume_info.append(f"  ... and {len(training_segments)-3} more")
        
        if len(time_diffs_hours) > 1:
            longest_break = time_diffs_hours.max()
            resume_info.append(f"Longest Break: {longest_break:.1f} hours")
        
        y_pos = 0.95
        for info in resume_info:
            fontsize = 10 if info.startswith('  ') else 12
            weight = 'normal' if info.startswith('  ') else 'bold'
            ax.text(0.05, y_pos, info, fontsize=fontsize, fontweight=weight, 
                   transform=ax.transAxes)
            y_pos -= 0.08
        
        ax.set_title('Resume Statistics', fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图片
        resume_file = self.vis_dir / "resume_analysis.png"
        plt.savefig(resume_file, dpi=300, bbox_inches='tight')
        print(f"   - Resume分析保存到: {resume_file}")
        
        plt.show()
    
    def create_config_summary(self):
        """创建配置信息摘要"""
        print("⚙️ 生成配置信息摘要...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))
        fig.suptitle(f'Model Configuration Summary - {self.output_dir.name}', 
                    fontsize=14, fontweight='bold')
        
        # 1. 主要配置信息
        ax = axes[0]
        ax.axis('off')
        
        if 'config' in self.config:
            config_data = self.config['config']
            
            main_config = [
                f"Model: {config_data.get('model', 'N/A')}",
                f"Image Size: {config_data.get('img_size', 'N/A')}",
                f"Batch Size: {config_data.get('batch_size', 'N/A')}",
                f"Epochs: {config_data.get('epochs', 'N/A')}",
                f"Learning Rate: {config_data.get('lr', 'N/A')}",
                f"Optimizer: {config_data.get('optimizer', 'N/A')}",
                f"Scheduler: {config_data.get('scheduler', 'N/A')}",
                f"Weight Decay: {config_data.get('weight_decay', 'N/A')}",
                f"Val Ratio: {config_data.get('val_ratio', 'N/A')}",
                f"Num Classes: {config_data.get('num_classes', 'N/A')}",
                f"Classification: {config_data.get('classification_scheme', 'N/A')}",
                f"Augmentation: {config_data.get('augment', 'N/A')}",
                f"Flip Prob: {config_data.get('flip_prob', 'N/A')}",
                f"Rotation: {config_data.get('rotation_degree', 'N/A')}°",
                f"FOV Mask: {config_data.get('apply_fov_mask', 'N/A')}",
                f"Early Stopping: {config_data.get('early_stopping', 'N/A')}",
                f"Patience: {config_data.get('patience', 'N/A')}"
            ]
            
            y_pos = 0.95
            for config_item in main_config:
                ax.text(0.05, y_pos, config_item, fontsize=11, 
                       transform=ax.transAxes, fontweight='bold')
                y_pos -= 0.05
        
        ax.set_title('Training Configuration', fontweight='bold', pad=20)
        
        # 2. 最佳模型性能
        ax = axes[1]
        ax.axis('off')
        
        if self.best_model_info:
            best_metrics = self.best_model_info.get('best_metrics', {})
            
            performance_info = [
                f"Best Epoch: {self.best_model_info.get('best_epoch', 'N/A')}",
                f"Best Val Loss: {best_metrics.get('val_loss', 'N/A'):.6f}",
                f"Best mIoU: {best_metrics.get('miou', 'N/A'):.4f}",
                f"Best mDice: {best_metrics.get('mdice', 'N/A'):.4f}",
                f"Best mAcc: {best_metrics.get('macc', 'N/A'):.4f}",
                "",
                "Per-Class IoU:",
                f"  Class 0: {best_metrics.get('iou_per_class', [0,0,0])[0]:.4f}",
                f"  Class 1: {best_metrics.get('iou_per_class', [0,0,0])[1]:.4f}",
                f"  Class 2: {best_metrics.get('iou_per_class', [0,0,0])[2]:.4f}",
                "",
                "Per-Class Dice:",
                f"  Class 0: {best_metrics.get('dice_per_class', [0,0,0])[0]:.4f}",
                f"  Class 1: {best_metrics.get('dice_per_class', [0,0,0])[1]:.4f}",
                f"  Class 2: {best_metrics.get('dice_per_class', [0,0,0])[2]:.4f}",
                "",
                f"Updated: {self.best_model_info.get('updated_at', 'N/A')}"
            ]
            
            y_pos = 0.95
            for perf_item in performance_info:
                color = 'red' if 'Best' in perf_item and perf_item != "" else 'black'
                weight = 'bold' if 'Best' in perf_item or 'Class' in perf_item else 'normal'
                ax.text(0.05, y_pos, perf_item, fontsize=11, color=color,
                       transform=ax.transAxes, fontweight=weight)
                y_pos -= 0.05
        
        ax.set_title('Best Model Performance', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # 保存图片
        config_file = self.vis_dir / "config_summary.png"
        plt.savefig(config_file, dpi=300, bbox_inches='tight')
        print(f"   - 配置摘要保存到: {config_file}")
        
        plt.show()
    
    def copy_existing_visualizations(self):
        """复制现有的可视化文件"""
        print("📋 复制现有可视化文件...")
        
        vis_source = self.output_dir / "visualizations"
        if vis_source.exists():
            # 复制整个visualizations目录
            vis_dest = self.vis_dir / "original_visualizations"
            if vis_dest.exists():
                shutil.rmtree(vis_dest)
            shutil.copytree(vis_source, vis_dest)
            print(f"   - 原始可视化文件复制到: {vis_dest}")
        else:
            print("   - ⚠️ 未找到原始可视化文件")
    
    def generate_summary_report(self, resume_points):
        """生成文本摘要报告"""
        print("📝 生成摘要报告...")
        
        report_file = self.vis_dir / "training_summary_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"训练结果摘要报告 - {self.output_dir.name}\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 基本信息
            f.write("【基本信息】\n")
            f.write(f"模型输出目录: {self.output_dir}\n")
            f.write(f"总训练轮数: {len(self.metrics_df)} epochs\n")
            f.write(f"训练时间跨度: {self.metrics_df['timestamp'].iloc[0]} -> {self.metrics_df['timestamp'].iloc[-1]}\n")
            
            # Resume信息
            f.write(f"\n【Resume信息】\n")
            if resume_points:
                f.write(f"检测到 {len(resume_points)} 次训练中断恢复\n")
                f.write(f"Resume epochs: {', '.join(map(str, resume_points))}\n")
                
                # 计算每段训练时间
                segments = []
                start_epoch = 1
                for resume_epoch in resume_points + [len(self.metrics_df) + 1]:
                    end_epoch = resume_epoch - 1
                    segments.append((start_epoch, end_epoch))
                    start_epoch = resume_epoch
                
                f.write("训练段落:\n")
                for i, (start, end) in enumerate(segments[:-1]):
                    f.write(f"  段落 {i+1}: Epoch {start}-{end}\n")
            else:
                f.write("无训练中断，连续训练完成\n")
            
            # 最佳性能
            f.write(f"\n【最佳性能】\n")
            if self.best_model_info:
                best_metrics = self.best_model_info['best_metrics']
                f.write(f"最佳epoch: {self.best_model_info['best_epoch']}\n")
                f.write(f"最佳验证损失: {best_metrics['val_loss']:.6f}\n")
                f.write(f"最佳mIoU: {best_metrics['miou']:.4f}\n")
                f.write(f"最佳mDice: {best_metrics['mdice']:.4f}\n")
                f.write(f"最佳mAcc: {best_metrics['macc']:.4f}\n")
            
            # 最终性能
            f.write(f"\n【最终性能】\n")
            final_metrics = self.metrics_df.iloc[-1]
            f.write(f"最终epoch: {final_metrics['epoch']}\n")
            f.write(f"最终训练损失: {final_metrics['train_loss']:.6f}\n")
            f.write(f"最终验证损失: {final_metrics['val_loss']:.6f}\n")
            f.write(f"最终mIoU: {final_metrics['miou']:.4f}\n")
            f.write(f"最终mDice: {final_metrics['mdice']:.4f}\n")
            f.write(f"最终mAcc: {final_metrics['macc']:.4f}\n")
            
            # 配置信息
            if 'config' in self.config:
                config_data = self.config['config']
                f.write(f"\n【训练配置】\n")
                key_configs = ['model', 'img_size', 'batch_size', 'epochs', 'lr', 
                              'optimizer', 'scheduler', 'weight_decay', 'val_ratio', 
                              'num_classes', 'classification_scheme']
                for key in key_configs:
                    if key in config_data:
                        f.write(f"{key}: {config_data[key]}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"   - 摘要报告保存到: {report_file}")
    
    def run_complete_visualization(self):
        """运行完整的可视化流程"""
        print(f"\n🎨 开始完整可视化流程...")
        print(f"目标目录: {self.output_dir}")
        print(f"保存目录: {self.vis_dir}")
        print("-" * 60)
        
        # 1. 绘制训练曲线并检测resume点
        resume_points = self.plot_training_curves()
        
        # 2. 绘制resume分析
        self.plot_resume_analysis(resume_points)
        
        # 3. 创建配置摘要
        self.create_config_summary()
        
        # 4. 复制现有可视化文件
        self.copy_existing_visualizations()
        
        # 5. 生成摘要报告
        self.generate_summary_report(resume_points)
        
        print("-" * 60)
        print(f"✅ 可视化完成！所有文件保存在: {self.vis_dir}")
        print(f"📁 生成的文件:")
        for file_path in self.vis_dir.rglob("*"):
            if file_path.is_file():
                print(f"   - {file_path.relative_to(self.vis_dir)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='可视化训练结果 - 独立的metrics分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 自动检测resume点
  python scripts/visualize_metrics.py --output_dir outputs/unet_plus_plus_20250911_230321
  
  # 手动指定resume点（推荐，更准确）
  python scripts/visualize_metrics.py --output_dir outputs/unet_plus_plus_20250911_230321 --resume_epochs 4,11
  
  # 对于训练在epoch 3和10中断的情况，resume点应该是4和11
  python scripts/visualize_metrics.py --output_dir outputs/unet_plus_plus_20250911_230321 --resume_epochs 4,11

注意：
  - resume_epochs 指的是恢复训练后的第一个epoch
  - 如果在epoch 3中断，那么resume点是epoch 4
  - 支持多个resume点，用逗号分隔
        """)
    
    parser.add_argument('--output_dir', type=str, required=True,
                       help='模型输出目录路径，例如: outputs/unet_plus_plus_20250911_230321')
    parser.add_argument('--resume_epochs', type=str, default=None,
                       help='手动指定resume epochs，用逗号分隔。注意：这是恢复后的epoch，不是中断的epoch。例如: 4,11')
    
    args = parser.parse_args()
    
    # 解析resume epochs
    resume_epochs = None
    if args.resume_epochs:
        try:
            resume_epochs = [int(x.strip()) for x in args.resume_epochs.split(',')]
            resume_epochs = sorted(list(set(resume_epochs)))  # 去重并排序
            print(f"🎯 将使用手动指定的resume点: {resume_epochs}")
        except ValueError:
            print("❌ resume_epochs格式错误，应该是用逗号分隔的数字，例如: 4,11")
            print("💡 提示：resume_epochs是恢复训练后的第一个epoch，不是中断的epoch")
            return
    
    try:
        # 创建可视化器并运行
        visualizer = MetricsVisualizer(args.output_dir, resume_epochs)
        visualizer.run_complete_visualization()
        
        print("\n🎉 可视化完成！")
        
    except Exception as e:
        print(f"❌ 可视化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()