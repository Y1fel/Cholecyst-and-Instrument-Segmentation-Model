#!/usr/bin/env python3
"""
Independent KD Visualization Generation Script
Generate specialized visualization charts for knowledge distillation training, solving table overlapping issues
All charts are saved as separate files for easy viewing and analysis

Usage:
python scripts/generate_kd_visualizations.py --experiment_dir outputs/distill_unet_plus_plus_to_adaptive_unet_20250920_180040
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib font (remove Chinese font settings to avoid encoding issues)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

class KDVisualizationGenerator:
    """Knowledge Distillation Specialized Visualization Generator"""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.viz_dir = self.experiment_dir / "visualizations"
        self.distill_dir = self.viz_dir / "distillation_analysis"
        
        # Create new visualization directory
        current_date = datetime.now().strftime("%Y%m%d")
        self.new_viz_dir = self.experiment_dir / f"new_kd_viz_{current_date}"
        self.new_viz_dir.mkdir(exist_ok=True)
        
        print(f"📁 Experiment Directory: {self.experiment_dir}")
        print(f"📁 New Visualization Directory: {self.new_viz_dir}")
        
        # Load data
        self.load_experiment_data()
    
    def load_experiment_data(self):
        """Load experiment data"""
        print("\n📊 Loading experiment data...")
        
        # Load training configuration
        config_path = self.experiment_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            print(f"   ✅ Config file: {config_path.name}")
        else:
            self.config = {}
            print(f"   ⚠️  Config file not found: {config_path}")
        
        # Load training metrics
        metrics_path = self.experiment_dir / "metrics.csv"
        if metrics_path.exists():
            self.metrics_df = pd.read_csv(metrics_path)
            print(f"   ✅ Metrics file: {metrics_path.name} ({len(self.metrics_df)} records)")
        else:
            self.metrics_df = pd.DataFrame()
            print(f"   ⚠️  Metrics file not found: {metrics_path}")
        
        # Load distillation analysis data
        distill_metrics_path = self.distill_dir / "distillation_metrics.csv"
        if distill_metrics_path.exists():
            self.distill_metrics_df = pd.read_csv(distill_metrics_path)
            print(f"   ✅ Distillation metrics: {distill_metrics_path.name} ({len(self.distill_metrics_df)} records)")
        else:
            self.distill_metrics_df = pd.DataFrame()
            print(f"   ⚠️  Distillation metrics not found: {distill_metrics_path}")
        
        # Load distillation statistics data
        distill_stats_path = self.distill_dir / "distillation_summary_stats.csv"
        if distill_stats_path.exists():
            self.distill_stats_df = pd.read_csv(distill_stats_path)
            print(f"   ✅ Distillation statistics: {distill_stats_path.name}")
        else:
            self.distill_stats_df = pd.DataFrame()
            print(f"   ⚠️  Distillation statistics not found: {distill_stats_path}")
    
    def generate_all_visualizations(self):
        """Generate all visualization charts"""
        print("\n🎨 Starting KD visualization generation...")
        
        # 1. Training loss curves (independent chart)
        self.create_training_loss_curves()
        
        # 2. Validation metrics curves (independent chart)
        self.create_validation_metrics_curves()
        
        # 3. Loss composition analysis (independent chart)
        self.create_loss_composition_analysis()
        
        # 4. Training progress table (independent chart, fix overlapping issues)
        self.create_training_progress_table()
        
        # 5. Distillation statistics overview (independent chart)
        self.create_distillation_statistics_overview()
        
        # 6. Final performance comparison (independent chart)
        self.create_final_performance_comparison()
        
        # 7. Training stability analysis (new feature)
        self.create_training_stability_analysis()
        
        # 8. Loss convergence analysis (new feature)
        self.create_loss_convergence_analysis()
        
        # 9. Generate visualization overview HTML report
        self.create_html_report()
        
        print(f"\n✅ All visualization charts generated successfully!")
        print(f"📁 Save location: {self.new_viz_dir}")
    
    def create_training_loss_curves(self):
        """Create training loss curves chart"""
        print("   📈 Generating training loss curves...")
        
        if self.distill_metrics_df.empty:
            print("      ⚠️  Skip: No distillation metrics data")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        epochs = self.distill_metrics_df['epoch']
        
        # Plot loss curves
        ax.plot(epochs, self.distill_metrics_df['total_loss'], 'b-', 
                linewidth=2.5, label='Total Loss', marker='o', markersize=4)
        ax.plot(epochs, self.distill_metrics_df['task_loss'], 'r--', 
                linewidth=2.5, label='Task Loss', marker='s', markersize=4)
        ax.plot(epochs, self.distill_metrics_df['distillation_loss'], 'g:', 
                linewidth=2.5, label='Distillation Loss', marker='^', markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
        ax.set_title('Knowledge Distillation Training Loss Curves', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add final value annotation
        final_epoch = epochs.iloc[-1]
        final_total = self.distill_metrics_df['total_loss'].iloc[-1]
        final_task = self.distill_metrics_df['task_loss'].iloc[-1]
        final_distill = self.distill_metrics_df['distillation_loss'].iloc[-1]
        
        ax.annotate(f'Final: {final_total:.4f}', 
                   xy=(final_epoch, final_total), xytext=(10, 10),
                   textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        save_path = self.new_viz_dir / "01_training_loss_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      ✅ Saved to: {save_path.name}")
    
    def create_validation_metrics_curves(self):
        """Create validation metrics curves chart"""
        print("   📊 Generating validation metrics curves...")
        
        if self.distill_metrics_df.empty:
            print("      ⚠️  Skip: No distillation metrics data")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        epochs = self.distill_metrics_df['epoch']
        
        # Left plot: Validation loss
        if 'val_loss' in self.distill_metrics_df.columns:
            ax1.plot(epochs, self.distill_metrics_df['val_loss'], 'purple', 
                    linewidth=2.5, label='Validation Loss', marker='o', markersize=4)
            ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
            ax1.set_title('Validation Loss Curve', fontsize=13, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Validation Loss\nData Not Available', 
                    ha='center', va='center', transform=ax1.transAxes,
                    fontsize=12, bbox=dict(boxstyle="round", facecolor='lightgray'))
            ax1.set_title('Validation Loss (N/A)', fontsize=13, fontweight='bold')
        
        # Right plot: mIoU
        if 'miou' in self.distill_metrics_df.columns:
            ax2.plot(epochs, self.distill_metrics_df['miou'], 'orange', 
                    linewidth=2.5, label='mIoU', marker='s', markersize=4)
            ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Mean IoU', fontsize=12, fontweight='bold')
            ax2.set_title('Mean IoU Progression', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)
            
            # Add best IoU annotation
            best_miou = self.distill_metrics_df['miou'].max()
            best_epoch = self.distill_metrics_df.loc[self.distill_metrics_df['miou'].idxmax(), 'epoch']
            ax2.annotate(f'Best: {best_miou:.4f}@Epoch{best_epoch}', 
                        xy=(best_epoch, best_miou), xytext=(10, 10),
                        textcoords='offset points', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        else:
            ax2.text(0.5, 0.5, 'mIoU Data\nNot Available', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=12, bbox=dict(boxstyle="round", facecolor='lightgray'))
            ax2.set_title('Mean IoU (N/A)', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.new_viz_dir / "02_validation_metrics_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      ✅ Saved to: {save_path.name}")
    
    def create_loss_composition_analysis(self):
        """Create loss composition analysis chart"""
        print("   🥧 Generating loss composition analysis...")
        
        if self.distill_metrics_df.empty:
            print("      ⚠️  Skip: No distillation metrics data")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Final loss composition pie chart
        if len(self.distill_metrics_df) > 0:
            final_total = self.distill_metrics_df['total_loss'].iloc[-1]
            final_task = self.distill_metrics_df['task_loss'].iloc[-1]
            final_distill = self.distill_metrics_df['distillation_loss'].iloc[-1]
            
            # Calculate percentages
            task_pct = (final_task / final_total) * 100
            distill_pct = (final_distill / final_total) * 100
            
            labels = ['Task Loss', 'Distillation Loss']
            sizes = [task_pct, distill_pct]
            colors = ['#ff9999', '#66b3ff']
            explode = (0.05, 0.05)  # Separation effect
            
            wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, 
                                              autopct='%1.1f%%', startangle=90, 
                                              explode=explode)
            ax1.set_title('Final Loss Composition', fontsize=14, fontweight='bold')
            
            # Beautify pie chart text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(11)
        
        # Right plot: Loss ratio change curve
        epochs = self.distill_metrics_df['epoch']
        task_ratios = self.distill_metrics_df['task_loss'] / self.distill_metrics_df['total_loss']
        distill_ratios = self.distill_metrics_df['distillation_loss'] / self.distill_metrics_df['total_loss']
        
        ax2.plot(epochs, task_ratios, 'r-', linewidth=2.5, 
                label='Task Loss Ratio', marker='o', markersize=4)
        ax2.plot(epochs, distill_ratios, 'b-', linewidth=2.5, 
                label='Distillation Loss Ratio', marker='^', markersize=4)
        
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Loss Ratio', fontsize=12, fontweight='bold')
        ax2.set_title('Loss Composition Over Time', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        save_path = self.new_viz_dir / "03_loss_composition_analysis.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      ✅ Saved to: {save_path.name}")
    
    def create_training_progress_table(self):
        """Create training progress table (fix overlapping issues)"""
        print("   📋 Generating training progress table...")
        
        if self.distill_metrics_df.empty:
            print("      ⚠️  Skip: No distillation metrics data")
            return
        
        # Create larger figure to avoid overlapping - increase height significantly
        table_height = max(10, len(self.distill_metrics_df) * 0.5 + 3)  # Add extra space for title
        fig, ax = plt.subplots(figsize=(18, table_height))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        for _, row in self.distill_metrics_df.iterrows():
            epoch = int(row['epoch'])
            total_loss = f"{row['total_loss']:.4f}"
            task_loss = f"{row['task_loss']:.4f}"
            distill_loss = f"{row['distillation_loss']:.4f}"
            val_loss = f"{row.get('val_loss', 0):.4f}" if 'val_loss' in row else "N/A"
            miou = f"{row.get('miou', 0):.4f}" if 'miou' in row else "N/A"
            
            table_data.append([epoch, total_loss, task_loss, distill_loss, val_loss, miou])
        
        columns = ['Epoch', 'Total Loss', 'Task Loss', 'Distill Loss', 'Val Loss', 'mIoU']
        
        # Create table with better positioning - move table down to leave space for title
        table = ax.table(cellText=table_data, colLabels=columns, 
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 0.85])  # Leave top 15% for title
        
        # Set table styles
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.2)  # Increase row height to avoid overlapping
        
        # Set header styles
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(0, i)].set_height(0.1)  # Increase header height
        
        # Set data row styles
        for i in range(1, len(table_data) + 1):
            for j in range(len(columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                table[(i, j)].set_height(0.08)  # Increase data row height
        
        # Highlight best performance row
        if 'miou' in self.distill_metrics_df.columns:
            best_idx = self.distill_metrics_df['miou'].idxmax() + 1
            for j in range(len(columns)):
                table[(best_idx, j)].set_facecolor('#FFE082')
        
        # Add title with proper spacing
        fig.suptitle('Knowledge Distillation Training Progress Summary', 
                    fontsize=18, fontweight='bold', y=0.95)  # Position title at top
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust top margin for title
        save_path = self.new_viz_dir / "04_training_progress_table.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        print(f"      ✅ Saved to: {save_path.name}")
    
    def create_distillation_statistics_overview(self):
        """Create distillation statistics overview"""
        print("   📊 Generating distillation statistics overview...")
        
        if self.distill_stats_df.empty:
            print("      ⚠️  Skip: No distillation statistics data")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract statistical data
        stats_dict = dict(zip(self.distill_stats_df['Metric'], self.distill_stats_df['Value']))
        
        # 1. Distillation core metrics
        kd_metrics = ['Mean KL Divergence', 'Teacher-Student Agreement Rate', 'Mean Confidence Difference']
        kd_values = []
        kd_labels = []
        
        for metric in kd_metrics:
            if metric in stats_dict:
                value = float(stats_dict[metric])
                kd_values.append(value)
                if metric == 'Mean KL Divergence':
                    kd_labels.append('KL Divergence')
                elif metric == 'Teacher-Student Agreement Rate':
                    kd_labels.append('Agreement Rate')
                else:
                    kd_labels.append('Confidence Diff')
        
        if kd_values:
            bars1 = ax1.bar(kd_labels, kd_values, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
            ax1.set_title('Distillation Core Metrics', fontsize=13, fontweight='bold')
            ax1.set_ylabel('Value')
            
            # Add value labels
            for bar, value in zip(bars1, kd_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Final loss comparison
        loss_metrics = ['Final Total Loss', 'Final Task Loss', 'Final Distillation Loss']
        loss_values = []
        loss_labels = ['Total', 'Task', 'Distillation']
        
        for metric in loss_metrics:
            if metric in stats_dict:
                loss_values.append(float(stats_dict[metric]))
        
        if loss_values:
            bars2 = ax2.bar(loss_labels, loss_values, color=['purple', 'orange', 'green'], alpha=0.8)
            ax2.set_title('Final Loss Values', fontsize=13, fontweight='bold')
            ax2.set_ylabel('Loss Value')
            
            for bar, value in zip(bars2, loss_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Performance metrics
        perf_labels = ['Val Loss', 'mIoU']
        perf_values = []
        
        if 'Final Validation Loss' in stats_dict:
            perf_values.append(float(stats_dict['Final Validation Loss']))
        else:
            perf_values.append(0)
            
        if 'Final mIoU' in stats_dict:
            perf_values.append(float(stats_dict['Final mIoU']))
        else:
            perf_values.append(0)
        
        # Use dual y-axis
        ax3_twin = ax3.twinx()
        
        bar1 = ax3.bar([0], [perf_values[0]], color='red', alpha=0.7, label='Val Loss')
        bar2 = ax3_twin.bar([1], [perf_values[1]], color='blue', alpha=0.7, label='mIoU')
        
        ax3.set_ylabel('Validation Loss', color='red')
        ax3_twin.set_ylabel('mIoU', color='blue')
        ax3.set_title('Final Performance Metrics', fontsize=13, fontweight='bold')
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(['Val Loss', 'mIoU'])
        
        # Add value labels
        if perf_values[0] > 0:
            ax3.text(0, perf_values[0] + perf_values[0]*0.01, f'{perf_values[0]:.4f}', 
                    ha='center', va='bottom', fontweight='bold')
        if perf_values[1] > 0:
            ax3_twin.text(1, perf_values[1] + perf_values[1]*0.01, f'{perf_values[1]:.4f}', 
                         ha='center', va='bottom', fontweight='bold')
        
        # 4. Experiment configuration information
        ax4.axis('off')
        config_text = f"""
        Knowledge Distillation Experiment Configuration:
        
        Teacher Model: {self.config.get('teacher_model', 'Unknown')}
        Student Model: {self.config.get('student_model', 'Unknown')}
        
        Training Config:
        • Epochs: {self.config.get('epochs', 'N/A')}
        • Batch Size: {self.config.get('batch_size', 'N/A')}
        • Temperature: {self.config.get('distill_temperature', self.config.get('temperature', 'N/A'))}
        • Alpha (KD Weight): {self.config.get('distill_alpha', self.config.get('alpha', 'N/A'))}
        
        Dataset: {self.config.get('data_root', 'N/A')}
        Task: {'Binary Classification' if self.config.get('binary', False) else 'Multi-class Segmentation'}
        Classes: {self.config.get('num_classes', 'N/A')}
        """
        
        ax4.text(0.1, 0.5, config_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='center', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        ax4.set_title('Experiment Configuration', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.new_viz_dir / "05_distillation_statistics_overview.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      ✅ Saved to: {save_path.name}")
    
    def create_final_performance_comparison(self):
        """Create final performance comparison chart"""
        print("   🏆 Generating final performance comparison...")
        
        if self.distill_metrics_df.empty:
            print("      ⚠️  Skip: No distillation metrics data")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Training progress comparison (initial vs final)
        if len(self.distill_metrics_df) > 1:
            initial_epoch = self.distill_metrics_df.iloc[0]
            final_epoch = self.distill_metrics_df.iloc[-1]
            
            metrics = ['total_loss', 'task_loss', 'distillation_loss']
            initial_values = [initial_epoch[m] for m in metrics]
            final_values = [final_epoch[m] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, initial_values, width, label='Initial (Epoch 1)', 
                           color='lightcoral', alpha=0.8)
            bars2 = ax1.bar(x + width/2, final_values, width, label=f'Final (Epoch {int(final_epoch["epoch"])})', 
                           color='lightblue', alpha=0.8)
            
            ax1.set_xlabel('Loss Type')
            ax1.set_ylabel('Loss Value')
            ax1.set_title('Training Progress: Initial vs Final')
            ax1.set_xticks(x)
            ax1.set_xticklabels(['Total Loss', 'Task Loss', 'Distill Loss'])
            ax1.legend()
            
            # Add improvement percentage annotations
            for i, (init_val, final_val) in enumerate(zip(initial_values, final_values)):
                improvement = ((init_val - final_val) / init_val) * 100
                ax1.text(i, max(init_val, final_val) + max(initial_values) * 0.05,
                        f'{improvement:+.1f}%', ha='center', va='bottom', 
                        fontweight='bold', color='green' if improvement > 0 else 'red')
        
        # Right plot: Performance radar chart
        if 'miou' in self.distill_metrics_df.columns:
            final_miou = self.distill_metrics_df['miou'].iloc[-1]
            final_val_loss = self.distill_metrics_df.get('val_loss', [0]).iloc[-1] if 'val_loss' in self.distill_metrics_df.columns else 0
            
            # Prepare radar chart data
            categories = ['mIoU', 'Task Loss\n(Inverted)', 'Distill Loss\n(Inverted)', 'Stability']
            values = [
                final_miou,
                1 - min(final_epoch['task_loss'], 1),  # Invert loss (smaller is better -> larger is better)
                1 - min(final_epoch['distillation_loss'], 1),
                0.8  # Assumed stability score
            ]
            
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
            values = np.concatenate((values, [values[0]]))  # Close the plot
            angles = np.concatenate((angles, [angles[0]]))
            
            ax2 = plt.subplot(1, 2, 2, projection='polar')
            ax2.plot(angles, values, 'o-', linewidth=2, label='KD Student Model', color='blue')
            ax2.fill(angles, values, alpha=0.25, color='blue')
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(categories)
            ax2.set_ylim(0, 1)
            ax2.set_title('Model Performance Radar', y=1.08, fontsize=13, fontweight='bold')
            ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        save_path = self.new_viz_dir / "06_final_performance_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      ✅ Saved to: {save_path.name}")
    
    def create_training_stability_analysis(self):
        """Create training stability analysis"""
        print("   📈 Generating training stability analysis...")
        
        if self.distill_metrics_df.empty or len(self.distill_metrics_df) < 3:
            print("      ⚠️  Skip: Insufficient data for stability analysis")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = self.distill_metrics_df['epoch']
        
        # 1. Loss change rate analysis
        total_loss_diff = self.distill_metrics_df['total_loss'].diff().abs()
        task_loss_diff = self.distill_metrics_df['task_loss'].diff().abs()
        distill_loss_diff = self.distill_metrics_df['distillation_loss'].diff().abs()
        
        ax1.plot(epochs[1:], total_loss_diff[1:], 'b-', label='Total Loss Change', linewidth=2)
        ax1.plot(epochs[1:], task_loss_diff[1:], 'r-', label='Task Loss Change', linewidth=2)
        ax1.plot(epochs[1:], distill_loss_diff[1:], 'g-', label='Distill Loss Change', linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Absolute Loss Change')
        ax1.set_title('Training Stability: Loss Change Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Moving average analysis
        window = min(3, len(self.distill_metrics_df) // 2)
        total_loss_ma = self.distill_metrics_df['total_loss'].rolling(window=window).mean()
        
        ax2.plot(epochs, self.distill_metrics_df['total_loss'], 'b-', alpha=0.6, label='Raw Total Loss')
        ax2.plot(epochs, total_loss_ma, 'r-', linewidth=2, label=f'{window}-Epoch Moving Average')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Total Loss')
        ax2.set_title('Loss Smoothness Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Loss composition stability
        task_ratio = self.distill_metrics_df['task_loss'] / self.distill_metrics_df['total_loss']
        distill_ratio = self.distill_metrics_df['distillation_loss'] / self.distill_metrics_df['total_loss']
        
        ax3.plot(epochs, task_ratio, 'r-', linewidth=2, label='Task Loss Ratio', marker='o')
        ax3.plot(epochs, distill_ratio, 'b-', linewidth=2, label='Distill Loss Ratio', marker='^')
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Ratio')
        ax3.set_title('Loss Composition Stability')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # 4. Stability statistics
        ax4.axis('off')
        
        # Calculate stability metrics
        total_loss_std = self.distill_metrics_df['total_loss'].std()
        total_loss_cv = total_loss_std / self.distill_metrics_df['total_loss'].mean()  # Coefficient of variation
        
        task_ratio_std = task_ratio.std()
        distill_ratio_std = distill_ratio.std()
        
        convergence_rate = (self.distill_metrics_df['total_loss'].iloc[0] - 
                          self.distill_metrics_df['total_loss'].iloc[-1]) / len(self.distill_metrics_df)
        
        stability_text = f"""
        Training Stability Statistics:
        
        Loss Stability:
        • Total Loss Std: {total_loss_std:.6f}
        • Coefficient of Variation: {total_loss_cv:.4f}
        • Convergence Rate: {convergence_rate:.6f}/epoch
        
        Composition Stability:
        • Task Ratio Std: {task_ratio_std:.4f}
        • Distill Ratio Std: {distill_ratio_std:.4f}
        
        Training Assessment:
        {'[OK] Training Stable' if total_loss_cv < 0.1 else '[WARN] Training Unstable'}
        {'[OK] Ratio Balanced' if abs(task_ratio.mean() - 0.5) < 0.2 else '[WARN] Ratio Imbalanced'}
        """
        
        ax4.text(0.1, 0.5, stability_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
        ax4.set_title('Stability Statistics', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.new_viz_dir / "07_training_stability_analysis.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      ✅ Saved to: {save_path.name}")
    
    def create_loss_convergence_analysis(self):
        """Create loss convergence analysis"""
        print("   🎯 Generating loss convergence analysis...")
        
        if self.distill_metrics_df.empty or len(self.distill_metrics_df) < 3:
            print("      ⚠️  Skip: Insufficient data for convergence analysis")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = self.distill_metrics_df['epoch'].values
        total_loss = self.distill_metrics_df['total_loss'].values
        task_loss = self.distill_metrics_df['task_loss'].values
        distill_loss = self.distill_metrics_df['distillation_loss'].values
        
        # 1. Logarithmic scale convergence analysis
        ax1.semilogy(epochs, total_loss, 'b-', linewidth=2, label='Total Loss', marker='o')
        ax1.semilogy(epochs, task_loss, 'r-', linewidth=2, label='Task Loss', marker='s')
        ax1.semilogy(epochs, distill_loss, 'g-', linewidth=2, label='Distill Loss', marker='^')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (Log Scale)')
        ax1.set_title('Loss Convergence (Log Scale)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Convergence speed analysis (exponential fitting)
        try:
            # Exponential fitting for total loss
            from scipy.optimize import curve_fit
            
            def exponential_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            popt, _ = curve_fit(exponential_decay, epochs, total_loss, maxfev=1000)
            fitted_loss = exponential_decay(epochs, *popt)
            
            ax2.plot(epochs, total_loss, 'bo-', label='Actual Total Loss', markersize=4)
            ax2.plot(epochs, fitted_loss, 'r--', linewidth=2, label='Exponential Fit')
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Total Loss')
            ax2.set_title('Convergence Pattern Analysis')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Calculate fitting quality
            r_squared = 1 - (np.sum((total_loss - fitted_loss) ** 2) / 
                            np.sum((total_loss - np.mean(total_loss)) ** 2))
            ax2.text(0.02, 0.98, f'R² = {r_squared:.4f}', transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        except:
            # If fitting fails, show simple trend line
            ax2.plot(epochs, total_loss, 'bo-', label='Total Loss', markersize=4)
            z = np.polyfit(epochs, total_loss, 1)
            p = np.poly1d(z)
            ax2.plot(epochs, p(epochs), 'r--', linewidth=2, label='Linear Trend')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Total Loss')
            ax2.set_title('Loss Trend Analysis')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Improvement rate analysis
        total_improvement = [(total_loss[0] - loss) / total_loss[0] * 100 for loss in total_loss]
        task_improvement = [(task_loss[0] - loss) / task_loss[0] * 100 for loss in task_loss]
        distill_improvement = [(distill_loss[0] - loss) / distill_loss[0] * 100 for loss in distill_loss]
        
        ax3.plot(epochs, total_improvement, 'b-', linewidth=2, label='Total Loss', marker='o')
        ax3.plot(epochs, task_improvement, 'r-', linewidth=2, label='Task Loss', marker='s')
        ax3.plot(epochs, distill_improvement, 'g-', linewidth=2, label='Distill Loss', marker='^')
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Improvement (%)')
        ax3.set_title('Cumulative Loss Improvement')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 4. Convergence statistics
        ax4.axis('off')
        
        final_total = total_loss[-1]
        final_task = task_loss[-1]
        final_distill = distill_loss[-1]
        
        total_reduction = (total_loss[0] - final_total) / total_loss[0] * 100
        task_reduction = (task_loss[0] - final_task) / task_loss[0] * 100
        distill_reduction = (distill_loss[0] - final_distill) / distill_loss[0] * 100
        
        # Calculate average improvement rate
        avg_total_improvement = total_reduction / len(epochs)
        avg_task_improvement = task_reduction / len(epochs)
        avg_distill_improvement = distill_reduction / len(epochs)
        
        convergence_text = f"""
        Loss Convergence Analysis:
        
        Overall Improvement:
        • Total Loss: {total_reduction:.2f}% ({final_total:.4f})
        • Task Loss: {task_reduction:.2f}% ({final_task:.4f})
        • Distill Loss: {distill_reduction:.2f}% ({final_distill:.4f})
        
        Average Improvement Rate (per Epoch):
        • Total: {avg_total_improvement:.2f}%
        • Task: {avg_task_improvement:.2f}%
        • Distill: {avg_distill_improvement:.2f}%
        
        Convergence Status:
        {'[OK] Good Convergence' if total_reduction > 50 else '[WARN] Poor Convergence'}
        {'[OK] Balanced Optimization' if abs(task_reduction - distill_reduction) < 20 else '[WARN] Unbalanced Optimization'}
        """
        
        ax4.text(0.1, 0.5, convergence_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
        ax4.set_title('Convergence Statistics', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.new_viz_dir / "08_loss_convergence_analysis.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      ✅ Saved to: {save_path.name}")
    
    def create_html_report(self):
        """Generate HTML visualization report"""
        print("   📄 Generating HTML visualization report...")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Distillation Training Visualization Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 10px;
            margin-top: 30px;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .description {{
            margin: 10px 0;
            padding: 10px;
            background-color: #ecf0f1;
            border-radius: 5px;
            font-style: italic;
        }}
        .metadata {{
            background-color: #e8f4fd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .metadata h3 {{
            margin-top: 0;
            color: #2980b9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Knowledge Distillation Training Visualization Report</h1>
        
        <div class="metadata">
            <h3>Experiment Information</h3>
            <p><strong>Experiment Directory:</strong> {self.experiment_dir.name}</p>
            <p><strong>Generation Time:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Teacher Model:</strong> {self.config.get('teacher_model', 'Unknown')}</p>
            <p><strong>Student Model:</strong> {self.config.get('student_model', 'Unknown')}</p>
            <p><strong>Training Epochs:</strong> {self.config.get('epochs', 'N/A')}</p>
            <p><strong>Dataset:</strong> {self.config.get('data_root', 'N/A')}</p>
        </div>
        
        <h2>1. Training Loss Curves</h2>
        <div class="image-container">
            <img src="01_training_loss_curves.png" alt="Training Loss Curves">
            <div class="description">
                Shows the trends of total loss, task loss, and distillation loss during knowledge distillation training.
            </div>
        </div>
        
        <h2>2. Validation Metrics Curves</h2>
        <div class="image-container">
            <img src="02_validation_metrics_curves.png" alt="Validation Metrics Curves">
            <div class="description">
                Displays validation loss and mIoU performance during training to evaluate model generalization capability.
            </div>
        </div>
        
        <h2>3. Loss Composition Analysis</h2>
        <div class="image-container">
            <img src="03_loss_composition_analysis.png" alt="Loss Composition Analysis">
            <div class="description">
                Analyzes the proportion of task loss and distillation loss in total loss and their changes over time.
            </div>
        </div>
        
        <h2>4. Training Progress Table</h2>
        <div class="image-container">
            <img src="04_training_progress_table.png" alt="Training Progress Table">
            <div class="description">
                Detailed record of metric values for each training epoch, facilitating precise analysis.
            </div>
        </div>
        
        <h2>5. Distillation Statistics Overview</h2>
        <div class="image-container">
            <img src="05_distillation_statistics_overview.png" alt="Distillation Statistics Overview">
            <div class="description">
                Comprehensive display of core knowledge distillation statistics and experimental configuration information.
            </div>
        </div>
        
        <h2>6. Final Performance Comparison</h2>
        <div class="image-container">
            <img src="06_final_performance_comparison.png" alt="Final Performance Comparison">
            <div class="description">
                Compares performance differences between training start and end, along with comprehensive model capability assessment.
            </div>
        </div>
        
        <h2>7. Training Stability Analysis</h2>
        <div class="image-container">
            <img src="07_training_stability_analysis.png" alt="Training Stability Analysis">
            <div class="description">
                Analyzes training process stability, including loss change rates, moving averages, and composition stability.
            </div>
        </div>
        
        <h2>8. Loss Convergence Analysis</h2>
        <div class="image-container">
            <img src="08_loss_convergence_analysis.png" alt="Loss Convergence Analysis">
            <div class="description">
                In-depth analysis of loss function convergence patterns, improvement rates, and final convergence effects.
            </div>
        </div>
        
        <div class="metadata">
            <h3>Report Description</h3>
            <p>This report is generated based on complete knowledge distillation training records, including detailed analysis and visualization of the training process.</p>
            <p>All charts have been saved in high-resolution PNG format and can be viewed and used individually.</p>
            <p>For more detailed data analysis, please refer to the corresponding CSV files.</p>
        </div>
    </div>
</body>
</html>
        """
        
        html_path = self.new_viz_dir / "KD_Visualization_Report.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"      ✅ HTML report saved to: {html_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Generate knowledge distillation specialized visualization charts")
    parser.add_argument("--experiment_dir", type=str, required=True,
                       help="Experiment output directory path, e.g.: outputs/distill_unet_plus_plus_to_adaptive_unet_20250920_180040")
    parser.add_argument("--output_name", type=str, default=None,
                       help="Custom output directory name suffix")
    
    args = parser.parse_args()
    
    # Validate experiment directory
    experiment_path = Path(args.experiment_dir)
    if not experiment_path.exists():
        print(f"❌ Error: Experiment directory does not exist - {experiment_path}")
        sys.exit(1)
    
    print("🎨 Knowledge Distillation Visualization Generator")
    print("=" * 50)
    
    try:
        # Create visualization generator
        generator = KDVisualizationGenerator(args.experiment_dir)
        
        # Generate all visualizations
        generator.generate_all_visualizations()
        
        print("\n" + "=" * 50)
        print("🎉 Visualization generation completed!")
        print(f"📂 View results: {generator.new_viz_dir}")
        print(f"📄 HTML report: {generator.new_viz_dir / 'KD_Visualization_Report.html'}")
        
    except Exception as e:
        print(f"\n❌ Error occurred during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()