"""
通用训练监控模块
提供单行刷新、GPU监控、进度条等功能
"""
import os, sys, time, torch, psutil
from typing import Dict, Optional

class TrainMonitor:
    def __init__(self, enable_gpu_monitor: bool = True):
        self.enable_gpu_monitor = enable_gpu_monitor
        self.start_time = None
        
        # 尝试导入GPU监控库
        self.gpu_available = False
        if enable_gpu_monitor:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.pynvml = pynvml
                self.gpu_available = True
            except ImportError:
                print("pynvml not available, GPU monitoring disabled")
    
    def get_gpu_stats(self) -> Dict:
        """获取GPU使用率和内存信息"""
        if not self.gpu_available:
            return {}
        
        try:
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            return {
                'gpu_util': utilization.gpu,
                'gpu_memory_used': memory_info.used // (1024**2),  # MB
                'gpu_memory_total': memory_info.total // (1024**2),  # MB
                'gpu_memory_percent': (memory_info.used / memory_info.total) * 100
            }
        except:
            return {}
    
    def get_system_stats(self) -> Dict:
        """获取系统资源信息"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent
        }
    
    def start_timing(self):
        """开始计时"""
        self.start_time = time.time()
    
    def get_elapsed_time(self) -> str:
        """获取已用时间"""
        if self.start_time is None:
            return "00:00:00"
        
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def print_progress(self, 
                      epoch: int, 
                      total_epochs: int,
                      batch: int, 
                      total_batches: int,
                      metrics: Dict,
                      refresh: bool = True):
        """打印训练进度（单行刷新）"""
        
        # 构建进度信息
        progress_str = f"Epoch {epoch}/{total_epochs} [{batch}/{total_batches}]"
        
        # 添加指标信息
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        # 添加系统信息
        sys_stats = self.get_system_stats()
        gpu_stats = self.get_gpu_stats()
        
        system_str = f"CPU: {sys_stats['cpu_percent']:.1f}%"
        if gpu_stats:
            system_str += f" | GPU: {gpu_stats['gpu_util']}% ({gpu_stats['gpu_memory_used']}/{gpu_stats['gpu_memory_total']}MB)"
        
        # 添加时间信息
        time_str = f"Time: {self.get_elapsed_time()}"
        
        # 组合完整信息
        full_str = f"{progress_str} | {metrics_str} | {system_str} | {time_str}"
        
        if refresh:
            # 单行刷新
            sys.stdout.write(f"\r{full_str}")
            sys.stdout.flush()
        else:
            # 正常打印
            print(full_str)
    
    def print_epoch_summary(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """打印epoch总结（换行）"""
        print()  # 换行
        print(f"Epoch {epoch} Summary:")
        print(f"  Train - {' | '.join([f'{k}: {v:.4f}' for k, v in train_metrics.items()])}")
        print(f"  Val   - {' | '.join([f'{k}: {v:.4f}' for k, v in val_metrics.items()])}")
        print("-" * 80)
