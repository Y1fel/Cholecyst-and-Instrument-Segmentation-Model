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
        self.epoch_times = []  # 记录每个epoch的时间
        self.batch_times = []  # 记录batch时间用于计算ETA
        self.last_system_check = 0  # 上次系统监控的时间
        self.system_check_interval = 5.0  # 系统监控间隔（秒）
        self.cached_stats = {}  # 缓存的系统状态
        self.metrics_history = []  # 新增：存储历史指标用于绘图
        
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
        except Exception as e:
            # GPU监控失败时返回空字典，避免阻塞训练
            return {}
    
    def get_system_stats(self) -> Dict:
        """获取系统资源信息"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=None),  # 非阻塞调用
                'memory_percent': psutil.virtual_memory().percent
            }
        except Exception as e:
            return {'cpu_percent': 0.0, 'memory_percent': 0.0}
    
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
    
    def calculate_eta(self, current_epoch: int, total_epochs: int, current_batch: int, total_batches: int) -> str:
        """计算剩余时间ETA"""
        if self.start_time is None:
            return "00:00:00"
        
        elapsed = time.time() - self.start_time
        
        # 计算总进度 (基于epoch和batch)
        total_progress = ((current_epoch - 1) * total_batches + current_batch) / (total_epochs * total_batches)
        
        if total_progress <= 0:
            return "00:00:00"
        
        # 估算总时间和剩余时间
        estimated_total_time = elapsed / total_progress
        remaining_time = estimated_total_time - elapsed
        
        if remaining_time < 0:
            remaining_time = 0
        
        hours = int(remaining_time // 3600)
        minutes = int((remaining_time % 3600) // 60)
        seconds = int(remaining_time % 60)
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
        
        # 添加系统信息（降低频率避免阻塞）
        current_time = time.time()
        if current_time - self.last_system_check >= self.system_check_interval:
            # 更新缓存的系统状态
            self.cached_stats['sys'] = self.get_system_stats()
            self.cached_stats['gpu'] = self.get_gpu_stats()
            self.last_system_check = current_time
        
        sys_stats = self.cached_stats.get('sys', {'cpu_percent': 0})
        gpu_stats = self.cached_stats.get('gpu', {})
        
        system_str = f"CPU: {sys_stats.get('cpu_percent', 0):.1f}%"
        if gpu_stats:
            system_str += f" | GPU: {gpu_stats.get('gpu_util', 0)}% ({gpu_stats.get('gpu_memory_used', 0)}/{gpu_stats.get('gpu_memory_total', 0)}MB)"
        
        # 添加时间信息
        time_str = f"Time: {self.get_elapsed_time()}"
        eta_str = f"ETA: {self.calculate_eta(epoch, total_epochs, batch, total_batches)}"
        
        # 组合完整信息
        full_str = f"{progress_str} | {metrics_str} | {system_str} | {time_str} | {eta_str}"
        
        try:
            if refresh:
                # 单行刷新
                sys.stdout.write(f"\r{full_str}")
                sys.stdout.flush()
            else:
                # 正常打印
                print(full_str)
        except Exception as e:
            # 输出失败时使用简化版本
            simple_str = f"Epoch {epoch}/{total_epochs} [{batch}/{total_batches}] | {metrics_str}"
            print(simple_str)
    
    def print_epoch_summary(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """打印epoch总结（换行）"""
        print()  # 换行
        print(f"Epoch {epoch} Summary:")
        print(f"  Train - {' | '.join([f'{k}: {v:.4f}' for k, v in train_metrics.items()])}")
        print(f"  Val   - {' | '.join([self._format_metric(k, v) for k, v in val_metrics.items()])}")
        print("-" * 80)
        
        # 记录指标历史用于后续绘图
        epoch_record = {
            'epoch': epoch,
            'train_loss': train_metrics.get('loss', 0.0),
            'val_loss': val_metrics.get('loss', 0.0)
        }
        # 添加验证集的其他指标
        for key, value in val_metrics.items():
            if key != 'loss' and not isinstance(value, (list, tuple)):
                epoch_record[key] = value
        
        self.metrics_history.append(epoch_record)
    
    def get_metrics_history(self):
        """获取指标历史记录"""
        return self.metrics_history

    def _format_metric(self, key: str, value) -> str:
        """格式化单个指标，处理数字和列表类型"""
        if isinstance(value, (list, tuple)):
            if key.endswith('_per_class'):
                # 对于per_class指标，只显示平均值
                avg_val = sum(value) / len(value) if value else 0.0
                return f"{key}_avg: {avg_val:.4f}"
            else:
                # 其他列表类型，显示前几个元素
                formatted_vals = [f"{v:.3f}" for v in value[:3]]
                return f"{key}: [{', '.join(formatted_vals)}{'...' if len(value) > 3 else ''}]"
        else:
            # 数字类型正常格式化
            try:
                return f"{key}: {value:.4f}"
            except (TypeError, ValueError):
                return f"{key}: {value}"
