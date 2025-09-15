# src/training/resume_manager.py
import os
import glob
import torch
import json
from typing import Optional, Dict, Any, Tuple

class ResumeManager:
    def __init__(self, resume_path: str):
        """
        Args:
            resume_path: checkpoint目录路径，如 'outputs/unet_plus_plus_20250911_230321/checkpoints'
        """
        self.resume_path = os.path.abspath(resume_path)
        self.checkpoint_dir = resume_path
        self.run_dir = os.path.dirname(resume_path)  # 上级目录

        if not os.path.exists(self.checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dir}")
    
    def find_latest_checkpoint(self, prefer_best: bool = False) -> str:
        """
        查找最新的checkpoint文件
        Args:
            prefer_best: 是否优先选择best checkpoint
        Returns:
            checkpoint文件的完整路径
        """
        if prefer_best:
            # 查找best checkpoint
            best_patterns = ["*_best.pth", "best.pth"]
            for pattern in best_patterns:
                matches = glob.glob(os.path.join(self.checkpoint_dir, pattern))
                if matches:
                    return matches[0]
        
        # 查找epoch checkpoint (排除best)
        epoch_pattern = "*.pth"
        all_checkpoints = glob.glob(os.path.join(self.checkpoint_dir, epoch_pattern))
        
        # 过滤掉best checkpoint
        epoch_checkpoints = [f for f in all_checkpoints if 'best' not in os.path.basename(f)]
        
        if not epoch_checkpoints:
            raise FileNotFoundError(f"No epoch checkpoints found in {self.checkpoint_dir}")
        
        # 按修改时间排序，返回最新的
        latest_checkpoint = max(epoch_checkpoints, key=os.path.getmtime)
        return latest_checkpoint
    
    def load_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        加载checkpoint信息
        Args:
            checkpoint_path: checkpoint文件路径
        Returns:
            checkpoint信息字典
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        info = {
            'checkpoint_path': checkpoint_path,
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'model_state_dict': checkpoint.get('model_state_dict'),
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
            'scheduler_state_dict': checkpoint.get('scheduler_state_dict'),
        }
        
        return info
    
    def load_original_config(self) -> Optional[Dict[str, Any]]:
        """
        加载原始训练配置
        Returns:
            原始配置字典，如果不存在则返回None
        """
        config_path = os.path.join(self.run_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return None
    
    def get_resume_info(self, prefer_best: bool = False) -> Dict[str, Any]:
        """
        获取完整的恢复信息
        Args:
            prefer_best: 是否优先选择best checkpoint
        Returns:
            包含所有恢复信息的字典
        """
        latest_checkpoint = self.find_latest_checkpoint(prefer_best)
        checkpoint_info = self.load_checkpoint_info(latest_checkpoint)
        original_config = self.load_original_config()
        
        resume_info = {
            'checkpoint_info': checkpoint_info,
            'original_config': original_config,
            'run_dir': self.run_dir,
            'checkpoint_dir': self.checkpoint_dir,
        }
        
        print(f"=== RESUME INFO ===")
        print(f"Checkpoint: {os.path.basename(latest_checkpoint)}")
        print(f"Resume from epoch: {checkpoint_info['epoch']}")
        print(f"Original run dir: {self.run_dir}")
        if checkpoint_info['metrics']:
            print(f"Last metrics: {checkpoint_info['metrics']}")
        
        return resume_info