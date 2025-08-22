# 
import os, csv, json, torch
from datetime import datetime
from typing import Dict, Any

class OutputManager:
    def __init__(self, model_type: str = "baseline", output_dir: str = "./outputs"):
        self.model_type = model_type
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.output_dir, f"{self.model_type}_{self.timestamp}")

        # build basic directory structure
        self._setup_directories()

    def _setup_directories(self):
        # construct basic dir
        dirs = [
            self.run_dir,
            self.get_vis_dir(),
            self.get_checkpoints_dir(),
        ]

        for dir_path in dirs: # loop and make
            os.makedirs(dir_path, exist_ok=True)

    def get_vis_dir(self) -> str:
        return os.path.join(self.run_dir, "visualizations")

    def get_checkpoints_dir(self) -> str:
        return os.path.join(self.run_dir, "checkpoints")

    def get_metrics_csv_path(self) -> str:
        return os.path.join(self.run_dir, "metrics.csv")

    def save_config(self, config: Dict):
        config_path = os.path.join(self.run_dir, "config.json")
        config_with_meta = {
            "timestamp": self.timestamp,
            "model_type": self.model_type,
            "config": config
        }

        with open(config_path, "w") as f:
            json.dump(config_with_meta, f, indent=2)

        print(f"Config saved to: {config_path}")

    def save_metrics_csv(self, metrics: Dict, epoch: int):
        # Append metrics to CSV
        csv_path = self.get_metrics_csv_path()

        # data row
        row_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch,
        }
        row_data.update(metrics)  # 🐛 修复：添加实际的指标数据

        # check header
        write_header = not os.path.exists(csv_path)

        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            if write_header: # if file is new, write header first
                writer.writeheader()
            writer.writerow(row_data) # write the data row

    def save_model(self, model, epoch: int, metrics: Dict):
        checkpoint = { # Create a checkpoint dictionary
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': self.timestamp
        }

        # save the newest model
        model_path = os.path.join(
            self.get_checkpoints_dir(),
            f"{self.model_type}_epoch_{epoch:03d}.pth"
        )
        torch.save(checkpoint, model_path) # Save the model checkpoint
        print(f"Model saved to: {model_path}")

        return model_path

    def get_run_summary(self) -> Dict: # get summary of current run
        return {
            "run_dir": self.run_dir,
            "timestamp": self.timestamp,
            "model_type": self.model_type,
            "file": {
                "metrics_csv": self.get_metrics_csv_path(),
                "vis_dir": self.get_vis_dir(),
                "checkpoints_dir": self.get_checkpoints_dir(),
            }
        }

    # get checkpoint with best performance
    def get_best_model_path(self) -> str:
        checkpoints_dir = self.get_checkpoints_dir()
        csv_path = self.get_metrics_csv_path()

        if not os.path.exists(checkpoints_dir) or not os.path.exists(csv_path):
            return ""

        try:
            import pandas as pd
            df = pd.read_csv(csv_path)

            if df.empty:
                return ""
            
            # by val_loss find the best epoch
            if 'val_loss' in df.columns:
                best_epoch = df.loc[df['val_loss'].idxmin(), 'epoch']
                print(f"Best epoch by validation loss: {best_epoch} with val_loss: {df['val_loss'].min():.4f}")
            elif 'iou' in df.columns:
                best_epoch = df.loc[df['iou'].idxmax(), 'epoch']
                print(f"Best epoch by IoU: {best_epoch} with IoU: {df['iou'].max():.4f}")
            else:
                best_epoch = df['epoch'].max()
                print(f"No specific metric found, using latest epoch: {best_epoch}")

            model_fname = f"{self.model_type}_epoch_{best_epoch:03d}.pth"
            model_path  = os.path.join(checkpoints_dir, model_fname)

            if os.path.exists(model_path):
                print(f"Best model found: {model_path}")
                return model_path
            else:
                print(f"Best model file not found: {model_path}, using latest")
                return self._get_latest_checkpoint()  # Fallback to latest checkpoint
        
        except Exception as e:
            print(f"Error occurred while getting best model path: {e}, using latest")
            return self._get_latest_checkpoint()  # Fallback to latest checkpoint

    # get latest checkpoint
    def _get_latest_checkpoint(self) -> str:
        # get path
        checkpoints_dir = self.get_checkpoints_dir()

        # locate all ck files
        if os.path.exists(checkpoints_dir):
            checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")]
            # retrive latest model
            latest = max(checkpoints, key = lambda x: os.path.getctime(os.path.join(checkpoints_dir, x)))
            return os.path.join(checkpoints_dir, latest)

        return ""

    # gain metrics history
    def get_metrics_history(self) -> list:
        # get csv path
        csv_path = self.get_metrics_csv_path()
        if not os.path.exists(csv_path):
            return []
        
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            return df.to_dict('records')  # convert to list of dicts
        except Exception as e:
            print(f"Error reading metrics CSV: {e}")
            return []
        

# ---------- 预留扩展接口 ---------- 
    def save_advanced_checkpoint(self, model, optimizer, scheduler, **kwargs):
        """预留接口：保存完整检查点"""
        # TODO: 未来实现完整的检查点保存（包含优化器、调度器等）
        pass
    
    def setup_tensorboard(self, log_dir=None):
        """预留接口：设置TensorBoard日志"""
        # TODO: 未来添加TensorBoard支持
        pass
    
    def export_results_summary(self, format='json'):
        """预留接口：导出结果摘要"""
        # TODO: 未来实现结果导出功能
        pass
