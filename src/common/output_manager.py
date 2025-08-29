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

    # Setup directories for the current run
    def _setup_directories(self):
        # construct basic dir
        dirs = [
            self.run_dir,
            self.get_vis_dir(),
            self.get_checkpoints_dir(),
        ]

        for dir_path in dirs: # loop and make
            os.makedirs(dir_path, exist_ok=True)

    # Get path for visualizations
    def get_vis_dir(self) -> str:
        return os.path.join(self.run_dir, "visualizations")

    # Get path for checkpoints
    def get_checkpoints_dir(self) -> str:
        return os.path.join(self.run_dir, "checkpoints")

    # Get path for metrics CSV
    def get_metrics_csv_path(self) -> str:
        return os.path.join(self.run_dir, "metrics.csv")

    # Save config to JSON
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

    # Save metrics to CSV
    def save_metrics_csv(self, metrics: Dict, epoch: int):
        # Append metrics to CSV
        csv_path = self.get_metrics_csv_path()

        # data row
        row_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch,
        }
        row_data.update(metrics)  #修复：添加实际的指标数据

        # check header
        write_header = not os.path.exists(csv_path)

        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            if write_header: # if file is new, write header first
                writer.writeheader()
            writer.writerow(row_data) # write the data row

    # obtain run summary
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

    # save model
    def save_model(self, model, epoch: int, metrics:Dict, is_best: bool = False):
        checkpoint = { # Create a checkpoint dictionary
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': self.timestamp,
            'is_best': is_best
        }

        # save every checkpoint
        epoch_path = os.path.join(
            self.get_checkpoints_dir(),
            f"{self.model_type}_epoch_{epoch:03d}.pth"
        )
        torch.save(checkpoint, epoch_path)
        print(f"✓ Saved checkpoint: {os.path.basename(epoch_path)}")

        # if best, save a copy as best
        if is_best:
            best_checkpoint = checkpoint.copy()
            best_checkpoint['best_epoch'] = epoch
            
            best_path = os.path.join(
                self.get_checkpoints_dir(),
                f"{self.model_type}_best.pth"
            )
            torch.save(best_checkpoint, best_path)
            print(f"BEST: Saved best model: {os.path.basename(best_path)}")
            
            # update best model record
            self._update_best_model_record(epoch, metrics, best_path)
            
            return best_path
        
        return epoch_path

    # save checkpoint by interval
    def save_checkpoint_if_needed(self, model, epoch: int, metrics: Dict, 
                                current_best_metric: float, metric_name: str = 'val_loss',
                                minimize: bool = True, save_interval: int = 5):
        """
        智能保存checkpoint，自动判断是否为最佳模型
        Args:
            model: 模型
            epoch: epoch
            metrics: 指标
            current_best_metric: 当前最佳指标值
            metric_name: 用于判断的指标名称
            minimize: True表示指标越小越好，False表示越大越好
            save_interval: 常规保存间隔
        """
        current_metric = metrics.get(metric_name)
        if current_metric is None:
            print(f"Metric '{metric_name}' not found in metrics")
            return self.save_model(model, epoch, metrics, is_best=False), current_best_metric
        
        # check if best
        is_best = False
        new_best_metric = current_best_metric

        if minimize: # switch metric valuation standard by minimize
            if current_metric < current_best_metric:
                is_best = True
                new_best_metric = current_metric
        else:
            if current_metric > current_best_metric: # such as acc, the greater the better
                is_best = True
                new_best_metric = current_metric

        # save the model
        saved_path = self.save_model(model, epoch, metrics, is_best=is_best)

        if is_best:
            print(f"BEST: New best model at epoch {epoch} with {metric_name}: {current_metric:.4f}")
        elif epoch % save_interval == 0:
            print(f"SAVED: Regular checkpoint saved at epoch {epoch}")

        return saved_path, new_best_metric


    def _update_best_model_record(self, epoch: int, metrics: Dict, model_path: str):
        best_record_path = os.path.join(self.run_dir, "best_model_info.json")
        best_info = {
            'best_epoch': epoch,
            'best_metrics': metrics,
            'model_path': model_path,
            'timestamp': self.timestamp,
            'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(best_record_path, 'w') as f:
            json.dump(best_info, f, indent=2)  

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
