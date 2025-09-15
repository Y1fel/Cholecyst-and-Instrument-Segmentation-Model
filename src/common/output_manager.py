# 
import os, csv, json, torch
from datetime import datetime
from typing import Dict, Any

class OutputManager:
    def __init__(self, model_type: str = "baseline", output_dir: str = "./outputs", run_dir: str = None):
        self.model_type = model_type
        self.output_dir = output_dir
        
        if run_dir is not None:
            # 使用指定的run目录（恢复训练时）
            self.run_dir = run_dir
            self.timestamp = os.path.basename(run_dir).split('_')[-1] if '_' in os.path.basename(run_dir) else "resumed"
        else:
            # 创建新的run目录（正常训练时）
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

    # Get run directory path
    def get_run_dir(self) -> str:
        return self.run_dir

    # Get path for visualizations
    def get_vis_dir(self) -> str:
        return os.path.join(self.run_dir, "visualizations")

    # Get path for specific visualization file
    def get_viz_path(self, filename: str) -> str:
        return os.path.join(self.get_vis_dir(), filename)

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
    def save_model(self, model, epoch: int, metrics:Dict, is_best: bool = False, model_suffix: str = ""):
        checkpoint = { # Create a checkpoint dictionary
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': self.timestamp,
            'is_best': is_best
        }

        # 构建文件名，支持model_suffix
        model_name = f"{self.model_type}_{model_suffix}" if model_suffix else self.model_type

        # save every checkpoint
        epoch_path = os.path.join(
            self.get_checkpoints_dir(),
            f"{model_name}_epoch_{epoch:03d}.pth"
        )
        torch.save(checkpoint, epoch_path)
        suffix_info = f" ({model_suffix})" if model_suffix else ""
        print(f"✓ Saved checkpoint{suffix_info}: {os.path.basename(epoch_path)}")

        # if best, save a copy as best
        if is_best:
            best_checkpoint = checkpoint.copy()
            best_checkpoint['best_epoch'] = epoch
            
            best_path = os.path.join(
                self.get_checkpoints_dir(),
                f"{model_name}_best.pth"
            )
            torch.save(best_checkpoint, best_path)
            print(f"BEST: Saved best model{suffix_info}: {os.path.basename(best_path)}")
            
            # update best model record
            self._update_best_model_record(epoch, metrics, best_path)
            
            return best_path
        
        return epoch_path

    # save checkpoint by interval
    def save_checkpoint_if_needed(self, model, epoch: int, metrics: Dict, 
                                current_best_metric: float, metric_name: str = 'val_loss',
                                minimize: bool = True, save_interval: int = 5, 
                                model_suffix: str = "", save_best: bool = True):
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
            model_suffix: 模型后缀（如"teacher", "student"）
            save_best: 是否保存最佳模型
        """
        current_metric = metrics.get(metric_name)
        if current_metric is None:
            print(f"Metric '{metric_name}' not found in metrics")
            return self.save_model(model, epoch, metrics, is_best=False, model_suffix=model_suffix), current_best_metric
        
        # check if best
        is_best = False
        new_best_metric = current_best_metric

        if save_best:  # 只有当save_best=True时才判断是否为最佳模型
            if minimize: # switch metric valuation standard by minimize
                if current_metric < current_best_metric:
                    is_best = True
                    new_best_metric = current_metric
            else:
                if current_metric > current_best_metric: # such as acc, the greater the better
                    is_best = True
                    new_best_metric = current_metric

        # save the model
        saved_path = self.save_model(model, epoch, metrics, is_best=is_best, model_suffix=model_suffix)

        if is_best and save_best:
            print(f"BEST: New best model at epoch {epoch} with {metric_name}: {current_metric:.4f}")
        elif epoch % save_interval == 0:
            suffix_info = f" ({model_suffix})" if model_suffix else ""
            print(f"SAVED: Regular checkpoint{suffix_info} saved at epoch {epoch}")

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
        
    def get_kd_experiment_dir(self) -> str:
        """获取KD实验专用目录"""
        kd_dir = os.path.join(self.run_dir, "kd_experiment")
        os.makedirs(kd_dir, exist_ok=True)
        return kd_dir
    
    def get_kd_comparison_csv_path(self) -> str:
        """获取KD对比表格CSV路径"""
        return os.path.join(self.get_kd_experiment_dir(), "kd_comparison_results.csv")
    
    def get_calibration_analysis_dir(self) -> str:
        """获取校准分析目录"""
        cal_dir = os.path.join(self.run_dir, "calibration_analysis")
        os.makedirs(cal_dir, exist_ok=True)
        return cal_dir
    
    def get_reliability_diagram_path(self, regime_name: str) -> str:
        """获取可靠性图保存路径"""
        return os.path.join(self.get_calibration_analysis_dir(), 
                           f"reliability_diagram_{regime_name}.png")
    
    def save_kd_experiment_summary(self, experiment_data: Dict):
        """保存KD实验总结"""
        summary_path = os.path.join(self.get_kd_experiment_dir(), "experiment_summary.json")
        
        summary = {
            "timestamp": self.timestamp,
            "experiment_type": "knowledge_distillation_comparison",
            "run_dir": self.run_dir,
            "experiment_data": experiment_data,
            "files": {
                "comparison_csv": self.get_kd_comparison_csv_path(),
                "calibration_dir": self.get_calibration_analysis_dir(),
                "visualizations": self.get_vis_dir()
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"KD experiment summary saved to: {summary_path}")
        return summary_path