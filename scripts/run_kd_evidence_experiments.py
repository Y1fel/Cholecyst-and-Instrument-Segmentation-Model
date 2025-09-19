#!/usr/bin/env python3
"""
KD Evidence Package Runner
Automated script to run three experimental configurations for comprehensive KD evidence generation

Experiments:
1. S-Equal: Student baseline with equal budget (video-aware split)
2. S-Long: Student with 3x training budget (video-aware split)
3. KD-Student: Knowledge distillation with video-aware split

Usage:
    python run_kd_evidence_experiments.py --data_root /path/to/data [--quick_test]
    python run_kd_evidence_experiments.py --data_root /path/to/data --only kd_student
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def run_experiment(config_path, data_root, experiment_name, quick_test=False):
    """Run a single experiment with the given configuration"""
    
    print(f"\n{'='*60}")
    print(f"🧪 RUNNING: {experiment_name}")
    print(f"📄 Config: {os.path.basename(config_path)}")
    print(f"{'='*60}")
    
    # Build command
    train_script = "src/training/offline/train_offline_universal.py"
    cmd = [
        sys.executable, train_script,
        "--config", config_path,
        "--data_root", data_root
    ]
    
    # Quick test mode: reduce epochs and samples
    if quick_test:
        cmd.extend([
            "--epochs", "3",
            "--evidence_samples", "50"
        ])
        print(f"🚀 Quick test mode: 3 epochs, 50 samples")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {experiment_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {experiment_name} failed (code {e.returncode})")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run KD Evidence Package Experiments")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory containing the dataset")
    parser.add_argument("--quick_test", action="store_true",
                       help="Run quick test with reduced epochs")
    parser.add_argument("--only", type=str, 
                       choices=["s_equal", "kd_student", "s_long"],
                       help="Run only specific experiment")
    
    args = parser.parse_args()
    
    # Configuration files (using existing ones)
    configs = {
        "S-Equal": "configs/experiments/s_equal_config.yaml",
        "S-Long": "configs/experiments/s_long_config.yaml",
        "KD-Student": "configs/experiments/kd_student_config.yaml"
    }
    
    # Check config files exist
    missing = [c for c in configs.values() if not os.path.exists(c)]
    if missing:
        print(f"❌ Missing configs: {missing}")
        return 1
    
    print(f"🔬 KD Evidence Package Experiments")
    print(f"📁 Data: {args.data_root}")
    print(f"⏰ Start: {datetime.now().strftime('%H:%M:%S')}")
    
    # Run experiments
    results = {}
    experiments_to_run = configs.items()
    
    if args.only:
        key_map = {
            "s_equal": "S-Equal", 
            "s_long": "S-Long",
            "kd_student": "KD-Student"
        }
        exp_name = key_map[args.only]
        experiments_to_run = [(exp_name, configs[exp_name])]
    
    for exp_name, config_path in experiments_to_run:
        results[exp_name] = run_experiment(config_path, args.data_root, exp_name, args.quick_test)
    
    # Summary
    successful = sum(results.values())
    print(f"\n📋 Results: {successful}/{len(results)} experiments successful")
    
    if successful > 0:
        print(f"💡 Check outputs/ directory for evidence packages")
        print(f"📊 Compare *_evidence_summary.csv files for analysis")
    
    return 0 if successful == len(results) else 1

if __name__ == "__main__":
    exit(main())
