#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single model training script - smart_parallel.py adapter
Supports command line arguments: --model, --type, --gpu, --log-dir
"""

import os
import sys

# Fix MKL conflict before importing any other modules
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
# 确定性 CUDA/cuBLAS（需在首次 CUDA 使用前设置，利于复现）
if os.getenv('CUBLAS_WORKSPACE_CONFIG') is None:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Import numpy to initialize MKL first
import numpy as np

import argparse
import torch
import pandas as pd
from datetime import datetime

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Single model training script')
    parser.add_argument('--model', type=str, required=True, help='Model name, s_mamba, s_mamba_full, ...')
    parser.add_argument('--type', type=str, default='standard', choices=['standard'], help='Model type')
    parser.add_argument('--test', action='store_true', help='仅进行测试，不进行训练')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number')
    parser.add_argument('--log-dir', type=str, default='./trash/smart_parallel_logs_single_model', help='Log directory')
    parser.add_argument('--wandb-name', type=str, default=None, help='WandB run 名：可填完整名如 MambaHSI_zx，或仅后缀如 zx（会变成 model_zx）')
    parser.add_argument('--pth-dir', type=str, default='/mnt/raid/zhengsen/pths/new_dataset_pths/new_experiments_10to1/', help='测试模式下：待评估的 .pth 模型所在目录')
    zscore_group = parser.add_mutually_exclusive_group()
    zscore_group.add_argument('--use-zscore', dest='use_zscore', action='store_true',
                              help='Enable train-only channel-wise z-score standardization')
    zscore_group.add_argument('--no-zscore', dest='use_zscore', action='store_false',
                              help='Disable channel-wise z-score standardization')
    parser.set_defaults(
        use_zscore=os.getenv('WF_USE_ZSCORE', '0').strip().lower() in ('1', 'true', 'yes', 'on')
    )
    parser.add_argument('--zscore-stats', type=str, default=os.getenv('WF_ZSCORE_STATS_PATH', ''),
                        help='Z-score H5 path; default is generated beside the yearly H5 files')
    parser.add_argument('--force-zscore-recompute', action='store_true',
                        help='Recompute training statistics before training/testing')
    return parser.parse_args()

def setup_environment(gpu_id):
    """Set up training environment"""
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Wait a moment to ensure environment variables take effect
    import time
    time.sleep(0.1)
    
    return True

def train_single_model_task(model_name, model_type, gpu_id, log_dir, wandb_run_name=None):
    """Train a single model task"""
    print(f"🚀 Starting single model training: {model_name} ({model_type}) on GPU {gpu_id}")
    
    # Set environment (including CUDA_VISIBLE_DEVICES)
    setup_environment(gpu_id)
    
    # Import training related modules
    from train_all_h5 import (
        set_seed, TRAINING_CONFIG, prepare_data_loaders, FIRMSNormalizer,
        DATA_CONFIG, train_single_model, test_model, save_structured_results_to_csv,
        worker_init_fn, configure_channel_zscore_normalizer
    )
    from torch.utils.data import DataLoader
    
    # Initialize（固定种子并生成 DataLoader 用的 generator，保证每次运行一致）
    seed = TRAINING_CONFIG['seed']
    set_seed(seed)
    train_generator = torch.Generator().manual_seed(seed)
    
    # Verify GPU settings
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # Since CUDA_VISIBLE_DEVICES is set, it will always be 0
        actual_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(actual_gpu)
        print(f"🖥️  Using device: Physical GPU {gpu_id} -> cuda:0 ({gpu_name})")
        
        # Verify GPU memory
        gpu_memory = torch.cuda.get_device_properties(actual_gpu).total_memory / 1024**3
        print(f"💾 GPU memory: {gpu_memory:.1f} GB")
    else:
        device = torch.device('cpu')
        print(f"⚠️  CUDA not available, using CPU")
        return False
    
    try:
        # Prepare data
        print(" Preparing data...")
        train_dataset, val_dataset, test_dataset, data_loader_obj, data_loader_test = prepare_data_loaders()
        
        # Initialize FIRMS normalizer
        print("🔧 Initializing FIRMS normalizer...")
        firms_normalizer = FIRMSNormalizer(
            method='minmax',
            firms_min=0,
            firms_max=4
        )
        
        # WF_REPRODUCIBLE=1 时强制单进程，保证数据顺序完全确定
        force_repro = os.getenv('WF_REPRODUCIBLE', '0') == '1'
        if force_repro:
            train_workers = val_workers = test_workers = 0
            print("🔒 WF_REPRODUCIBLE=1: num_workers=0 for deterministic order")
        else:
            cpu_count = os.cpu_count() or 8
            train_workers = int(os.getenv('WF_TRAIN_WORKERS', str(min(12, cpu_count))))
            val_workers = int(os.getenv('WF_VAL_WORKERS', str(min(6, cpu_count))))
            test_workers = int(os.getenv('WF_TEST_WORKERS', str(min(6, cpu_count))))
        use_persistent = (train_workers > 0)
        
        # Create a temporary data loader for quick fitting (reduce worker count)
        temp_loader = DataLoader(
            train_dataset, batch_size=1024, shuffle=False,
            num_workers=1 if not force_repro else 0,
            collate_fn=data_loader_obj.dataset.custom_collate_fn
        )
        firms_normalizer.fit(temp_loader)
        configure_channel_zscore_normalizer(firms_normalizer, train_dataset)
        
        # Create data loaders (optimized performance settings)
        config_key = model_type
        train_config = TRAINING_CONFIG[config_key]
        
        train_loader = DataLoader(
            train_dataset, batch_size=train_config['batch_size'], shuffle=True,
            num_workers=train_workers, collate_fn=data_loader_obj.dataset.custom_collate_fn,
            worker_init_fn=worker_init_fn, pin_memory=True, persistent_workers=use_persistent,
            prefetch_factor=4,
            generator=train_generator,  # 固定 shuffle 顺序
        )
        val_loader = DataLoader(
            val_dataset, batch_size=train_config['batch_size'], shuffle=False,
            num_workers=val_workers, collate_fn=data_loader_obj.dataset.custom_collate_fn,
            worker_init_fn=worker_init_fn, pin_memory=True, persistent_workers=use_persistent,
            prefetch_factor=4
        )
        # Use appropriate collate_fn for test dataset
        if data_loader_test is not None:
            test_collate_fn = data_loader_test.dataset.custom_collate_fn
        else:
            test_collate_fn = data_loader_obj.dataset.custom_collate_fn
            
        test_loader = DataLoader(
            test_dataset, batch_size=train_config['batch_size'], shuffle=False,
            num_workers=test_workers, collate_fn=test_collate_fn,
            worker_init_fn=worker_init_fn, pin_memory=True, persistent_workers=use_persistent,
            prefetch_factor=4
        )
        
        # Train model
        print(f"🔥 Starting training {model_name}...")
        result = train_single_model(
            model_name, device, train_loader, val_loader, test_loader, firms_normalizer, model_type,
            wandb_run_name=wandb_run_name, finish_wandb=False
        )
        
        if result is None:
            print(f"❌ Training failed for {model_name} ({model_type})")
            return False
        
        print(f"✅ Training completed for {model_name} ({model_type})")
        
        # Test all saved models
        print(f"🧪 Starting to test all saved models for {model_name}...")
        
        # Dictionary to store structured test results
        structured_results = {model_name: {
            'f1': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'recall': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'pr_auc': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'mae': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'mse': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None},
            'final_epoch': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'mse': None, 'mae': None}
        }}
        
        # Test best models and final epoch models
        for metric_name, metric_info in result.items():
            if metric_info['path'] is not None:
                if metric_name == 'final_epoch':
                    print(f"📊 Testing final_epoch model...")
                else:
                    print(f"📊 Testing {metric_name} model...")
                
                try:
                    test_result = test_model(model_name, metric_info['path'], device, test_loader, firms_normalizer, model_type)
                    if test_result:
                        # Save to structured results
                        structured_results[model_name][metric_name] = {
                            'precision': test_result['precision'],
                            'recall': test_result['recall'],
                            'f1': test_result['f1'],
                            'pr_auc': test_result['pr_auc'],
                            'roc_auc': test_result.get('roc_auc', 0.0),
                            'fpr': test_result.get('fpr', 1.0),
                            'mse': test_result['mse'],
                            'mae': test_result['mae'],
                            'accuracy': test_result.get('accuracy', 0.0)
                        }
                        print(f"   P={test_result['precision']:.4f}, R={test_result['recall']:.4f}, F1={test_result['f1']:.4f}, PR-AUC={test_result['pr_auc']:.4f}, MSE={test_result['mse']:.6f}, MAE={test_result['mae']:.6f}")
                except Exception as e:
                    print(f"❌ Testing failed for {model_name} ({metric_name}): {str(e)}")
        
        # Save results to CSV file
        print(f"💾 Saving test results...")
        
        # Prepare CSV data
        csv_data = []
        columns = ['Model']
        metric_types = ['f1', 'recall', 'pr_auc', 'mae', 'mse', 'final_epoch']
        metric_names = ['precision', 'recall', 'f1', 'pr_auc', 'mse', 'mae']
        
        for metric_type in metric_types:
            for metric_name in metric_names:
                display_type = "final_epoch" if metric_type == 'final_epoch' else f"best_{metric_type}"
                columns.append(f"{display_type}_{metric_name}")
        
        # Add data rows
        row = [model_name]
        for metric_type in metric_types:
            for metric_name in metric_names:
                value = structured_results[model_name][metric_type][metric_name]
                if value is not None:
                    row.append(f"{value:.6f}")
                else:
                    row.append("N/A")
        csv_data.append(row)
        
        # Save to CSV file
        df = pd.DataFrame(csv_data, columns=columns)
        csv_filename = os.path.join(log_dir, f"{model_name}_{model_type}_results.csv")
        df.to_csv(csv_filename, index=False)
        
        # Save summary file
        summary_filename = os.path.join(log_dir, f"{model_name}_{model_type}_summary.txt")
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write(f"Model training and testing summary - {model_name} ({model_type})\n")
            f.write(f"{'='*60}\n")
            f.write(f"Training time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"GPU device: {gpu_id}\n\n")
            
            f.write("Test results:\n")
            f.write("-" * 40 + "\n")
            for metric_type in metric_types:
                metrics = structured_results[model_name][metric_type]
                if metrics['precision'] is not None:
                    display_type = "FINAL" if metric_type == 'final_epoch' else metric_type.upper()
                    f.write(f"{display_type:<12} P={metrics['precision']:<8.4f} R={metrics['recall']:<8.4f} F1={metrics['f1']:<8.4f} PR-AUC={metrics['pr_auc']:<8.4f} MSE={metrics['mse']:<10.6f} MAE={metrics['mae']:<10.6f}\n")
        
        print(f"📄 Summary saved: {summary_filename}")
        
        # 同一 run 内以柱状图写入 test/ 分组（与 train/val 区分），然后 finish
        try:
            from train_all_h5 import TRAINING_CONFIG, WANDB_AVAILABLE, _wandb_log_test_bar_chart
            if TRAINING_CONFIG.get('use_wandb') and WANDB_AVAILABLE:
                import wandb
                test_log = {}
                for metric_type, metrics in structured_results[model_name].items():
                    if metrics.get('precision') is not None:
                        prefix = f"test/{metric_type}"
                        test_log[f"{prefix}/accuracy"] = metrics.get('accuracy', 0.0)
                        test_log[f"{prefix}/precision"] = metrics['precision']
                        test_log[f"{prefix}/recall"] = metrics['recall']
                        test_log[f"{prefix}/f1"] = metrics['f1']
                        test_log[f"{prefix}/pr_auc"] = metrics['pr_auc']
                        test_log[f"{prefix}/roc_auc"] = metrics.get('roc_auc', 0.0)
                        test_log[f"{prefix}/fpr"] = metrics.get('fpr', 1.0)
                        test_log[f"{prefix}/mse"] = metrics['mse']
                        test_log[f"{prefix}/mae"] = metrics['mae']
                if test_log and wandb.run is not None:
                    _wandb_log_test_bar_chart(test_log)
                    print("✅ Test metrics (test/ group) logged as bar chart")
                if wandb.run is not None:
                    wandb.finish()
        except Exception as e:
            print(f"⚠️ WandB test logging failed: {e}")
        
        print(f"🎉 Training and testing completed for {model_name} ({model_type})!")
        
        return True
        
    except Exception as e:
        print(f"💥 An exception occurred during training: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_only_task(model_name, model_type, gpu_id, log_dir, pth_dir):
    """只进行测试：加载指定目录下的 .pth 模型并在统一的 test_loader 上评估"""
    print(f"🧪 Test-only mode for {model_name} ({model_type}) on GPU {gpu_id}")
    setup_environment(gpu_id)

    from train_all_h5 import (
        set_seed, TRAINING_CONFIG, prepare_data_loaders, FIRMSNormalizer,
        DATA_CONFIG, test_model, worker_init_fn, configure_channel_zscore_normalizer
    )
    from torch.utils.data import DataLoader

    seed = TRAINING_CONFIG['seed']
    set_seed(seed)
    train_generator = torch.Generator().manual_seed(seed)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        actual_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(actual_gpu)
        print(f"🖥️  Using device: Physical GPU {gpu_id} -> cuda:0 ({gpu_name})")
    else:
        device = torch.device('cpu')
        print(f"⚠️  CUDA not available, using CPU")
        return False

    # 数据与归一化器准备：与训练模式保持一致，确保 test_loader 一致
    print(" Preparing data (test-only)...")
    train_dataset, val_dataset, test_dataset, data_loader_obj, data_loader_test = prepare_data_loaders()

    print("🔧 Initializing FIRMS normalizer (test-only)...")
    firms_normalizer = FIRMSNormalizer(
        method='minmax',
        firms_min=0,
        firms_max=4
    )

    force_repro = os.getenv('WF_REPRODUCIBLE', '0') == '1'
    if force_repro:
        train_workers = val_workers = test_workers = 0
        print("🔒 WF_REPRODUCIBLE=1: num_workers=0 for deterministic order")
    else:
        cpu_count = os.cpu_count() or 8
        train_workers = int(os.getenv('WF_TRAIN_WORKERS', str(min(12, cpu_count))))
        val_workers = int(os.getenv('WF_VAL_WORKERS', str(min(6, cpu_count))))
        test_workers = int(os.getenv('WF_TEST_WORKERS', str(min(6, cpu_count))))
    use_persistent = (train_workers > 0)

    temp_loader = DataLoader(
        train_dataset, batch_size=1024, shuffle=False,
        num_workers=1 if not force_repro else 0,
        collate_fn=data_loader_obj.dataset.custom_collate_fn
    )
    firms_normalizer.fit(temp_loader)
    configure_channel_zscore_normalizer(firms_normalizer, train_dataset)

    config_key = model_type
    train_config = TRAINING_CONFIG[config_key]

    # 仅构建 test_loader，train/val loader 在 test-only 模式中不使用，但重用设置以保证 batch_size 一致
    if data_loader_test is not None:
        test_collate_fn = data_loader_test.dataset.custom_collate_fn
    else:
        test_collate_fn = data_loader_obj.dataset.custom_collate_fn

    test_loader = DataLoader(
        test_dataset, batch_size=train_config['batch_size'], shuffle=False,
        num_workers=test_workers, collate_fn=test_collate_fn,
        worker_init_fn=worker_init_fn, pin_memory=True, persistent_workers=use_persistent,
        prefetch_factor=4
    )

    # 扫描指定目录下的 .pth 模型
    if not os.path.isdir(pth_dir):
        print(f"❌ pth 目录不存在: {pth_dir}")
        return False

    all_files = sorted([
        os.path.join(pth_dir, f) for f in os.listdir(pth_dir)
        if f.endswith('.pth')
    ])
    if not all_files:
        print(f"❌ 在目录中未找到任何 .pth 文件: {pth_dir}")
        return False

    print(f"🧪 Found {len(all_files)} checkpoint(s) in {pth_dir}")

    results_rows = []
    columns = ['ckpt_name', 'precision', 'recall', 'f1', 'pr_auc', 'roc_auc', 'fpr', 'mse', 'mae', 'accuracy']

    for ckpt_path in all_files:
        ckpt_name = os.path.basename(ckpt_path)
        print(f"📊 Testing checkpoint: {ckpt_name}")
        try:
            test_result = test_model(model_name, ckpt_path, device, test_loader, firms_normalizer, model_type)
            if not test_result:
                print(f"⚠️  No result returned for {ckpt_name}")
                continue
            print(f"   P={test_result['precision']:.4f}, R={test_result['recall']:.4f}, "
                  f"F1={test_result['f1']:.4f}, PR-AUC={test_result['pr_auc']:.4f}, "
                  f"MSE={test_result['mse']:.6f}, MAE={test_result['mae']:.6f}")
            row = [
                ckpt_name,
                test_result['precision'],
                test_result['recall'],
                test_result['f1'],
                test_result['pr_auc'],
                test_result.get('roc_auc', 0.0),
                test_result.get('fpr', 1.0),
                test_result['mse'],
                test_result['mae'],
                test_result.get('accuracy', 0.0),
            ]
            results_rows.append(row)
        except Exception as e:
            print(f"❌ Testing failed for {ckpt_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    if not results_rows:
        print("⚠️ 所有 checkpoint 测试均失败或无有效结果")
        return False

    # 将结果保存为 CSV
    df = pd.DataFrame(results_rows, columns=columns)
    csv_filename = os.path.join(log_dir, f"{model_name}_{model_type}_test_only_results.csv")
    df.to_csv(csv_filename, index=False)
    print(f"💾 Test-only results saved to: {csv_filename}")

    print(f"🎉 Test-only evaluation completed for {model_name} ({model_type})!")
    return True

def main():
    """Main function"""
    args = parse_args()
    os.environ['WF_USE_ZSCORE'] = '1' if args.use_zscore else '0'
    if args.zscore_stats:
        os.environ['WF_ZSCORE_STATS_PATH'] = args.zscore_stats
    else:
        os.environ.pop('WF_ZSCORE_STATS_PATH', None)
    if args.force_zscore_recompute:
        os.environ['WF_ZSCORE_FORCE_RECOMPUTE'] = '1'
    else:
        os.environ.pop('WF_ZSCORE_FORCE_RECOMPUTE', None)
    
    if args.test:
        print(f"🔥 Single model TEST-ONLY")
    else:
        print(f"🔥 Single model trainer")
    print(f"📋 Model: {args.model}")
    print(f"📋 Type: {args.type}")
    print(f"📋 GPU: {args.gpu}")
    print(f"📋 Log directory: {args.log_dir}")
    print(f"📋 Channel z-score: {'enabled' if args.use_zscore else 'disabled'}")
    if args.use_zscore:
        print(f"📋 Z-score stats: {args.zscore_stats or 'auto (H5 directory)'}")
    if args.test:
        print(f"📋 Test checkpoints dir: {args.pth_dir}")
    print("=" * 50)
    
    # Ensure log directory exists
    os.makedirs(args.log_dir, exist_ok=True)
    
    if args.test:
        if args.pth_dir is None:
            print("❌ --test 模式下必须提供 --pth-dir")
            sys.exit(1)
        success = test_only_task(args.model, args.type, args.gpu, args.log_dir, args.pth_dir)
    else:
        # Set environment and train model
        success = train_single_model_task(args.model, args.type, args.gpu, args.log_dir, wandb_run_name=args.wandb_name)
    
    if success:
        print("🎉 Training completed successfully!")
        sys.exit(0)
    else:
        print("❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 
