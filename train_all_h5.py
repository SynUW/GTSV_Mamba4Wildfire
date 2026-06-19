#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Wildfire Forecasting Model Benchmark - Unified Version
Supports early stopping, F1 evaluation metric, best model testing, and CSV result export
"""
import os
# 确定性 cuBLAS，需在首次 import torch 之前设置
if os.getenv('CUBLAS_WORKSPACE_CONFIG') is None:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from dataload_h5 import YearTimeSeriesDataLoader, YearTimeSeriesPixelDataset
from model_adapter_unified import UnifiedModelAdapter

from torch.utils.data import Dataset, DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
import random
import math
import importlib
from datetime import datetime, timedelta
import glob
import torch.nn.functional as F
import warnings
import pandas as pd
import sys
import time
import argparse

from typing import Tuple, Dict

from channel_zscore import (
    env_flag,
    ensure_training_zscore_normalizer,
)


# Dynamically import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb is not installed, will skip wandb monitoring")


# 只展示 best_f1 的 7 个评价指标，每个指标一个 panel
TEST_METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "pr_auc", "mse", "mae")


def _wandb_log_test_bar_chart(test_log):
    """仅用 best_f1 结果：7 个评价指标，每个指标一个柱状图 panel（单柱 best_f1）。"""
    if not test_log or not WANDB_AVAILABLE or wandb.run is None:
        return
    try:
        # 只取 best_f1（key 为 test/f1/xxx）
        metrics_by_type = {}
        for key, value in test_log.items():
            parts = key.split("/")
            if len(parts) >= 3 and parts[0] == "test" and parts[1] == "f1":
                metric_type = parts[2]
                if metric_type in TEST_METRIC_NAMES:
                    metrics_by_type[metric_type] = float(value)
        
        charts = {}
        for metric_type in TEST_METRIC_NAMES:
            if metric_type not in metrics_by_type:
                continue
            val = metrics_by_type[metric_type]
            # 单柱：best_f1 对应一个值
            table = wandb.Table(data=[["best_f1", val]], columns=["checkpoint", "value"])
            bar_plot = wandb.plot.bar(
                table=table,
                label="checkpoint",
                value="value",
                title=f"Test {metric_type}",
            )
            charts[f"test/{metric_type}"] = bar_plot
        
        if charts:
            wandb.log(charts)
    except Exception as e:
        print(f"⚠️ WandB test bar chart failed: {e}")

# STAtten 等脉冲网络需在每步前 reset 神经元状态，避免 "backward through the graph a second time"
try:
    from spikingjelly.clock_driven import functional as sj_functional
    SJ_RESET_NET = getattr(sj_functional, "reset_net", None)
except Exception:
    SJ_RESET_NET = None

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration Parameters
# =============================================================================

# Global training configuration - unified management of all training-related parameters
# Whether to enable WandB monitoring
WANDB_ENABLED = True               # Whether to enable WandB monitoring
# 是否在脚本中使用 WANDB_API_KEY（True=按下面的 key/环境变量自动登录；False=完全依赖终端提前执行过 `wandb login`）
WANDB_USE_API_KEY = False
# 是否静默 WandB 控制台输出（True=不在终端打印 run history/summary 等信息）
WANDB_SILENT = True
# WandB 账户配置：
# 1. 从环境变量读取（推荐）：export WANDB_API_KEY="your_api_key"
# 2. 或在此处直接设置：WANDB_API_KEY = "your_api_key"
# 3. 获取 API key：登录 https://wandb.ai/settings 获取
WANDB_API_KEY = os.getenv('Untitled Key', "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")  # 环境变量优先，否则用此处默认 key (Lincoln Xu)
WANDB_ENTITY = None                # WandB entity/team name (可选，留 None 使用默认账户)
GLOBAL_SEED = 42                   # Global random seed
# DataLoader worker 复现用：set_seed() 会写入，worker_init_fn 用 base + worker_id
_WORKER_BASE_SEED = None
DEFAULT_PATIENCE = 6              # Default early stopping patience
METRIC_WARMUP_EPOCHS = 5           # First N epochs do not participate in best-metric comparison, checkpoint save, or early-stopping monitoring
DEFAULT_MAX_PARALLEL_PER_GPU = 2   # Default maximum parallel tasks per GPU

# Multi-task learning configuration
MULTITASK_CONFIG = {
    'firms_weight': 1,           # Loss weight for FIRMS prediction. Typical loss combination (other drivers loss*weight): FIRMS loss: 0.3112890124320984, Other drivers loss: 0.0020517727825790644
    'other_drivers_weight': 1.0,   # Loss weight for other drivers prediction
    'ignore_zero_values': True,    # Whether to ignore zero values in other drivers
    'loss_function': 'mse',       # Loss function type: 'huber', 'mse', 'mae'
    'loss_type': 'focal'          # Loss type selection: 'focal'(MultiTaskFocalLoss), 'kldiv'(MultiTaskKLDivLoss), or 'multitask'(MultiTaskLoss)
}

# Dataset year configuration
DEFAULT_TRAIN_YEARS = [2000,2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 
                      2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
DEFAULT_VAL_YEARS = [2021, 2022]
DEFAULT_TEST_YEARS = [2023, 2024]


# Model directory configuration
# target_all_channels = target_all_channels.clone()
# target_all_channels[:, :, 0] = (target_all_channels[:, :, 0] > 10).float() Don't forget to remove these 2 lines
STANDARD_MODEL_DIR = '/mnt/raid/zhengsen/pths/new_dataset_pths/new_experiments_10to1'

def print_config_status():
    """Print current configuration status"""
    print("📋 Current training configuration:")
    print(f"   WandB monitoring: {'✅ Enabled' if WANDB_ENABLED else '❌ Disabled'}")
    print(f"   Random seed: {GLOBAL_SEED}")
    print(f"   Default parallelism: {DEFAULT_MAX_PARALLEL_PER_GPU}/GPU")
    print(f"   Early Stopping patience: {DEFAULT_PATIENCE}")
    print(f"   Metric warmup epochs (no best/early-stop): {METRIC_WARMUP_EPOCHS}")
    print(f"   Multi-task Loss type: {MULTITASK_CONFIG['loss_type'].upper()}")
    print(f"   FIRMS weight: {MULTITASK_CONFIG['firms_weight']}")
    print(f"   Other drivers weight: {MULTITASK_CONFIG['other_drivers_weight']}")
    print(f"   Ignore zero values: {'✅' if MULTITASK_CONFIG['ignore_zero_values'] else '❌'}")
    print(f"   Regression loss function: {MULTITASK_CONFIG['loss_function']}")
    if MULTITASK_CONFIG['loss_type'] == 'focal':
        print(f"   Focal Loss α: {TRAINING_CONFIG['focal_alpha']}")
        print(f"   Focal Loss γ: {TRAINING_CONFIG['focal_gamma']}")
    elif MULTITASK_CONFIG['loss_type'] == 'kldiv':
        print(f"   KL divergence temperature parameter: 1.0")
    elif MULTITASK_CONFIG['loss_type'] == 'multitask':
        print(f"   Unified loss function: {MULTITASK_CONFIG['loss_function']}")
    
    # 🔥 New: Position and weather feature status
    print(f"\n🔧 Data feature configuration:")
    print(f"   Position features: {'✅ Enabled' if DATA_CONFIG['enable_position_features'] else '❌ Disabled'}")
    print(f"   Future weather data: {'✅ Enabled' if DATA_CONFIG['enable_future_weather'] else '❌ Disabled'}")
    print(f"   Channel z-score: {'✅ Enabled' if DATA_CONFIG['use_zscore'] else '❌ Disabled'}")
    if DATA_CONFIG['use_zscore']:
        print(f"   Z-score stats: {DATA_CONFIG.get('zscore_stats_path') or 'auto (H5 directory)'}")
    if DATA_CONFIG['enable_future_weather']:
        channels_str = ','.join(map(str, DATA_CONFIG['weather_channels']))
        print(f"   Weather channels: [{channels_str}] (Total {len(DATA_CONFIG['weather_channels'])})")
    
    # Calculate total input channels
    base_channels = 38
    additional_channels = 0
    if DATA_CONFIG['enable_position_features']:
        additional_channels += 4  # 🔥 Updated: 4 channels for position embedding
    if DATA_CONFIG['enable_future_weather']:
        additional_channels += len(DATA_CONFIG['weather_channels'])
    
    total_channels = base_channels + additional_channels
    if additional_channels > 0:
        print(f"   Input channels: {base_channels} (base) + {additional_channels} (features) = {total_channels} (total)")
    else:
        print(f"   Input channels: {total_channels} (standard)")

def get_model_save_dir(model_type='standard'):
    """Return model save directory: env MODEL_SAVE_DIR overrides TRAINING_CONFIG."""
    return os.environ.get('MODEL_SAVE_DIR') or TRAINING_CONFIG[model_type]['model_save_dir']

def is_model_trained(model_name, model_type='standard'):
    """
    Check if the model has been trained
    Determine by checking if the final_epoch model file exists
    """
    model_save_dir = get_model_save_dir(model_type)
    final_model_path = os.path.join(model_save_dir, f'{model_name}_final_epoch.pth')
    return os.path.exists(final_model_path)

def get_trained_model_paths(model_name, model_type='standard'):
    """
    Get all saved paths for trained models
    Returns a dictionary containing metric_name and path
    """
    model_save_dir = get_model_save_dir(model_type)
    metric_types = ['precision', 'recall', 'f1', 'pr_auc', 'roc_auc', 'fpr', 'mae', 'mse', 'final_epoch']
    
    trained_paths = {}
    for metric_type in metric_types:
        if metric_type == 'final_epoch':
            path = os.path.join(model_save_dir, f'{model_name}_final_epoch.pth')
        else:
            path = os.path.join(model_save_dir, f'{model_name}_best_{metric_type}.pth')
        
        if os.path.exists(path):
            trained_paths[metric_type] = {'path': path, 'score': 0.0}  # score will be updated during testing
    
    return trained_paths

def filter_trained_models(model_list, model_type='standard', force_retrain=False):
    """
    Filter trained models
    Returns (list of models to train, dictionary of trained models)
    """
    if force_retrain:
        print(f"🔄 Force retrain mode: will train all {len(model_list)} {model_type} models")
        return model_list, {}
    
    models_to_train = []
    trained_models = {}
    
    print(f"🔍 Checking {model_type} model training status...")
    
    for model_name in model_list:
        if is_model_trained(model_name, model_type):
            trained_paths = get_trained_model_paths(model_name, model_type)
            trained_models[model_name] = trained_paths
            print(f"✅ {model_name}: Training completed ({len(trained_paths)} saved versions)")
        else:
            models_to_train.append(model_name)
            print(f"❌ {model_name}: Needs training")
    
    print(f"\n📊 {model_type} model status statistics:")
    print(f"   Need training: {len(models_to_train)} models")
    print(f"   Training completed: {len(trained_models)} models")
    
    if models_to_train:
        print(f"   Will train: {', '.join(models_to_train)}")
    if trained_models:
        print(f"   Skip training: {', '.join(trained_models.keys())}")
    
    return models_to_train, trained_models

def get_all_models(model_zoo_path):
    """Get all available models in the specified model_zoo"""
    model_files = []
    if os.path.exists(model_zoo_path):
        for file in os.listdir(model_zoo_path):
            if file.endswith('.py') and not file.startswith('__') and file != 'trash':
                model_name = file[:-3]  # Remove .py extension
                model_files.append(model_name)
    return sorted(model_files)

# Get standard model list
MODEL_LIST_STANDARD = get_all_models('model_zoo')

# Filter out models containing "mamba_enex" in their names (but keep s_mamba, Mamba, etc.)
MODEL_LIST_STANDARD = [m for m in MODEL_LIST_STANDARD if 'mamba_enex' not in m.lower()]

print(f"Found {len(MODEL_LIST_STANDARD)} standard models: {MODEL_LIST_STANDARD}")

# Training configuration
# wandb_run_name: 若设置则作为本次运行的 WandB run 名称；否则为 f"{model_name}_{model_type}"
# 例如设为 "MambaHSI_zx" 或 "zx"（仅后缀时会在 train_single_model 中与 model_name 拼接）
TRAINING_CONFIG = {
    'use_wandb': WANDB_ENABLED,         # Use WandB configuration
    'wandb_run_name': 'Zhengsen',             # 自定义 WandB run 名，None 则用 model_name_model_type
    'seed': GLOBAL_SEED,                # Use random seed
    'patience': DEFAULT_PATIENCE,       # Use patience configuration
    'seq_len': 10,                      # Input sequence length
    'pred_len': 1,                      # Prediction sequence length
    'focal_alpha': 0.5,                 # Use optimal Focal Loss positive sample weight
    'focal_gamma': 2.0,                 # Focal Loss focus parameter
    
    # Standard model configuration
    'standard': {
        'epochs': 50,
        'batch_size': 16,  # batch size  16 
        'learning_rate': 1e-5,          # Lower learning rate
        'weight_decay': 1e-4,
        'T_0': 20,
        'T_mult': 2,
        'eta_min':1e-5,
        'max_grad_norm': 0.0,           # Enable gradient clipping to prevent gradient explosion; 0.0 means no clipping
        'model_save_dir': STANDARD_MODEL_DIR,
    },
}

# Data configuration
DATA_CONFIG = {
    'h5_dir': os.getenv('WF_H5_DIR', '/caribou/zhengsen/h5_dataset/newset_dataset_v7'),
    'train_years': DEFAULT_TRAIN_YEARS,
    'val_years': DEFAULT_VAL_YEARS,
    'test_years': DEFAULT_TEST_YEARS,
    
    # Underlying dataset configuration (load full data)
    'positive_ratio': 1.0,           # Load all positive samples at the bottom layer
    'pos_neg_ratio': 2.0,            # Positive to negative sample ratio 1:1 at the bottom layer. 负样本数量 = 正样本数量 × pos_neg_ratio
    
    # Test sampling override
    # If test_use_full_test is True -> use all negatives for test (pos_neg_ratio=999999)
    # Else if test_pos_neg_ratio is a number -> use that ratio for test
    # Else (None) -> inherit train/val setting (current behavior)
    'test_use_full_test': False,
    'test_pos_neg_ratio': 1.0,
    
    'resample_each_epoch': False,    # Disable resampling at the bottom layer, so always set to False
    'firms_min': 0,                  # Minimum value of FIRMS data (skip statistics)
    'firms_max': 4,                # Maximum value of FIRMS data (skip statistics)
    
    # Dynamic sampling configuration (sample per epoch)
    'enable_dynamic_sampling': True,   # Whether to enable dynamic sampling for training set
    'sampling_ratio': 0.5,            # Proportion of data to sample per epoch (0.0-1.0)
    
    # 🔥 New: Position information feature configuration
    'enable_position_features': False,  # Whether to enable position information feature (default enabled)
    'raster_size': (1392, 652),         # Image size (height, width), used for normalization of position 278, 130 or 130, 278 ???? 557, 261
    
    # 🔥 New: Future weather data feature configuration  
    'enable_future_weather': False,    # Whether to enable future weather data feature (default disabled)
    'weather_channels': list(range(1, 13)),  # Weather data channel indices: 2-13 bands (indices 1-12)
    # Optional: imputed H5 to override past window
    'imputed_h5': '',              # Path to per-sample H5 (key: YYYYMMDD_row_col, data [39,L])
    'impute_mode': 'full',         # 'full' or 'replace_missing'

    # Optional train-only channel-wise z-score standardization.
    'use_zscore': env_flag('WF_USE_ZSCORE', False),
    'zscore_stats_path': os.getenv('WF_ZSCORE_STATS_PATH', ''),
    'zscore_force_recompute': env_flag('WF_ZSCORE_FORCE_RECOMPUTE', False),
    'zscore_batch_size': int(os.getenv('WF_ZSCORE_BATCH_SIZE', '8')),
    'patch_size': 13,

}

# =============================================================================
# Custom dynamic sampling dataset class
# =============================================================================

class DynamicSamplingSubset(Dataset):
    """
    Support dynamic sampling of subset (simplified version)
    Each epoch randomly samples a specified proportion of data from a balanced dataset
    Since the underlying dataset is already 1:1 balanced, random sampling will maintain a similar proportion
    """
    def __init__(self, dataset, full_indices, sampling_ratio=1.0, enable_dynamic_sampling=False):
        """
        Args:
            dataset: Original dataset (already 1:1 balanced)
            full_indices: List of complete indices
            sampling_ratio: Proportion of data to use per epoch (0.0-1.0)
            enable_dynamic_sampling: Whether to enable dynamic sampling
        """
        self.dataset = dataset
        self.full_indices = full_indices
        self.sampling_ratio = sampling_ratio
        self.enable_dynamic_sampling = enable_dynamic_sampling
        
        # Current indices being used
        if enable_dynamic_sampling and sampling_ratio < 1.0:
            self.current_indices = self._sample_indices(epoch_seed=42)
        else:
            self.current_indices = full_indices
            
        print(f"📊 DynamicSamplingSubset initialization:")
        print(f"   Total indices: {len(full_indices)}")
        print(f"   Current use: {len(self.current_indices)}")
        print(f"   Sampling ratio: {sampling_ratio:.1%}")
        print(f"   Dynamic sampling: {'Enabled' if enable_dynamic_sampling else 'Disabled'}")
    
    def _sample_indices(self, epoch_seed):
        """Randomly sample indices based on epoch seed"""
        if not self.enable_dynamic_sampling or self.sampling_ratio >= 1.0:
            return self.full_indices
            
        # Set random seed for reproducibility
        np.random.seed(epoch_seed)
        random.seed(epoch_seed)
        
        # Calculate sampling quantity
        sample_size = int(len(self.full_indices) * self.sampling_ratio)
        sample_size = max(1, sample_size)  # At least ensure 1 sample
        sample_size = min(sample_size, len(self.full_indices))  # No more than available quantity
        
        # Random sampling
        sampled_indices = np.random.choice(self.full_indices, size=sample_size, replace=False)
        return sampled_indices.tolist()
    
    def resample_for_epoch(self, epoch):
        """Resample for new epoch - random sampling order"""
        if not self.enable_dynamic_sampling:
            return
            
        # old_size = len(self.current_indices)
        sampled_indices = self._sample_indices(epoch_seed=42 + epoch)
        
        # Keep random order (no sorting) for complete randomness
        self.current_indices = sampled_indices
        
        # new_size = len(self.current_indices)
        # print(f"🔄 Epoch {epoch+1}: Re-sampling completed {old_size} → {new_size} samples (ratio: {self.sampling_ratio:.1%})")
    
    def __len__(self):
        return len(self.current_indices)
    
    def __getitem__(self, idx):
        # Map current indices to actual indices in the original dataset
        actual_idx = self.current_indices[idx]
        return self.dataset[actual_idx]

# =============================================================================
# Utility functions
# =============================================================================

def set_seed(seed):
    """Set random seed，尽量保证多次运行可复现。"""
    global _WORKER_BASE_SEED
    _WORKER_BASE_SEED = seed
    # 确定性 cuBLAS（需在首次 CUDA 运算前设置）
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 强制确定性算法（可能更慢或部分 op 报错，warn_only 仅警告不抛错）
    torch.use_deterministic_algorithms(True, warn_only=True)

def worker_init_fn(worker_id):
    """DataLoader worker 初始化：用 base_seed + worker_id 保证各 worker 种子确定、可复现。"""
    base = _WORKER_BASE_SEED if _WORKER_BASE_SEED is not None else GLOBAL_SEED
    worker_seed = (base + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class FIRMSNormalizer:
    """FIRMS data normalizer - simplified to use fixed min/max normalization"""
    
    def __init__(self, method='minmax', firms_min=0, firms_max=4):
        # Fixed min/max values
        self.method = method
        self.firms_min = firms_min
        self.firms_max = firms_max
        self.fitted = False
        self.channel_zscore = None
        
    def fit(self, data_loader):
        """Fit normalizer - no-op since we use fixed min/max"""
        self.fitted = True
        print(f"✅ FIRMS Normalizer initialized with fixed range: [{self.firms_min}, {self.firms_max}]")
        
    def normalize(self, firms_data):
        """Normalize FIRMS data using fixed min/max normalization"""
        if not self.fitted:
            raise ValueError("Normalizer not fitted, please call fit() method")
        
        # Convert 255 (NoData) and negative values to 0 before normalization
        firms_data = firms_data.clone()
        firms_data[firms_data == 255] = 0.0
        
        # Simple min/max normalization: (x - min) / (max - min)
        return (firms_data - self.firms_min) / (self.firms_max - self.firms_min)
    
    def transform_tensor(self, tensor_data):
        """Apply normalization transformation to tensor data (compatible method)"""
        return self.normalize(tensor_data)
    
    def inverse_transform_numpy(self, normalized_data):
        """Inverse transform normalized numpy data"""
        if not self.fitted:
            raise ValueError("Normalizer not fitted, please call fit() method")
        
        if isinstance(normalized_data, torch.Tensor):
            normalized_data = normalized_data.cpu().numpy()
        
        # Inverse min/max normalization: x = normalized * (max - min) + min
        return normalized_data * (self.firms_max - self.firms_min) + self.firms_min

class PositionEmbedding(nn.Module):
    """
    Learnable position embedding layer
    Maps 2D position coordinates (row, col) to a 4-dimensional embedding vector
    """
    def __init__(self, max_height=1392, max_width=652, embedding_dim=4):
        """
        Args:
            max_height: Maximum height of the raster (for row normalization)
            max_width: Maximum width of the raster (for col normalization)
            embedding_dim: Dimension of the output embedding (default: 4)
        """
        super(PositionEmbedding, self).__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.embedding_dim = embedding_dim
        
        # Create embedding tables for row and col positions
        # We use separate embeddings for row and col, then concatenate them
        # This allows the model to learn spatial relationships independently
        self.row_embedding = nn.Embedding(max_height, embedding_dim // 2)
        self.col_embedding = nn.Embedding(max_width, embedding_dim // 2)
        
        # Initialize embeddings with small random values
        nn.init.normal_(self.row_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.col_embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, row_indices, col_indices):
        """
        Forward pass: embed position coordinates
        
        Args:
            row_indices: Tensor of row indices [batch_size] (int, 0 to max_height-1)
            col_indices: Tensor of col indices [batch_size] (int, 0 to max_width-1)
        
        Returns:
            Position embeddings [batch_size, embedding_dim]
        """
        # Clamp indices to valid range
        row_indices = torch.clamp(row_indices, 0, self.max_height - 1)
        col_indices = torch.clamp(col_indices, 0, self.max_width - 1)
        
        # Get embeddings
        row_emb = self.row_embedding(row_indices)  # [batch_size, embedding_dim // 2]
        col_emb = self.col_embedding(col_indices)  # [batch_size, embedding_dim // 2]
        
        # Concatenate row and col embeddings
        pos_emb = torch.cat([row_emb, col_emb], dim=-1)  # [batch_size, embedding_dim]
        
        return pos_emb

# Global position embedding instance (will be initialized in training)
_position_embedding = None

def get_position_embedding(raster_size=(1392, 652), embedding_dim=4, device=None):
    """
    Get or create the global position embedding instance
    
    Args:
        raster_size: Image size (height, width)
        embedding_dim: Dimension of the embedding (default: 4)
        device: Device to place the embedding on
    
    Returns:
        PositionEmbedding instance
    """
    global _position_embedding
    if _position_embedding is None:
        height, width = raster_size
        _position_embedding = PositionEmbedding(
            max_height=height,
            max_width=width,
            embedding_dim=embedding_dim
        )
        if device is not None:
            _position_embedding = _position_embedding.to(device)
    return _position_embedding

def reset_position_embedding():
    """Reset the global position embedding (useful for testing or reinitialization)"""
    global _position_embedding
    _position_embedding = None

def add_position_features(data, metadata_list, raster_size, position_embedding=None):
    """
    Add position information feature to data using learnable embedding.
    
    统一按“空间模型”处理：
    - 输入 data 视为 5D 张量: (batch_size, channels, height, width, time_steps)
    - 位置 embedding 为每个样本生成一个 4 维向量，并在 (H, W, T) 上广播
    
    Args:
        data: Input data (batch_size, channels, height, width, time_steps)
        metadata_list: List of metadata, containing position information
        raster_size: Image size (height, width)
        position_embedding: PositionEmbedding instance (if None, uses global instance)
    
    Returns:
        Enhanced data with position features (batch_size, channels+4, height, width, time_steps)
    """
    batch_size, channels, height, width, time_steps = data.shape
    
    # Get position embedding instance
    if position_embedding is None:
        position_embedding = get_position_embedding(raster_size=raster_size, device=data.device)
    
    # Ensure embedding is on the correct device
    if next(position_embedding.parameters()).device != data.device:
        position_embedding = position_embedding.to(data.device)
    
    # Extract position indices from metadata
    row_indices = []
    col_indices = []
    
    for metadata in metadata_list:
        try:
            if isinstance(metadata, dict):
                row = metadata.get('row', 0)
                col = metadata.get('col', 0)
            elif hasattr(metadata, '__len__') and len(metadata) >= 3:
                if len(metadata) >= 3:
                    try:
                        row, col = int(metadata[1]), int(metadata[2])
                    except (ValueError, IndexError):
                        try:
                            row, col = int(metadata[2]), int(metadata[3])
                        except (ValueError, IndexError):
                            row, col = 0, 0
                else:
                    row, col = 0, 0
            else:
                row, col = 0, 0
        except Exception:
            row, col = 0, 0
        
        # Clamp to valid range
        row = max(0, min(int(row), height - 1))
        col = max(0, min(int(col), width - 1))
        
        row_indices.append(row)
        col_indices.append(col)
    
    # Convert to tensors
    row_indices_tensor = torch.tensor(row_indices, dtype=torch.long, device=data.device)
    col_indices_tensor = torch.tensor(col_indices, dtype=torch.long, device=data.device)
    
    # Get position embeddings [batch_size, embedding_dim=4]
    pos_embeddings = position_embedding(row_indices_tensor, col_indices_tensor)  # (B, 4)
    
    # Broadcast to match spatial-temporal dimensions:
    # (B, 4) -> (B, 4, 1, 1, 1) -> (B, 4, H, W, T)
    position_features = (
        pos_embeddings
        .unsqueeze(-1)  # (B, 4, 1)
        .unsqueeze(-1)  # (B, 4, 1, 1)
        .unsqueeze(-1)  # (B, 4, 1, 1, 1)
        .expand(-1, -1, height, width, time_steps)
    )
    
    # Concatenate position features to original data: channel dim +4
    enhanced_data = torch.cat([data, position_features], dim=1)
    return enhanced_data

def add_weather_features(past_data, future_data, weather_channels):
    """
    Extract weather features from future data and add to past data
    
    Args:
        past_data: Past data (batch_size, channels, past_time_steps)
        future_data: Future data (batch_size, channels, future_time_steps)  
        weather_channels: List of weather data channel indices
    
    Returns:
        Enhanced past data with weather features (batch_size, channels+len(weather_channels), past_time_steps)
    """
    batch_size, channels, past_time_steps = past_data.shape
    future_time_steps = future_data.shape[2]
    
    # Extract future weather data (batch_size, len(weather_channels), future_time_steps)
    future_weather = future_data[:, weather_channels, :]
    
    # Repeat or interpolate future weather data to match past time step length
    if future_time_steps != past_time_steps:
        # Use linear interpolation to adjust time dimension
        future_weather = F.interpolate(
            future_weather, 
            size=past_time_steps, 
            mode='linear', 
            align_corners=False
        )
    
    # Concatenate weather features to past data
    enhanced_past = torch.cat([past_data, future_weather], dim=1)
    return enhanced_past

def normalize_batch(past, future, firms_normalizer=None, metadata_list=None, position_embedding=None):
    """
    Normalize batch data and optionally add position information and weather data features
    
    Args:
        past: Past data (batch_size, channels, past_time_steps)
        future: Future data (batch_size, channels, future_time_steps)
        firms_normalizer: FIRMS data normalizer
        metadata_list: List of metadata, used for extracting position information
        position_embedding: PositionEmbedding instance (optional, will use global if None)
    
    Returns:
        Processed (past, future) data tuple
    """
    # 🔥 Key: First handle all NaN values, replace them with 0
    nan_mask_past = torch.isnan(past)
    past[nan_mask_past] = 0.0
    nan_mask_future = torch.isnan(future)
    future[nan_mask_future] = 0.0
    
    # Normalize the 0th channel (FIRMS) for both past and future
    # Note: FIRMSNormalizer.normalize() will handle 255->0 conversion internally
    if firms_normalizer is not None:
        past[:, 0, :] = firms_normalizer.normalize(past[:, 0, :])
        future[:, 0, :] = firms_normalizer.normalize(future[:, 0, :])

        channel_zscore = getattr(firms_normalizer, 'channel_zscore', None)
        if channel_zscore is not None:
            past = channel_zscore.transform(past, inplace=True)
    
    past[:, 0, :] = past[:, 0, :]
    future[:, 0, :] = future[:, 0, :]
    
    # 🔥 New: Add position information feature using learnable embedding
    if DATA_CONFIG['enable_position_features'] and metadata_list is not None:
        past = add_position_features(past, metadata_list, DATA_CONFIG['raster_size'], position_embedding=position_embedding)
        # Note: Future data usually doesn't need position feature addition, as position information is mainly used as input
        
    # 🔥 New: Add future weather data feature
    if DATA_CONFIG['enable_future_weather']:
        past = add_weather_features(past, future, DATA_CONFIG['weather_channels'])
    
    return past, future


def configure_channel_zscore_normalizer(firms_normalizer, train_dataset):
    """Attach train-only channel statistics to the existing normalizer."""
    if not DATA_CONFIG.get('use_zscore', False):
        firms_normalizer.channel_zscore = None
        print("ℹ️ Channel z-score disabled")
        return None

    stats_path = DATA_CONFIG.get('zscore_stats_path') or None
    normalizer = ensure_training_zscore_normalizer(
        train_dataset=train_dataset,
        h5_dir=DATA_CONFIG['h5_dir'],
        train_years=DATA_CONFIG['train_years'],
        lookback_seq=TRAINING_CONFIG['seq_len'],
        forecast_hor=TRAINING_CONFIG['pred_len'],
        patch_size=DATA_CONFIG.get('patch_size', 13),
        stats_path=stats_path,
        force=DATA_CONFIG.get('zscore_force_recompute', False),
        batch_size=DATA_CONFIG.get('zscore_batch_size', 8),
    )
    firms_normalizer.channel_zscore = normalizer
    DATA_CONFIG['zscore_stats_path'] = normalizer.stats_path
    print(f"✅ Channel z-score enabled: {normalizer.stats_path}")
    print(
        f"   channels={normalizer.num_channels}, "
        f"mean=[{normalizer.mean.min():.6g}, {normalizer.mean.max():.6g}], "
        f"std=[{normalizer.std.min():.6g}, {normalizer.std.max():.6g}]"
    )
    return normalizer

def load_model(model_name, configs, model_type='standard'):
    """Load model dynamically (using model_zoo)"""
    try:
        # Check for special dependencies
        if model_name in ['Mamba', 'Reformer', 'Transformer', 'iTransformer', 's_mamba']:
            try:
                import mamba_ssm
            except ImportError:
                print(f"⚠️ Model {model_name} requires mamba_ssm library")
                print(f"💡 Suggest using mamba_env environment: conda activate mamba_env")
                raise ImportError(f"Model {model_name} requires mamba_ssm library, please run in mamba_env environment")
        
        # Use model_zoo uniformly
        model_zoo_path = os.path.join(os.getcwd(), 'model_zoo')
        module_name = f'model_zoo.{model_name}'
        
        if model_zoo_path not in sys.path:
            sys.path.insert(0, model_zoo_path)
        
        module = importlib.import_module(module_name)
        Model = getattr(module, 'Model')
        
        return Model(configs), model_type
    except Exception as e:
        print(f"Failed to load {model_type} model {model_name}: {e}")
        raise

def calculate_detailed_metrics(output, target):
    """Calculate detailed regression and binary classification metrics, including MSE, MAE, PR-AUC, ROC-AUC, FPR"""
    # Original output values used for regression metrics
    output_raw = output.view(-1).cpu().numpy()
    target_np = target.view(-1).cpu().numpy()
    
    # Calculate MSE and MAE (regression metrics, using original output values)
    mse = np.mean((output_raw - target_np) ** 2)
    mae = np.mean(np.abs(output_raw - target_np))
    
    # Probability values from Sigmoid processing used for classification metrics
    pred_probs = torch.sigmoid(output).view(-1).cpu().numpy()
    pred_binary = (pred_probs > 0.5).astype(int)
    target_binary = (target_np > 0).astype(int)
    
    unique_targets = np.unique(target_binary)
    if len(unique_targets) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, mse, mae  # fpr defaults to 1.0 (worst case)
    
    try:
        precision = precision_score(target_binary, pred_binary, average='binary', zero_division=0)
        recall = recall_score(target_binary, pred_binary, average='binary', zero_division=0)
        f1 = f1_score(target_binary, pred_binary, average='binary', zero_division=0)
        pr_auc = average_precision_score(target_binary, pred_probs)
        roc_auc = roc_auc_score(target_binary, pred_probs)
        
        # Calculate FPR (False Positive Rate) = FP / (FP + TN)
        tp = np.sum((pred_binary == 1) & (target_binary == 1))
        fp = np.sum((pred_binary == 1) & (target_binary == 0))
        fn = np.sum((pred_binary == 0) & (target_binary == 1))
        tn = np.sum((pred_binary == 0) & (target_binary == 0))
        
        if fp + tn > 0:
            fpr = fp / (fp + tn)
        else:
            fpr = 0.0  # No negatives, FPR is 0
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, mse, mae
    
    return precision, recall, f1, pr_auc, roc_auc, fpr, mse, mae

def calculate_metrics_fixed_threshold(probs, targets, threshold=0.5):
    """
    使用固定阈值计算指标（输入是概率值，不是logits）
    
    参数:
        probs: 预测概率 [N] 或 [N, L] 或 [N, 1] (Tensor或numpy array)
        targets: 真实标签 [N] 或 [N, L] 或 [N, 1] (Tensor或numpy array, 0或1)
        threshold: 固定阈值，默认0.5
    
    返回:
        precision, recall, f1, pr_auc, roc_auc, fpr, mse, mae, accuracy
    """
    # 展平
    if isinstance(probs, torch.Tensor):
        probs_flat = probs.view(-1).cpu().numpy()
    else:
        probs_flat = np.array(probs).reshape(-1)
    
    if isinstance(targets, torch.Tensor):
        targets_flat = targets.view(-1).cpu().numpy()
    else:
        targets_flat = np.array(targets).reshape(-1)
    
    # 计算MSE和MAE（回归指标）
    mse = np.mean((probs_flat - targets_flat) ** 2)
    mae = np.mean(np.abs(probs_flat - targets_flat))
    
    # 二值化预测（使用固定阈值）
    pred_binary = (probs_flat > threshold).astype(int)
    target_binary = (targets_flat > 0.5).astype(int)
    
    # Accuracy = (TP+TN)/(TP+TN+FP+FN)
    accuracy = np.mean(pred_binary == target_binary).item() if pred_binary.size > 0 else 0.0

    unique_targets = np.unique(target_binary)
    if len(unique_targets) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, mse, mae, accuracy  # fpr defaults to 1.0 (worst case)
    
    try:
        precision = precision_score(target_binary, pred_binary, average='binary', zero_division=0)
        recall = recall_score(target_binary, pred_binary, average='binary', zero_division=0)
        f1 = f1_score(target_binary, pred_binary, average='binary', zero_division=0)
        pr_auc = average_precision_score(target_binary, probs_flat)
        roc_auc = roc_auc_score(target_binary, probs_flat)
        
        # Calculate FPR (False Positive Rate) = FP / (FP + TN)
        tp = np.sum((pred_binary == 1) & (target_binary == 1))
        fp = np.sum((pred_binary == 1) & (target_binary == 0))
        fn = np.sum((pred_binary == 0) & (target_binary == 1))
        tn = np.sum((pred_binary == 0) & (target_binary == 0))
        
        if fp + tn > 0:
            fpr = fp / (fp + tn)
        else:
            fpr = 0.0  # No negatives, FPR is 0
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, mse, mae, accuracy
    
    return precision, recall, f1, pr_auc, roc_auc, fpr, mse, mae, accuracy

# def calculate_optimal_f1_metrics(output, target):
#     """Calculate detailed metrics at optimal F1 threshold for testing - debugging version"""
#     # Original output values used for regression metrics
#     output_raw = output.view(-1).cpu().numpy()
#     target_np = target.view(-1).cpu().numpy()
    
#     # Calculate MSE and MAE (regression metrics, using original output values)
#     mse = np.mean((output_raw - target_np) ** 2)
#     mae = np.mean(np.abs(output_raw - target_np))
    
#     # Probability values from Sigmoid processing used for classification metrics
#     pred_probs = torch.sigmoid(output).view(-1).cpu().numpy()
#     target_binary = (target_np > 0).astype(int)
    
#     # 🔍 Debugging information: Analyze input data characteristics
#     print(f"   🔍 Data statistics:")
#     print(f"       Number of prediction samples: {len(pred_probs)}")
#     print(f"       Number of true positive samples: {np.sum(target_binary)}")
#     print(f"       Proportion of true positives: {np.sum(target_binary) / len(target_binary):.4f}")
#     print(f"       Range of predicted probabilities: [{np.min(pred_probs):.4f}, {np.max(pred_probs):.4f}]")
#     print(f"       Mean of predicted probabilities: {np.mean(pred_probs):.4f}")
#     print(f"       Standard deviation of predicted probabilities: {np.std(pred_probs):.4f}")
    
#     unique_targets = np.unique(target_binary)
#     if len(unique_targets) < 2:
#         return 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, mse, mae  # fpr defaults to 1.0 (worst case)
    
#     try:
#         # Calculate PR-AUC and ROC-AUC
#         pr_auc = average_precision_score(target_binary, pred_probs)
#         roc_auc = roc_auc_score(target_binary, pred_probs)
        
#         # Find optimal F1 threshold
#         thresholds = np.linspace(0, 1, 100)  # Search using 100 threshold points
#         best_f1 = 0.0
#         best_precision = 0.0
#         best_recall = 0.0
#         best_threshold = 0.5
#         best_fpr = 1.0  # Initialize with worst FPR
        
#         # 🔍 Debugging: Record metrics for all thresholds
#         all_recalls = []
#         all_precisions = []
#         all_f1s = []
        
#         for threshold in thresholds:
#             pred_binary_thresh = (pred_probs > threshold).astype(int)
            
#             # 🔍 Prevent division by zero error, add more detailed check
#             tp = np.sum((pred_binary_thresh == 1) & (target_binary == 1))
#             fp = np.sum((pred_binary_thresh == 1) & (target_binary == 0))
#             fn = np.sum((pred_binary_thresh == 0) & (target_binary == 1))
#             tn = np.sum((pred_binary_thresh == 0) & (target_binary == 0))
            
#             if tp + fp > 0:
#                 precision = tp / (tp + fp)
#             else:
#                 precision = 0.0
                
#             if tp + fn > 0:
#                 recall = tp / (tp + fn)
#             else:
#                 recall = 0.0
                
#             if precision + recall > 0:
#                 f1 = 2 * (precision * recall) / (precision + recall)
#             else:
#                 f1 = 0.0
            
#             # Calculate FPR for this threshold
#             if fp + tn > 0:
#                 fpr = fp / (fp + tn)
#             else:
#                 fpr = 0.0
            
#             all_recalls.append(recall)
#             all_precisions.append(precision)
#             all_f1s.append(f1)
            
#             if f1 >= best_f1:
#                 best_f1 = f1
#                 best_precision = precision
#                 best_recall = recall
#                 best_threshold = threshold
#                 best_fpr = fpr
        
#         # 🔍 Debugging information: Analyze distribution of recalls
#         all_recalls = np.array(all_recalls)
#         unique_recalls = np.unique(all_recalls)
#         print(f"       Found {len(unique_recalls)} different recall values")
#         print(f"      Range of recalls: [{np.min(all_recalls):.4f}, {np.max(all_recalls):.4f}]")
#         print(f"       Highest recall: {np.max(all_recalls):.6f}")
#         print(f"       Best F1 threshold: {best_threshold:.3f} (F1={best_f1:.4f})")
        
#         # 🔍 If all recalls are the same, there's a problem
#         if len(unique_recalls) == 1:
#             print(f"      ⚠️ Warning: All thresholds have the same recall = {unique_recalls[0]:.6f}")
#             print(f"      Possible cause: Model predictions are too concentrated or data distribution is abnormal")
            
#         # 🔍 Analyze threshold distribution
#         recall_counts = {}
#         for r in all_recalls:
#             r_rounded = round(r, 6)
#             recall_counts[r_rounded] = recall_counts.get(r_rounded, 0) + 1
        
#         print(f"      Top 5 recall values frequency:")
#         for r, count in sorted(recall_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
#             print(f"         {r:.6f}: {count} times")
        
#     except Exception as e:
#         print(f"Error calculating optimal F1 metrics: {e}")
#         return 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, mse, mae
    
#     return best_precision, best_recall, best_f1, pr_auc, roc_auc, best_fpr, mse, mae

class Config:
    """Configuration class - fixing type safety issues"""
    def __init__(self, model_name, model_type='standard'):
        self.model_name = model_name  # Add model name attribute
        self.model_type = model_type
        config = TRAINING_CONFIG[model_type]
        
        # Basic training parameters - ensure type safety
        self.epochs = int(config['epochs'])
        self.batch_size = int(config['batch_size'])
        self.learning_rate = float(config['learning_rate'])
        self.weight_decay = float(config['weight_decay'])
        self.T_0 = int(config['T_0'])
        self.T_mult = int(config['T_mult'])
        self.eta_min = float(config['eta_min'])
        self.max_grad_norm = float(config['max_grad_norm'])
        
        # Sequence parameters - ensure they are integers, avoiding Config object issues
        self.seq_len = int(TRAINING_CONFIG['seq_len'])
        self.pred_len = int(TRAINING_CONFIG['pred_len'])
        self.label_len = 0  # Default label length
        
        # Get configuration based on model type (using unified adapter)
        try:
            from model_adapter_unified import get_unified_model_configs
            model_configs = get_unified_model_configs(model_name, model_type)
            
            # Safely set configuration, ensuring correct numerical types
            for key, value in model_configs.items():
                if key in ['seq_len', 'pred_len']:
                    continue  # Skip, using fixed values we've set
                elif isinstance(value, (int, float, str, bool)):
                    setattr(self, key, value)
                elif value is None:
                    setattr(self, key, None)
                else:
                    # For complex types, try converting to basic type
                    try:
                        if isinstance(value, list):
                            setattr(self, key, value)
                        else:
                            setattr(self, key, value)
                    except:
                        print(f"⚠️ Skipping configuration {key}={value} (type: {type(value)})")
                        
        except Exception as e:
            print(f"⚠️ Dynamic configuration import failed: {e}, using default configuration")
            # Use default configuration
            self.d_model = 512
            self.n_heads = 8
            self.d_ff = 2048
            self.e_layers = 2
            self.d_layers = 2
            self.d_state = 16
            self.d_conv = 4
            self.expand = 2
            
            # General model parameters
            self.dropout = 0.1
            self.activation = 'gelu'
            self.output_attention = False
            self.enc_in = 38
            self.dec_in = 38
            self.c_out = 38
            self.embed = 'timeF'
            self.freq = 'd'
            self.factor = 1
            self.moving_avg = 25
            self.channel_independence = False
            self.use_norm = True  # True by default
            self.distil = True
            self.label_len = 3 if model_name in ['Autoformer', 'Autoformer_M'] else 0
        
        # Add special configuration needed for new models
        self.task_name = 'long_term_forecast'  # New models generally need this parameter
        
        # Add special configuration for specific models
        if model_name == 'DLinear':
            self.moving_avg = 25  # DLinear needs moving_avg for series_decomp
            self.individual = False  # DLinear's individual parameter
            
        elif model_name == 'CrossLinear':
            self.features = 'M'  # CrossLinear needs features parameter
            self.patch_len = 16  # CrossLinear needs patch-related parameters
            self.alpha = 0.5
            self.beta = 0.5
            
        elif model_name == 'TimesNet':
            self.top_k = 5  # TimesNet needs parameter
            self.num_kernels = 6
            
        elif model_name == 'Mamba':
            # Special parameters for Mamba are already set in the base configuration
            pass

        elif model_name == 'T4Fire':
            # ViT 规模：'tiny' | 'small' | 'base' | 'large'，默认 base
            self.vit_size = getattr(self, 'vit_size', 'base')
        
        # FIRMS data normalization parameters
        self.normalize_firms = True
        self.firms_normalization_method = 'divide_by_100'
        self.binarization_threshold = 0.0
        self.firms_min = int(DATA_CONFIG['firms_min'])
        self.firms_max = int(DATA_CONFIG['firms_max'])
        
        # Focal Loss parameters  
        self.focal_alpha = float(TRAINING_CONFIG['focal_alpha'])
        self.focal_gamma = float(TRAINING_CONFIG['focal_gamma'])
        
        # Multi-task learning parameters
        self.firms_weight = float(MULTITASK_CONFIG['firms_weight'])
        self.other_drivers_weight = float(MULTITASK_CONFIG['other_drivers_weight'])
        self.ignore_zero_values = MULTITASK_CONFIG['ignore_zero_values']
        self.loss_function = MULTITASK_CONFIG['loss_function']
        self.loss_type = MULTITASK_CONFIG['loss_type']  # New: Loss function type selection
        
        # Dataset split
        self.train_years = DATA_CONFIG['train_years']
        self.val_years = DATA_CONFIG['val_years']
        self.test_years = DATA_CONFIG['test_years']
        
        # 🔥 New: Dynamic update of model channel configuration
        self.update_model_channels()
    
    # 🔥 New: Dynamic calculation of input channel number
    def calculate_input_channels(self):
        """
        Calculate input channel number dynamically based on configuration
        Base channel number + position feature channel number + weather data channel number
        Position features: 4 channels (row_norm, col_norm, row_sin, col_sin)
        """
        base_channels = 38  # Base channel number
        additional_channels = 0
        
        # Position information feature (+4 channels) - use config object attributes first, otherwise use global configuration
        enable_position = getattr(self, 'enable_position_features', DATA_CONFIG['enable_position_features'])
        if enable_position:
            additional_channels += 4  # 🔥 Updated: 4 channels for position embedding
            
        # Weather data feature - use config object attributes first, otherwise use global configuration
        enable_weather = getattr(self, 'enable_future_weather', DATA_CONFIG['enable_future_weather'])
        if enable_weather:
            weather_channels = getattr(self, 'weather_channels', DATA_CONFIG['weather_channels'])
            additional_channels += len(weather_channels)
            
        return base_channels + additional_channels
    
    def update_model_channels(self):
        """Update model's input/output channel configuration"""
        # Dynamically calculate input channel number
        dynamic_enc_in = self.calculate_input_channels()
        
        # Update encoder input channel number
        self.enc_in = dynamic_enc_in
        
        # Decoder input channel number usually matches encoder
        self.dec_in = dynamic_enc_in
        
        # Output channel number remains 39 (predict all original channels)
        self.c_out = 38
        
        # Print channel information for debugging - use config object attributes instead of global configuration
        features_info = []
        enable_position = getattr(self, 'enable_position_features', DATA_CONFIG['enable_position_features'])
        enable_weather = getattr(self, 'enable_future_weather', DATA_CONFIG['enable_future_weather'])
        
        if enable_position:
            features_info.append("Position information(+4)")
        if enable_weather:
            weather_channels = getattr(self, 'weather_channels', DATA_CONFIG['weather_channels'])
            features_info.append(f"Weather data(+{len(weather_channels)})")
        
        if features_info:
            print(f"🔧 {self.model_name} Dynamic channel configuration: {self.enc_in} input -> {self.c_out} output (additional features: {', '.join(features_info)})")
        else:
            print(f"🔧 {self.model_name} Standard channel configuration: {self.enc_in} input -> {self.c_out} output")


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MultiTaskFocalLoss(nn.Module):
    def __init__(self, firms_weight=1.0, other_drivers_weight=0.1, 
                 focal_alpha=0.25, focal_gamma=2.0,
                 ignore_zero_values=True, regression_loss='mse'):
        super().__init__()
        self.firms_weight = firms_weight
        self.other_drivers_weight = other_drivers_weight
        self.ignore_zero_values = ignore_zero_values

        # 不要手动 sigmoid；用 logits + BCEWithLogits
        # 为了逐元素加权，这里不平均
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='none')
        
        self.focal_loss = nn.BCEWithLogitsLoss()

        if regression_loss == 'huber':
            self.regression_loss_fn = nn.HuberLoss(reduction='none')
        elif regression_loss == 'mse':
            self.regression_loss_fn = nn.MSELoss(reduction='none')
        elif regression_loss == 'mae':
            self.regression_loss_fn = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported regression loss function type: {regression_loss}")
        self.regression_loss_type = regression_loss

    def forward(self, predictions, targets):
        B, C_pred = predictions.shape
        B, C_tgt  = targets.shape
        if C_pred > C_tgt:
            predictions = predictions[:, :C_tgt]

        # --- 分类通道（FIRMS） ---
        firms_logits = predictions# [:, :, 0]          # (B, T) logits，不能先 sigmoid
        firms_target = targets[:, :1]            # (B, T) 连续值
        y = (firms_target > 0).float()               # 二分类标签 {0,1}
        
        # mse_loss = F.mse_loss(x_rec, x_before_manifold)

        if 1 == 0:
            # 使用历史记录加权
            
            # 正负样本权重
            pos_mask = (firms_target > 0).float()
            neg_mask = 1.0 - pos_mask

            # 防止 log(0) / 负值：先 clamp
            # 这里假设 firms_target 单位是“事件强度/面积比例”等非负量
            pos_weights = 1.0 + torch.log1p(torch.clamp(firms_target * 100.0 / 20.0, min=0.0))
            neg_weights = torch.full_like(firms_target, 1.02)

            sample_weights = pos_mask * pos_weights + neg_mask * neg_weights  # (B, T)

            # 逐元素 BCE（内部做 sigmoid），再乘权重、再平均
            elem_loss = self.bce_logits(firms_logits, y)          # (B, T)
            firms_loss = (elem_loss * sample_weights).mean() * self.firms_weight
        else:
            firms_loss = self.focal_loss(firms_logits, y) # * self.firms_weight

        # --- 其余通道（回归） ---
        # other_pred   = predictions[:, :, 1:]
        # other_target = targets[:,     :, 1:]
        other_loss = 0.0

        # elem_reg = self.regression_loss_fn(other_pred, other_target)  # (B, T, C-1)
        # if self.ignore_zero_values:
        #     mask = (other_target != 0.0).float()
        #     valid = mask.sum()
        #     other_loss = (elem_reg * mask).sum() / (valid + 1e-8) if valid > 0 else torch.tensor(0.0, device=predictions.device)
        # else:
        #     other_loss = elem_reg.mean()
        # other_loss = other_loss * self.other_drivers_weight

        total_loss = firms_loss # + mse_loss*0.5  # + other_loss

        loss_components = {
            'total_loss': float(total_loss.detach()),
            'firms_loss': float(firms_loss.detach()),
            'other_drivers_loss': float(other_loss if isinstance(other_loss, float) else other_loss.detach()),
            'firms_weight': self.firms_weight,
            'other_drivers_weight': self.other_drivers_weight,
            'regression_loss_type': self.regression_loss_type,
            'loss_type': 'bce_logits_weighted'
        }
        return total_loss, loss_components

class MultiTaskKLDivLoss(nn.Module):
    """
    Multi-task KL divergence Loss:
    - Use KL divergence for FIRMS channel (0th channel)
    - Use KL divergence for regression of other drivers
    - Support weight adjustment and ignore0 value functionality
    """
    def __init__(self, firms_weight=1.0, other_drivers_weight=0.1, 
                 ignore_zero_values=True, temperature=1.0, epsilon=1e-8):
        super(MultiTaskKLDivLoss, self).__init__()
        self.firms_weight = firms_weight
        self.other_drivers_weight = other_drivers_weight
        self.ignore_zero_values = ignore_zero_values
        self.temperature = temperature  # Temperature parameter, used to control smoothness of distribution
        self.epsilon = epsilon  # Small constant to prevent numerical instability
        
        # KL divergence loss function (reduction='none' for manual handling)
        self.kldiv_loss = nn.KLDivLoss(reduction='none')
    
    def _to_probability_distribution(self, x, is_classification=False):
        """
        Convert input to probability distribution
        
        Args:
            x: Input tensor
            is_classification: Whether this is a classification task (FIRMS channel)
            
        Returns:
            Probability distribution tensor
        """
        if is_classification:
            # For classification tasks, use sigmoid+normalization
            # x shape: (...,) or (..., 1)
            if x.dim() > 0 and x.shape[-1] == 1:
                x = x.squeeze(-1)  # Remove last dimension if it's 1
            
            prob = torch.sigmoid(x / self.temperature)
            # Create binomial distribution: [1-p, p]
            prob_neg = 1 - prob
            prob_dist = torch.stack([prob_neg, prob], dim=-1)  # (..., 2)
            # Normalize to ensure it's a probability distribution
            prob_dist = prob_dist / (prob_dist.sum(dim=-1, keepdim=True) + self.epsilon)
        else:
            # For regression tasks, convert values to positive then normalize
            # Use softplus to ensure positive values: softplus(x) = log(1 + exp(x))
            positive_vals = F.softplus(x / self.temperature)
            # Normalize to probability distribution
            prob_dist = positive_vals / (positive_vals.sum(dim=-1, keepdim=True) + self.epsilon)
        
        # Add small constant to prevent log(0)
        prob_dist = prob_dist + self.epsilon
        prob_dist = prob_dist / prob_dist.sum(dim=-1, keepdim=True)
        
        return prob_dist
    
    def forward(self, predictions, targets):
        """
        Calculate multi-task KL divergence loss
        
        Args:
            predictions: (B, T, C) Model prediction results, C may be greater than 39 (if there are additional features)
            targets: (B, T, 39) True labels, always 39 channels
            
        Returns:
            total_loss: Total loss
            loss_components: Dictionary of loss components
        """
        batch_size, seq_len, pred_channels = predictions.shape
        _, _, target_channels = targets.shape
        
        # 🔥 Key fix: If predicted channel number is greater than target channel number, only take the first target_channels channels
        # This is because the extra channels (e.g., weather data) have already been used as input features, so we shouldn't calculate loss for them
        if pred_channels > target_channels:
            predictions = predictions[:, :, :target_channels]
            print(f"🔧 KL divergence loss calculation: Predicted channel number ({pred_channels}) > Target channel number ({target_channels}), only calculating loss for the first {target_channels} channels")
        
        # Separate FIRMS and other drivers
        firms_pred = predictions[:, :, 0]      # (B, T) - FIRMS channel
        firms_target = targets[:, :, 0]        # (B, T)
        other_pred = predictions[:, :, 1:]     # (B, T, 38) - Other channels
        other_target = targets[:, :, 1:]       # (B, T, 38)
        
        # 1. Calculate KL divergence loss for FIRMS (classification task)
        # Convert FIRMS target to binary classification labels (1 if >0, 0 if =0)
        firms_binary_target = (firms_target > 0).float()
        
        # Convert to probability distribution
        firms_pred_dist = self._to_probability_distribution(firms_pred, is_classification=True)  # (B, T, 2)
        firms_target_dist = self._to_probability_distribution(firms_binary_target, is_classification=True)  # (B, T, 2)
        
        # Calculate KL divergence: KL(target || pred)
        firms_kl = self.kldiv_loss(firms_pred_dist.log(), firms_target_dist)  # (B, T, 2)
        firms_loss = firms_kl.sum(dim=-1).mean() * self.firms_weight  # Sum over distribution dimensions then average
        
        # 2. Calculate KL divergence loss for other drivers (regression task)
        # Convert to probability distribution
        other_pred_dist = self._to_probability_distribution(other_pred, is_classification=False)  # (B, T, 38)
        other_target_dist = self._to_probability_distribution(other_target, is_classification=False)  # (B, T, 38)
        
        # Calculate KL divergence
        other_kl = self.kldiv_loss(other_pred_dist.log(), other_target_dist)  # (B, T, 38)
        
        if self.ignore_zero_values:
            # Create non-zero mask to ignore0 values
            non_zero_mask = (other_target != 0.0).float()  # (B, T, 38)
            
            # Calculate effective number of samples
            valid_samples = non_zero_mask.sum()
            
            if valid_samples > 0:
                # Only calculate loss for non-zero values
                masked_kl = other_kl * non_zero_mask
                other_loss = masked_kl.sum() / valid_samples
            else:
                # If there are no valid samples, loss is 0
                other_loss = torch.tensor(0.0, device=predictions.device)
        else:
            # Don't ignore0 values, just calculate average loss
            other_loss = other_kl.mean()
        
        other_loss = other_loss * self.other_drivers_weight
        
        # Total loss
        total_loss = firms_loss + other_loss
        
        # Return loss component information
        loss_components = {
            'total_loss': total_loss.item(),
            'firms_loss': firms_loss.item(),
            'other_drivers_loss': other_loss.item(),
            'firms_weight': self.firms_weight,
            'other_drivers_weight': self.other_drivers_weight,
            'temperature': self.temperature,
            'loss_type': 'kldiv'
        }
        
        return total_loss, loss_components

class MultiMetricEarlyStopping:
    """
    Multi-metric Early Stopping: Monitor F1, Recall, PR-AUC simultaneously
    Any improvement in any metric resets the counter
    """
    def __init__(self, patience=7, min_delta=0.0001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_metrics = {
            'f1': 0.0,
            'recall': 0.0,
            'pr_auc': 0.0,
            'mae': float('inf')  # Lower MAE is better
        }
        self.best_weights = None
        self.should_stop = False
    
    def __call__(self, metrics, model):
        """
        Check if training should stop
        Args:
            metrics: dict containing 'f1', 'recall', 'pr_auc', 'mae'
            model: Model instance
        Returns:
            bool: Whether training should stop
        """
        f1_improved = metrics['f1'] > (self.best_metrics['f1'] + self.min_delta)
        recall_improved = metrics['recall'] > (self.best_metrics['recall'] + self.min_delta)
        pr_auc_improved = metrics['pr_auc'] > (self.best_metrics['pr_auc'] + self.min_delta)
        mae_improved = metrics['mae'] < (self.best_metrics['mae'] - self.min_delta)  # Lower MAE is better
        
        # Any improvement resets the counter
        if f1_improved or recall_improved or pr_auc_improved or mae_improved:
            # Update best metrics
            if f1_improved:
                self.best_metrics['f1'] = metrics['f1']
            if recall_improved:
                self.best_metrics['recall'] = metrics['recall']
            if pr_auc_improved:
                self.best_metrics['pr_auc'] = metrics['pr_auc']
            if mae_improved:
                self.best_metrics['mae'] = metrics['mae']
                
            self.counter = 0
            if self.restore_best_weights:
                self.save_checkpoint(model)
            print(f"📈 Metrics improved! F1: {metrics['f1']:.4f}, Recall: {metrics['recall']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}, MAE: {metrics['mae']:.6f}")
        else:
            self.counter += 1
            print(f"⏳ No improvement ({self.counter}/{self.patience}): F1: {metrics['f1']:.4f}, Recall: {metrics['recall']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}, MAE: {metrics['mae']:.6f}")
        
        if self.counter >= self.patience:
            self.should_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                print("🔄 Restored best weights")
        
        return self.should_stop
    
    def save_checkpoint(self, model):
        """Save best weights"""
        self.best_weights = model.state_dict().copy()

class MultiTaskLoss(nn.Module):
    """Multi-task loss function, supports weighted loss calculation for different channels"""
    
    def __init__(self, firms_weight=1.0, other_drivers_weight=0.1, 
                 ignore_zero_values=True, loss_function='huber'):
        super(MultiTaskLoss, self).__init__()
        self.firms_weight = firms_weight
        self.other_drivers_weight = other_drivers_weight
        self.ignore_zero_values = ignore_zero_values
        
        # Select loss function
        if loss_function == 'huber':
            self.loss_fn = nn.HuberLoss(reduction='none')
        elif loss_function == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_function == 'mae':
            self.loss_fn = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss function type: {loss_function}")
    
    def forward(self, predictions, targets):
        """
        Calculate multi-task loss
        
        Args:
            predictions: (B, T, C) Model prediction results, C may be greater than 39 (if there are additional features)
            targets: (B, T, 39) True labels, always 39 channels
            
        Returns:
            total_loss: Total loss
            loss_components: Dictionary of loss components
        """
        batch_size, seq_len, pred_channels = predictions.shape
        _, _, target_channels = targets.shape
        
        # 🔥 Key fix: If predicted channel number is greater than target channel number, only take the first target_channels channels
        # This is because the extra channels (e.g., weather data) have already been used as input features, so we shouldn't calculate loss for them
        if pred_channels > target_channels:
            predictions = predictions[:, :, :target_channels]
            print(f"🔧 Multi-task loss calculation: Predicted channel number ({pred_channels}) > Target channel number ({target_channels}), only calculating loss for the first {target_channels} channels")
        
        # Separate FIRMS and other drivers
        firms_pred = predictions[:, :, 0:1]  # (B, T, 1)
        firms_target = targets[:, :, 0:1]    # (B, T, 1)
        other_pred = predictions[:, :, 1:]   # (B, T, 38)
        other_target = targets[:, :, 1:]     # (B, T, 38)
        
        # Calculate FIRMS loss
        firms_loss = self.loss_fn(firms_pred, firms_target)  # (B, T, 1)
        firms_loss = firms_loss.mean() * self.firms_weight
        
        # Calculate other drivers loss
        other_loss = self.loss_fn(other_pred, other_target)  # (B, T, 38)
        
        if self.ignore_zero_values:
            # Create non-zero mask to ignore0 values
            non_zero_mask = (other_target != 0.0).float()  # (B, T, 38)
            
            # Calculate effective number of samples
            valid_samples = non_zero_mask.sum()
            
            if valid_samples > 0:
                # Only calculate loss for non-zero values
                masked_loss = other_loss * non_zero_mask
                other_loss = masked_loss.sum() / valid_samples
            else:
                # If there are no valid samples, loss is 0
                other_loss = torch.tensor(0.0, device=predictions.device)
        else:
            # Don't ignore0 values, just calculate average loss
            other_loss = other_loss.mean()
        
        other_loss = other_loss * self.other_drivers_weight
        
        # Total loss
        total_loss = firms_loss   # + other_loss
        
        # Return loss component information
        loss_components = {
            'total_loss': total_loss.item(),
            'firms_loss': firms_loss.item(),
            'other_drivers_loss': other_loss.item(),
            'firms_weight': self.firms_weight,
            'other_drivers_weight': self.other_drivers_weight,
            'loss_type': 'multitask'  # New: Loss function type identifier
        }
        # print(firms_loss, other_loss)
        return total_loss, loss_components

# =============================================================================
# Progress display utility functions
# =============================================================================

class SimpleProgressTracker:
    """Simplified progress tracker, mimicking tqdm default effect but without progress bar"""
    def __init__(self):
        self.start_time = None
        
    def update(self, current, total, prefix="Progress", clear_on_complete=True):
        """
        Update progress display - tqdm style but without progress bar
        """
        if self.start_time is None:
            self.start_time = time.time()
            
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Calculate speed (items/second)
        speed = current / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate percentage
        percent = int((current / total) * 100)
        
        # tqdm style display format
        if current == total:
            # Format when complete
            progress_text = f"\r{prefix}: {percent:3d}%|{current}/{total} [{self._format_time(elapsed_time)}, {speed:.2f}it/s]"
        else:
            # Format while in progress, calculate estimated remaining time
            if speed > 0:
                remaining_time = (total - current) / speed
                progress_text = f"\r{prefix}: {percent:3d}%|{current}/{total} [{self._format_time(elapsed_time)}<{self._format_time(remaining_time)}, {speed:.2f}it/s]"
            else:
                # If speed is 0, use simplified format
                progress_text = f"\r{prefix}: {percent:3d}%|{current}/{total} [{self._format_time(elapsed_time)}<?, ?it/s]"
        
        print(progress_text, end='', flush=True)
        
        # Handle completion
        if current == total:
            if clear_on_complete:
                # Clear progress bar
                print('\r' + ' ' * len(progress_text) + '\r', end='', flush=True)
            else:
                print()  # Keep final state and add newline
    
    def _format_time(self, seconds):
        """Format time display - tqdm style"""
        if seconds < 0:
            return "00s"
        elif seconds < 60:
            return f"{int(seconds):02d}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}:{minutes:02d}:00"

def print_dynamic_progress(current, total, prefix="Progress", show_percent=True):
    """
    Compatibility function - maintain simple dynamic progress display
    """
    if show_percent:
        percent = (current / total) * 100
        progress_text = f"\r{prefix}: {current}/{total} ({percent:.1f}%)"
    else:
        progress_text = f"\r{prefix}: {current}/{total}"
    
    print(progress_text, end='', flush=True)
    
    # Clear progress bar after completion
    if current == total:
        print('\r' + ' ' * len(progress_text) + '\r', end='', flush=True)

def save_epoch_metrics_to_log(epoch_metrics, log_file, model_name, model_type):
    """
    Save training and validation metrics for each epoch to log file
    """
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Detailed training log - {model_name} ({model_type})\n")
            f.write(f"{'='*80}\n")
            f.write(f"Record time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write header
            f.write(f"{'Epoch':<6} {'Train_Loss':<11} {'Train_P':<8} {'Train_R':<8} {'Train_F1':<9} {'Train_PRAUC':<11} {'Train_MSE':<10} {'Train_MAE':<10} ")
            f.write(f"{'Val_Loss':<9} {'Val_P':<6} {'Val_R':<6} {'Val_F1':<7} {'Val_PRAUC':<9} {'Val_MSE':<8} {'Val_MAE':<8} {'LR':<10}\n")
            f.write("-" * 150 + "\n")
            
            # Write data for each epoch
            for metrics in epoch_metrics:
                f.write(f"{metrics['epoch']:<6} ")
                f.write(f"{metrics['train_loss']:<11.6f} ")
                f.write(f"{metrics['train_precision']:<8.4f} ")
                f.write(f"{metrics['train_recall']:<8.4f} ")
                f.write(f"{metrics['train_f1']:<9.4f} ")
                f.write(f"{metrics['train_pr_auc']:<11.4f} ")
                f.write(f"{metrics['train_mse']:<10.6f} ")
                f.write(f"{metrics['train_mae']:<10.6f} ")
                f.write(f"{metrics['val_loss']:<9.6f} ")
                f.write(f"{metrics['val_precision']:<6.4f} ")
                f.write(f"{metrics['val_recall']:<6.4f} ")
                f.write(f"{metrics['val_f1']:<7.4f} ")
                f.write(f"{metrics['val_pr_auc']:<9.4f} ")
                f.write(f"{metrics['val_mse']:<8.6f} ")
                f.write(f"{metrics['val_mae']:<8.6f} ")
                f.write(f"{metrics['learning_rate']:<10.2e}\n")
            
            f.write("\n")
            
        print(f"📝 Detailed epoch log saved to: {log_file}")
        
    except Exception as e:
        print(f"⚠️ Failed to save epoch log: {e}")

def save_test_results_to_log(test_results, log_file, model_name):
    """
    Save test results to log file
    """
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Test Results - {model_name}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Record time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write test results for each model version
            for version, results in test_results.items():
                if results.get('precision') is not None:
                    version_name = version.upper() if version != 'final_epoch' else 'FINAL_EPOCH'
                    f.write(f"{version_name} Results:\n")
                    f.write(f"  Precision: {results['precision']:.6f}\n")
                    f.write(f"  Recall:    {results['recall']:.6f}\n")
                    f.write(f"  F1-Score:  {results['f1']:.6f}\n")
                    f.write(f"  PR-AUC:    {results['pr_auc']:.6f}\n")
                    f.write(f"  ROC-AUC:   {results.get('roc_auc', 0.0):.6f}\n")
                    f.write(f"  FPR:       {results.get('fpr', 1.0):.6f}\n")
                    f.write(f"  MSE:       {results['mse']:.6f}\n")
                    f.write(f"  MAE:       {results['mae']:.6f}\n")
                    f.write("\n")
            
            f.write("\n")
            
        print(f"📝 Test results saved to: {log_file}")
        
    except Exception as e:
        print(f"⚠️ Failed to save test results: {e}")

def save_structured_results_to_csv(structured_results, model_type):
    """
    Save structured test results as CSV files for classification
    Save separately: best_precision.csv, best_recall.csv, best_f1.csv, best_pr_auc.csv, best_roc_auc.csv, best_fpr.csv, final_epoch.csv
    Each CSV contains: Model, precision, recall, f1, pr_auc, roc_auc, fpr, mse, mae
    """
    if not structured_results:
        print("⚠️ No results to save")
        return
    
    # Determine save directory
    save_dir = STANDARD_MODEL_DIR
    
    # Create save directory if it doesn't exist
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # All types of models to save
    model_categories = ['precision', 'recall', 'f1', 'pr_auc', 'roc_auc', 'fpr', 'final_epoch']
    classification_metrics = ['precision', 'recall', 'f1', 'pr_auc', 'roc_auc', 'fpr', 'mse', 'mae']  # All metrics
    
    saved_files = []
    
    for category in model_categories:
        # Prepare CSV data
        csv_data = []
        columns = ['Model'] + classification_metrics
        
        # Add data rows
        for model_name, model_results in structured_results.items():
            if category in model_results and model_results[category]['precision'] is not None:
                row = [model_name]
                
                for metric_name in classification_metrics:
                    value = model_results[category][metric_name]
                    if value is not None:
                        row.append(f"{value:.6f}")
                    else:
                        row.append("N/A")
                
                csv_data.append(row)
        
        # Generate filename and save
        if category == 'final_epoch':
            filename = f"final_epoch.csv"
        else:
            filename = f"best_{category}.csv"
        
        csv_filepath = os.path.join(save_dir, filename)
        
        if csv_data:  # Only save if there's data
            df = pd.DataFrame(csv_data, columns=columns)
            df.to_csv(csv_filepath, index=False)
            saved_files.append(csv_filepath)
            
            print(f"📊 {filename}: {len(csv_data)} model results saved")
        else:
            print(f"⚠️  {filename}: No data available")
    
    # Summarize save situation
    print(f"\n✅ Total {len(saved_files)} CSV files saved to: {save_dir}")
    for filepath in saved_files:
        print(f"   📄 {os.path.basename(filepath)}")
    
    print(f"\n📋 CSV file structure explanation:")
    print(f"   best_precision.csv: Performance evaluation of the best Precision model")
    print(f"   best_recall.csv: Performance evaluation of the best Recall model")
    print(f"   best_f1.csv: Performance evaluation of the best F1 model")
    print(f"   best_pr_auc.csv: Performance evaluation of the best PR-AUC model")
    print(f"   best_roc_auc.csv: Performance evaluation of the best ROC-AUC model")
    print(f"   best_fpr.csv: Performance evaluation of the best FPR (lowest false positive rate) model")
    print(f"   final_epoch.csv: Performance evaluation of the final epoch model")
    print(f"   Each file contains: Model, precision, recall, f1, pr_auc, roc_auc, fpr, mse, mae")

# =============================================================================
# Core training and testing functions
# =============================================================================
def train_single_model(model_name, device, train_loader, val_loader, test_loader, firms_normalizer, model_type='standard', log_file=None, wandb_run_name=None, finish_wandb=True):
    """Train a single model. If finish_wandb=False, caller 负责在测试并 log test 后调用 wandb.finish()。"""
    # 确保每次训练模型初始化前随机状态一致（复现权重初始化）
    set_seed(TRAINING_CONFIG.get('seed', GLOBAL_SEED))
    
    print(f"\n🔥 Training {model_type} model: {model_name}")
    
    config = Config(model_name, model_type)
    
    # Create detailed logger
    epoch_metrics = []  # Record metrics for each epoch
    
    # Initialize wandb (if enabled)
    wandb_run = None
    if TRAINING_CONFIG['use_wandb'] and WANDB_AVAILABLE:
        # WandB 仅支持 API key 登录（不支持在脚本里用账号密码）。两种方式二选一：
        # 1) 终端执行 wandb login，粘贴 API key（从 https://wandb.ai/settings 复制），之后脚本会用本地缓存的 key
        # 2) 设置环境变量 WANDB_API_KEY 或在本文件顶部设置 WANDB_API_KEY。新版 key 为 wandb_v1_ 开头、长度>40，只能用环境变量；旧版 40 字符 key 可用 login(key=...)
        if WANDB_SILENT:
            os.environ["WANDB_SILENT"] = "true"
        # 是否使用脚本内配置的 API_KEY 登录
        if WANDB_USE_API_KEY and WANDB_API_KEY is not None:
            if len(WANDB_API_KEY) > 40:
                os.environ["WANDB_API_KEY"] = WANDB_API_KEY
                print("✅ WandB 已设置 API key（通过环境变量，供 wandb.init 使用）")
            else:
                try:
                    wandb.login(key=WANDB_API_KEY)
                    print("✅ WandB 已使用指定 API key 登录")
                except Exception as e:
                    print(f"⚠️ WandB 登录失败: {e}，将尝试使用已登录账户")
        else:
            print("ℹ️ 当前未使用脚本内 API_KEY，将依赖本地已登录账户（需先在终端执行 `wandb login` 并粘贴 API key）")
        # 自定义 run 名：优先用参数 wandb_run_name，其次 TRAINING_CONFIG['wandb_run_name']
        custom_name = wandb_run_name or TRAINING_CONFIG.get('wandb_run_name')
        if custom_name is not None:
            run_name = custom_name if '_' in custom_name or custom_name == model_name else f"{model_name}_{custom_name}"
        else:
            run_name = f"{model_name}_{model_type}"
        wandb_run = wandb.init(
            project="hackathon_wildfire",
            name=run_name,
            entity=WANDB_ENTITY,  # 使用指定 entity/team（如提供）
            config={
                "model_name": model_name,
                "model_type": model_type,
                "seq_len": config.seq_len,
                "pred_len": config.pred_len,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "focal_alpha": config.focal_alpha,
                "focal_gamma": config.focal_gamma,
                # Multi-task learning configuration
                "multitask_enabled": True,
                "firms_weight": config.firms_weight,
                "other_drivers_weight": config.other_drivers_weight,
                "ignore_zero_values": config.ignore_zero_values,
                "loss_function": config.loss_function,
            },
            reinit=True
        )
        print(f"✅ WandB initialization completed: {wandb_run.name}")
    
    # Use unified adapter
    adapter = UnifiedModelAdapter(config)
    
    # 🔥 New: Initialize position embedding if enabled
    position_embedding = None
    if DATA_CONFIG['enable_position_features']:
        position_embedding = get_position_embedding(
            raster_size=DATA_CONFIG['raster_size'],
            embedding_dim=4,
            device=device
        )
        print(f"✅ Position embedding initialized: {DATA_CONFIG['raster_size']} -> 4D embedding")
    
    try:
        model, _ = load_model(model_name, config, model_type)
        model = model.to(device)
    except Exception as e:
        print(f"❌ {model_type} model {model_name} failed to load: {e}")
        if wandb_run:
            wandb_run.finish()
        return None
    
    # Optimizer and loss function
    # 🔥 New: Include position embedding parameters in optimizer if enabled
    if position_embedding is not None:
        optimizer = optim.AdamW(
            list(model.parameters()) + list(position_embedding.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        print(f"✅ Optimizer includes position embedding parameters")
    else:
        # Split offset_mlp params (deformable attention) into a separate group
        # with LR x10 to compensate for weak gradient signal in multi-stream nets.
        offset_params = [p for n, p in model.named_parameters() if 'offset_mlp' in n and p.requires_grad]
        other_params  = [p for n, p in model.named_parameters() if 'offset_mlp' not in n and p.requires_grad]
        if offset_params:
            optimizer = optim.AdamW(
                [
                    {'params': other_params,  'lr': config.learning_rate},
                    {'params': offset_params, 'lr': config.learning_rate * 10},
                ],
                weight_decay=config.weight_decay,
            )
            n_off = sum(p.numel() for p in offset_params)
            print(f"✅ Optimizer: {n_off} offset_mlp params at LR x10 = {config.learning_rate * 10:.1e}")
        else:
            optimizer = optim.AdamW(model.parameters(),
                                    lr=config.learning_rate,
                                    weight_decay=config.weight_decay)
    
    # Select loss function type based on configuration
    if config.loss_type == 'focal':
        # Use multi-task Focal Loss
        criterion = MultiTaskFocalLoss(
            firms_weight=config.firms_weight,
            other_drivers_weight=config.other_drivers_weight,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            ignore_zero_values=config.ignore_zero_values,
            regression_loss=config.loss_function  # 'mse', 'huber', 'mae'
        )
        
        print(f"🔍 Multi-task Focal Loss configuration:")
        print(f"   FIRMS weight: {config.firms_weight}, other drivers weight: {config.other_drivers_weight}")
        print(f"   Focal α: {config.focal_alpha}, Focal γ: {config.focal_gamma}")
        print(f"   Regression loss: {config.loss_function}, Ignore zero values: {config.ignore_zero_values}")
        
    elif config.loss_type == 'kldiv':
        # Use multi-task KL divergence Loss
        criterion = MultiTaskKLDivLoss(
            firms_weight=config.firms_weight,
            other_drivers_weight=config.other_drivers_weight,
            ignore_zero_values=config.ignore_zero_values,
            temperature=1.0,  # Can be added to configuration later
            epsilon=1e-8
        )
        
        print(f"🔍 Multi-task KL divergence Loss configuration:")
        print(f"   FIRMS weight: {config.firms_weight}, other drivers weight: {config.other_drivers_weight}")
        print(f"   温度参数: 1.0, 忽略0值: {config.ignore_zero_values}")
        
    elif config.loss_type == 'multitask':
        # Use multi-task loss function
        criterion = MultiTaskLoss(
            firms_weight=config.firms_weight,
            other_drivers_weight=config.other_drivers_weight,
            ignore_zero_values=config.ignore_zero_values,
            loss_function=config.loss_function
        )
        
        print(f"🔍 Multi-task loss function configuration:")
        print(f"   FIRMS weight: {config.firms_weight}, other drivers weight: {config.other_drivers_weight}")
        print(f"   忽略0值: {config.ignore_zero_values}")
        print(f"   损失函数: {config.loss_function}")
    
    else:
        raise ValueError(f"Unsupported loss function type: {config.loss_type}. Supported types: 'focal', 'kldiv', 'multitask'")
    
    print(f"🎯 Current loss function being used: {config.loss_type.upper()}")
    
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=config.T_0, T_mult=config.T_mult, eta_min=config.eta_min
    # )
    
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Early stopping
    early_stopping = MultiMetricEarlyStopping(patience=TRAINING_CONFIG['patience'], min_delta=0.0001, restore_best_weights=True)
    
    # Track best metrics and model paths
    best_metrics = {
        'precision': {'score': 0.0, 'path': None},
        'recall': {'score': 0.0, 'path': None},
        'f1': {'score': 0.0, 'path': None},
        'pr_auc': {'score': 0.0, 'path': None},
        'roc_auc': {'score': 0.0, 'path': None},
        'fpr': {'score': float('inf'), 'path': None}  # Lower FPR is better, initialized to infinity
    }
    
    print(f"🚀 Starting training {config.epochs} epochs...")
    
    for epoch in range(config.epochs):
        # Resample training set for each epoch if enabled
        if hasattr(train_loader.dataset, 'resample_for_epoch'):
            train_loader.dataset.resample_for_epoch(epoch)
        
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        # Training phase - simplify progress display for performance
        # train_progress = SimpleProgressTracker()
        for i, batch in enumerate(train_loader):
            # Comment out detailed training progress display to reduce CPU overhead
            # train_progress.update(i+1, len(train_loader), f"🔥 Epoch {epoch+1}/{config.epochs} Training")
            
            past, future, metadata_list = batch[0], batch[1], batch[2]
            burn_history = batch[3] if len(batch) >= 4 else None  # (B, 1, H, W, T) 或 None
            past[:, 0, :] = past[:, 0, :] * 0.0
            past[:, -1, :] = past[:, -1, :]/ 17.0   # 最后一个channel除以17
            past, future = past.to(device), future.to(device)  # B, C, T
            
            # print(f"future: {future[:, 0, 0].min(), future[:, 0, 0].max()}")
            
            
            # 🔥 Fix: Don't delete the 0th channel, just set its data to 0, keeping the completeness of 39 channels
            # past[:, 0, :] = 0.0  # Set the 0th channel (FIRMS) to 0 instead of deleting
          
            if firms_normalizer is not None:
                past, future = normalize_batch(past, future, firms_normalizer, metadata_list, position_embedding=position_embedding)
            date_strings = [str(int(metadata[0])) for metadata in metadata_list]  # B 1, yyyymmdd
            # print(date_strings)
            
            future_truncated = future[:, 0, 13//2, 13//2, :]

            x_enc, x_mark_enc, x_dec, x_mark_dec = adapter.adapt_inputs(past, future, date_strings)
            x_enc, x_mark_enc, x_dec, x_mark_dec = x_enc.to(device), x_mark_enc.to(device), x_dec.to(device), x_mark_dec.to(device)
            # STAtten：每步前重置脉冲神经元状态，避免跨 batch 计算图导致二次 backward
            if model_name == 'STAtten' and SJ_RESET_NET is not None:
                SJ_RESET_NET(model)
            # 前向传播（SSM 等支持时传入 burn_history）
            if burn_history is not None:
                burn_history = burn_history.to(device)
                model_output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, burn_history=burn_history)
            # if getattr(config, 'model_name', None) == 'SSM' and burn_history is not None:
            #     model_output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, burn_history=burn_history)
            # else:
            #    model_output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # 检测是否是 uncertainty 模型 (mu, log_var)
            if isinstance(model_output, tuple) and len(model_output) == 2:
                output = model_output[0]
            else:
                output = model_output
            
            target_all_channels = future_truncated  # (B, L)，criterion 接受 2D (B, C)
            # 使用标准的loss函数（STAtten 的 (B,1) 直接传入，criterion 内用 targets[:, :1]）
            loss, loss_components = criterion(output, target_all_channels)
            
            # print(f"FIRMS loss: {loss_components['firms_loss']}, Other drivers loss: {loss_components['other_drivers_loss']}")
            optimizer.zero_grad()
            loss.backward()
            
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            # STAtten：backward 后重置脉冲神经元状态，下一 batch 不携带旧图
            if model_name == 'STAtten' and SJ_RESET_NET is not None:
                SJ_RESET_NET(model)
            
            train_loss += loss.item()
            # Only save prediction for FIRMS channel（STAtten 等 (B,1) 需 expand 成 (B,L)）
            if output.dim() == 3:
                train_preds.append(output[:, 0].detach())
            elif output.dim() == 2 and output.size(1) == 1:
                train_preds.append(output.expand(-1, target_all_channels.size(1)).detach())
            else:
                train_preds.append(output.detach())
            train_targets.append(target_all_channels.detach())
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        train_preds = torch.cat(train_preds, dim=0)
        train_targets = torch.cat(train_targets, dim=0)
        train_precision, train_recall, train_f1, train_pr_auc, train_roc_auc, train_fpr, train_mse, train_mae = calculate_detailed_metrics(train_preds, train_targets)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            # Validation phase - simplify progress display for performance
            # val_progress = SimpleProgressTracker()
            for i, batch in enumerate(val_loader):
                # Comment out detailed validation progress display to reduce CPU overhead
                # val_progress.update(i+1, len(val_loader), f"📊 Epoch {epoch+1}/{config.epochs} Validation")
                
                past, future, metadata_list = batch[0], batch[1], batch[2]
                burn_history = batch[3] if len(batch) >= 4 else None
                past[:, 0, :] = past[:, 0, :] * 0.0
                past[:, -1, :] = past[:, -1, :] / 17.0   # 最后一个channel除以17
                past, future = past.to(device), future.to(device)
                
                # 🔥 Fix: Don't delete the 0th channel, just set its data to 0, keeping the completeness of 39 channels
                # past[:, 0, :] = 0.0  # Set the 0th channel (FIRMS) to 0 instead of deleting
                
                # Why normalize future data!?!
                if firms_normalizer is not None:
                    past, future = normalize_batch(past, future, firms_normalizer, metadata_list, position_embedding=position_embedding)
                
                date_strings = [str(int(metadata[0])) for metadata in metadata_list]
                
                future_truncated = future[:, 0, 13//2, 13//2, :]
                # target = (target > config.binarization_threshold).float()
                
                # if model_name == 's_mamba':
                #     past_transposed = past.transpose(1, 2)
                #     past_truncated = past_transposed[:, -config.seq_len:, :]
                #     output = model(past_truncated, date_strings)
                # else:
                
                # x_mark_enc: B T 5 (year_norm, month_sin, month_cos, day_sin, day_cos, weekday_norm)
                x_enc, x_mark_enc, x_dec, x_mark_dec = adapter.adapt_inputs(past, future, date_strings)
                x_enc, x_mark_enc, x_dec, x_mark_dec = x_enc.to(device), x_mark_enc.to(device), x_dec.to(device), x_mark_dec.to(device)
                # x_enc = torch.flip(x_enc, dims=[1])
                
                # if epoch <= 30:
                #     x_enc[:, :, 0] = x_enc[:, :, 0] * 0.0
                
                if model_name == 'STAtten' and SJ_RESET_NET is not None:
                    SJ_RESET_NET(model)
                # 前向传播（SSM 等支持时传入 burn_history）
                if burn_history is not None:
                    burn_history = burn_history.to(device)
                    model_output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, burn_history=burn_history)
                else:
                    model_output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                # 检测是否是 uncertainty 模型 (mu, log_var)
                if isinstance(model_output, tuple) and len(model_output) == 2:
                    output = model_output[0]
                else:
                    output = model_output
                # Multi-task learning: Predict all 39 channels
                target_all_channels = future_truncated  # Use all channels as target
                # Calculate multi-task Focal loss
                # target_all_channels = target_all_channels.clone()
                # target_all_channels[:, :, 0] = (target_all_channels[:, :, 0] > 10).float()
                # 使用标准的loss函数
                loss, loss_components = criterion(output, target_all_channels)
                val_loss += loss.item()
                
                # Only save prediction for FIRMS channel（STAtten 等 (B,1) 需 expand 成 (B,L)）
                if output.dim() == 3:
                    val_preds.append(output[:, 0].detach())
                elif output.dim() == 2 and output.size(1) == 1:
                    val_preds.append(output.expand(-1, target_all_channels.size(1)).detach())
                else:
                    val_preds.append(output.detach())
                val_targets.append(target_all_channels.detach())
        
        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        val_precision, val_recall, val_f1, val_pr_auc, val_roc_auc, val_fpr, val_mse, val_mae = calculate_detailed_metrics(val_preds, val_targets)
        
        # Record metrics for current epoch
        epoch_data = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_f1": train_f1,
                "train_pr_auc": train_pr_auc,
                "train_roc_auc": train_roc_auc,
                "train_fpr": train_fpr,
                "train_mse": train_mse,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "val_pr_auc": val_pr_auc,
                "val_roc_auc": val_roc_auc,
                "val_fpr": val_fpr,
                "val_mse": val_mse,
                "val_mae": val_mae,
                "learning_rate": optimizer.param_groups[0]["lr"],
                # Add multi-task loss component information
                "firms_weight": config.firms_weight,
                "other_drivers_weight": config.other_drivers_weight,
                "loss_function": config.loss_function,
                "ignore_zero_values": config.ignore_zero_values
        }
        epoch_metrics.append(epoch_data)
        
        # Record to wandb
        if wandb_run:
            wandb.log(epoch_data)
        
        # Display training progress
        print(f"Epoch {epoch+1}/{config.epochs} - "
              f"Train Loss: {train_loss:.4f} (F1: {train_f1:.4f}) - "
              f"Val Loss: {val_loss:.4f} (F1: {val_f1:.4f}) - "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Display multi-task loss component information (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            print(f"    Multi-task loss components - FIRMS: {config.firms_weight:.1f}, "
                  f"other drivers: {config.other_drivers_weight:.1f}, "
                  f"loss function: {config.loss_function}")
        
        # Save best model for each metric (first METRIC_WARMUP_EPOCHS do not participate in comparison/save)
        model_save_dir = get_model_save_dir(model_type)
        os.makedirs(model_save_dir, exist_ok=True)
        
        if epoch >= METRIC_WARMUP_EPOCHS:
            metrics_to_save = {
                'precision': val_precision,
                'recall': val_recall,
                'f1': val_f1,
                'pr_auc': val_pr_auc,
                'roc_auc': val_roc_auc,
                'fpr': val_fpr
            }
            for metric_name, score in metrics_to_save.items():
                # Lower FPR is better, other metrics are better if higher
                if metric_name in ['fpr']:
                    if score <= best_metrics[metric_name]['score']:
                        best_metrics[metric_name]['score'] = score
                        model_path = os.path.join(model_save_dir, f'{model_name}_best_{metric_name}.pth')
                        torch.save(model.state_dict(), model_path)
                        best_metrics[metric_name]['path'] = model_path
                else:
                    if score >= best_metrics[metric_name]['score']:
                        best_metrics[metric_name]['score'] = score
                        model_path = os.path.join(model_save_dir, f'{model_name}_best_{metric_name}.pth')
                        torch.save(model.state_dict(), model_path)
                        best_metrics[metric_name]['path'] = model_path
        
        # Print epoch summary
        print(f'Epoch {epoch+1:3d}/{config.epochs} | Train: Loss={train_loss:.4f}, F1={train_f1:.4f}, MSE={train_mse:.6f}, MAE={train_mae:.6f} | '
              f'Val: Loss={val_loss:.4f}, P={val_precision:.4f}, R={val_recall:.4f}, F1={val_f1:.4f}, PR-AUC={val_pr_auc:.4f}, MSE={val_mse:.6f}, MAE={val_mae:.6f} | '
              f'LR={optimizer.param_groups[0]["lr"]:.2e}')
        
        # Early stopping: do not monitor metric change in first METRIC_WARMUP_EPOCHS
        if epoch >= METRIC_WARMUP_EPOCHS:
            if early_stopping({'f1': val_f1, 'recall': val_recall, 'pr_auc': val_pr_auc, 'mae': val_mae, 'mse': val_mse}, model):
                print(f"⏹️  Early stopping triggered at epoch {epoch+1} (patience={TRAINING_CONFIG['patience']}, counter={early_stopping.counter})")
                break
        
        lr_scheduler.step()
    
    # Save model parameters for last epoch
    final_model_path = os.path.join(model_save_dir, f'{model_name}_final_epoch.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"💾 Last epoch model saved: {final_model_path}")
    
    # Add last epoch path to return result
    best_metrics['final_epoch'] = {
        'score': epoch + 1,  # Record final epoch number
        'path': final_model_path
    }
    
    # Save detailed epoch training log
    if log_file:
        save_epoch_metrics_to_log(epoch_metrics, log_file, model_name, model_type)
    
    # Close wandb（若 finish_wandb=False 则由调用方在 test 并 log 后 finish）
    if finish_wandb and wandb_run:
        wandb.finish()
    
    return best_metrics

def test_model(model_name, model_path, device, test_loader, firms_normalizer, model_type='standard'):
    """Test model with standard single forward pass"""
    print(f"\n📊 Testing {model_type} model: {model_name}")

    # === 配置与适配器 ===
    config = Config(model_name, model_type)
    from model_adapter_unified import UnifiedModelAdapter
    adapter = UnifiedModelAdapter(config)
    
    # 🔥 New: Initialize position embedding if enabled
    position_embedding = None
    if DATA_CONFIG['enable_position_features']:
        position_embedding = get_position_embedding(
            raster_size=DATA_CONFIG['raster_size'],
            embedding_dim=4,
            device=device
        )
        # Load position embedding state if available (for consistency with training)
        # Note: Position embedding is shared, so we use the global instance

    # === 加载模型 ===
    try:
        model, _ = load_model(model_name, config, model_type)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ {model_type} model {model_name} failed to load for testing: {e}")
        return None

    # === 评估缓存 ===
    test_probs = []     # 预测概率（只第一个变量） -> [B, L]
    test_targets = []   # 第一个变量的标签 -> [B, L]

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            past, future, metadata_list = batch[0], batch[1], batch[2]
            burn_history = batch[3] if len(batch) >= 4 else None
            past[:, 0, :] = past[:, 0, :] * 0.0
            past[:, -1, :] = past[:, -1, :] / 17.0   # 最后一个channel除以17
            past, future = past.to(device), future.to(device)
            # 归一化（如有）
            if firms_normalizer is not None:
                past, future = normalize_batch(past, future, firms_normalizer, metadata_list, position_embedding=position_embedding)

            # 时间标记
            date_strings = [str(int(metadata[0])) for metadata in metadata_list]

            # 目标：只取第一个变量，且裁到 pred_len
            future_trunc = future[:, 0, 13//2, 13//2, :]  # [B, L]
            target_first = (future_trunc>0).float()                  # [B, L] 0/1

            # 构造模型输入
            x_enc, x_mark_enc, x_dec, x_mark_dec = adapter.adapt_inputs(past, future, date_strings)
            x_enc, x_mark_enc = x_enc.to(device), x_mark_enc.to(device)
            x_dec, x_mark_dec = x_dec.to(device), x_mark_dec.to(device)

            # STAtten：每 batch 前重置脉冲神经元，避免上一 batch=16 的状态与当前 batch=8 冲突导致 8 vs 16
            if model_name == 'STAtten' and SJ_RESET_NET is not None:
                SJ_RESET_NET(model)
            # 单次前向传播（SSM 等支持时传入 burn_history）
            if burn_history is not None:
                burn_history = burn_history.to(device)
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, burn_history=burn_history)  # [B, L, N]
            else:
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B, L, N]
            
            # 检测是否是 uncertainty 模型 (mu, log_var)，取 FIRMS 通道
            if isinstance(output, tuple) and len(output) == 2:
                y_hat = torch.sigmoid(output[0][:, :, 0])  # [B, L]
            else:
                y_hat = torch.sigmoid(output)
                if y_hat.dim() == 2 and y_hat.size(1) == 1 and target_first.size(1) > 1:
                    y_hat = y_hat.expand(-1, target_first.size(1))  # [B, L]

            test_probs.append(y_hat.detach())
            test_targets.append(target_first.detach().float())

    # === 评估指标（使用固定阈值0.5） ===
    test_probs   = torch.cat(test_probs, dim=0)   # [N_samples, L]
    test_targets = torch.cat(test_targets, dim=0) # [N_samples, L]

    # 展示标签分布
    test_targets_np = test_targets.cpu().numpy().reshape(-1)
    pos_cnt = np.sum(test_targets_np > 0.5)
    tot_cnt = test_targets_np.size
    print(f"   🔍 Test data distribution before metrics: {pos_cnt}/{tot_cnt} positive samples ({pos_cnt/tot_cnt:.4f})")
    print(f"   Using fixed threshold: 0.5 (instead of optimal F1 threshold)")

    # 使用固定阈值0.5计算指标（输入已经是概率值）
    precision, recall, f1, pr_auc, roc_auc, fpr, mse, mae, accuracy = calculate_metrics_fixed_threshold(test_probs, test_targets, threshold=0.5)

    print(f"✅ {model_name} {model_type} test: Acc={accuracy:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, PR-AUC={pr_auc:.4f}, ROC-AUC={roc_auc:.4f}, FPR={fpr:.4f}, MSE={mse:.6f}, MAE={mae:.6f}")

    return {
        'model': model_name,
        'model_type': model_type,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'mse': mse,
        'mae': mae,
        'accuracy': accuracy
    }

def train_and_test_models(model_list, model_type, device, train_dataset, val_dataset, test_dataset, 
                          data_loader_obj, data_loader_test, firms_normalizer, force_retrain=False, parallel_models=1):
    """Train and test a group of models"""
    print(f"\n🔥 Starting training {model_type} model group")
    print(f"📋 Original model list: {len(model_list)} {model_type} models")
    print(f"📊 {model_type} model list: {', '.join(model_list)}")
    
    # 创建log文件
    log_dir = STANDARD_MODEL_DIR
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    print(f"📝 Test results will be saved to: {log_file}")
    
    # Filter trained models
    models_to_train, trained_models = filter_trained_models(model_list, model_type, force_retrain)
    
    # Train new models
    model_results = []
    failed_models = []
    # 提前初始化，顺序/并行训练时在训练后立即测试并填入，便于 test 指标写入同一 wandb run
    structured_results = {}
    
    # 🔥 Initialize test_loader to None (will be created if needed)
    test_loader = None
    
    # Prepare DataLoader configuration (will be used to create per-thread DataLoaders)
    standard_config = TRAINING_CONFIG[model_type]
    dl_seed = standard_config.get('seed', GLOBAL_SEED)  # 固定 DataLoader shuffle 复现
    # WF_REPRODUCIBLE=1 时强制 num_workers=0，避免多进程导致 batch 顺序不确定
    force_reproducible = os.getenv('WF_REPRODUCIBLE', '0') == '1'
    if force_reproducible:
        train_workers = val_workers = test_workers = 0
        if not os.environ.get('WF_REPRODUCIBLE_QUIET'):
            print("🔒 WF_REPRODUCIBLE=1: num_workers=0 for deterministic data order")
    else:
        train_workers = int(os.getenv('WF_TRAIN_WORKERS', '4'))
        val_workers = int(os.getenv('WF_VAL_WORKERS', '4'))
        test_workers = int(os.getenv('WF_TEST_WORKERS', '8'))
    prefetch = int(os.getenv('WF_PREFETCH_FACTOR', '4'))
    use_persistent_workers = (train_workers > 0)
    
    # Get collate function
    if data_loader_test is not None:
        test_collate_fn = data_loader_test.dataset.custom_collate_fn
    else:
        test_collate_fn = data_loader_obj.dataset.custom_collate_fn
    train_collate_fn = data_loader_obj.dataset.custom_collate_fn
    
    if models_to_train:
        print(f"\n🚀 Starting training {len(models_to_train)} {model_type} models that need training...")
        
        # 🔥 New: Parallel training support
        if parallel_models > 1 and len(models_to_train) > 1:
            print(f"⚡ Parallel training enabled: {parallel_models} models will be trained simultaneously")
            
            # Check GPU availability for parallel training
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                if num_gpus >= parallel_models:
                    print(f"✅ {num_gpus} GPU(s) available, can run {parallel_models} models in parallel")
                else:
                    print(f"⚠️ Only {num_gpus} GPU(s) available, but {parallel_models} parallel models requested")
                    print(f"   Will use {min(parallel_models, num_gpus)} parallel models")
                    parallel_models = min(parallel_models, num_gpus)
            else:
                print(f"⚠️ No GPU available, falling back to sequential training")
                parallel_models = 1
            
            # Use ThreadPoolExecutor for parallel training
            from concurrent.futures import ThreadPoolExecutor
            
            def create_dataloaders_for_thread():
                """Create independent DataLoader instances for each thread"""
                g = torch.Generator().manual_seed(dl_seed)
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=standard_config['batch_size'], 
                    shuffle=True, 
                    generator=g,
                    num_workers=train_workers,
                    collate_fn=train_collate_fn, 
                    worker_init_fn=worker_init_fn,
                    pin_memory=True,
                    persistent_workers=use_persistent_workers,
                    prefetch_factor=prefetch
                )
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=standard_config['batch_size'], 
                    shuffle=False,
                    num_workers=val_workers,
                    collate_fn=train_collate_fn, 
                    worker_init_fn=worker_init_fn,
                    pin_memory=True,
                    persistent_workers=use_persistent_workers,
                    prefetch_factor=prefetch
                )
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=standard_config['batch_size'], 
                    shuffle=False,
                    num_workers=test_workers,
                    collate_fn=test_collate_fn, 
                    worker_init_fn=worker_init_fn,
                    pin_memory=True,
                    persistent_workers=use_persistent_workers,
                    prefetch_factor=prefetch
                )
                return train_loader, val_loader, test_loader
            
            _empty_test_template_par = {
                'precision': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'recall': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'f1': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'pr_auc': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'roc_auc': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'fpr': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'mae': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'mse': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'final_epoch': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None}
            }
            def train_model_wrapper(model_name, gpu_id=None):
                """Wrapper: train -> test -> log test/ to same run -> finish -> return (model_name, result, structured_single)."""
                try:
                    if gpu_id is not None:
                        thread_device = torch.device(f'cuda:{gpu_id}')
                    else:
                        thread_device = device
                    train_loader_thread, val_loader_thread, test_loader_thread = create_dataloaders_for_thread()
                    result = train_single_model(
                        model_name, thread_device, train_loader_thread, val_loader_thread, test_loader_thread,
                        firms_normalizer, model_type, log_file=log_file, finish_wandb=False
                    )
                    if result is None:
                        return (model_name, None, None, None)
                    struct_single = {model_name: {k: dict(v) for k, v in _empty_test_template_par.items()}}
                    test_log = {}
                    for metric_name, metric_info in result.items():
                        if metric_info.get('path') is not None:
                            try:
                                r = test_model(model_name, metric_info['path'], thread_device, test_loader_thread, firms_normalizer, model_type)
                                if r:
                                    struct_single[model_name][metric_name] = {
                                        'precision': r['precision'], 'recall': r['recall'], 'f1': r['f1'], 'pr_auc': r['pr_auc'],
                                        'roc_auc': r.get('roc_auc', 0.0), 'fpr': r.get('fpr', 1.0), 'mse': r['mse'], 'mae': r['mae'], 'accuracy': r.get('accuracy', 0.0)
                                    }
                                    pre = f"test/{metric_name}"
                                    test_log[f"{pre}/accuracy"] = r.get('accuracy', 0.0)
                                    test_log[f"{pre}/precision"] = r['precision']
                                    test_log[f"{pre}/recall"] = r['recall']
                                    test_log[f"{pre}/f1"] = r['f1']
                                    test_log[f"{pre}/pr_auc"] = r['pr_auc']
                                    test_log[f"{pre}/roc_auc"] = r.get('roc_auc', 0.0)
                                    test_log[f"{pre}/fpr"] = r.get('fpr', 1.0)
                                    test_log[f"{pre}/mse"] = r['mse']
                                    test_log[f"{pre}/mae"] = r['mae']
                            except Exception:
                                pass
                    if test_log and TRAINING_CONFIG.get('use_wandb') and WANDB_AVAILABLE and wandb.run is not None:
                        _wandb_log_test_bar_chart(test_log)
                    if WANDB_AVAILABLE and wandb.run is not None:
                        wandb.finish()
                    return (model_name, result, struct_single, None)
                except Exception as e:
                    import traceback
                    return (model_name, None, None, f"{str(e)}\n{traceback.format_exc()}")
            
            # Train models in parallel batches
            with ThreadPoolExecutor(max_workers=parallel_models) as executor:
                futures = {}
                remaining_models = list(models_to_train)
                completed_count = len(trained_models)
                
                # Submit initial batch
                for i in range(min(parallel_models, len(remaining_models))):
                    model_name = remaining_models.pop(0)
                    # Assign GPU if multiple GPUs available
                    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                        gpu_id = i % torch.cuda.device_count()
                    else:
                        gpu_id = None
                    
                    future = executor.submit(train_model_wrapper, model_name, gpu_id)
                    futures[future] = (model_name, gpu_id)
                    print(f"🚀 Submitted {model_name} for training (GPU: {gpu_id if gpu_id is not None else 'default'})")
                
                # Process completed tasks and submit new ones
                while futures or remaining_models:
                    # Wait for at least one task to complete
                    done, not_done = [], []
                    for future in list(futures.keys()):
                        if future.done():
                            done.append(future)
                        else:
                            not_done.append(future)
                    
                    # Process completed futures
                    for future in done:
                        model_name, result, struct_single, error = future.result()
                        completed_count += 1
                        if struct_single:
                            structured_results.update(struct_single)
                        if error:
                            print(f"❌ {model_name} {model_type} model training failed: {error}")
                            failed_models.append(model_name)
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        elif result is not None:
                            best_metrics = result
                            print(f"✅ {model_name} {model_type} model training completed ({completed_count}/{len(model_list)}), saved model:")
                            for metric_name, metric_info in best_metrics.items():
                                if metric_name == 'final_epoch':
                                    print(f"  Final epoch model (epoch {metric_info['score']}): {metric_info['path']}")
                                else:
                                    print(f"  Best {metric_name} model ({metric_info['score']:.4f}): {metric_info['path']}")
                            model_results.append((model_name, best_metrics))
                        else:
                            failed_models.append(model_name)
                        
                        # Remove completed future
                        del futures[future]
                        
                        # Submit next model if available
                        if remaining_models:
                            next_model = remaining_models.pop(0)
                            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                                gpu_id = (completed_count - len(trained_models)) % torch.cuda.device_count()
                            else:
                                gpu_id = None
                            
                            new_future = executor.submit(train_model_wrapper, next_model, gpu_id)
                            futures[new_future] = (next_model, gpu_id)
                            print(f"🚀 Submitted {next_model} for training (GPU: {gpu_id if gpu_id is not None else 'default'})")
                        
                        # Clear GPU memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # If no tasks completed, wait a bit
                    if not done:
                        time.sleep(0.1)
        else:
            # Sequential training (original code) - create DataLoaders once
            train_generator = torch.Generator().manual_seed(dl_seed)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=standard_config['batch_size'], 
                shuffle=True, 
                generator=train_generator,
                num_workers=train_workers,
                collate_fn=train_collate_fn, 
                worker_init_fn=worker_init_fn,
                pin_memory=True,
                persistent_workers=use_persistent_workers,
                prefetch_factor=prefetch
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=standard_config['batch_size'], 
                shuffle=False,
                num_workers=val_workers,
                collate_fn=train_collate_fn, 
                worker_init_fn=worker_init_fn,
                pin_memory=True,
                persistent_workers=use_persistent_workers,
                prefetch_factor=prefetch
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=standard_config['batch_size'], 
                shuffle=False,
                num_workers=test_workers,
                collate_fn=test_collate_fn, 
                worker_init_fn=worker_init_fn,
                pin_memory=True,
                persistent_workers=use_persistent_workers,
                prefetch_factor=prefetch
            )
            
            _empty_test_template = {
                'precision': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'recall': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'f1': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'pr_auc': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'roc_auc': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'fpr': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'mae': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'mse': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'final_epoch': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None}
            }
            for i, model_name in enumerate(models_to_train):
                print(f"\n🔄 {model_type} training progress: {i+1}/{len(models_to_train)} (overall: {i+1+len(trained_models)}/{len(model_list)})")
                try:
                    result = train_single_model(
                        model_name, device, train_loader, val_loader, test_loader, firms_normalizer, model_type, log_file=log_file, finish_wandb=False
                    )
                    if result is not None:
                        best_metrics = result
                        print(f"✅ {model_name} {model_type} model training completed, saved model:")
                        for mn, minfo in best_metrics.items():
                            if mn == 'final_epoch':
                                print(f"  Final epoch model (epoch {minfo['score']}): {minfo['path']}")
                            else:
                                print(f"  Best {mn} model ({minfo['score']:.4f}): {minfo['path']}")
                        model_results.append((model_name, best_metrics))
                        # 同一 run 内：立即测试并写入 test/ 分组（区别于 train/val）
                        structured_results[model_name] = {k: dict(v) for k, v in _empty_test_template.items()}
                        test_log = {}
                        for metric_name, metric_info in best_metrics.items():
                            if metric_info.get('path') is not None:
                                try:
                                    r = test_model(model_name, metric_info['path'], device, test_loader, firms_normalizer, model_type)
                                    if r:
                                        structured_results[model_name][metric_name] = {
                                            'precision': r['precision'], 'recall': r['recall'], 'f1': r['f1'], 'pr_auc': r['pr_auc'],
                                            'roc_auc': r.get('roc_auc', 0.0), 'fpr': r.get('fpr', 1.0), 'mse': r['mse'], 'mae': r['mae'], 'accuracy': r.get('accuracy', 0.0)
                                        }
                                        pre = f"test/{metric_name}"
                                        test_log[f"{pre}/accuracy"] = r.get('accuracy', 0.0)
                                        test_log[f"{pre}/precision"] = r['precision']
                                        test_log[f"{pre}/recall"] = r['recall']
                                        test_log[f"{pre}/f1"] = r['f1']
                                        test_log[f"{pre}/pr_auc"] = r['pr_auc']
                                        test_log[f"{pre}/roc_auc"] = r.get('roc_auc', 0.0)
                                        test_log[f"{pre}/fpr"] = r.get('fpr', 1.0)
                                        test_log[f"{pre}/mse"] = r['mse']
                                        test_log[f"{pre}/mae"] = r['mae']
                                        print(f"✅ {model_name} ({metric_name}) test completed")
                                except Exception as e:
                                    print(f"❌ {model_name} ({metric_name}) test failed: {str(e)}")
                        if test_log and TRAINING_CONFIG.get('use_wandb') and WANDB_AVAILABLE and wandb.run is not None:
                            _wandb_log_test_bar_chart(test_log)
                            print("✅ Test metrics (test/ group) logged as bar chart")
                        if WANDB_AVAILABLE and wandb.run is not None:
                            wandb.finish()
                    else:
                        failed_models.append(model_name)
                except Exception as e:
                    print(f"❌ {model_name} {model_type} model training failed: {e}")
                    failed_models.append(model_name)
                    if WANDB_AVAILABLE and wandb.run is not None:
                        wandb.finish()
                    # Clear GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    else:
        print(f"\n✅ All {model_type} models have already been trained, skipping training phase")
    
    # Add trained models to results
    for model_name, trained_paths in trained_models.items():
        model_results.append((model_name, trained_paths))
        print(f"📋 Loaded trained model: {model_name} ({len(trained_paths)} saved versions)")
    
    print(f"\n📈 {model_type} model preparation completed!")
    print(f"    New training: {len(models_to_train)} models")
    print(f"    Trained: {len(trained_models)} models") 
    print(f"    Training failed: {len(failed_models)} models")
    print(f"    Total available: {len(model_results)} models")
    
    if failed_models:
        print(f"❌ Failed {model_type} models: {', '.join(failed_models)}")
    
    # 🔥 Fix: Create test_loader for testing phase (needed regardless of training mode or whether models were trained)
    # This ensures test_loader exists even when all models are already trained
    if test_loader is None:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=standard_config['batch_size'], 
            shuffle=False,
            num_workers=test_workers,
            collate_fn=test_collate_fn, 
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            persistent_workers=use_persistent_workers,
            prefetch_factor=prefetch
        )
        print(f"✅ Test loader created for testing phase")
    
    # Testing phase
    print("\n" + "="*60)
    print("🧪 Testing phase - evaluate trained models")
    print("="*60)
    
    # Dictionary to store structured test results
    structured_results = {}
    
    for model_name, metrics in model_results:
        print(f"\n📋 Testing model: {model_name}")
        print("-" * 40)
        
        # Initialize dictionary for model's results
        structured_results[model_name] = {
            'precision': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
            'recall': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
            'f1': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
            'pr_auc': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
            'roc_auc': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
            'fpr': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
            'mae': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
            'mse': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
            'final_epoch': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None}
        }
        
        # Test all saved models (including best model and final epoch model)
        for metric_name, metric_info in metrics.items():
            if metric_info['path'] is not None:
                if metric_name == 'final_epoch':
                    print(f"\n🎯 Testing final epoch model (epoch: {metric_info['score']})")
                else:
                    print(f"\n🎯 Testing best {metric_name.upper()} model (score: {metric_info['score']:.4f})")
                try:
                    result = test_model(model_name, metric_info['path'], device, test_loader, firms_normalizer, model_type)
                    if result:
                        # Save to structured results
                        structured_results[model_name][metric_name] = {
                            'precision': result['precision'],
                            'recall': result['recall'],
                            'f1': result['f1'],
                            'pr_auc': result['pr_auc'],
                            'roc_auc': result.get('roc_auc', 0.0),
                            'fpr': result.get('fpr', 1.0),
                            'mse': result['mse'],
                            'mae': result['mae'],
                            'accuracy': result.get('accuracy', 0.0)
                        }
                        print(f"✅ {model_name} ({metric_name}) test completed")
                except Exception as e:
                    print(f"❌ {model_name} ({metric_name}) test failed: {str(e)}")
        
        # 保存测试结果到log文件
        save_test_results_to_log(structured_results[model_name], log_file, model_name)
    
    if not structured_results:
        print("⚠️ No models passed testing!")
        return None
    
    # Save structured results to CSV
    save_structured_results_to_csv(structured_results, model_type)
    
    # Output final summary of results
    print("\n" + "="*80)
    print("📊 Final test results summary")
    print("="*80)
    
    # Display results in tabular format
    for model_name, model_results in structured_results.items():
        print(f"\n🔥 Model: {model_name}")
        print("-" * 80)
        print(f"{'Metric type':<12} {'Precision':<8} {'Recall':<8} {'F1 score':<8} {'PR-AUC':<8} {'ROC-AUC':<8} {'FPR':<8} {'MSE':<10} {'MAE':<10}")
        print("-" * 100)
        for metric_type, metrics in model_results.items():
            if metrics['precision'] is not None:
                display_type = "FINAL" if metric_type == 'final_epoch' else metric_type.upper()
                print(f"{display_type:<12} {metrics['precision']:<8.4f} {metrics['recall']:<8.4f} {metrics['f1']:<8.4f} {metrics['pr_auc']:<8.4f} {metrics.get('roc_auc', 0.0):<8.4f} {metrics.get('fpr', 1.0):<8.4f} {metrics['mse']:<10.6f} {metrics['mae']:<10.6f}")
    
    print(f"\n🎉 Training and testing completed! Total {len(model_results)} models trained")
    
    if failed_models:
        print(f"\n⚠️ Failed models: {failed_models}")
    
    print("\n📁 All models saved to corresponding directories")
    save_dir = STANDARD_MODEL_DIR
    print(f"Test results saved to directory: {save_dir}")
    print(f"Log file: {log_file}")
    
    return structured_results

def prepare_data_loaders():
    """Prepare data loaders"""
    print("📂 Loading data...")
    data_loader = YearTimeSeriesDataLoader(

        # hackathon data, the only survival data
        # h5_dir = '/beluga/wildfire_prediction/all_data_masked_10x_without_qa_masked_clip_min_max_normalized',
        
        # new set
        # h5_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/h5_dataset/newset_dataset_v4',
        # h5_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/h5_dataset/newset_dataset_v7',  # NFS — slow, kept as fallback
        h5_dir = DATA_CONFIG['h5_dir'],  # local copy on /caribou (sdc1)
        
        # h5_dir='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/year_datasets_h5_masked_10x',
        
        # the most stable one, and with normalization
        # h5_dir='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/full_datasets',
        
        # try to impute the data
        # h5_dir='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/full_datasets_imputed',
        
        # with true LAI and distance map but without normalization
        # h5_dir='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/h5_dataset/all_data_masked_10x_without_qa_Norm_masked_pixel_clean_10',
        
        # with norm and right data
        # h5_dir='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/h5_dataset/all_data_masked_10x_without_qa_masked_clip_min_max_normalized',
        # h5_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/h5_dataset/all_data_masked_10x_without_qa_masked_clip_min_max_normalized_withoutThrmalFiltering',
        
        # without downsampling withhout test seems not correct as it does not sample the data
        # h5_dir='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked_result_undownsampled',
                
        # with qa and different data format
        # h5_dir='/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data_masked_result_10x_QAapplied',
        

        positive_ratio=DATA_CONFIG['positive_ratio'],
        pos_neg_ratio=DATA_CONFIG['pos_neg_ratio'],
        resample_each_epoch=False,  # Disable resampling at the bottom layer, use dynamic sampling instead
        force_resample=False,  # 🚀 禁用强制重新采样，使用缓存避免每次训练都重新生成样本
        min_fire_threshold=0.001,  # Set fire threshold to 10 (changed from default 0.001)
        
        # 🔥 关键修复：传递正确的序列长度参数
        lookback_seq=TRAINING_CONFIG['seq_len'],    # 使用训练配置中的seq_len
        forecast_hor=TRAINING_CONFIG['pred_len'],    # 使用训练配置中的pred_len
        patch_size=DATA_CONFIG.get('patch_size', 13),
        
    )
    
    # 🚀 数据预加载策略（根据样本数量自适应）
    # enable_preload = os.getenv('WF_ENABLE_PRELOAD', 'true').lower() == 'true'  # 默认禁用
    # total_samples = len(data_loader.dataset.sample_index)
    
    # if enable_preload and total_samples < 50000:
    #     # 只有样本数<50000时才预加载（否则太慢）
        # print(f"🚀 Enabling data preloading for {total_samples} samples...")
        # data_loader.enable_preload(max_samples=None, num_workers=None)
        # print("✅ Data preloading completed!")
    # else:
    #     if enable_preload and total_samples >= 50000:
    #         print(f"⚠️  Skipping preload: {total_samples} samples too many (>50000)")
    #     print("💡 Using DataLoader caching strategy instead (第一个epoch慢，后续快)")
    #     print("   第一个epoch: DataLoader会缓存数据到workers")
    #     print("   后续epoch: 直接使用缓存，速度很快 ✅")
    
    # Dataset split
    train_indices, val_indices, test_indices = data_loader.get_year_based_split(
        train_years=DATA_CONFIG['train_years'],
        val_years=DATA_CONFIG['val_years'],
        test_years=DATA_CONFIG['test_years']
    )
    
    print(f"🔍 Year-based split results:")
    print(f"   Train: {len(train_indices)} samples")
    print(f"   Val: {len(val_indices)} samples")
    print(f"   Test: {len(test_indices)} samples")
    
    # Use custom dynamic sampling dataset instead of standard Subset
    train_dataset = DynamicSamplingSubset(
        dataset=data_loader.dataset,
        full_indices=train_indices,
        sampling_ratio=DATA_CONFIG['sampling_ratio'],
        enable_dynamic_sampling=DATA_CONFIG['enable_dynamic_sampling']
    )
    
    # Validation set: use split on the same underlying dataset (no dynamic sampling)
    val_dataset = Subset(data_loader.dataset, val_indices)

    # Test set: allow overriding pos_neg_ratio or using full negatives
    test_pos_neg_ratio = DATA_CONFIG.get('test_pos_neg_ratio', None)
    test_use_full = DATA_CONFIG.get('test_use_full_test', False)
    
    # Debug: Print configuration values
    print(f"🔍 Test sampling configuration:")
    print(f"   test_pos_neg_ratio: {test_pos_neg_ratio} (type: {type(test_pos_neg_ratio)})")
    print(f"   test_use_full: {test_use_full} (type: {type(test_use_full)})")
    print(f"   Condition check: (test_pos_neg_ratio is None) = {test_pos_neg_ratio is None}")
    print(f"   Condition check: (not test_use_full) = {not test_use_full}")
    print(f"   Combined condition: {(test_pos_neg_ratio is None) and (not test_use_full)}")
    
    # Initialize variables for test dataset handling
    data_loader_test = None
    test_indices2 = None

    if (test_pos_neg_ratio is None) and (not test_use_full):
        # Default: use the same underlying dataset and indices as train/val
        test_dataset = Subset(data_loader.dataset, test_indices)
    else:
        # Build a dedicated loader/dataset for test with overridden sampling rules
        # Use the same underlying dataset to avoid regenerating cache
        override_ratio = float('inf') if test_use_full else float(test_pos_neg_ratio)
        print(f"🔍 Creating test data loader with pos_neg_ratio={override_ratio}")
        
        # Create a copy of the existing dataset with different sampling parameters
        test_dataset = data_loader.dataset.copy_with_sampling_params(
            positive_ratio=DATA_CONFIG['positive_ratio'],
            pos_neg_ratio=override_ratio
        )
        
        # Create test data loader using PyTorch DataLoader with the copied dataset
        from torch.utils.data import DataLoader
        tw = 0 if os.getenv('WF_REPRODUCIBLE', '0') == '1' else int(os.getenv('WF_TEST_WORKERS', '4'))
        data_loader_test = DataLoader(
            test_dataset,
            batch_size=TRAINING_CONFIG['standard']['batch_size'],
            shuffle=False,
            num_workers=tw,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=(tw > 0),
            worker_init_fn=worker_init_fn
        )
        print(f"🔍 Test data loader created with pos_neg_ratio={test_dataset.pos_neg_ratio}")
        # For test dataset, we need to filter by test years since the sampling may have changed the indices
        # Create test year filter for the new dataset
        test_years = DATA_CONFIG['test_years']
        test_indices_filtered = []
        
        base_test_dataset = test_dataset
        print(f"🔍 Filtering test samples: test_years={test_years}, total_samples={len(base_test_dataset.all_samples)}")
        
        # Debug: Check year distribution
        year_counts = {}
        for idx, sample in enumerate(base_test_dataset.all_samples):
            year = sample.get('year', -1)
            year_counts[year] = year_counts.get(year, 0) + 1
            if year in test_years:
                test_indices_filtered.append(idx)
        
        print(f"🔍 Year distribution in test dataset: {sorted(year_counts.items())}")
        print(f"🔍 Filtered test indices count: {len(test_indices_filtered)}")
        
        if len(test_indices_filtered) == 0:
            print(f"⚠️ WARNING: No test samples found for years {test_years}!")
            print(f"   Available years: {sorted(set(year_counts.keys()))}")
            # Fallback: use original test_indices if available
            if len(test_indices) > 0:
                print(f"   Falling back to original test_indices: {len(test_indices)} samples")
                test_dataset = Subset(data_loader.dataset, test_indices)
                test_indices2 = test_indices
            else:
                raise ValueError(f"No test samples found for years {test_years} and no fallback available!")
        else:
            final_test_indices = list(test_indices_filtered)
            if not math.isinf(override_ratio):
                pos_indices = [idx for idx in test_indices_filtered
                               if base_test_dataset.all_samples[idx].get('sample_type') == 'positive']
                neg_indices = [idx for idx in test_indices_filtered
                               if base_test_dataset.all_samples[idx].get('sample_type') != 'positive']
                
                # 调试：检查 sample_type 的实际值
                sample_types = {}
                for idx in test_indices_filtered[:100]:  # 只检查前100个样本
                    st = base_test_dataset.all_samples[idx].get('sample_type', 'MISSING')
                    sample_types[st] = sample_types.get(st, 0) + 1
                print(f"🔍 Sample type distribution (first 100): {sample_types}")
                
                print(f"🔍 Before ratio filtering: {len(pos_indices)} pos, {len(neg_indices)} neg")
                desired_neg = int(len(pos_indices) * override_ratio)
                
                # 如果正样本为 0，说明标签可能有问题，使用原始 test_indices
                if len(pos_indices) == 0 and len(test_indices) > 0:
                    print(f"⚠️ WARNING: No positive samples found after filtering, falling back to original test_indices")
                    test_dataset = Subset(data_loader.dataset, test_indices)
                    test_indices2 = test_indices
                elif desired_neg < len(neg_indices):
                    rng = random.Random(getattr(data_loader.dataset, 'epoch_seed', 42))
                    neg_indices = rng.sample(neg_indices, desired_neg)
                    final_test_indices = pos_indices + neg_indices
                    print(f"🔍 After ratio filtering: {len(pos_indices)} pos, {len(neg_indices)} neg, total={len(final_test_indices)}")
                    epoch_seed = getattr(data_loader.dataset, 'epoch_seed', 42)
                    if epoch_seed is None:
                        epoch_seed = 42
                    rng = random.Random(epoch_seed + 1)
                    rng.shuffle(final_test_indices)
                    
                    # Use filtered indices for test dataset
                    test_dataset = Subset(base_test_dataset, final_test_indices)
                    test_indices2 = final_test_indices
                else:
                    final_test_indices = pos_indices + neg_indices
                    print(f"🔍 After ratio filtering: {len(pos_indices)} pos, {len(neg_indices)} neg, total={len(final_test_indices)}")
                    epoch_seed = getattr(data_loader.dataset, 'epoch_seed', 42)
                    if epoch_seed is None:
                        epoch_seed = 42
                    rng = random.Random(epoch_seed + 1)
                    rng.shuffle(final_test_indices)
                    
                    # Use filtered indices for test dataset
                    test_dataset = Subset(base_test_dataset, final_test_indices)
                    test_indices2 = final_test_indices

    # ===== Optionally apply imputed H5 to override past windows =====
    if DATA_CONFIG.get('imputed_h5'):
        imputed_path = DATA_CONFIG['imputed_h5']
        impute_mode = DATA_CONFIG.get('impute_mode', 'full')
        if os.path.isfile(imputed_path):
            print(f"🧩 Using imputed H5 to override past windows: {imputed_path} (mode={impute_mode})")

            import h5py
            from torch.utils.data import Dataset

            class ImputedWrapperDataset(Dataset):
                def __init__(self, base_dataset, indices):
                    self.base = base_dataset
                    self.indices = list(indices)
                    self.h5 = h5py.File(imputed_path, 'r')
                    self.mode = impute_mode

                def __len__(self):
                    return len(self.indices)

                def __getitem__(self, i):
                    idx = self.indices[i]
                    past, future, meta = self.base[idx]
                    date_int, row, col = meta
                    key = f"{int(date_int)}_{int(row)}_{int(col)}"
                    if key in self.h5:
                        arr = self.h5[key][()]  # [39, L]
                        import numpy as np
                        x_imp = torch.from_numpy(arr.astype(np.float32))  # [C,L]
                        if self.mode == 'full':
                            past = x_imp
                        else:  # replace_missing
                            # 原始 [C,L] 中 0 视作缺失（通道0 特殊：0 为合法观测）
                            obs = (past != 0)
                            obs[0, :] = True
                            past = past * obs + x_imp * (~obs)
                    return past, future, meta

                def __del__(self):
                    try:
                        self.h5.close()
                    except Exception:
                        pass

            train_dataset = ImputedWrapperDataset(data_loader.dataset, train_indices)
            val_dataset = ImputedWrapperDataset(data_loader.dataset, val_indices)
            # For test dataset, use the correct indices based on whether we overrode sampling
            if (test_pos_neg_ratio is None) and (not test_use_full):
                test_dataset = ImputedWrapperDataset(data_loader.dataset, test_indices)
            else:
                test_dataset = ImputedWrapperDataset(data_loader_test.dataset, test_indices2)
        else:
            print(f"⚠️ imputed_h5 not found: {imputed_path}. Ignoring.")
    
    print(f"📊 Dataset size:")
    print(f"    Training set: {len(train_dataset)} (full: {len(train_indices)})")
    print(f"    Validation set: {len(val_dataset)} (full data)")
    if (test_pos_neg_ratio is None) and (not test_use_full):
        print(f"    Test set: {len(test_dataset)} (inherit train/val sampling rules)")
    else:
        mode = 'FULL' if test_use_full else f"1:{override_ratio}"
        print(f"    Test set: {len(test_dataset)} (override pos_neg_ratio={mode})")
        
        # Debug: Print test dataset statistics based on split indices
        if data_loader_test is not None:
            # For test dataset with independent sampling, use the entire dataset
            # test_stats = calculate_split_stats(data_loader_test.dataset, None)
            test_stats = calculate_split_stats(data_loader_test.dataset, test_indices2)  # 只统计测试年子集
            print(f"    Test dataset stats: {test_stats['positive_samples']} pos, {test_stats['negative_samples']} neg, ratio=1:{test_stats['pos_neg_ratio']:.2f}")
            
            # Calculate statistics for the actual train split (train_indices)
            train_stats = calculate_split_stats(data_loader.dataset, train_indices)
            print(f"    Train dataset stats (full): {train_stats['positive_samples']} pos, {train_stats['negative_samples']} neg, ratio=1:{train_stats['pos_neg_ratio']:.2f}")
            
            # Calculate statistics for the actual used train dataset (after dynamic sampling)
            # actual_train_stats = calculate_split_stats(data_loader.dataset, list(range(len(train_dataset))))
            actual_train_stats = calculate_split_stats(data_loader.dataset, train_dataset.current_indices)
            sampling_ratio_pct = DATA_CONFIG['sampling_ratio'] * 100
            print(f"    Train dataset stats ({sampling_ratio_pct:.1f}%): {actual_train_stats['positive_samples']} pos, {actual_train_stats['negative_samples']} neg, ratio=1:{actual_train_stats['pos_neg_ratio']:.2f}")
            print(f"    Test/Train ratio difference: {abs(test_stats['pos_neg_ratio'] - train_stats['pos_neg_ratio']):.3f}")
    print(f"    Dynamic sampling: {'Enabled' if DATA_CONFIG['enable_dynamic_sampling'] else 'Disabled'}")
    if DATA_CONFIG['enable_dynamic_sampling']:
        print(f"    Sampling configuration: Randomly use {DATA_CONFIG['sampling_ratio']:.1%} of training data per epoch")
    
    return train_dataset, val_dataset, test_dataset, data_loader, data_loader_test

def calculate_split_stats(dataset, split_indices):
    """Calculate statistics for a specific split based on indices"""
    positive_count = 0
    negative_count = 0

    if not hasattr(dataset, "sample_index"):
        base_dataset = None
        mapped_indices = split_indices

        if isinstance(dataset, Subset):
            base_dataset = dataset.dataset
            subset_indices = list(dataset.indices)
            if split_indices is None or len(split_indices) == 0:
                mapped_indices = subset_indices
            else:
                mapped_indices = [subset_indices[i] for i in split_indices if 0 <= i < len(subset_indices)]
        elif hasattr(dataset, "dataset"):
            base_dataset = dataset.dataset
        elif hasattr(dataset, "base"):
            base_dataset = dataset.base
            if hasattr(dataset, "indices"):
                base_map = list(dataset.indices)
                if split_indices is None or len(split_indices) == 0:
                    mapped_indices = base_map
                else:
                    mapped_indices = [base_map[i] for i in split_indices if 0 <= i < len(base_map)]

        if base_dataset is not None:
            return calculate_split_stats(base_dataset, mapped_indices)
        raise AttributeError("Dataset does not expose sample_index and cannot be unwrapped for statistics.")

    sample_index = dataset.sample_index
    def _iter_indices():
        if split_indices is None or len(split_indices) == 0:
            return range(len(sample_index))
        return (idx for idx in split_indices if idx < len(sample_index))

    total_samples = 0
    for idx in _iter_indices():
        try:
            _, _, metadata = sample_index[idx]
        except IndexError:
            continue
        total_samples += 1
        sample_type = metadata.get('sample_type')
        if sample_type is None:
            firms_value = metadata.get('firms_value', 0.0)
            sample_type = 'positive' if firms_value >= dataset.min_fire_threshold else 'negative'
        if sample_type == 'positive':
            positive_count += 1
        else:
            negative_count += 1

    pos_neg_ratio = (negative_count / positive_count) if positive_count > 0 else (float('inf') if negative_count > 0 else 0.0)
    positive_ratio = (positive_count / total_samples) if total_samples > 0 else 0.0

    return {
        'total_samples': total_samples,
        'positive_samples': positive_count,
        'negative_samples': negative_count,
        'positive_ratio': positive_ratio,
        'pos_neg_ratio': pos_neg_ratio
    }

def main():
    """Main function - train standard models"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Script for training wildfire prediction models')
    
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retrain all models, ignoring existing model files')
    
    # Multi-task regression learning parameters
    parser.add_argument('--firms-weight', type=float, default=0.005,  # 0.005 for focal loss, 0.1 for multitask 
                       help='Loss weight for FIRMS prediction (default: 1.0)')
    parser.add_argument('--other-drivers-weight', type=float, default=1.0,
                       help='Loss weight for other drivers prediction (default: 1.0)')
    parser.add_argument('--loss-function', type=str, default='mse',
                       choices=['huber', 'mse', 'mae'],
                       help='Type of regression loss function for other drivers (default: mse)')
    parser.add_argument('--no-ignore-zero', action='store_true',
                       help='Do not ignore0 values in other drivers')
    
    # Focal Loss parameters
    parser.add_argument('--focal-alpha', type=float, default=0.5,
                       help='Alpha parameter for Focal Loss (default: 0.5)')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                       help='Gamma parameter for Focal Loss (default: 2.0)')
    
    # Loss function type selection parameter
    parser.add_argument('--loss-type', type=str, default='focal',  ####################################### use focal by default
                       choices=['focal', 'kldiv', 'multitask'],
                       help='Type of loss function to use (default: focal)')
    
    # 🔥 New: Position information and weather data feature parameters
    parser.add_argument('--enable-position-features', action='store_true',
                       help='Enable position information feature (default: disabled)')
    parser.add_argument('--enable-future-weather', action='store_true', 
                       help='Enable future weather data feature (default: disabled)')
    parser.add_argument('--weather-channels', type=str, default='1-12',
                       help='Range of weather data channels, format like "1-12" or "1,3,5-8" (default: 1-12)')

    zscore_group = parser.add_mutually_exclusive_group()
    zscore_group.add_argument('--use-zscore', dest='use_zscore', action='store_true',
                              help='Enable train-only channel-wise z-score standardization')
    zscore_group.add_argument('--no-zscore', dest='use_zscore', action='store_false',
                              help='Disable channel-wise z-score standardization')
    parser.set_defaults(use_zscore=env_flag('WF_USE_ZSCORE', False))
    parser.add_argument('--zscore-stats', type=str, default=os.getenv('WF_ZSCORE_STATS_PATH', ''),
                        help='Z-score H5 path; default is generated beside the yearly H5 files')
    parser.add_argument('--force-zscore-recompute', action='store_true',
                        default=env_flag('WF_ZSCORE_FORCE_RECOMPUTE', False),
                        help='Recompute training statistics even when the stats H5 already exists')
    
    # GPU selection parameter
    parser.add_argument('--gpu', type=int, default=1,
                       help='GPU device ID to use (default: 0)')
    
    # Model selection parameter
    parser.add_argument('--models', type=str, default=None,
                       help='Specify model names to train, comma-separated (e.g., --models s_mamba_org,autoformer,DLinear). If not specified, train all models.')
    
    # Parallel training parameter
    parser.add_argument('--parallel-models', type=int, default=1,
                       help='Number of models to train in parallel (default: 1). If set to 2, will train 2 models simultaneously when memory allows.')
    
    # Test mode parameter
    parser.add_argument('--test', action='store_true',
                       help='Test mode: Skip training and only test models from STANDARD_MODEL_DIR')
    
    args = parser.parse_args()
    
    # 🔥 New: Parse range of weather data channels
    def parse_channel_range(channel_str):
        """Parse channel range string, return list of channel indices"""
        channels = []
        for part in channel_str.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                channels.extend(range(start, end + 1))
            else:
                channels.append(int(part))
        return channels
    
    # Update data configuration
    global DATA_CONFIG
    DATA_CONFIG['enable_position_features'] = args.enable_position_features
    DATA_CONFIG['enable_future_weather'] = args.enable_future_weather
    DATA_CONFIG['use_zscore'] = bool(args.use_zscore)
    DATA_CONFIG['zscore_stats_path'] = args.zscore_stats or ''
    DATA_CONFIG['zscore_force_recompute'] = bool(args.force_zscore_recompute)
    
    if args.enable_future_weather:
        try:
            DATA_CONFIG['weather_channels'] = parse_channel_range(args.weather_channels)
        except ValueError as e:
            print(f"❌ Error: Invalid format for weather channel range: {args.weather_channels}")
            print(f"    Error message: {e}")
            print(f"    Correct format example: '1-12' or '1,3,5-8'")
            return
    
    # Update multi-task learning configuration
    global MULTITASK_CONFIG, TRAINING_CONFIG
    MULTITASK_CONFIG['firms_weight'] = args.firms_weight
    MULTITASK_CONFIG['other_drivers_weight'] = args.other_drivers_weight
    MULTITASK_CONFIG['loss_function'] = args.loss_function
    MULTITASK_CONFIG['ignore_zero_values'] = not args.no_ignore_zero
    MULTITASK_CONFIG['loss_type'] = args.loss_type  # New: Loss function type configuration
    
    # Update Focal Loss configuration
    TRAINING_CONFIG['focal_alpha'] = args.focal_alpha
    TRAINING_CONFIG['focal_gamma'] = args.focal_gamma
    
    print("🔥 Comprehensive wildfire forecasting model benchmark - unified version")
    
    # Display shared configuration status
    print_config_status()
    print()
    
    # 🔥 New: Filter models based on command line arguments
    models_to_train = MODEL_LIST_STANDARD.copy()
    if args.models:
        # Parse comma-separated model names
        specified_models = [m.strip() for m in args.models.split(',')]
        
        # Remove .py extension if present
        specified_models = [m[:-3] if m.endswith('.py') else m for m in specified_models]
        
        # Remove empty strings
        specified_models = [m for m in specified_models if m]
        
        if not specified_models:
            print(f"❌ Error: No valid model names provided after parsing")
            return
        
        # Validate that all specified models exist
        available_models = set(MODEL_LIST_STANDARD)
        invalid_models = [m for m in specified_models if m not in available_models]
        
        if invalid_models:
            print(f"❌ Error: The following models are not available:")
            for m in invalid_models:
                print(f"   - {m}")
            print(f"\n📋 Available models:")
            print(f"   {', '.join(sorted(MODEL_LIST_STANDARD))}")
            return
        
        # Filter to only specified models
        models_to_train = [m for m in MODEL_LIST_STANDARD if m in specified_models]
        print(f"✅ Model filter applied: Training {len(models_to_train)} specified model(s)")
        print(f"   Selected models: {', '.join(models_to_train)}")
    else:
        print(f"📋 No model filter specified: Training all {len(models_to_train)} available models")
    
    # Validate parallel_models parameter
    if args.parallel_models < 1:
        print(f"⚠️ Invalid parallel_models value: {args.parallel_models}, setting to 1")
        args.parallel_models = 1
    
    if args.parallel_models > 1:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if args.parallel_models > num_gpus:
                print(f"⚠️ Warning: parallel_models ({args.parallel_models}) > available GPUs ({num_gpus})")
                print(f"   Will use {num_gpus} parallel models (one per GPU)")
                args.parallel_models = num_gpus
        else:
            print(f"⚠️ Warning: No GPU available, parallel training may not be efficient")
            print(f"   Consider using sequential training (--parallel-models 1)")
    
    print(f"⚙️ Parallel training: {args.parallel_models} model(s) will be trained simultaneously")
    
    # Based on command line arguments, decide which models to train
    train_standard = True
    
    print("📋 Training plan: Standard models ✅")
    if args.force_retrain:
        print("🔄 Force retrain mode enabled, will ignore existing model files")
    
    # Initialize
    set_seed(TRAINING_CONFIG['seed'])
    
    # GPU selection
    if torch.cuda.is_available():
        gpu_id = args.gpu
        if gpu_id >= torch.cuda.device_count():
            print(f"⚠️  Warning: GPU {gpu_id} not available. Available GPUs: 0-{torch.cuda.device_count()-1}")
            print(f"   Falling back to GPU 0")
            gpu_id = 0
        device = torch.device(f'cuda:{gpu_id}')
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"🖥️  Using device: cuda:{gpu_id} ({gpu_name})")
        print(f"🔍 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    else:
        device = torch.device('cpu')
        print(f"🖥️  Using device: {device}")
    
    # WandB configuration check
    if TRAINING_CONFIG['use_wandb']:
        if WANDB_AVAILABLE:
            print("✅ WandB monitoring enabled")
        else:
            print("⚠️ WandB monitoring configured but wandb not installed, will skip monitoring functionality")
    else:
        print("ℹ️ WandB monitoring disabled")
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"💾 GPU memory: {gpu_memory:.1f} GB")
    
    # Prepare data
    train_dataset, val_dataset, test_dataset, data_loader_obj, data_loader_test = prepare_data_loaders()
    
    # Initialize FIRMS normalizer
    print("🔧 Initializing FIRMS normalizer...")
    firms_normalizer = FIRMSNormalizer(
        method='divide_by_100',
        firms_min=DATA_CONFIG['firms_min'],
        firms_max=DATA_CONFIG['firms_max']
    )
    
    # Create temporary data loader for fitting normalizer
    temp_loader = DataLoader(
        train_dataset, batch_size=512, shuffle=False, 
        num_workers=2, collate_fn=data_loader_obj.dataset.custom_collate_fn
    )
    firms_normalizer.fit(temp_loader)
    configure_channel_zscore_normalizer(firms_normalizer, train_dataset)
    
    all_results = {}
    
    # ========== Test mode: Only test models without training ==========
    if args.test:
        print(f"\n{'='*80}")
        print("🧪 Test mode: Testing models from STANDARD_MODEL_DIR (skipping training)")
        print(f"{'='*80}")
        
        # Find all available model files in STANDARD_MODEL_DIR
        model_files = []
        if os.path.exists(STANDARD_MODEL_DIR):
            model_files.extend(glob.glob(os.path.join(STANDARD_MODEL_DIR, '*_best_*.pth')))
            model_files.extend(glob.glob(os.path.join(STANDARD_MODEL_DIR, '*_final_epoch.pth')))
        else:
            print(f"❌ Model directory does not exist: {STANDARD_MODEL_DIR}")
            return
        
        # Extract model names and organize by model
        available_models = {}
        for model_file in model_files:
            basename = os.path.basename(model_file)
            # Parse model name from filename (e.g., "WaveletMixer_best_f1.pth" -> "WaveletMixer")
            if '_best_' in basename:
                model_name = basename.split('_best_')[0]
                metric_type = basename.split('_best_')[1].replace('.pth', '')
            elif '_final_epoch.pth' in basename:
                model_name = basename.replace('_final_epoch.pth', '')
                metric_type = 'final_epoch'
            else:
                continue
            
            if model_name not in available_models:
                available_models[model_name] = {}
            
            available_models[model_name][metric_type] = {
                'path': model_file,
                'score': 0.0  # Score will be updated during testing
            }
        
        if not available_models:
            print(f"❌ No model files found in {STANDARD_MODEL_DIR}")
            print(f"   Please check the directory path and ensure model files exist")
            return
        
        print(f"📋 Found {len(available_models)} models with {sum(len(v) for v in available_models.values())} saved versions")
        for model_name, versions in available_models.items():
            print(f"   {model_name}: {len(versions)} versions ({', '.join(versions.keys())})")
        
        # Filter models if --models argument is provided
        if args.models:
            specified_models = [m.strip() for m in args.models.split(',')]
            specified_models = [m[:-3] if m.endswith('.py') else m for m in specified_models]
            specified_models = [m for m in specified_models if m]
            
            # Filter available models
            filtered_models = {k: v for k, v in available_models.items() if k in specified_models}
            missing_models = [m for m in specified_models if m not in available_models]
            
            if missing_models:
                print(f"⚠️ Warning: The following models were not found in {STANDARD_MODEL_DIR}:")
                for m in missing_models:
                    print(f"   - {m}")
            
            if filtered_models:
                available_models = filtered_models
                print(f"✅ Filtered to {len(available_models)} specified model(s)")
            else:
                print(f"❌ No specified models found in {STANDARD_MODEL_DIR}")
                return
        
        # Create test loader
        standard_config = TRAINING_CONFIG['standard']
        test_workers = 0 if os.getenv('WF_REPRODUCIBLE', '0') == '1' else int(os.getenv('WF_TEST_WORKERS', '8'))
        prefetch = int(os.getenv('WF_PREFETCH_FACTOR', '4'))
        use_persistent = (test_workers > 0)
        if data_loader_test is not None:
            test_collate_fn = data_loader_test.dataset.custom_collate_fn
        else:
            test_collate_fn = data_loader_obj.dataset.custom_collate_fn
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=standard_config['batch_size'], 
            shuffle=False,
            num_workers=test_workers,
            collate_fn=test_collate_fn, 
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            persistent_workers=use_persistent,
            prefetch_factor=prefetch
        )
        print(f"✅ Test loader created for testing phase")
        
        # Test all available models
        structured_results = {}
        
        for model_name, model_versions in available_models.items():
            print(f"\n📋 Testing model: {model_name}")
            print("-" * 40)
            
            # Initialize dictionary for model's results
            structured_results[model_name] = {
                'precision': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'recall': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'f1': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'pr_auc': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'roc_auc': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'fpr': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'mae': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'mse': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None},
                'final_epoch': {'precision': None, 'recall': None, 'f1': None, 'pr_auc': None, 'roc_auc': None, 'fpr': None, 'mse': None, 'mae': None}
            }
            
            # Test all saved model versions
            for metric_name, metric_info in model_versions.items():
                if metric_info['path'] is not None:
                    if metric_name == 'final_epoch':
                        print(f"\n🎯 Testing final epoch model")
                    else:
                        print(f"\n🎯 Testing best {metric_name.upper()} model")
                    try:
                        result = test_model(model_name, metric_info['path'], device, test_loader, firms_normalizer, 'standard')
                        if result:
                            # Save to structured results
                            structured_results[model_name][metric_name] = {
                                'precision': result['precision'],
                                'recall': result['recall'],
                                'f1': result['f1'],
                                'pr_auc': result['pr_auc'],
                                'roc_auc': result.get('roc_auc', 0.0),
                                'fpr': result.get('fpr', 1.0),
                                'mse': result['mse'],
                                'mae': result['mae']
                            }
                            print(f"✅ {model_name} ({metric_name}) test completed")
                    except Exception as e:
                        print(f"❌ {model_name} ({metric_name}) test failed: {str(e)}")
        
        if not structured_results:
            print("⚠️ No models passed testing!")
            return
        
        # Save structured results to CSV
        save_structured_results_to_csv(structured_results, 'standard')
        
        # Output final summary of results
        print("\n" + "="*80)
        print("📊 Final test results summary")
        print("="*80)
        
        # Display results in tabular format
        for model_name, model_results in structured_results.items():
            print(f"\n🔥 Model: {model_name}")
            print("-" * 80)
            print(f"{'Metric type':<12} {'Precision':<8} {'Recall':<8} {'F1 score':<8} {'PR-AUC':<8} {'ROC-AUC':<8} {'FPR':<8} {'MSE':<10} {'MAE':<10}")
            print("-" * 100)
            for metric_type, metrics in model_results.items():
                if metrics['precision'] is not None:
                    display_type = "FINAL" if metric_type == 'final_epoch' else metric_type.upper()
                    print(f"{display_type:<12} {metrics['precision']:<8.4f} {metrics['recall']:<8.4f} {metrics['f1']:<8.4f} {metrics['pr_auc']:<8.4f} {metrics.get('roc_auc', 0.0):<8.4f} {metrics.get('fpr', 1.0):<8.4f} {metrics['mse']:<10.6f} {metrics['mae']:<10.6f}")
        
        print(f"\n🎉 Testing completed! Total {len(structured_results)} models tested")
        print(f"📁 Test results saved to: {STANDARD_MODEL_DIR}")
        
        all_results['standard'] = structured_results
    
    # ========== First stage: Train standard model_zoo models ==========
    elif train_standard and models_to_train:
        print(f"\n{'='*80}")
        print("🚀 First stage: Train standard model_zoo models")
        print(f"{'='*80}")
        
        # Note: DataLoaders will be created inside train_and_test_models
        # This allows each parallel thread to have its own DataLoader instances
        
        standard_results = train_and_test_models(
            models_to_train, 'standard', device, train_dataset, val_dataset, test_dataset,
            data_loader_obj, data_loader_test, firms_normalizer, args.force_retrain, args.parallel_models
        )
        all_results['standard'] = standard_results
    

    
    # ========== Final summary ==========
    print(f"\n{'='*80}")
    print("🎉 All models experiment completed!")
    print(f"{'='*80}")
    
    for model_type, results in all_results.items():
        if results:
            # 🔥 Fix: Convert structured_results to DataFrame format for summary
            # structured_results format: {model_name: {metric_type: {precision, recall, f1, pr_auc, mse, mae}}}
            summary_data = []
            for model_name, model_results in results.items():
                # Try to find best F1 score across all metric types
                best_f1 = None
                best_metric_type = None
                for metric_type, metrics in model_results.items():
                    if metrics.get('f1') is not None:
                        if best_f1 is None or metrics['f1'] > best_f1:
                            best_f1 = metrics['f1']
                            best_metric_type = metric_type
                
                if best_f1 is not None:
                    best_metrics = model_results[best_metric_type]
                    summary_data.append({
                        'model': model_name,
                        'precision': best_metrics.get('precision'),
                        'recall': best_metrics.get('recall'),
                        'f1': best_metrics.get('f1'),
                        'pr_auc': best_metrics.get('pr_auc'),
                        'mse': best_metrics.get('mse'),
                        'mae': best_metrics.get('mae')
                    })
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                df = df.sort_values('f1', ascending=False)
                best_model = df.iloc[0]
                print(f"\n🏆 Best {model_type} model: {best_model['model']}")
                print(f"   F1-Score: {best_model['f1']:.4f}")
                print(f"   Precision: {best_model['precision']:.4f}")
                print(f"   Recall: {best_model['recall']:.4f}")
                print(f"   PR-AUC: {best_model['pr_auc']:.4f}")
            else:
                print(f"\n⚠️ No valid test results for {model_type} models")
    
    print("\n📊 All results saved to corresponding CSV files!")

if __name__ == "__main__":
    main() 
