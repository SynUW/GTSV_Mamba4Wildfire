# Wildfire Forecasting Model Training System - User Guide


## Model Training
Before model training, you need to change the wandb setting:
1. find out **'wandb_run_name': 'Zhengsen'** in train_all_h5.py
2. replace **'Zhengsen'** with your name

You also need to change the pth sotre path to your own path
1. find out **"STANDARD_MODEL_DIR = '/mnt/raid/zhengsen/pths/new_dataset_pths/new_experiments_10to1'"** in train_all_h5.py
2. repalce **'/mnt/raid/zhengsen/pths/new_dataset_pths/new_experiments_10to1'** with your own path

```
python train_single_model_h5.py --model {your model name without .py} --gpu {0 or 1}
```

## 1. Environment Requirements

### 1.1 Python Version
- Recommended: Python 3.10

### 1.2 CUDA Environment
- Requires NVIDIA GPU and CUDA drivers, recommended CUDA 11.8
- Driver and CUDA Toolkit must be compatible with your PyTorch version

### 1.3 Required Dependencies
Please strictly follow the installation order below to ensure environment consistency.

#### 1.3.1 Create a new conda environment
```bash
conda create -n wildfire python=3.10 -y
conda activate wildfire
```

#### 1.3.2 Install PyTorch (with CUDA support)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 1.3.3 Install core dependencies
```bash
conda install numpy pandas scikit-learn tqdm h5py matplotlib
pip install wandb
```

#### 1.3.4 Install Mamba model dependencies (required for s_mamba/Mamba series)
```bash
pip install mamba-ssm
```

#### 1.3.5 Other notes
- It is recommended to use conda/pip for all dependencies to avoid version conflicts.
- If you need wandb experiment tracking, register and run `wandb login` in advance.

## 3. Model Training

### 3.1 Train a Single Model
```bash
# Train a specific model
python train_single_model_h5.py --model DLinear

# List available models
python train_single_model.py --list-models
```
- Suitable for testing or debugging a single model

### 3.2 Sequential Batch Training
```bash
# Train all standard models (sequential)
python train_all_models_combined.py --force-retrain
```
- Automatically trains all models, covering all mainstream time series architectures
- Standard model results are saved in `/mnt/raid/zhengsen/pths/7to1_Focal_woFirms_onlyFirmsLoss_newloadertest/`