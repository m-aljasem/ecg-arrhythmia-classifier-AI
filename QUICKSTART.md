"""Quick Start Guide for ECG Classification Project"""

# ECG Classification Project - Quick Start Guide

## Files Overview

### Core Modules (`src/`)
- **data/loader.py**: `ECGDataLoader` class for loading PTB-XL dataset
- **data/preprocessing.py**: `DataPreprocessor` and `AugmentedDataGenerator` for data preparation
- **models/ecg_classifiers.py**: Neural network model definitions and factory

### Main Scripts
- **train.py**: Complete training pipeline for all three models
- **evaluate.py**: Model evaluation and result visualization

### Configuration
- **config.py**: All project settings and hyperparameters

## Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset
Get PTB-XL from Kaggle and note the download path.

### Step 3: Prepare Data
```bash
python train.py --data-path /full/path/to/ptbxl_database.csv/parent --model all
```

This will:
1. Load raw ECG signals and metadata (takes ~5-10 minutes)
2. Extract features and create target labels
3. Split into train/valid/test sets (fold 8/9/10)
4. Standardize all features
5. Save to `data/processed/data.npz`

### Step 4: Train Models
All three models train automatically in Step 3, or train individually:

```bash
# Train only Model02 (best performance)
python train.py --data-path /path/to/dataset --model model02 --skip-preprocessing
```

### Step 5: Evaluate Results
```bash
python evaluate.py --model all
```

This shows:
- Confusion matrices for each diagnostic class
- Classification reports (Precision, Recall, F1)
- Overall accuracy

## Model Selection Guide

- **model01**: Quick baseline (metadata only) - ~2 min training
- **model02**: Best performance (metadata + ECG) - ~20 min training  
- **model03**: Augmentation experiment (ECG windowing + noise) - ~20 min training

## Data Preparation Details

The pipeline handles:
1. **Raw Loading**: WFDB format ECG signals (5 leads, 1000 samples each)
2. **Feature Engineering**:
   - Metadata: age, sex, height, weight, infarction stage, pacemaker status
   - Targets: 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP)
3. **Data Splitting**: 80% train / 10% validation / 10% test (by strat_fold)
4. **Standardization**: Z-score normalization (fit on train set)
5. **Augmentation** (model03): sliding windows + Gaussian noise

## Output Results

Trained models saved in `models/`:
- `model01.keras` - Metadata classifier
- `model02.keras` - Combined classifier  
- `model03.keras` - Augmented classifier

Processed data saved in `data/processed/`:
- `data.npz` - All train/valid/test splits

## Troubleshooting

**Issue**: WFDB library not found
```bash
pip install wfdb
```

**Issue**: TensorFlow GPU not detected
- Install CUDA/cuDNN or use CPU (slower but works)
- Don't worry, training still works on CPU

**Issue**: Out of memory during training
- Reduce batch_size in config.py (default 32)
- Or train model01 first (requires less memory)

## Project Organization

```
project/
├── src/data/           # Data loading and preprocessing
├── src/models/         # Model definitions
├── src/utils/          # Utilities and visualization
├── data/processed/     # Processed data (created after running)
├── models/             # Model checkpoints (created after training)
├── train.py            # Main training script
├── evaluate.py         # Evaluation script
└── config.py           # Configuration constants
```

## Configuration Customization

Edit `config.py` to change:
```python
# Model training hyperparameters
MODEL_CONFIGS['model02'] = {
    'batch_size': 32,      # Reduce if out of memory
    'epochs': 100,         # Max epochs
    'patience': 20,        # Early stopping patience
}

# Data paths
PROCESSED_DATA_FILE = 'data/processed/data.npz'
MODEL02_CHECKPOINT = 'models/model02.keras'
```

## Next Steps

After successful training:
1. Review the confusion matrices and classification reports
2. Analyze per-class performance
3. Try different hyperparameters in config.py
4. Implement additional augmentations
5. Fine-tune on specific diagnostic classes

Enjoy your ECG classification project! 🏥🔬
"""
