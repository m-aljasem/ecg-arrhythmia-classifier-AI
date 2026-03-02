# ECG Classification Project

A professional Python project for ECG signal classification using deep learning, built from the PTB-XL dataset.

## Overview

This project implements multiple neural network models for multi-label ECG classification. The models combine patient metadata and raw ECG signals to predict diagnostic superclasses:

- **NORM**: Normal ECG
- **MI**: Myocardial Infarction
- **STTC**: ST/T Change  
- **CD**: Conduction Disturbance
- **HYP**: Hypertrophy

## Project Structure

```
project/
├── src/
│   ├── data/              # Data loading and preprocessing
│   │   ├── loader.py      # ECG dataset loader
│   │   └── preprocessing.py # Data preprocessing and augmentation
│   ├── models/            # Model definitions
│   │   └── ecg_classifiers.py # Neural network models
│   └── utils/             # Utility functions
│       └── visualization.py # Visualization tools
├── data/                  # Data directory (populated after processing)
├── models/                # Trained model checkpoints
├── notebooks/             # Jupyter notebooks for exploration
├── config.py              # Project configuration
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. Clone or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install as a package:
```bash
pip install -e .
```

## Data Setup

Download the PTB-XL dataset from Kaggle:
- [PTB-XL Dataset](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1-0-1)

Extract it and note the path for the next steps.

## Usage

### 1. Data Preparation

Prepare the data (loads dataset, preprocesses, and saves to NPZ):

```bash
python train.py --data-path /path/to/ptbxl/dataset --model all --skip-preprocessing false
```

Or just prepare data without training:
```bash
python train.py --data-path /path/to/ptbxl/dataset --skip-preprocessing false
```

### 2. Training Models

Train all models:
```bash
python train.py --data-path /path/to/ptbxl/dataset --model all
```

Train specific model:
```bash
python train.py --data-path /path/to/ptbxl/dataset --model model02
```

Available models:
- **model01**: Metadata-only classifier
- **model02**: Combined metadata + ECG signals (1D CNN)
- **model03**: Model02 with data augmentation

### 3. Evaluation

Evaluate all trained models:
```bash
python evaluate.py --model all
```

Evaluate specific model:
```bash
python evaluate.py --model model02
```

## Models

### Model01: Metadata Only
- Uses only patient metadata features (age, sex, height, weight, etc.)
- 2 Dense layers (32 units) + 2 Dense layers (64 units)
- Baseline model for comparison

### Model02: Combined (Metadata + ECG)
- Metadata: 2 Dense layers (32 units)
- ECG: 3 Conv1D layers (64→128→256 filters) with BatchNorm and MaxPool
- Concatenates both embeddings for final classification
- Best overall performance (~89% binary accuracy)

### Model03: With Data Augmentation
- Same architecture as Model02
- Training-time augmentation: sliding windows + Gaussian noise
- Aims to improve generalization

## Configuration

Edit `config.py` to customize:
- Data paths
- Model hyperparameters (batch size, epochs, dropout rates, etc.)
- Sampling rate and random seed

## Key Results

- **Model01**: Limited performance (predicts only NORM and MI)
- **Model02**: Good performance on all classes (89% binary accuracy)
- **Model03**: Alternative approach with augmentation

## Development

### Adding New Models

1. Add model function to `src/models/ecg_classifiers.py`
2. Add factory method to `ECGClassifierFactory`
3. Create training function in `train.py`
4. Update configuration in `config.py`

### Data Processing Pipeline

1. **Loader**: `ECGDataLoader` - loads raw dataset
2. **Preprocessor**: `DataPreprocessor` - creates features and targets
3. **Generator**: `AugmentedDataGenerator` - provides augmented batches

## Requirements

- Python 3.8+
- TensorFlow 2.9+
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- wfdb (for reading ECG files)

## References

PTB-XL: A Large Publicly Available Electrocardiography Dataset:
https://physionet.org/content/ptb-xl/1.0.1/

## License

MIT License

## Notes

- Original notebook (`../main.ipynb`) kept intact as reference
- All models use binary crossentropy loss for multi-label classification
- Early stopping and model checkpointing during training
- Test set evaluation on stratification fold 10 (as per dataset authors)
