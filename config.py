"""Project Configuration"""

import os

# Data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'processed', 'data.npz')

# Model paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
MODEL01_CHECKPOINT = os.path.join(MODELS_DIR, 'model01.keras')
MODEL02_CHECKPOINT = os.path.join(MODELS_DIR, 'model02.keras')
MODEL03_CHECKPOINT = os.path.join(MODELS_DIR, 'model03.keras')

# Model hyperparameters
MODEL_CONFIGS = {
    'model01': {
        'batch_size': 32,
        'epochs': 40,
        'patience': 10,
        'metadata_units': 32,
        'metadata_dropout': 0.3,
    },
    'model02': {
        'batch_size': 32,
        'epochs': 100,
        'patience': 20,
        'metadata_units': 32,
        'metadata_dropout': 0.3,
        'ecg_filters': (64, 128, 256),
        'ecg_kernel_size': (7, 3, 3),
    },
    'model03': {
        'batch_size': 32,
        'epochs': 100,
        'patience': 20,
        'window_size': 800,
        'window_shift': -1,  # Random shift
        'sigma': 0.05,  # Noise std dev
    }
}

# Target superclasses
SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

# Sampling rate for ECG signals
SAMPLING_RATE = 100

# Random seed for reproducibility
RANDOM_SEED = 42
