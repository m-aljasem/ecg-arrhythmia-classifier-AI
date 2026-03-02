"""Example notebook for data exploration and model usage"""

# This notebook demonstrates how to use the ECG classification project
# Uncomment and run cells as needed for exploration

# %%
# Import required libraries
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data import ECGDataLoader, DataPreprocessor
from src.models import ECGClassifierFactory
from src.utils import plot_samples, plot_training_history
import config

# %%
# Example: Load data and create features
# path_to_data = '/path/to/ptbxl/dataset'
# 
# loader = ECGDataLoader(path_to_data)
# ecg_df, scp_df = loader.load_metadata()
# loader.add_diagnostic_classes()
# ecg_data = loader.load_raw_data(sampling_rate=100)
# 
# # Explore data
# print(f"ECG Data shape: {ecg_data.shape}")
# print(f"ECG Dataframe shape: {ecg_df.shape}")
# print(f"SCP Statements: {scp_df.shape}")

# %%
# Example: Preprocess data
# preprocessor = DataPreprocessor(ecg_df, ecg_data)
# X = preprocessor.create_metadata_features()
# Z = preprocessor.create_target_labels()
# 
# print(f"Features shape: {X.shape}")
# print(f"Targets shape: {Z.shape}")
# print(f"\nTarget distribution:")
# print(Z.sum(axis=0))

# %%
# Example: Visualize ECG sample
# plot_samples(ecg_data, sample_idx=0)

# %%
# Example: Load trained model and make predictions
# import tensorflow.keras as keras
# model = keras.models.load_model(config.MODEL02_CHECKPOINT)
# predictions = model.predict([X_test, Y_test])
# print(f"Prediction shape: {predictions.shape}")
# print(f"Sample predictions: {predictions[0]}")

# %%
# Example: Create new model programmatically
# from config import PROCESSED_DATA_FILE
# 
# # Load data
# with np.load(PROCESSED_DATA_FILE) as data:
#     X_train = data['X_train'].astype(float)
#     Y_train = data['Y_train'].astype(float)
#     Z_train = data['Z_train'].astype(float)
# 
# # Create model
# model = ECGClassifierFactory.create_combined(X_train.shape, Y_train.shape, Z_train.shape)
# ECGClassifierFactory.compile_model(model)
# model.summary()
