"""Evaluation script for ECG classification models"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import sklearn.metrics
from src.utils import print_confusion_matrix
from config import MODEL01_CHECKPOINT, MODEL02_CHECKPOINT, MODEL03_CHECKPOINT, \
    PROCESSED_DATA_FILE, SUPERCLASSES


def load_test_data():
    """Load test data"""
    with np.load(PROCESSED_DATA_FILE) as data:
        return {
            'X_test': data['X_test'].astype(float),
            'Y_test': data['Y_test'].astype(float),
            'Z_test': data['Z_test'].astype(int),
        }


def evaluate_model(model_name: str, model_path: str, test_data: dict):
    """Evaluate a single model"""
    print(f"\n=== Evaluating {model_name} ===")
    
    model = keras.models.load_model(model_path)
    
    if model_name == 'model01':
        predictions = model.predict(test_data['X_test']).round().astype(int)
    elif model_name in ['model02', 'model03']:
        if model_name == 'model03':
            # Truncate Y_test to 800 samples for model03
            predictions = model.predict([
                test_data['X_test'],
                test_data['Y_test'][:, :800, :]
            ]).round().astype(int)
        else:
            predictions = model.predict([
                test_data['X_test'],
                test_data['Y_test']
            ]).round().astype(int)
    
    Z_test = test_data['Z_test']
    
    # Print confusion matrices
    fig, ax = plt.subplots(1, 5, figsize=(16, 3))
    
    for axes, cfs_matrix, label in zip(
        ax.flatten(),
        sklearn.metrics.multilabel_confusion_matrix(Z_test, predictions),
        SUPERCLASSES
    ):
        print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])
    
    fig.tight_layout()
    plt.show()
    
    # Print classification report
    print(sklearn.metrics.classification_report(Z_test, predictions,
                                               target_names=SUPERCLASSES,
                                               zero_division=0))
    
    # Print overall metrics
    accuracy = sklearn.metrics.accuracy_score(Z_test, predictions)
    print(f"Overall Accuracy: {accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate ECG classification models')
    parser.add_argument('--model', type=str, choices=['model01', 'model02', 'model03', 'all'],
                       default='all', help='Which model to evaluate')
    
    args = parser.parse_args()
    
    # Load test data
    print("Loading test data...")
    test_data = load_test_data()
    
    # Evaluate models
    if args.model in ['model01', 'all']:
        evaluate_model('model01', MODEL01_CHECKPOINT, test_data)
    
    if args.model in ['model02', 'all']:
        evaluate_model('model02', MODEL02_CHECKPOINT, test_data)
    
    if args.model in ['model03', 'all']:
        evaluate_model('model03', MODEL03_CHECKPOINT, test_data)


if __name__ == '__main__':
    main()
