"""
Training script for ECG classification models

Classification of Life-Threatening Arrhythmia ECG Signals Using Deep Learning

Author: Mohamad AlJasem
Website: https://aljasem.eu.org
GitHub: https://github.com/m-aljasem/ecg-arrhythmia-classifier-AI
Contact: mohamad@aljasem.eu.org
"""

import os
import argparse
import pickle
import numpy as np
import tensorflow.keras as keras
from src.data import ECGDataLoader, DataPreprocessor, AugmentedDataGenerator
from src.models import ECGClassifierFactory
from config import MODEL_CONFIGS, PROCESSED_DATA_FILE, MODEL01_CHECKPOINT, \
    MODEL02_CHECKPOINT, MODEL03_CHECKPOINT, SAMPLING_RATE, SUPERCLASSES


def prepare_data(data_path: str):
    """Load and prepare data for training"""
    # Load data
    loader = ECGDataLoader(data_path)
    ecg_df, scp_df = loader.load_metadata()
    loader.add_diagnostic_classes()
    ecg_data = loader.load_raw_data(sampling_rate=SAMPLING_RATE)
    
    # Preprocess data
    preprocessor = DataPreprocessor(ecg_df, ecg_data)
    X = preprocessor.create_metadata_features()
    Z = preprocessor.create_target_labels()
    
    # Split data
    (X_train, X_valid, X_test), (Y_train, Y_valid, Y_test), \
    (Z_train, Z_valid, Z_test) = preprocessor.split_data(X, ecg_data, Z)
    
    # Standardize data
    (X_train, X_valid, X_test), (Y_train, Y_valid, Y_test) = \
        preprocessor.standardize_data(X_train, X_valid, X_test, 
                                     Y_train, Y_valid, Y_test)
    
    # Save processed data
    os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)
    preprocessor.save_data(PROCESSED_DATA_FILE, X_train, X_valid, X_test,
                          Y_train, Y_valid, Y_test, Z_train, Z_valid, Z_test)
    
    return (X_train, X_valid, X_test), (Y_train, Y_valid, Y_test), \
           (Z_train, Z_valid, Z_test)


def load_preprocessed_data():
    """Load preprocessed data from saved file"""
    with np.load(PROCESSED_DATA_FILE) as data:
        return {k: data[k].astype(float) for k in data.keys()}


def train_model01(data_dict):
    """Train metadata-only model"""
    config = MODEL_CONFIGS['model01']
    X_train, X_valid, X_test = data_dict['X_train'], data_dict['X_valid'], \
                               data_dict['X_test']
    Z_train, Z_valid, Z_test = data_dict['Z_train'], data_dict['Z_valid'], \
                               data_dict['Z_test']
    
    # Create and compile model
    model = ECGClassifierFactory.create_metadata_only(X_train.shape, Z_train.shape)
    ECGClassifierFactory.compile_model(model)
    print(model.summary())
    
    # Setup callbacks
    os.makedirs(os.path.dirname(MODEL01_CHECKPOINT), exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_binary_accuracy',
                                     patience=config['patience']),
        keras.callbacks.ModelCheckpoint(filepath=MODEL01_CHECKPOINT,
                                       monitor='val_binary_accuracy',
                                       save_best_only=True)
    ]
    
    # Train model
    history = model.fit(X_train, Z_train, epochs=config['epochs'],
                       batch_size=config['batch_size'], callbacks=callbacks,
                       validation_data=(X_valid, Z_valid))
    
    # Evaluate
    model = keras.models.load_model(MODEL01_CHECKPOINT)
    results = model.evaluate(X_test, Z_test)
    print(f"Model01 Test Results: {results}")
    
    return model, history


def train_model02(data_dict):
    """Train combined metadata and ECG model"""
    config = MODEL_CONFIGS['model02']
    X_train, X_valid, X_test = data_dict['X_train'], data_dict['X_valid'], \
                               data_dict['X_test']
    Y_train, Y_valid, Y_test = data_dict['Y_train'], data_dict['Y_valid'], \
                               data_dict['Y_test']
    Z_train, Z_valid, Z_test = data_dict['Z_train'], data_dict['Z_valid'], \
                               data_dict['Z_test']
    
    # Create and compile model
    model = ECGClassifierFactory.create_combined(X_train.shape, Y_train.shape,
                                                Z_train.shape)
    ECGClassifierFactory.compile_model(model)
    print(model.summary())
    
    # Setup callbacks
    os.makedirs(os.path.dirname(MODEL02_CHECKPOINT), exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_binary_accuracy',
                                     patience=config['patience']),
        keras.callbacks.ModelCheckpoint(filepath=MODEL02_CHECKPOINT,
                                       monitor='val_binary_accuracy',
                                       save_best_only=True)
    ]
    
    # Train model
    history = model.fit([X_train, Y_train], Z_train,
                       epochs=config['epochs'],
                       batch_size=config['batch_size'],
                       callbacks=callbacks,
                       validation_data=([X_valid, Y_valid], Z_valid))
    
    # Evaluate
    model = keras.models.load_model(MODEL02_CHECKPOINT)
    results = model.evaluate([X_test, Y_test], Z_test)
    print(f"Model02 Test Results: {results}")
    
    return model, history


def train_model03_with_augmentation(data_dict):
    """Train model with data augmentation"""
    config = MODEL_CONFIGS['model03']
    X_train, X_valid, X_test = data_dict['X_train'], data_dict['X_valid'], \
                               data_dict['X_test']
    Y_train, Y_valid, Y_test = data_dict['Y_train'], data_dict['Y_valid'], \
                               data_dict['Y_test']
    Z_train, Z_valid, Z_test = data_dict['Z_train'], data_dict['Z_valid'], \
                               data_dict['Z_test']
    
    # Create generators
    train_gen = AugmentedDataGenerator(X_train, Y_train, Z_train,
                                      window_size=config['window_size'],
                                      window_shift=config['window_shift'],
                                      sigma=config['sigma'])
    valid_gen = AugmentedDataGenerator(X_valid, Y_valid, Z_valid,
                                      window_size=config['window_size'])
    test_gen = AugmentedDataGenerator(X_test, Y_test, Z_test,
                                     window_size=config['window_size'])
    
    # Create and compile model
    model = ECGClassifierFactory.create_combined(train_gen.x_shape,
                                                train_gen.y_shape,
                                                train_gen.z_shape)
    ECGClassifierFactory.compile_model(model)
    print(model.summary())
    
    # Setup callbacks
    os.makedirs(os.path.dirname(MODEL03_CHECKPOINT), exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_binary_accuracy',
                                     patience=config['patience']),
        keras.callbacks.ModelCheckpoint(filepath=MODEL03_CHECKPOINT,
                                       monitor='val_binary_accuracy',
                                       save_best_only=True)
    ]
    
    # Train model
    history = model.fit(train_gen, epochs=config['epochs'],
                       callbacks=callbacks, validation_data=valid_gen)
    
    # Evaluate
    model = keras.models.load_model(MODEL03_CHECKPOINT)
    results = model.evaluate(test_gen)
    print(f"Model03 Test Results: {results}")
    
    return model, history


def save_scalers(data_dict, output_path='models/scalers.pkl'):
    """Save scalers for production use"""
    print("\n=== Saving Scalers ===")
    
    # Recreate scalers from training data
    from sklearn.preprocessing import StandardScaler
    
    X_train = data_dict['X_train']
    Y_train = data_dict['Y_train']
    
    # Recreate the scalers
    x_scaler = StandardScaler()
    x_scaler.fit(X_train)
    
    y_scaler = StandardScaler()
    y_scaler.fit(Y_train.reshape(-1, Y_train.shape[-1]))
    
    # Save the scalers
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump({
            'x_scaler': x_scaler,
            'y_scaler': y_scaler,
            'superclasses': SUPERCLASSES,
            'feature_names': ['age', 'sex', 'height', 'weight', 
                            'infarction_stadium1', 'infarction_stadium2', 'pacemaker']
        }, f)
    
    print(f"✓ Scalers saved to: {output_path}")
    print("\nFor your application:")
    print("  1. Load model: keras.models.load_model('model03.keras')")
    print("  2. Load scalers: pickle.load(open('scalers.pkl', 'rb'))")
    print("  3. Use x_scaler to normalize metadata")
    print("  4. Use y_scaler to normalize ECG signals")


def main():
    parser = argparse.ArgumentParser(description='Train ECG classification models')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to PTB-XL dataset')
    parser.add_argument('--model', type=str, choices=['model01', 'model02', 'model03', 'all'],
                       default='all', help='Which model to train')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip data preprocessing')
    parser.add_argument('--save-scalers', action='store_true',
                       help='Save scalers for production use')
    
    args = parser.parse_args()
    
    # Prepare data
    if not args.skip_preprocessing:
        print("Preparing data...")
        prepare_data(args.data_path)
    else:
        print("Loading preprocessed data...")
    
    data_dict = load_preprocessed_data()
    
    # Train models
    if args.model in ['model01', 'all']:
        print("\n=== Training Model01 (Metadata Only) ===")
        train_model01(data_dict)
    
    if args.model in ['model02', 'all']:
        print("\n=== Training Model02 (Combined Metadata + ECG) ===")
        train_model02(data_dict)
    
    if args.model in ['model03', 'all']:
        print("\n=== Training Model03 (With Data Augmentation) ===")
        train_model03_with_augmentation(data_dict)
    
    # Save scalers if requested
    if args.save_scalers or args.model == 'all':
        save_scalers(data_dict)


if __name__ == '__main__':
    main()
