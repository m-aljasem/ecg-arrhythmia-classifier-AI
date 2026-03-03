"""
Kaggle Cloud Training Script for ECG Classification

This script is designed to run directly in Kaggle without needing to download
or manage data locally. It handles the complete training pipeline.

KAGGLE SETUP INSTRUCTIONS:
1. Create a new Kaggle notebook
2. Add the PTB-XL dataset as input data source:
   https://www.kaggle.com/datasets/khyeh0719/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1-0-1
3. Upload this script or copy-paste the code into the notebook
4. If using the project package, upload the entire 'src' folder to the notebook
5. Run the script

Note: Kaggle paths:
- Input data: /kaggle/input/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1-0-1/
- Working directory: /kaggle/working/
- Model outputs will be saved to: /kaggle/working/models/
- Processed data will be saved to: /kaggle/working/data/
"""

import os
import sys
import ast
import math
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler
import sklearn.metrics

# ============================================================================
# KAGGLE PATHS CONFIGURATION
# ============================================================================

KAGGLE_INPUT = '/kaggle/input/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1-0-1/'
KAGGLE_WORKING = '/kaggle/working/'

# Ensure output directories exist
os.makedirs(os.path.join(KAGGLE_WORKING, 'models'), exist_ok=True)
os.makedirs(os.path.join(KAGGLE_WORKING, 'data'), exist_ok=True)

# Model paths
MODEL01_CHECKPOINT = os.path.join(KAGGLE_WORKING, 'models', 'model01.keras')
MODEL02_CHECKPOINT = os.path.join(KAGGLE_WORKING, 'models', 'model02.keras')
MODEL03_CHECKPOINT = os.path.join(KAGGLE_WORKING, 'models', 'model03.keras')
NUMPY_DATA_FILE = os.path.join(KAGGLE_WORKING, 'data', 'data.npz')

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
SAMPLING_RATE = 100
RANDOM_SEED = 42

# Model hyperparameters
MODEL_CONFIGS = {
    'model01': {
        'batch_size': 32,
        'epochs': 40,
        'patience': 10,
    },
    'model02': {
        'batch_size': 32,
        'epochs': 100,
        'patience': 20,
    },
    'model03': {
        'batch_size': 32,
        'epochs': 100,
        'patience': 20,
        'window_size': 800,
        'window_shift': -1,
        'sigma': 0.05,
    }
}

sns.set_style('darkgrid')

print("=" * 80)
print("ECG CLASSIFICATION - KAGGLE CLOUD TRAINING")
print("=" * 80)
print(f"Input Data Path: {KAGGLE_INPUT}")
print(f"Working Directory: {KAGGLE_WORKING}")
print(f"Random Seed: {RANDOM_SEED}")

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_data():
    """Load ECG metadata and raw signals from Kaggle input"""
    print("\n[1/5] Loading ECG Metadata...")
    
    ecg_df = pd.read_csv(
        os.path.join(KAGGLE_INPUT, 'ptbxl_database.csv'), 
        index_col='ecg_id'
    )
    ecg_df.scp_codes = ecg_df.scp_codes.apply(lambda x: ast.literal_eval(x))
    ecg_df.patient_id = ecg_df.patient_id.astype(int)
    ecg_df.nurse = ecg_df.nurse.astype('Int64')
    ecg_df.site = ecg_df.site.astype('Int64')
    ecg_df.validated_by = ecg_df.validated_by.astype('Int64')
    
    print(f"  ✓ Loaded {len(ecg_df)} ECG records")
    
    # Load SCP statements
    scp_df = pd.read_csv(
        os.path.join(KAGGLE_INPUT, 'scp_statements.csv'), 
        index_col=0
    )
    scp_df = scp_df[scp_df.diagnostic == 1]
    print(f"  ✓ Loaded {len(scp_df)} SCP diagnostic statements")
    
    return ecg_df, scp_df


def add_diagnostic_classes(ecg_df, scp_df):
    """Add diagnostic superclasses to ECG dataframe"""
    def diagnostic_class(scp):
        res = set()
        for k in scp.keys():
            if k in scp_df.index:
                res.add(scp_df.loc[k].diagnostic_class)
        return list(res)
    
    ecg_df['scp_classes'] = ecg_df.scp_codes.apply(diagnostic_class)
    print("  ✓ Added diagnostic classes to records")


def load_raw_data(ecg_df):
    """Load raw ECG signal data"""
    print("\n[2/5] Loading ECG Signals (this may take a few minutes)...")
    
    data = [wfdb.rdsamp(os.path.join(KAGGLE_INPUT, f)) 
           for f in ecg_df.filename_lr]
    ecg_data = np.array([signal for signal, meta in data])
    
    print(f"  ✓ Loaded {ecg_data.shape[0]} signals, shape: {ecg_data.shape}")
    return ecg_data


# ============================================================================
# DATA PREPROCESSING FUNCTIONS
# ============================================================================

def create_metadata_features(ecg_df):
    """Create features from patient metadata"""
    print("\n[3/5] Creating Features...")
    
    X = pd.DataFrame(index=ecg_df.index)
    
    X['age'] = ecg_df.age
    X['age'].fillna(0, inplace=True)
    
    X['sex'] = ecg_df.sex.astype(float)
    X['sex'].fillna(0, inplace=True)
    
    X['height'] = ecg_df.height
    X.loc[X['height'] < 50, 'height'] = np.nan
    X['height'].fillna(0, inplace=True)
    
    X['weight'] = ecg_df.weight
    X['weight'].fillna(0, inplace=True)
    
    X['infarction_stadium1'] = ecg_df.infarction_stadium1.replace({
        'unknown': 0, 'Stadium I': 1, 'Stadium I-II': 2,
        'Stadium II': 3, 'Stadium II-III': 4, 'Stadium III': 5
    }).fillna(0)
    
    X['infarction_stadium2'] = ecg_df.infarction_stadium2.replace({
        'unknown': 0, 'Stadium I': 1, 'Stadium II': 2, 'Stadium III': 3
    }).fillna(0)
    
    X['pacemaker'] = (ecg_df.pacemaker == 'ja, pacemaker').astype(float)
    
    print(f"  ✓ Created metadata features: {X.shape}")
    return X


def create_target_labels(ecg_df):
    """Create target labels from diagnostic classes"""
    Z = pd.DataFrame(0, index=ecg_df.index, columns=SUPERCLASSES, dtype='int')
    
    for i in Z.index:
        for k in ecg_df.loc[i].scp_classes:
            if k in Z.columns:
                Z.loc[i, k] = 1
    
    print(f"  ✓ Created target labels: {Z.shape}")
    print(f"    Class distribution: {dict(Z.sum(axis=0))}")
    return Z


def split_data(ecg_df, X, ecg_data, Z):
    """Split data into train/valid/test sets"""
    train_mask = ecg_df.strat_fold <= 8
    valid_mask = ecg_df.strat_fold == 9
    test_mask = ecg_df.strat_fold == 10
    
    X_train, X_valid, X_test = X[train_mask], X[valid_mask], X[test_mask]
    
    Y_train = ecg_data[np.array(X_train.index) - 1]
    Y_valid = ecg_data[np.array(X_valid.index) - 1]
    Y_test = ecg_data[np.array(X_test.index) - 1]
    
    Z_train, Z_valid, Z_test = Z[train_mask], Z[valid_mask], Z[test_mask]
    
    print("  ✓ Data split:")
    print(f"    Train: {X_train.shape[0]} samples")
    print(f"    Valid: {X_valid.shape[0]} samples")
    print(f"    Test:  {X_test.shape[0]} samples")
    
    return (X_train, X_valid, X_test), (Y_train, Y_valid, Y_test), \
           (Z_train, Z_valid, Z_test)


def standardize_data(X_train, X_valid, X_test, Y_train, Y_valid, Y_test):
    """Standardize input data"""
    # Standardize metadata
    x_scaler = StandardScaler()
    x_scaler.fit(X_train)
    
    X_train_std = pd.DataFrame(x_scaler.transform(X_train), columns=X_train.columns)
    X_valid_std = pd.DataFrame(x_scaler.transform(X_valid), columns=X_valid.columns)
    X_test_std = pd.DataFrame(x_scaler.transform(X_test), columns=X_test.columns)
    
    # Standardize ECG signals
    y_scaler = StandardScaler()
    y_scaler.fit(Y_train.reshape(-1, Y_train.shape[-1]))
    
    Y_train_std = y_scaler.transform(Y_train.reshape(-1, Y_train.shape[-1])).reshape(Y_train.shape)
    Y_valid_std = y_scaler.transform(Y_valid.reshape(-1, Y_valid.shape[-1])).reshape(Y_valid.shape)
    Y_test_std = y_scaler.transform(Y_test.reshape(-1, Y_test.shape[-1])).reshape(Y_test.shape)
    
    print("  ✓ Data standardized")
    return (X_train_std, X_valid_std, X_test_std), (Y_train_std, Y_valid_std, Y_test_std)


def save_data(X_train, X_valid, X_test, Y_train, Y_valid, Y_test, 
             Z_train, Z_valid, Z_test):
    """Save preprocessed data to NPZ file"""
    save_args = {
        'X_train': X_train.to_numpy().astype('float32') if isinstance(X_train, pd.DataFrame) else X_train.astype('float32'),
        'X_valid': X_valid.to_numpy().astype('float32') if isinstance(X_valid, pd.DataFrame) else X_valid.astype('float32'),
        'X_test': X_test.to_numpy().astype('float32') if isinstance(X_test, pd.DataFrame) else X_test.astype('float32'),
        'Y_train': Y_train.astype('float32'),
        'Y_valid': Y_valid.astype('float32'),
        'Y_test': Y_test.astype('float32'),
        'Z_train': Z_train.to_numpy().astype('float32') if isinstance(Z_train, pd.DataFrame) else Z_train.astype('float32'),
        'Z_valid': Z_valid.to_numpy().astype('float32') if isinstance(Z_valid, pd.DataFrame) else Z_valid.astype('float32'),
        'Z_test': Z_test.to_numpy().astype('float32') if isinstance(Z_test, pd.DataFrame) else Z_test.astype('float32'),
    }
    np.savez(NUMPY_DATA_FILE, **save_args)
    print(f"  ✓ Data saved to {NUMPY_DATA_FILE}")


# ============================================================================
# MODEL DEFINITION FUNCTIONS
# ============================================================================

def create_X_model(X_input, units=32, dropouts=0.3):
    """Create metadata processing model"""
    X = keras.layers.Dense(units, activation='relu', name='X_dense_1')(X_input)
    X = keras.layers.Dropout(dropouts, name='X_drop_1')(X)
    X = keras.layers.Dense(units, activation='relu', name='X_dense_2')(X)
    X = keras.layers.Dropout(dropouts, name='X_drop_2')(X)
    return X


def create_Y_model(Y_input, filters=(32, 64, 128), kernel_size=(5, 3, 3), strides=(1, 1, 1)):
    """Create 1D CNN model for ECG signals"""
    f1, f2, f3 = filters
    k1, k2, k3 = kernel_size
    s1, s2, s3 = strides
    
    X = keras.layers.Conv1D(f1, k1, strides=s1, padding='same', name='Y_conv_1')(Y_input)
    X = keras.layers.BatchNormalization(name='Y_norm_1')(X)
    X = keras.layers.ReLU(name='Y_relu_1')(X)
    X = keras.layers.MaxPool1D(2, name='Y_pool_1')(X)
    
    X = keras.layers.Conv1D(f2, k2, strides=s2, padding='same', name='Y_conv_2')(X)
    X = keras.layers.BatchNormalization(name='Y_norm_2')(X)
    X = keras.layers.ReLU(name='Y_relu_2')(X)
    X = keras.layers.MaxPool1D(2, name='Y_pool_2')(X)
    
    X = keras.layers.Conv1D(f3, k3, strides=s3, padding='same', name='Y_conv_3')(X)
    X = keras.layers.BatchNormalization(name='Y_norm_3')(X)
    X = keras.layers.ReLU(name='Y_relu_3')(X)
    
    X = keras.layers.GlobalAveragePooling1D(name='Y_aver')(X)
    X = keras.layers.Dropout(0.5, name='Y_drop')(X)
    
    return X


def create_model01(X_shape, Z_shape):
    """Create metadata-only classifier"""
    X_inputs = keras.Input(X_shape[1:], name='X_inputs')
    
    X = create_X_model(X_inputs)
    X = keras.layers.Dense(64, activation='relu', name='Z_dense_1')(X)
    X = keras.layers.Dense(64, activation='relu', name='Z_dense_2')(X)
    X = keras.layers.Dropout(0.5, name='Z_drop_1')(X)
    outputs = keras.layers.Dense(Z_shape[-1], activation='sigmoid', name='Z_outputs')(X)
    
    model = keras.Model(inputs=X_inputs, outputs=outputs, name='model01')
    return model


def create_model02(X_shape, Y_shape, Z_shape):
    """Create combined metadata and ECG signal classifier"""
    X_inputs = keras.Input(X_shape[1:], name='X_inputs')
    Y_inputs = keras.Input(Y_shape[1:], name='Y_inputs')
    
    X = keras.layers.Concatenate(name='Z_concat')([
        create_X_model(X_inputs),
        create_Y_model(Y_inputs, filters=(64, 128, 256), kernel_size=(7, 3, 3))
    ])
    X = keras.layers.Dense(64, activation='relu', name='Z_dense_1')(X)
    X = keras.layers.Dense(64, activation='relu', name='Z_dense_2')(X)
    X = keras.layers.Dropout(0.5, name='Z_drop_1')(X)
    outputs = keras.layers.Dense(Z_shape[-1], activation='sigmoid', name='Z_outputs')(X)
    
    model = keras.Model(inputs=[X_inputs, Y_inputs], outputs=outputs, name='model02')
    return model


# ============================================================================
# DATA AUGMENTATION FUNCTIONS
# ============================================================================

def sliding_window(x, size, shift):
    """Apply sliding window to ECG signal"""
    if 0 < size < x.shape[0]:
        shift = np.random.randint(0, x.shape[0] - size) if shift < 0 else shift
        return x[shift:size + shift, :]
    else:
        return x


class AugmentedDataGenerator(keras.utils.Sequence):
    """Generate augmented ECG data with sliding window and noise"""
    
    def __init__(self, x, y, z, batch_size=32, window_size=0, 
                 window_shift=0, sigma=0.0, **kwargs):
        super(AugmentedDataGenerator, self).__init__(**kwargs)
        self.x = x
        self.y = y
        self.z = z
        self.batch_size = batch_size
        self.window_size = window_size
        self.window_shift = window_shift
        self.sigma = sigma
    
    @property
    def x_shape(self):
        return (self.batch_size,) + self.x.shape[1:]
    
    @property
    def y_shape(self):
        y_len = self.window_size if self.window_size > 0 else self.y.shape[1]
        return (self.batch_size, y_len) + self.y.shape[2:]
    
    @property
    def z_shape(self):
        return (self.batch_size,) + self.z.shape[1:]
    
    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)
    
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = np.array([
            sliding_window(r, self.window_size, self.window_shift) 
            for r in self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        ])
        batch_z = self.z[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        if self.sigma > 0:
            batch_y += np.random.normal(loc=0.0, scale=self.sigma, size=batch_y.shape)
        
        return (batch_x, batch_y), batch_z


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_model01(X_train, X_valid, X_test, Z_train, Z_valid, Z_test):
    """Train metadata-only model"""
    print("\n" + "=" * 80)
    print("TRAINING MODEL01 (Metadata Only)")
    print("=" * 80)
    
    config = MODEL_CONFIGS['model01']
    
    # Create and compile model
    model = create_model01(X_train.shape, Z_train.shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                 metrics=['binary_accuracy', 'Precision', 'Recall'])
    print(model.summary())
    
    # Setup callbacks
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
                       validation_data=(X_valid, Z_valid), verbose=1)
    
    # Evaluate
    model = keras.models.load_model(MODEL01_CHECKPOINT)
    results = model.evaluate(X_test, Z_test)
    print(f"\n✓ Model01 Test Results: Loss={results[0]:.4f}, Accuracy={results[1]:.4f}")
    
    return model, history


def train_model02(X_train, X_valid, X_test, Y_train, Y_valid, Y_test, 
                 Z_train, Z_valid, Z_test):
    """Train combined metadata and ECG model"""
    print("\n" + "=" * 80)
    print("TRAINING MODEL02 (Combined Metadata + ECG)")
    print("=" * 80)
    
    config = MODEL_CONFIGS['model02']
    
    # Create and compile model
    model = create_model02(X_train.shape, Y_train.shape, Z_train.shape)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                 metrics=['binary_accuracy', 'Precision', 'Recall'])
    print(model.summary())
    
    # Setup callbacks
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
                       validation_data=([X_valid, Y_valid], Z_valid), verbose=1)
    
    # Evaluate
    model = keras.models.load_model(MODEL02_CHECKPOINT)
    results = model.evaluate([X_test, Y_test], Z_test)
    print(f"\n✓ Model02 Test Results: Loss={results[0]:.4f}, Accuracy={results[1]:.4f}")
    
    return model, history


def train_model03_with_augmentation(X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
                                   Z_train, Z_valid, Z_test):
    """Train model with data augmentation"""
    print("\n" + "=" * 80)
    print("TRAINING MODEL03 (With Data Augmentation)")
    print("=" * 80)
    
    config = MODEL_CONFIGS['model03']
    
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
    model = create_model02(train_gen.x_shape, train_gen.y_shape, train_gen.z_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                 metrics=['binary_accuracy', 'Precision', 'Recall'])
    print(model.summary())
    
    # Setup callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_binary_accuracy',
                                     patience=config['patience']),
        keras.callbacks.ModelCheckpoint(filepath=MODEL03_CHECKPOINT,
                                       monitor='val_binary_accuracy',
                                       save_best_only=True)
    ]
    
    # Train model
    history = model.fit(train_gen, epochs=config['epochs'],
                       callbacks=callbacks, validation_data=valid_gen, verbose=1)
    
    # Evaluate
    model = keras.models.load_model(MODEL03_CHECKPOINT)
    results = model.evaluate(test_gen)
    print(f"\n✓ Model03 Test Results: Loss={results[0]:.4f}, Accuracy={results[1]:.4f}")
    
    return model, history


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""
    
    # Step 1: Load data
    print("\n" + "=" * 80)
    print("DATA LOADING & PREPROCESSING")
    print("=" * 80)
    
    ecg_df, scp_df = load_data()
    add_diagnostic_classes(ecg_df, scp_df)
    ecg_data = load_raw_data(ecg_df)
    
    # Step 2: Preprocess data
    X = create_metadata_features(ecg_df)
    Z = create_target_labels(ecg_df)
    
    (X_train, X_valid, X_test), (Y_train, Y_valid, Y_test), \
    (Z_train, Z_valid, Z_test) = split_data(ecg_df, X, ecg_data, Z)
    
    (X_train, X_valid, X_test), (Y_train, Y_valid, Y_test) = \
        standardize_data(X_train, X_valid, X_test, Y_train, Y_valid, Y_test)
    
    # Save processed data
    save_data(X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
             Z_train, Z_valid, Z_test)
    
    # Step 3: Train models
    model01, history01 = train_model01(X_train, X_valid, X_test, 
                                       Z_train, Z_valid, Z_test)
    
    model02, history02 = train_model02(X_train, X_valid, X_test, Y_train, Y_valid, Y_test,
                                       Z_train, Z_valid, Z_test)
    
    model03, history03 = train_model03_with_augmentation(X_train, X_valid, X_test,
                                                         Y_train, Y_valid, Y_test,
                                                         Z_train, Z_valid, Z_test)
    
    # Step 4: Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n✓ All models trained successfully!")
    print(f"\nModel Checkpoints:")
    print(f"  - Model01: {MODEL01_CHECKPOINT}")
    print(f"  - Model02: {MODEL02_CHECKPOINT}")
    print(f"  - Model03: {MODEL03_CHECKPOINT}")
    print(f"\nProcessed Data:")
    print(f"  - NPZ File: {NUMPY_DATA_FILE}")
    print(f"\nAll outputs saved to: {KAGGLE_WORKING}")
    print("\nYou can now download the models and data from Kaggle output!")
    print("=" * 80)


if __name__ == '__main__':
    main()
