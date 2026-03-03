"""Data Preprocessing and Augmentation Module"""

import math
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Handle data preprocessing and preparation for modeling"""
    
    SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    def __init__(self, ecg_df: pd.DataFrame, ecg_data: np.ndarray):
        """
        Initialize preprocessor.
        
        Args:
            ecg_df: ECG metadata dataframe
            ecg_data: ECG signal data array
        """
        self.ecg_df = ecg_df
        self.ecg_data = ecg_data
        self.x_scaler = None
        self.y_scaler = None
    
    def create_metadata_features(self) -> pd.DataFrame:
        """
        Create features from patient metadata.
        
        Returns:
            pd.DataFrame: Feature matrix X
        """
        X = pd.DataFrame(index=self.ecg_df.index)
        
        # Age feature
        X['age'] = self.ecg_df.age
        X['age'].fillna(0, inplace=True)
        
        # Sex feature
        X['sex'] = self.ecg_df.sex.astype(float)
        X['sex'].fillna(0, inplace=True)
        
        # Height feature
        X['height'] = self.ecg_df.height
        X.loc[X['height'] < 50, 'height'] = np.nan
        X['height'].fillna(0, inplace=True)
        
        # Weight feature
        X['weight'] = self.ecg_df.weight
        X['weight'].fillna(0, inplace=True)
        
        # Infarction stadium 1
        X['infarction_stadium1'] = self.ecg_df.infarction_stadium1.replace({
            'unknown': 0,
            'Stadium I': 1,
            'Stadium I-II': 2,
            'Stadium II': 3,
            'Stadium II-III': 4,
            'Stadium III': 5
        }).fillna(0)
        
        # Infarction stadium 2
        X['infarction_stadium2'] = self.ecg_df.infarction_stadium2.replace({
            'unknown': 0,
            'Stadium I': 1,
            'Stadium II': 2,
            'Stadium III': 3
        }).fillna(0)
        
        # Pacemaker feature
        X['pacemaker'] = (self.ecg_df.pacemaker == 'ja, pacemaker').astype(float)
        
        return X
    
    def create_target_labels(self) -> pd.DataFrame:
        """
        Create target labels from diagnostic classes.
        
        Returns:
            pd.DataFrame: Multi-label target matrix Z
        """
        Z = pd.DataFrame(0, index=self.ecg_df.index, 
                        columns=self.SUPERCLASSES, dtype='int')
        
        for i in Z.index:
            for k in self.ecg_df.loc[i].scp_classes:
                if k in Z.columns:
                    Z.loc[i, k] = 1
        
        return Z
    
    def split_data(self, X: pd.DataFrame, Y: np.ndarray, Z: pd.DataFrame):
        """
        Split data into train, validation, and test sets.
        
        Returns:
            tuple: (X_train, X_valid, X_test), (Y_train, Y_valid, Y_test), 
                   (Z_train, Z_valid, Z_test)
        """
        train_mask = self.ecg_df.strat_fold <= 8
        valid_mask = self.ecg_df.strat_fold == 9
        test_mask = self.ecg_df.strat_fold == 10
        
        X_train = X[train_mask]
        X_valid = X[valid_mask]
        X_test = X[test_mask]
        
        # Adjust indices for numpy array indexing
        Y_train = Y[np.array(X_train.index) - 1]
        Y_valid = Y[np.array(X_valid.index) - 1]
        Y_test = Y[np.array(X_test.index) - 1]
        
        Z_train = Z[train_mask]
        Z_valid = Z[valid_mask]
        Z_test = Z[test_mask]
        
        return (X_train, X_valid, X_test), (Y_train, Y_valid, Y_test), \
               (Z_train, Z_valid, Z_test)
    
    def standardize_data(self, X_train: pd.DataFrame, X_valid: pd.DataFrame, 
                        X_test: pd.DataFrame, Y_train: np.ndarray, 
                        Y_valid: np.ndarray, Y_test: np.ndarray):
        """
        Standardize input data.
        
        Returns:
            tuple: Standardized data arrays
        """
        # Standardize metadata
        self.x_scaler = StandardScaler()
        self.x_scaler.fit(X_train)
        
        X_train_std = pd.DataFrame(self.x_scaler.transform(X_train), 
                                   columns=X_train.columns)
        X_valid_std = pd.DataFrame(self.x_scaler.transform(X_valid), 
                                   columns=X_valid.columns)
        X_test_std = pd.DataFrame(self.x_scaler.transform(X_test), 
                                  columns=X_test.columns)
        
        # Standardize ECG signals
        self.y_scaler = StandardScaler()
        self.y_scaler.fit(Y_train.reshape(-1, Y_train.shape[-1]))
        
        Y_train_std = self.y_scaler.transform(
            Y_train.reshape(-1, Y_train.shape[-1])
        ).reshape(Y_train.shape)
        Y_valid_std = self.y_scaler.transform(
            Y_valid.reshape(-1, Y_valid.shape[-1])
        ).reshape(Y_valid.shape)
        Y_test_std = self.y_scaler.transform(
            Y_test.reshape(-1, Y_test.shape[-1])
        ).reshape(Y_test.shape)
        
        return (X_train_std, X_valid_std, X_test_std), \
               (Y_train_std, Y_valid_std, Y_test_std)
    
    def save_data(self, filepath: str, X_train, X_valid, X_test, 
                 Y_train, Y_valid, Y_test, Z_train, Z_valid, Z_test):
        """Save preprocessed data to NPZ file"""
        save_args = {
            'X_train': X_train.to_numpy().astype('float32') 
                      if isinstance(X_train, pd.DataFrame) else X_train.astype('float32'),
            'X_valid': X_valid.to_numpy().astype('float32') 
                      if isinstance(X_valid, pd.DataFrame) else X_valid.astype('float32'),
            'X_test': X_test.to_numpy().astype('float32') 
                     if isinstance(X_test, pd.DataFrame) else X_test.astype('float32'),
            'Y_train': Y_train.astype('float32'),
            'Y_valid': Y_valid.astype('float32'),
            'Y_test': Y_test.astype('float32'),
            'Z_train': Z_train.to_numpy().astype('float32') 
                      if isinstance(Z_train, pd.DataFrame) else Z_train.astype('float32'),
            'Z_valid': Z_valid.to_numpy().astype('float32') 
                      if isinstance(Z_valid, pd.DataFrame) else Z_valid.astype('float32'),
            'Z_test': Z_test.to_numpy().astype('float32') 
                     if isinstance(Z_test, pd.DataFrame) else Z_test.astype('float32'),
        }
        np.savez(filepath, **save_args)


def sliding_window(x: np.ndarray, size: int, shift: int) -> np.ndarray:
    """
    Apply sliding window to ECG signal.
    
    Args:
        x: Input signal array
        size: Window size
        shift: Window shift (negative for random)
    
    Returns:
        np.ndarray: Windowed signal
    """
    if 0 < size < x.shape[0]:
        shift = np.random.randint(0, x.shape[0] - size) if shift < 0 else shift
        return x[shift:size + shift, :]
    else:
        return x


class AugmentedDataGenerator(keras.utils.Sequence):
    """Generate augmented ECG data with sliding window and noise"""
    
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 batch_size: int = 32, window_size: int = 0, 
                 window_shift: int = 0, sigma: float = 0.0, **kwargs):
        """
        Initialize data generator.
        
        Args:
            x: Metadata features
            y: ECG signals
            z: Target labels
            batch_size: Batch size
            window_size: Sliding window size
            window_shift: Window shift (negative for random)
            sigma: Gaussian noise standard deviation
        """
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
        
        # Convert pandas DataFrames to numpy arrays if needed
        if isinstance(batch_x, pd.DataFrame):
            batch_x = batch_x.to_numpy()
        if isinstance(batch_z, pd.DataFrame):
            batch_z = batch_z.to_numpy()
        
        if self.sigma > 0:
            batch_y += np.random.normal(loc=0.0, scale=self.sigma, 
                                       size=batch_y.shape)
        
        return (batch_x, batch_y), batch_z
