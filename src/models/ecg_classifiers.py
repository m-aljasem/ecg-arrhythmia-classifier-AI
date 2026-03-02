"""ECG Classification Models"""

import tensorflow.keras as keras


def create_X_model(X_input, units=32, dropouts=0.3):
    """
    Create metadata processing model.
    
    Args:
        X_input: Input layer for metadata
        units: Number of units in dense layers
        dropouts: Dropout rate
    
    Returns:
        keras.layers: Output layer of metadata model
    """
    X = keras.layers.Dense(units, activation='relu', name='X_dense_1')(X_input)
    X = keras.layers.Dropout(dropouts, name='X_drop_1')(X)
    X = keras.layers.Dense(units, activation='relu', name='X_dense_2')(X)
    X = keras.layers.Dropout(dropouts, name='X_drop_2')(X)
    return X


def create_Y_model(Y_input, filters=(32, 64, 128), kernel_size=(5, 3, 3), 
                   strides=(1, 1, 1)):
    """
    Create 1D CNN model for ECG signals.
    
    Args:
        Y_input: Input layer for ECG signals
        filters: Number of filters in Conv1D layers
        kernel_size: Kernel sizes for Conv1D layers
        strides: Strides for Conv1D layers
    
    Returns:
        keras.layers: Output layer of ECG model
    """
    f1, f2, f3 = filters
    k1, k2, k3 = kernel_size
    s1, s2, s3 = strides
    
    X = keras.layers.Conv1D(f1, k1, strides=s1, padding='same', 
                            name='Y_conv_1')(Y_input)
    X = keras.layers.BatchNormalization(name='Y_norm_1')(X)
    X = keras.layers.ReLU(name='Y_relu_1')(X)
    X = keras.layers.MaxPool1D(2, name='Y_pool_1')(X)
    
    X = keras.layers.Conv1D(f2, k2, strides=s2, padding='same', 
                            name='Y_conv_2')(X)
    X = keras.layers.BatchNormalization(name='Y_norm_2')(X)
    X = keras.layers.ReLU(name='Y_relu_2')(X)
    X = keras.layers.MaxPool1D(2, name='Y_pool_2')(X)
    
    X = keras.layers.Conv1D(f3, k3, strides=s3, padding='same', 
                            name='Y_conv_3')(X)
    X = keras.layers.BatchNormalization(name='Y_norm_3')(X)
    X = keras.layers.ReLU(name='Y_relu_3')(X)
    
    X = keras.layers.GlobalAveragePooling1D(name='Y_aver')(X)
    X = keras.layers.Dropout(0.5, name='Y_drop')(X)
    
    return X


def create_model01(X_shape, Z_shape):
    """
    Create metadata-only classifier.
    
    Args:
        X_shape: Shape of metadata input
        Z_shape: Shape of target output
    
    Returns:
        keras.Model: Compiled model
    """
    X_inputs = keras.Input(X_shape[1:], name='X_inputs')
    
    X = create_X_model(X_inputs)
    X = keras.layers.Dense(64, activation='relu', name='Z_dense_1')(X)
    X = keras.layers.Dense(64, activation='relu', name='Z_dense_2')(X)
    X = keras.layers.Dropout(0.5, name='Z_drop_1')(X)
    outputs = keras.layers.Dense(Z_shape[-1], activation='sigmoid', 
                                 name='Z_outputs')(X)
    
    model = keras.Model(inputs=X_inputs, outputs=outputs, name='model01')
    return model


def create_model02(X_shape, Y_shape, Z_shape):
    """
    Create combined metadata and ECG signal classifier.
    
    Args:
        X_shape: Shape of metadata input
        Y_shape: Shape of ECG signal input
        Z_shape: Shape of target output
    
    Returns:
        keras.Model: Compiled model
    """
    X_inputs = keras.Input(X_shape[1:], name='X_inputs')
    Y_inputs = keras.Input(Y_shape[1:], name='Y_inputs')
    
    X = keras.layers.Concatenate(name='Z_concat')([
        create_X_model(X_inputs),
        create_Y_model(Y_inputs, filters=(64, 128, 256), 
                      kernel_size=(7, 3, 3))
    ])
    X = keras.layers.Dense(64, activation='relu', name='Z_dense_1')(X)
    X = keras.layers.Dense(64, activation='relu', name='Z_dense_2')(X)
    X = keras.layers.Dropout(0.5, name='Z_drop_1')(X)
    outputs = keras.layers.Dense(Z_shape[-1], activation='sigmoid', 
                                 name='Z_outputs')(X)
    
    model = keras.Model(inputs=[X_inputs, Y_inputs], outputs=outputs, 
                       name='model02')
    return model


class ECGClassifierFactory:
    """Factory for creating ECG classification models"""
    
    @staticmethod
    def create_metadata_only(X_shape, Z_shape):
        """Create metadata-only model"""
        return create_model01(X_shape, Z_shape)
    
    @staticmethod
    def create_combined(X_shape, Y_shape, Z_shape):
        """Create combined metadata and ECG signal model"""
        return create_model02(X_shape, Y_shape, Z_shape)
    
    @staticmethod
    def compile_model(model, optimizer='adam', loss='binary_crossentropy'):
        """Compile model with standard settings"""
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['binary_accuracy', 'Precision', 'Recall']
        )
        return model
