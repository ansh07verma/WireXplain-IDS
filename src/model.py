"""
Model Definition Module
Deep learning models for intrusion detection
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class IDSModel:
    """Deep Neural Network for Intrusion Detection"""
    
    def __init__(self, input_dim, num_classes):
        """
        Initialize IDS model
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self, architecture='deep'):
        """
        Build the neural network model
        
        Args:
            architecture: Model architecture type ('deep', 'wide', 'simple')
            
        Returns:
            Compiled Keras model
        """
        if architecture == 'deep':
            self.model = self._build_deep_model()
        elif architecture == 'wide':
            self.model = self._build_wide_model()
        else:
            self.model = self._build_simple_model()
        
        return self.model
    
    def _build_deep_model(self):
        """Build a deep neural network"""
        model = models.Sequential([
            layers.Input(shape=(self.input_dim,)),
            
            # First block
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second block
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third block
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Fourth block
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_wide_model(self):
        """Build a wide neural network"""
        model = models.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_simple_model(self):
        """Build a simple neural network"""
        model = models.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_callbacks(self, model_path='models/best_model.h5'):
        """
        Get training callbacks
        
        Args:
            model_path: Path to save best model
            
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")


if __name__ == "__main__":
    # Example usage
    print("Creating sample IDS model...")
    ids_model = IDSModel(input_dim=11, num_classes=5)
    model = ids_model.build_model(architecture='deep')
    ids_model.summary()
