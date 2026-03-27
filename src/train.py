"""
Training Module
Train the IDS model on network traffic data
"""

import numpy as np
from pathlib import Path
from tensorflow import keras

from data_loader import DataLoader
from model import IDSModel


def train_model(data_path, model_save_path='models/best_model.h5', 
                epochs=50, batch_size=32, architecture='deep'):
    """
    Train the IDS model
    
    Args:
        data_path: Path to training data CSV
        model_save_path: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        architecture: Model architecture type
        
    Returns:
        Training history
    """
    print("\n" + "=" * 60)
    print("WireXplain-IDS: Training Pipeline")
    print("=" * 60 + "\n")
    
    # Load and preprocess data
    loader = DataLoader()
    df = loader.load_csv(data_path)
    X_train, X_test, y_train, y_test = loader.preprocess(df)
    
    # Get dimensions
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    print(f"\nModel Configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Architecture: {architecture}")
    
    # Build model
    print("\nBuilding model...")
    ids_model = IDSModel(input_dim=input_dim, num_classes=num_classes)
    model = ids_model.build_model(architecture=architecture)
    ids_model.summary()
    
    # Create model directory
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Get callbacks
    callbacks = ids_model.get_callbacks(model_path=model_save_path)
    
    # Train model
    print(f"\nTraining for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print("\n" + "=" * 60)
    print("Training Results:")
    print("=" * 60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: {model_save_path}")
    print("=" * 60)
    
    return history


if __name__ == "__main__":
    # Example training
    print("Training module ready!")
    print("\nTo train a model, run:")
    print("  python main.py --mode train --data data/processed/sample_data.csv")
