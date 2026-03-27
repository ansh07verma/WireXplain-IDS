"""
Prediction Module
Make predictions on network traffic data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow import keras


def predict_traffic(input_path, model_path='models/best_model.h5'):
    """
    Predict on network traffic data
    
    Args:
        input_path: Path to input data (CSV or PCAP)
        model_path: Path to trained model
        
    Returns:
        Predictions and probabilities
    """
    print("\n" + "=" * 60)
    print("WireXplain-IDS: Prediction Pipeline")
    print("=" * 60 + "\n")
    
    # Load model
    print(f"Loading model from {model_path}...")
    try:
        model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None
    
    # Load input data
    print(f"\nLoading input data from {input_path}...")
    
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
        
        # Remove label column if present
        if 'label' in df.columns:
            X = df.drop(columns=['label'])
        else:
            X = df
        
        print(f"✓ Loaded {len(X)} samples")
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = model.predict(X, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        # Class names (should match training)
        class_names = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
        
        # Create results
        results = {
            'predictions': predicted_classes,
            'confidence': confidence_scores,
            'class_distribution': {}
        }
        
        # Calculate distribution
        for i, class_name in enumerate(class_names):
            count = np.sum(predicted_classes == i)
            percentage = (count / len(predicted_classes)) * 100
            results['class_distribution'][class_name] = {
                'count': int(count),
                'percentage': f"{percentage:.2f}%"
            }
        
        # Print results
        print("\n" + "=" * 60)
        print("Prediction Results:")
        print("=" * 60)
        print(f"Total samples: {len(predicted_classes)}")
        print(f"Average confidence: {np.mean(confidence_scores):.4f}")
        print("\nClass Distribution:")
        for class_name, stats in results['class_distribution'].items():
            print(f"  {class_name}: {stats['count']} ({stats['percentage']})")
        print("=" * 60)
        
        return results
        
    elif input_path.endswith('.pcap'):
        print("⚠ PCAP file prediction not yet implemented")
        print("Please convert PCAP to CSV format first")
        return None
    
    else:
        print(f"✗ Unsupported file format: {input_path}")
        return None


def predict_single(features, model_path='models/best_model.h5'):
    """
    Predict on a single sample
    
    Args:
        features: Feature vector (numpy array or list)
        model_path: Path to trained model
        
    Returns:
        Predicted class and confidence
    """
    model = keras.models.load_model(model_path)
    
    # Ensure features is 2D
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    prediction = model.predict(features, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction, axis=1)[0]
    
    class_names = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
    
    return {
        'class': class_names[predicted_class],
        'class_id': int(predicted_class),
        'confidence': float(confidence),
        'probabilities': {
            class_names[i]: float(prediction[0][i])
            for i in range(len(class_names))
        }
    }


if __name__ == "__main__":
    print("Prediction module ready!")
    print("\nTo make predictions, run:")
    print("  python main.py --mode predict --input data/processed/test_data.csv")
