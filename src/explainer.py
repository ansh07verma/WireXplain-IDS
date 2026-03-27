"""
Explainability Module
Generate explanations for IDS predictions using SHAP and LIME
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow import keras

# Note: SHAP and LIME require additional installation
# Placeholder implementation for now


class ExplainabilityEngine:
    """Generate explanations for model predictions"""
    
    def __init__(self, model_path, feature_names=None):
        """
        Initialize explainability engine
        
        Args:
            model_path: Path to trained model
            feature_names: List of feature names
        """
        self.model = keras.models.load_model(model_path)
        self.feature_names = feature_names or [f"feature_{i}" for i in range(11)]
        
    def explain_with_shap(self, X_sample, background_data=None):
        """
        Generate SHAP explanations
        
        Args:
            X_sample: Sample to explain
            background_data: Background dataset for SHAP
            
        Returns:
            SHAP values and explanation
        """
        print("⚠ SHAP explanation requires 'shap' package")
        print("Install with: pip install shap")
        
        # Placeholder for SHAP implementation
        # In production, you would use:
        # import shap
        # explainer = shap.DeepExplainer(self.model, background_data)
        # shap_values = explainer.shap_values(X_sample)
        
        return None
    
    def explain_with_lime(self, X_sample):
        """
        Generate LIME explanations
        
        Args:
            X_sample: Sample to explain
            
        Returns:
            LIME explanation
        """
        print("⚠ LIME explanation requires 'lime' package")
        print("Install with: pip install lime")
        
        # Placeholder for LIME implementation
        # In production, you would use:
        # from lime.lime_tabular import LimeTabularExplainer
        # explainer = LimeTabularExplainer(training_data, feature_names=self.feature_names)
        # explanation = explainer.explain_instance(X_sample, self.model.predict)
        
        return None
    
    def get_feature_importance(self, X_sample):
        """
        Get simple feature importance based on gradients
        
        Args:
            X_sample: Sample to analyze
            
        Returns:
            Feature importance scores
        """
        import tensorflow as tf
        
        # Convert to tensor
        X_tensor = tf.convert_to_tensor(X_sample.reshape(1, -1), dtype=tf.float32)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = self.model(X_tensor)
            predicted_class = tf.argmax(predictions, axis=1)
            class_output = predictions[0, predicted_class[0]]
        
        # Get gradients
        gradients = tape.gradient(class_output, X_tensor)
        importance = np.abs(gradients.numpy()[0])
        
        # Create importance dictionary
        importance_dict = {
            self.feature_names[i]: float(importance[i])
            for i in range(len(importance))
        }
        
        # Sort by importance
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_importance


def explain_predictions(input_path, model_path='models/best_model.h5', 
                       output_path='logs/explanations.txt'):
    """
    Generate explanations for predictions
    
    Args:
        input_path: Path to input data
        model_path: Path to trained model
        output_path: Path to save explanations
    """
    print("\n" + "=" * 60)
    print("WireXplain-IDS: Explanation Pipeline")
    print("=" * 60 + "\n")
    
    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    if 'label' in df.columns:
        X = df.drop(columns=['label']).values
    else:
        X = df.values
    
    # Initialize explainer
    print(f"Loading model from {model_path}...")
    feature_names = df.columns.tolist()
    if 'label' in feature_names:
        feature_names.remove('label')
    
    explainer = ExplainabilityEngine(model_path, feature_names)
    
    # Generate explanations for first few samples
    print("\nGenerating explanations for sample predictions...")
    
    explanations = []
    num_samples = min(5, len(X))
    
    for i in range(num_samples):
        print(f"\nSample {i+1}:")
        importance = explainer.get_feature_importance(X[i])
        
        print("Top 5 important features:")
        for j, (feature, score) in enumerate(list(importance.items())[:5]):
            print(f"  {j+1}. {feature}: {score:.4f}")
        
        explanations.append({
            'sample_id': i,
            'importance': importance
        })
    
    # Save explanations
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("WireXplain-IDS: Prediction Explanations\n")
        f.write("=" * 60 + "\n\n")
        
        for exp in explanations:
            f.write(f"Sample {exp['sample_id']}:\n")
            f.write("Feature Importance:\n")
            for feature, score in exp['importance'].items():
                f.write(f"  {feature}: {score:.4f}\n")
            f.write("\n")
    
    print(f"\n✓ Explanations saved to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    print("Explainability module ready!")
    print("\nTo generate explanations, run:")
    print("  python main.py --mode explain --input data/processed/test_data.csv")
