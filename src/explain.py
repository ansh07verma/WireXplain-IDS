"""
Model Explainability Module
Generate SHAP explanations for RandomForest intrusion detection model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import shap
import matplotlib.pyplot as plt
import logging
import sys
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/explain.log')
    ]
)
logger = logging.getLogger(__name__)

# Set matplotlib style for publication-ready plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


class ModelExplainer:
    """Generate SHAP explanations for trained models"""
    
    def __init__(self, model_path):
        """
        Initialize explainer
        
        Args:
            model_path: Path to trained model file
        """
        self.model_path = model_path
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.X_test = None
        self.y_test = None
        
    def load_model(self):
        """Load trained model"""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"✓ Model loaded: {type(self.model).__name__}")
            logger.info(f"  Number of trees: {self.model.n_estimators}")
            logger.info(f"  Number of features: {self.model.n_features_in_}")
            
            return self.model
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_test_data(self, data_path, sample_size=None):
        """
        Load test dataset
        
        Args:
            data_path: Path to test data CSV
            sample_size: Number of samples to use (None = all, for faster computation)
            
        Returns:
            X_test, y_test
        """
        logger.info(f"Loading test data from {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            logger.info(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
            
            # Separate features and target
            feature_cols = [col for col in df.columns if col not in ['label', 'is_anomaly']]
            X = df[feature_cols]
            y = df['label']
            
            # Sample if requested
            if sample_size and sample_size < len(X):
                logger.info(f"Sampling {sample_size:,} samples for faster computation...")
                indices = np.random.choice(len(X), sample_size, replace=False)
                X = X.iloc[indices].reset_index(drop=True)
                y = y.iloc[indices].reset_index(drop=True)
                logger.info(f"✓ Using {len(X):,} samples")
            
            self.X_test = X
            self.y_test = y
            
            logger.info(f"Features: {X.shape[1]}")
            logger.info(f"Samples: {len(X):,}")
            
            return X, y
            
        except FileNotFoundError:
            logger.error(f"Data file not found: {data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_explainer(self):
        """
        Create SHAP TreeExplainer
        
        Returns:
            SHAP explainer object
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Creating SHAP TreeExplainer...")
        
        try:
            # TreeExplainer is optimized for tree-based models
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("✓ TreeExplainer created successfully")
            
            return self.explainer
            
        except Exception as e:
            logger.error(f"Error creating explainer: {e}")
            raise
    
    def compute_shap_values(self, X=None):
        """
        Compute SHAP values for test data
        
        Args:
            X: Feature matrix (uses self.X_test if None)
            
        Returns:
            SHAP values
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        if X is None:
            X = self.X_test
        
        logger.info(f"Computing SHAP values for {len(X):,} samples...")
        logger.info("This may take a few minutes...")
        
        try:
            # Compute SHAP values
            self.shap_values = self.explainer.shap_values(X)
            
            # For binary classification, shap_values is a list [class_0, class_1]
            # We use class_1 (attack) for explanations
            if isinstance(self.shap_values, list):
                logger.info(f"✓ Computed SHAP values for {len(self.shap_values)} classes")
                self.shap_values = self.shap_values[1]  # Use attack class
            else:
                logger.info("✓ Computed SHAP values")
            
            return self.shap_values
            
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            raise
    
    def plot_global_importance(self, output_dir='outputs', max_display=15):
        """
        Generate global feature importance plot (summary plot)
        
        Args:
            output_dir: Directory to save plots
            max_display: Number of top features to display
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        logger.info("Generating global feature importance plot...")
        
        try:
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Summary plot (beeswarm)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values,
                self.X_test,
                max_display=max_display,
                show=False
            )
            plt.title('Global Feature Importance (SHAP Summary Plot)', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('SHAP Value (impact on model output)', fontsize=12)
            plt.tight_layout()
            
            output_path = Path(output_dir) / 'global_feature_importance.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            logger.info(f"✓ Saved global importance plot: {output_path}")
            
            # Bar plot (mean absolute SHAP values)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values,
                self.X_test,
                plot_type='bar',
                max_display=max_display,
                show=False
            )
            plt.title('Mean Feature Importance (SHAP Bar Plot)', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('Mean |SHAP Value|', fontsize=12)
            plt.tight_layout()
            
            output_path = Path(output_dir) / 'global_feature_importance_bar.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            logger.info(f"✓ Saved bar plot: {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating global importance plot: {e}")
            raise
    
    def plot_local_explanation(self, sample_idx=0, output_dir='outputs'):
        """
        Generate local explanation for a specific sample (waterfall plot)
        
        Args:
            sample_idx: Index of sample to explain
            output_dir: Directory to save plots
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        logger.info(f"Generating local explanation for sample {sample_idx}...")
        
        try:
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Get sample data
            sample_features = self.X_test.iloc[sample_idx]
            sample_label = self.y_test.iloc[sample_idx]
            sample_shap = self.shap_values[sample_idx]
            
            # Ensure SHAP values are 1D and match feature length
            if isinstance(sample_shap, np.ndarray):
                if len(sample_shap.shape) > 1:
                    # If 2D, take the first dimension (should match features)
                    if sample_shap.shape[0] == len(sample_features):
                        sample_shap = sample_shap[:, 0] if sample_shap.shape[1] > 0 else sample_shap.flatten()
                    else:
                        sample_shap = sample_shap.flatten()[:len(sample_features)]
            
            # Predict for this sample
            prediction = self.model.predict([sample_features])[0]
            probability = self.model.predict_proba([sample_features])[0]
            
            logger.info(f"Sample {sample_idx}:")
            logger.info(f"  True label: {sample_label} ({'Normal' if sample_label == 0 else 'Attack'})")
            logger.info(f"  Prediction: {prediction} ({'Normal' if prediction == 0 else 'Attack'})")
            logger.info(f"  Probability: {probability[1]:.4f} (attack)")
            
            # Create simple bar plot for local explanation
            plt.figure(figsize=(10, 6))
            
            # Get top 10 features by absolute SHAP value
            feature_importance = pd.DataFrame({
                'feature': list(sample_features.index),
                'shap_value': list(sample_shap)
            })
            feature_importance['abs_shap'] = feature_importance['shap_value'].abs()
            feature_importance = feature_importance.sort_values('abs_shap', ascending=False).head(10)
            
            # Plot
            colors = ['red' if x < 0 else 'green' for x in feature_importance['shap_value']]
            plt.barh(range(len(feature_importance)), feature_importance['shap_value'], color=colors)
            plt.yticks(range(len(feature_importance)), feature_importance['feature'])
            plt.xlabel('SHAP Value (impact on model output)', fontsize=11)
            plt.ylabel('Feature', fontsize=11)
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            plt.gca().invert_yaxis()
            
            label_text = 'Normal' if sample_label == 0 else 'Attack'
            pred_text = 'Normal' if prediction == 0 else 'Attack'
            plt.title(f'Local Explanation - Sample {sample_idx}\n'
                     f'True: {label_text}, Predicted: {pred_text} (P(Attack)={probability[1]:.3f})',
                     fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            output_path = Path(output_dir) / f'local_explanation_sample_{sample_idx}.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            logger.info(f"✓ Saved local explanation: {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating local explanation: {e}")
            raise
    
    def plot_multiple_local_explanations(self, n_samples=5, output_dir='outputs'):
        """
        Generate local explanations for multiple samples
        
        Args:
            n_samples: Number of samples to explain
            output_dir: Directory to save plots
        """
        logger.info(f"Generating {n_samples} local explanations...")
        
        # Select diverse samples (some normal, some attacks)
        normal_indices = self.y_test[self.y_test == 0].index[:n_samples//2]
        attack_indices = self.y_test[self.y_test == 1].index[:n_samples//2 + n_samples%2]
        
        selected_indices = list(normal_indices) + list(attack_indices)
        
        for i, idx in enumerate(selected_indices):
            logger.info(f"Processing sample {i+1}/{len(selected_indices)}...")
            self.plot_local_explanation(idx, output_dir)
        
        logger.info(f"✓ Generated {len(selected_indices)} local explanations")


def main():
    """Main explainability pipeline"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate SHAP Explanations for IDS Model"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/random_forest_model.pkl',
        help='Path to trained model'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/filtered_data.csv',
        help='Path to test data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Directory to save plots'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=1000,
        help='Number of samples to use for SHAP computation (default: 1000)'
    )
    parser.add_argument(
        '--n-local',
        type=int,
        default=4,
        help='Number of local explanations to generate (default: 4)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("WireXplain-IDS: Model Explainability Pipeline")
    print("=" * 80 + "\n")
    
    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path(args.output).mkdir(exist_ok=True)
    
    try:
        # Initialize explainer
        explainer = ModelExplainer(args.model)
        
        # Step 1: Load model
        model = explainer.load_model()
        
        # Step 2: Load test data
        X_test, y_test = explainer.load_test_data(args.data, sample_size=args.sample_size)
        
        # Step 3: Create SHAP explainer
        shap_explainer = explainer.create_explainer()
        
        # Step 4: Compute SHAP values
        shap_values = explainer.compute_shap_values()
        
        # Step 5: Generate global importance plots
        logger.info("\nGenerating global feature importance plots...")
        explainer.plot_global_importance(output_dir=args.output)
        
        # Step 6: Generate local explanations
        logger.info(f"\nGenerating {args.n_local} local explanations...")
        explainer.plot_multiple_local_explanations(n_samples=args.n_local, output_dir=args.output)
        
        print("\n" + "=" * 80)
        print("✅ Explainability Analysis Complete!")
        print("=" * 80)
        print(f"Model:        {args.model}")
        print(f"Data:         {args.data}")
        print(f"Samples used: {len(X_test):,}")
        print(f"Output dir:   {args.output}")
        print(f"\nGenerated plots:")
        print(f"  - global_feature_importance.png")
        print(f"  - global_feature_importance_bar.png")
        print(f"  - {args.n_local} local explanation plots")
        print("=" * 80 + "\n")
        
        return explainer
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    explainer = main()
