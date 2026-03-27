"""
Model Training Module
Train and evaluate RandomForestClassifier for intrusion detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib
import logging
import sys
import argparse
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/train_model.log')
    ]
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate machine learning models"""
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize model trainer
        
        Args:
            test_size: Proportion of test set (default: 0.2 = 20%)
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, file_path, exclude_anomalies=False):
        """
        Load training data
        
        Args:
            file_path: Path to filtered_data.csv
            exclude_anomalies: If True, remove samples flagged as anomalies
            
        Returns:
            X (features), y (labels)
        """
        logger.info(f"Loading data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
            
            # Check for anomaly column
            if 'is_anomaly' in df.columns and exclude_anomalies:
                original_len = len(df)
                df = df[df['is_anomaly'] == 0]
                removed = original_len - len(df)
                logger.info(f"Excluded {removed:,} anomalous samples")
                logger.info(f"Remaining samples: {len(df):,}")
            
            # Separate features and target
            if 'label' not in df.columns:
                raise ValueError("Label column not found in dataset")
            
            # Drop label and anomaly flag (if present)
            feature_cols = [col for col in df.columns if col not in ['label', 'is_anomaly']]
            X = df[feature_cols]
            y = df['label']
            
            logger.info(f"Features: {X.shape[1]}")
            logger.info(f"Samples: {len(X):,}")
            logger.info(f"Label distribution: {y.value_counts().to_dict()}")
            
            return X, y
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def split_data(self, X, y):
        """
        Split data into training and test sets
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Splitting data (test_size={self.test_size})...")
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y  # Maintain class distribution
            )
            
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            
            logger.info(f"✓ Training set: {len(X_train):,} samples")
            logger.info(f"✓ Test set: {len(X_test):,} samples")
            logger.info(f"Train label distribution: {y_train.value_counts().to_dict()}")
            logger.info(f"Test label distribution: {y_test.value_counts().to_dict()}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise
    
    def train_random_forest(self, n_estimators=100, max_depth=None, min_samples_split=2):
        """
        Train RandomForestClassifier
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Minimum samples required to split a node
            
        Returns:
            Trained model
        """
        if self.X_train is None:
            raise ValueError("Data not split. Call split_data() first.")
        
        logger.info("Training RandomForestClassifier...")
        logger.info(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}")
        
        try:
            # Initialize model
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=self.random_state,
                n_jobs=-1,  # Use all CPU cores
                verbose=1
            )
            
            # Train model
            start_time = time.time()
            self.model.fit(self.X_train, self.y_train)
            training_time = time.time() - start_time
            
            logger.info(f"✓ Model trained in {training_time:.2f} seconds")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def evaluate_model(self):
        """
        Evaluate model on test set
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_random_forest() first.")
        
        logger.info("Evaluating model on test set...")
        
        try:
            # Make predictions
            y_pred = self.model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='binary')
            recall = recall_score(self.y_test, y_pred, average='binary')
            f1 = f1_score(self.y_test, y_pred, average='binary')
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            
            # Classification report
            report = classification_report(self.y_test, y_pred, 
                                          target_names=['Normal', 'Attack'])
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'classification_report': report
            }
            
            logger.info(f"✓ Accuracy: {accuracy:.4f}")
            logger.info(f"✓ Precision: {precision:.4f}")
            logger.info(f"✓ Recall: {recall:.4f}")
            logger.info(f"✓ F1-Score: {f1:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def print_evaluation_report(self, metrics):
        """
        Print detailed evaluation report
        
        Args:
            metrics: Dictionary with evaluation metrics
        """
        print("\n" + "=" * 80)
        print("MODEL EVALUATION REPORT")
        print("=" * 80)
        
        # Overall metrics
        print("\n📊 Overall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        # Confusion matrix
        cm = metrics['confusion_matrix']
        print("\n📈 Confusion Matrix:")
        print(f"                Predicted")
        print(f"              Normal  Attack")
        print(f"  Actual Normal  {cm[0][0]:>6,}  {cm[0][1]:>6,}")
        print(f"        Attack   {cm[1][0]:>6,}  {cm[1][1]:>6,}")
        
        # Derived metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        print(f"\n  True Negatives:  {tn:,}")
        print(f"  False Positives: {fp:,}")
        print(f"  False Negatives: {fn:,}")
        print(f"  True Positives:  {tp:,}")
        
        # Classification report
        print("\n📋 Detailed Classification Report:")
        print(metrics['classification_report'])
        
        print("=" * 80 + "\n")
    
    def save_model(self, output_path):
        """
        Save trained model to disk
        
        Args:
            output_path: Path to save model file
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        logger.info(f"Saving model to {output_path}")
        
        try:
            # Create output directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model using joblib
            joblib.dump(self.model, output_path)
            
            file_size = Path(output_path).stat().st_size / 1024**2
            
            logger.info(f"✓ Model saved successfully")
            logger.info(f"  File size: {file_size:.2f} MB")
            
            print(f"\n✓ Model saved to: {output_path}")
            print(f"  File size: {file_size:.2f} MB\n")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise


def main():
    """Main model training pipeline"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train RandomForestClassifier for Intrusion Detection"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/filtered_data.csv',
        help='Path to input data file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/random_forest_model.pkl',
        help='Path to save trained model'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of trees (default: 100)'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=None,
        help='Maximum tree depth (default: None = unlimited)'
    )
    parser.add_argument(
        '--exclude-anomalies',
        action='store_true',
        help='Exclude samples flagged as anomalies'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("WireXplain-IDS: Model Training Pipeline")
    print("=" * 80 + "\n")
    
    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(
            test_size=args.test_size,
            random_state=42
        )
        
        # Step 1: Load data
        X, y = trainer.load_data(args.input, exclude_anomalies=args.exclude_anomalies)
        
        # Step 2: Split data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        # Step 3: Train model
        model = trainer.train_random_forest(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth
        )
        
        # Step 4: Evaluate model
        metrics = trainer.evaluate_model()
        
        # Step 5: Print detailed report
        trainer.print_evaluation_report(metrics)
        
        # Step 6: Save model
        trainer.save_model(args.output)
        
        print("=" * 80)
        print("✅ Model Training Complete!")
        print("=" * 80)
        print(f"Input:      {args.input}")
        print(f"Model:      {args.output}")
        print(f"Accuracy:   {metrics['accuracy']:.4f}")
        print(f"F1-Score:   {metrics['f1_score']:.4f}")
        print("=" * 80 + "\n")
        
        return trainer, metrics
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    trainer, metrics = main()
