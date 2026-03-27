"""
Visualization Module
Generate publication-ready plots for model evaluation and interpretation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import logging
import sys
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/visualize.log')
    ]
)
logger = logging.getLogger(__name__)

# Set consistent style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


class ModelVisualizer:
    """Generate comprehensive visualizations for IDS model"""
    
    def __init__(self, model_path, data_path, output_dir='outputs'):
        """
        Initialize visualizer
        
        Args:
            model_path: Path to trained model
            data_path: Path to test data
            output_dir: Directory to save plots
        """
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_model_and_data(self):
        """Load trained model and test data"""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Load model
            self.model = joblib.load(self.model_path)
            logger.info(f"✓ Model loaded: {type(self.model).__name__}")
            
            # Load data
            logger.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
            
            # Separate features and labels
            feature_cols = [col for col in df.columns if col not in ['label', 'is_anomaly']]
            self.X_test = df[feature_cols]
            self.y_test = df['label']
            
            # Make predictions
            logger.info("Generating predictions...")
            self.y_pred = self.model.predict(self.X_test)
            
            logger.info(f"✓ Loaded {len(self.X_test):,} test samples")
            
        except Exception as e:
            logger.error(f"Error loading model/data: {e}")
            raise
    
    def plot_confusion_matrix(self, normalize=False):
        """
        Generate confusion matrix plot
        
        Args:
            normalize: If True, normalize the confusion matrix
        """
        logger.info("Generating confusion matrix...")
        
        try:
            # Compute confusion matrix
            cm = confusion_matrix(self.y_test, self.y_pred)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot confusion matrix
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=['Normal', 'Attack']
            )
            disp.plot(ax=ax, cmap='Blues', values_format=',.0f')
            
            # Customize
            plt.title('Confusion Matrix - IDS Model', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            
            # Add accuracy text
            accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
            plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.2%}',
                    ha='center', transform=ax.transAxes,
                    fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            
            # Save
            output_path = Path(self.output_dir) / 'confusion_matrix.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            logger.info(f"✓ Saved confusion matrix: {output_path}")
            
            # Also create normalized version
            if normalize:
                self._plot_normalized_confusion_matrix(cm)
            
        except Exception as e:
            logger.error(f"Error generating confusion matrix: {e}")
            raise
    
    def _plot_normalized_confusion_matrix(self, cm):
        """Plot normalized confusion matrix"""
        logger.info("Generating normalized confusion matrix...")
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'],
                   cbar_kws={'label': 'Proportion'},
                   ax=ax)
        
        plt.title('Normalized Confusion Matrix - IDS Model',
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        # Save
        output_path = Path(self.output_dir) / 'confusion_matrix_normalized.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"✓ Saved normalized confusion matrix: {output_path}")
    
    def plot_feature_importance(self, top_n=15):
        """
        Generate feature importance plot from RandomForest
        
        Args:
            top_n: Number of top features to display
        """
        logger.info("Generating feature importance plot...")
        
        try:
            # Get feature importances
            if not hasattr(self.model, 'feature_importances_'):
                logger.warning("Model does not have feature_importances_ attribute")
                return
            
            importances = self.model.feature_importances_
            feature_names = self.X_test.columns
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Select top N
            top_features = importance_df.head(top_n)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot horizontal bar chart
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
            bars = ax.barh(range(len(top_features)), top_features['importance'], color=colors)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.invert_yaxis()
            
            # Labels and title
            ax.set_xlabel('Feature Importance (Gini Importance)', fontsize=12)
            ax.set_ylabel('Feature', fontsize=12)
            ax.set_title(f'Top {top_n} Feature Importances - RandomForest',
                        fontsize=14, fontweight='bold', pad=20)
            
            # Add value labels on bars
            for i, (idx, row) in enumerate(top_features.iterrows()):
                ax.text(row['importance'], i, f' {row["importance"]:.4f}',
                       va='center', fontsize=9)
            
            plt.tight_layout()
            
            # Save
            output_path = Path(self.output_dir) / 'feature_importance.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            logger.info(f"✓ Saved feature importance: {output_path}")
            
            # Log top features
            logger.info(f"Top {min(5, top_n)} features:")
            for idx, row in top_features.head(5).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
        except Exception as e:
            logger.error(f"Error generating feature importance: {e}")
            raise
    
    def plot_performance_metrics(self):
        """Generate comprehensive performance metrics visualization"""
        logger.info("Generating performance metrics plot...")
        
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score,
                f1_score, roc_auc_score, roc_curve
            )
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, self.y_pred)
            precision = precision_score(self.y_test, self.y_pred)
            recall = recall_score(self.y_test, self.y_pred)
            f1 = f1_score(self.y_test, self.y_pred)
            
            # Get probabilities for ROC
            y_proba = self.model.predict_proba(self.X_test)[:, 1]
            auc = roc_auc_score(self.y_test, y_proba)
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Left: Metrics bar chart
            metrics = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': auc
            }
            
            colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
            bars = ax1.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.8)
            ax1.set_ylim([0, 1.1])
            ax1.set_ylabel('Score', fontsize=12)
            ax1.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
            ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            ax1.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Right: ROC curve
            ax2.plot(fpr, tpr, color='#3498db', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
            ax2.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random')
            ax2.set_xlabel('False Positive Rate', fontsize=12)
            ax2.set_ylabel('True Positive Rate', fontsize=12)
            ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
            ax2.legend(loc='lower right', fontsize=10)
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            
            # Save
            output_path = Path(self.output_dir) / 'performance_metrics.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            logger.info(f"✓ Saved performance metrics: {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating performance metrics: {e}")
            raise
    
    def create_summary_dashboard(self):
        """Create a comprehensive summary dashboard"""
        logger.info("Generating summary dashboard...")
        
        try:
            from sklearn.metrics import classification_report
            
            # Create figure
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Confusion Matrix (top left)
            ax1 = fig.add_subplot(gs[0:2, 0])
            cm = confusion_matrix(self.y_test, self.y_pred)
            sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues',
                       xticklabels=['Normal', 'Attack'],
                       yticklabels=['Normal', 'Attack'],
                       ax=ax1, cbar=False)
            ax1.set_title('Confusion Matrix', fontweight='bold')
            ax1.set_ylabel('True Label')
            ax1.set_xlabel('Predicted Label')
            
            # 2. Feature Importance (top middle & right)
            ax2 = fig.add_subplot(gs[0:2, 1:])
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_names = self.X_test.columns
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False).head(10)
                
                ax2.barh(range(len(importance_df)), importance_df['importance'],
                        color=plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df))))
                ax2.set_yticks(range(len(importance_df)))
                ax2.set_yticklabels(importance_df['feature'])
                ax2.invert_yaxis()
                ax2.set_xlabel('Importance')
                ax2.set_title('Top 10 Feature Importances', fontweight='bold')
            
            # 3. Classification Report (bottom)
            ax3 = fig.add_subplot(gs[2, :])
            ax3.axis('off')
            
            report = classification_report(self.y_test, self.y_pred,
                                          target_names=['Normal', 'Attack'],
                                          output_dict=True)
            
            report_text = "Classification Report:\n\n"
            report_text += f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}\n"
            report_text += "-" * 60 + "\n"
            for label in ['Normal', 'Attack']:
                report_text += f"{label:<12} {report[label]['precision']:<12.4f} "
                report_text += f"{report[label]['recall']:<12.4f} {report[label]['f1-score']:<12.4f} "
                report_text += f"{int(report[label]['support']):<12,}\n"
            report_text += "-" * 60 + "\n"
            report_text += f"{'Accuracy':<12} {'':<12} {'':<12} {report['accuracy']:<12.4f} "
            report_text += f"{int(report['macro avg']['support']):<12,}\n"
            
            ax3.text(0.1, 0.5, report_text, fontsize=10, family='monospace',
                    verticalalignment='center')
            
            # Overall title
            fig.suptitle('WireXplain-IDS: Model Evaluation Dashboard',
                        fontsize=16, fontweight='bold', y=0.98)
            
            # Save
            output_path = Path(self.output_dir) / 'summary_dashboard.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            logger.info(f"✓ Saved summary dashboard: {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating summary dashboard: {e}")
            raise
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        logger.info("Generating all visualization plots...")
        
        # Load model and data
        self.load_model_and_data()
        
        # Generate plots
        self.plot_confusion_matrix(normalize=True)
        self.plot_feature_importance(top_n=15)
        self.plot_performance_metrics()
        self.create_summary_dashboard()
        
        logger.info("✓ All plots generated successfully")


def main():
    """Main visualization pipeline"""
    parser = argparse.ArgumentParser(
        description="Generate Visualizations for IDS Model"
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
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("WireXplain-IDS: Visualization Generation")
    print("=" * 80 + "\n")
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    try:
        # Initialize visualizer
        visualizer = ModelVisualizer(
            model_path=args.model,
            data_path=args.data,
            output_dir=args.output
        )
        
        # Generate all plots
        visualizer.generate_all_plots()
        
        print("\n" + "=" * 80)
        print("✅ Visualization Complete!")
        print("=" * 80)
        print(f"Model:      {args.model}")
        print(f"Data:       {args.data}")
        print(f"Output dir: {args.output}")
        print(f"\nGenerated plots:")
        print(f"  - confusion_matrix.png")
        print(f"  - confusion_matrix_normalized.png")
        print(f"  - feature_importance.png")
        print(f"  - performance_metrics.png")
        print(f"  - summary_dashboard.png")
        print("=" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
