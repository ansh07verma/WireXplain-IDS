"""
Feature Selection Module
Select the most important features using Mutual Information scores
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
import logging
import sys
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/feature_selection.log')
    ]
)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """Select important features using Mutual Information"""
    
    def __init__(self, top_n=10):
        """
        Initialize feature selector
        
        Args:
            top_n: Number of top features to select (default: 10)
        """
        self.top_n = top_n
        self.feature_scores = None
        self.selected_features = None
        
    def load_features(self, file_path):
        """
        Load feature dataset
        
        Args:
            file_path: Path to features.csv
            
        Returns:
            X (features), y (labels)
        """
        logger.info(f"Loading features from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
            
            # Separate features and target
            if 'label' not in df.columns:
                raise ValueError("Label column not found in dataset")
            
            X = df.drop(columns=['label'])
            y = df['label']
            
            logger.info(f"Features: {X.shape[1]}")
            logger.info(f"Samples: {len(X):,}")
            logger.info(f"Label distribution: {y.value_counts().to_dict()}")
            
            return X, y
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            raise
    
    def compute_mutual_information(self, X, y, random_state=42):
        """
        Compute Mutual Information scores for all features
        
        Args:
            X: Feature matrix
            y: Target labels
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with feature names and MI scores
        """
        logger.info("Computing Mutual Information scores...")
        logger.info(f"This may take a few minutes for {X.shape[1]} features...")
        
        try:
            # Compute MI scores
            mi_scores = mutual_info_classif(
                X, y, 
                discrete_features=False,
                random_state=random_state,
                n_neighbors=3
            )
            
            # Create DataFrame with scores
            feature_scores = pd.DataFrame({
                'feature': X.columns,
                'mi_score': mi_scores
            })
            
            # Sort by score (descending)
            feature_scores = feature_scores.sort_values(
                'mi_score', 
                ascending=False
            ).reset_index(drop=True)
            
            self.feature_scores = feature_scores
            
            logger.info(f"✓ Computed MI scores for {len(feature_scores)} features")
            logger.info(f"Score range: [{mi_scores.min():.6f}, {mi_scores.max():.6f}]")
            
            return feature_scores
            
        except Exception as e:
            logger.error(f"Error computing MI scores: {e}")
            raise
    
    def print_feature_ranking(self, top_n=None):
        """
        Print feature ranking by importance
        
        Args:
            top_n: Number of top features to display (default: all)
        """
        if self.feature_scores is None:
            logger.warning("No feature scores available. Run compute_mutual_information first.")
            return
        
        display_n = top_n or len(self.feature_scores)
        
        print("\n" + "=" * 80)
        print(f"Feature Ranking by Mutual Information (Top {display_n})")
        print("=" * 80)
        print(f"{'Rank':<6} {'Feature':<40} {'MI Score':<12}")
        print("-" * 80)
        
        for idx, row in self.feature_scores.head(display_n).iterrows():
            print(f"{idx+1:<6} {row['feature']:<40} {row['mi_score']:.8f}")
        
        print("=" * 80 + "\n")
        
        # Log to file as well
        logger.info(f"Top {display_n} features by MI score:")
        for idx, row in self.feature_scores.head(display_n).iterrows():
            logger.info(f"  {idx+1}. {row['feature']}: {row['mi_score']:.8f}")
    
    def select_top_features(self, n=None):
        """
        Select top N features
        
        Args:
            n: Number of features to select (default: self.top_n)
            
        Returns:
            List of selected feature names
        """
        if self.feature_scores is None:
            raise ValueError("No feature scores available. Run compute_mutual_information first.")
        
        n = n or self.top_n
        
        # Ensure n doesn't exceed available features
        n = min(n, len(self.feature_scores))
        
        selected = self.feature_scores.head(n)['feature'].tolist()
        self.selected_features = selected
        
        logger.info(f"✓ Selected top {n} features")
        logger.info(f"Selected features: {', '.join(selected[:5])}...")
        
        return selected
    
    def save_selected_features(self, X, y, output_path):
        """
        Save dataset with only selected features
        
        Args:
            X: Full feature matrix
            y: Target labels
            output_path: Path to save reduced dataset
        """
        if self.selected_features is None:
            raise ValueError("No features selected. Run select_top_features first.")
        
        logger.info(f"Saving selected features to {output_path}")
        
        # Create reduced dataset
        X_selected = X[self.selected_features].copy()
        
        # Add label column
        X_selected['label'] = y.values
        
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        X_selected.to_csv(output_path, index=False)
        
        file_size = Path(output_path).stat().st_size / 1024**2
        
        logger.info(f"✓ Saved {len(X_selected):,} rows, {len(X_selected.columns)} columns")
        logger.info(f"  Features: {len(self.selected_features)}")
        logger.info(f"  File size: {file_size:.2f} MB")
        
        print(f"\n✓ Reduced dataset saved to: {output_path}")
        print(f"  Original features: {X.shape[1]}")
        print(f"  Selected features: {len(self.selected_features)}")
        print(f"  Reduction: {(1 - len(self.selected_features)/X.shape[1])*100:.1f}%")
        print(f"  File size: {file_size:.2f} MB\n")
    
    def get_feature_importance_summary(self):
        """
        Get summary statistics of feature importance
        
        Returns:
            Dictionary with summary statistics
        """
        if self.feature_scores is None:
            return None
        
        scores = self.feature_scores['mi_score']
        
        summary = {
            'total_features': len(scores),
            'mean_score': scores.mean(),
            'median_score': scores.median(),
            'std_score': scores.std(),
            'min_score': scores.min(),
            'max_score': scores.max(),
            'top_10_mean': scores.head(10).mean(),
            'bottom_10_mean': scores.tail(10).mean()
        }
        
        return summary


def main():
    """Main feature selection pipeline"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Feature Selection using Mutual Information"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/features.csv',
        help='Path to input features file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/selected_features.csv',
        help='Path to output selected features file'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top features to select (default: 10)'
    )
    parser.add_argument(
        '--show-all',
        action='store_true',
        help='Show all features in ranking (not just top N)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("WireXplain-IDS: Feature Selection Pipeline")
    print("=" * 80 + "\n")
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    try:
        # Initialize selector
        selector = FeatureSelector(top_n=args.top_n)
        
        # Step 1: Load features
        X, y = selector.load_features(args.input)
        
        # Step 2: Compute Mutual Information scores
        feature_scores = selector.compute_mutual_information(X, y)
        
        # Step 3: Print feature ranking
        display_n = None if args.show_all else args.top_n * 2
        selector.print_feature_ranking(top_n=display_n)
        
        # Step 4: Select top N features
        selected = selector.select_top_features(n=args.top_n)
        
        # Step 5: Get summary statistics
        summary = selector.get_feature_importance_summary()
        if summary:
            print("\n" + "=" * 80)
            print("Feature Importance Summary")
            print("=" * 80)
            print(f"Total features:        {summary['total_features']}")
            print(f"Mean MI score:         {summary['mean_score']:.8f}")
            print(f"Median MI score:       {summary['median_score']:.8f}")
            print(f"Std MI score:          {summary['std_score']:.8f}")
            print(f"Min MI score:          {summary['min_score']:.8f}")
            print(f"Max MI score:          {summary['max_score']:.8f}")
            print(f"Top 10 mean:           {summary['top_10_mean']:.8f}")
            print(f"Bottom 10 mean:        {summary['bottom_10_mean']:.8f}")
            print("=" * 80 + "\n")
        
        # Step 6: Save selected features
        selector.save_selected_features(X, y, args.output)
        
        print("=" * 80)
        print("✅ Feature Selection Complete!")
        print("=" * 80)
        print(f"Input:            {args.input}")
        print(f"Output:           {args.output}")
        print(f"Features selected: {args.top_n}")
        print(f"Samples:          {len(X):,}")
        print("=" * 80 + "\n")
        
        return selector
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    selector = main()
