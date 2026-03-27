"""
Isolation Filter Module
Detect and filter outlier/anomalous samples using IsolationForest

Rationale:
-----------
IsolationForest is an unsupervised anomaly detection algorithm that works by:
1. Randomly selecting features and split values
2. Building isolation trees that partition the data
3. Anomalies are easier to isolate (fewer splits needed)
4. Normal samples require more splits to isolate

This helps identify:
- Rare attack patterns that may be noise
- Data collection errors or corrupted samples
- Extreme outliers that could skew model training

The module provides two modes:
- FILTER: Remove anomalous samples entirely
- FLAG: Keep all samples but add an 'is_anomaly' column
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
import logging
import sys
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/isolation_filter.log')
    ]
)
logger = logging.getLogger(__name__)


class IsolationFilter:
    """Filter outliers using IsolationForest algorithm"""
    
    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize isolation filter
        
        Args:
            contamination: Expected proportion of outliers (default: 0.1 = 10%)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.predictions = None
        
    def load_data(self, file_path):
        """
        Load selected features dataset
        
        Args:
            file_path: Path to selected_features.csv
            
        Returns:
            X (features), y (labels), df (full DataFrame)
        """
        logger.info(f"Loading data from {file_path}")
        
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
            
            return X, y, df
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def fit_isolation_forest(self, X):
        """
        Fit IsolationForest model to detect anomalies
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions (-1 for outliers, 1 for inliers)
        """
        logger.info("Fitting IsolationForest model...")
        logger.info(f"Contamination rate: {self.contamination:.2%}")
        logger.info(f"Expected outliers: ~{int(len(X) * self.contamination):,} samples")
        
        try:
            # Initialize IsolationForest
            # n_estimators: number of trees (more = better but slower)
            # max_samples: samples per tree (auto = min(256, n_samples))
            # contamination: expected proportion of outliers
            self.model = IsolationForest(
                n_estimators=100,
                max_samples='auto',
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,  # Use all CPU cores
                verbose=0
            )
            
            # Fit and predict
            # Returns: 1 for inliers, -1 for outliers
            predictions = self.model.fit_predict(X)
            self.predictions = predictions
            
            # Count outliers
            n_outliers = (predictions == -1).sum()
            n_inliers = (predictions == 1).sum()
            outlier_pct = (n_outliers / len(predictions)) * 100
            
            logger.info(f"✓ IsolationForest fitted successfully")
            logger.info(f"Inliers:  {n_inliers:,} ({100-outlier_pct:.2f}%)")
            logger.info(f"Outliers: {n_outliers:,} ({outlier_pct:.2f}%)")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error fitting IsolationForest: {e}")
            raise
    
    def analyze_outliers_by_class(self, y, predictions):
        """
        Analyze outlier distribution across classes
        
        Args:
            y: Target labels
            predictions: IsolationForest predictions
        """
        logger.info("\nAnalyzing outliers by class...")
        
        # Create DataFrame for analysis
        analysis_df = pd.DataFrame({
            'label': y,
            'is_outlier': predictions == -1
        })
        
        # Group by label
        summary = analysis_df.groupby('label').agg({
            'is_outlier': ['sum', 'count', 'mean']
        })
        
        print("\n" + "=" * 70)
        print("Outlier Distribution by Class")
        print("=" * 70)
        print(f"{'Class':<10} {'Total':<12} {'Outliers':<12} {'Percentage':<12}")
        print("-" * 70)
        
        for label in sorted(y.unique()):
            mask = analysis_df['label'] == label
            total = mask.sum()
            outliers = (analysis_df[mask]['is_outlier']).sum()
            pct = (outliers / total * 100) if total > 0 else 0
            
            class_name = "Normal" if label == 0 else "Attack"
            print(f"{class_name:<10} {total:<12,} {outliers:<12,} {pct:<12.2f}%")
            
            logger.info(f"{class_name}: {outliers:,}/{total:,} ({pct:.2f}%) outliers")
        
        print("=" * 70 + "\n")
    
    def filter_outliers(self, df, predictions, mode='filter'):
        """
        Filter or flag outliers based on mode
        
        Args:
            df: Original DataFrame
            predictions: IsolationForest predictions
            mode: 'filter' to remove outliers, 'flag' to add column
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Processing outliers in '{mode}' mode...")
        
        if mode == 'filter':
            # Remove outliers (keep only inliers where prediction == 1)
            df_filtered = df[predictions == 1].copy()
            removed = len(df) - len(df_filtered)
            
            logger.info(f"✓ Filtered {removed:,} outliers")
            logger.info(f"Remaining samples: {len(df_filtered):,}")
            
            return df_filtered
            
        elif mode == 'flag':
            # Add is_anomaly column (True for outliers)
            df_flagged = df.copy()
            df_flagged['is_anomaly'] = (predictions == -1).astype(int)
            
            n_flagged = df_flagged['is_anomaly'].sum()
            logger.info(f"✓ Flagged {n_flagged:,} samples as anomalies")
            logger.info(f"Total samples: {len(df_flagged):,}")
            
            return df_flagged
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'filter' or 'flag'")
    
    def save_filtered_data(self, df, output_path):
        """
        Save processed dataset
        
        Args:
            df: Processed DataFrame
            output_path: Path to save filtered data
        """
        logger.info(f"Saving filtered data to {output_path}")
        
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        file_size = Path(output_path).stat().st_size / 1024**2
        
        logger.info(f"✓ Saved {len(df):,} rows, {len(df.columns)} columns")
        logger.info(f"  File size: {file_size:.2f} MB")
        
        print(f"\n✓ Filtered dataset saved to: {output_path}")
        print(f"  Samples: {len(df):,}")
        print(f"  Features: {len(df.columns)}")
        print(f"  File size: {file_size:.2f} MB\n")


def main():
    """Main isolation filtering pipeline"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Outlier Detection and Filtering using IsolationForest"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/selected_features.csv',
        help='Path to input features file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/filtered_data.csv',
        help='Path to output filtered data file'
    )
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.05,
        help='Expected proportion of outliers (default: 0.05 = 5%%)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['filter', 'flag'],
        default='flag',
        help='Mode: filter (remove outliers) or flag (add is_anomaly column)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("WireXplain-IDS: Isolation Filter Pipeline")
    print("=" * 80 + "\n")
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    try:
        # Initialize filter
        iso_filter = IsolationFilter(
            contamination=args.contamination,
            random_state=42
        )
        
        # Step 1: Load data
        X, y, df = iso_filter.load_data(args.input)
        
        # Step 2: Fit IsolationForest and detect outliers
        predictions = iso_filter.fit_isolation_forest(X)
        
        # Step 3: Analyze outliers by class
        iso_filter.analyze_outliers_by_class(y, predictions)
        
        # Step 4: Filter or flag outliers
        df_processed = iso_filter.filter_outliers(df, predictions, mode=args.mode)
        
        # Step 5: Print summary
        print("=" * 80)
        print("Summary Statistics")
        print("=" * 80)
        print(f"Original samples:      {len(df):,}")
        print(f"Processed samples:     {len(df_processed):,}")
        
        if args.mode == 'filter':
            removed = len(df) - len(df_processed)
            print(f"Removed outliers:      {removed:,} ({removed/len(df)*100:.2f}%)")
        else:
            flagged = df_processed['is_anomaly'].sum()
            print(f"Flagged anomalies:     {flagged:,} ({flagged/len(df)*100:.2f}%)")
        
        print(f"Mode:                  {args.mode}")
        print(f"Contamination:         {args.contamination:.2%}")
        print("=" * 80 + "\n")
        
        # Step 6: Save processed data
        iso_filter.save_filtered_data(df_processed, args.output)
        
        print("=" * 80)
        print("✅ Isolation Filtering Complete!")
        print("=" * 80)
        print(f"Input:  {args.input}")
        print(f"Output: {args.output}")
        print(f"Mode:   {args.mode}")
        print("=" * 80 + "\n")
        
        return iso_filter, df_processed
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    iso_filter, df_processed = main()
