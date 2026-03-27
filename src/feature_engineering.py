"""
Feature Engineering Module
Transform raw network traffic data into ML-ready features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/feature_engineering.log')
    ]
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering pipeline for network traffic data"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_data(self, file_path):
        """
        Load cleaned dataset
        
        Args:
            file_path: Path to clean_data.csv
            
        Returns:
            pandas.DataFrame
        """
        logger.info(f"Loading data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_label_column(self, df, label_col='Label'):
        """
        Validate presence and format of label column
        
        Args:
            df: Input DataFrame
            label_col: Name of label column
            
        Returns:
            bool: True if validation passes
        """
        logger.info("Validating label column...")
        
        # Check if label column exists
        if label_col not in df.columns:
            logger.error(f"Label column '{label_col}' not found in dataset")
            raise ValueError(f"Missing label column: {label_col}")
        
        # Get unique labels
        unique_labels = df[label_col].unique()
        logger.info(f"Found {len(unique_labels)} unique labels: {unique_labels}")
        
        # Check label distribution
        label_counts = df[label_col].value_counts()
        logger.info("Label distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"  {label}: {count:,} ({percentage:.2f}%)")
        
        logger.info("✓ Label column validation passed")
        return True
    
    def encode_labels(self, df, label_col='Label'):
        """
        Encode labels to binary (0=normal, 1=attack)
        
        Args:
            df: Input DataFrame
            label_col: Name of label column
            
        Returns:
            DataFrame with encoded labels
        """
        logger.info("Encoding labels...")
        
        # Create binary labels: 0 for normal/benign, 1 for any attack
        df = df.copy()
        
        # Check if labels are already numeric
        if pd.api.types.is_numeric_dtype(df[label_col]):
            logger.info("Labels are already numeric")
            df['label_binary'] = df[label_col]
        else:
            # Convert string labels to binary
            # Benign/Normal = 0, everything else = 1
            df['label_binary'] = df[label_col].apply(
                lambda x: 0 if str(x).lower() in ['benign', 'normal'] else 1
            )
        
        # Also keep original encoded labels for multi-class
        df['label_encoded'] = self.label_encoder.fit_transform(df[label_col])
        
        logger.info(f"Binary label distribution:")
        binary_counts = df['label_binary'].value_counts()
        for label, count in binary_counts.items():
            label_name = "Normal" if label == 0 else "Attack"
            percentage = (count / len(df)) * 100
            logger.info(f"  {label_name} ({label}): {count:,} ({percentage:.2f}%)")
        
        logger.info("✓ Labels encoded successfully")
        return df
    
    def encode_categorical_features(self, df, exclude_cols=None):
        """
        Encode categorical features
        
        Args:
            df: Input DataFrame
            exclude_cols: Columns to exclude from encoding
            
        Returns:
            DataFrame with encoded features
        """
        logger.info("Encoding categorical features...")
        
        if exclude_cols is None:
            exclude_cols = ['Label', 'label_binary', 'label_encoded', 'Timestamp']
        
        df = df.copy()
        categorical_cols = df.select_dtypes(include=['object', 'str']).columns
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        if len(categorical_cols) == 0:
            logger.info("No categorical features to encode")
            return df
        
        logger.info(f"Encoding {len(categorical_cols)} categorical columns: {categorical_cols}")
        
        for col in categorical_cols:
            try:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                logger.info(f"  ✓ Encoded {col} ({len(le.classes_)} unique values)")
            except Exception as e:
                logger.warning(f"  ⚠ Failed to encode {col}: {e}")
        
        logger.info("✓ Categorical encoding complete")
        return df
    
    def select_features(self, df, exclude_cols=None):
        """
        Select meaningful features for ML
        
        Args:
            df: Input DataFrame
            exclude_cols: Columns to exclude
            
        Returns:
            DataFrame with selected features
        """
        logger.info("Selecting features...")
        
        if exclude_cols is None:
            exclude_cols = ['Label', 'Timestamp', 'Dst Port', 'Protocol']
        
        # Get all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove excluded columns and label columns
        feature_cols = [
            col for col in numeric_cols 
            if col not in exclude_cols 
            and col not in ['label_binary', 'label_encoded']
        ]
        
        # Handle missing values
        logger.info("Handling missing values...")
        df_features = df[feature_cols].copy()
        
        # Fill missing values with median
        missing_before = df_features.isnull().sum().sum()
        if missing_before > 0:
            logger.info(f"  Found {missing_before} missing values")
            df_features = df_features.fillna(df_features.median())
            logger.info(f"  ✓ Filled missing values with median")
        
        # Handle infinite values
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(df_features.median())
        
        self.feature_names = feature_cols
        logger.info(f"✓ Selected {len(feature_cols)} features")
        
        return df_features
    
    def engineer_features(self, df):
        """
        Create derived features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional engineered features
        """
        logger.info("Engineering additional features...")
        
        df = df.copy()
        engineered_count = 0
        
        # Example: Packet rate features
        if 'Flow Duration' in df.columns and 'Tot Fwd Pkts' in df.columns:
            df['fwd_packet_rate'] = df['Tot Fwd Pkts'] / (df['Flow Duration'] + 1)
            engineered_count += 1
        
        if 'Flow Duration' in df.columns and 'Tot Bwd Pkts' in df.columns:
            df['bwd_packet_rate'] = df['Tot Bwd Pkts'] / (df['Flow Duration'] + 1)
            engineered_count += 1
        
        # Example: Byte ratio features
        if 'TotLen Fwd Pkts' in df.columns and 'TotLen Bwd Pkts' in df.columns:
            total_bytes = df['TotLen Fwd Pkts'] + df['TotLen Bwd Pkts']
            df['fwd_byte_ratio'] = df['TotLen Fwd Pkts'] / (total_bytes + 1)
            engineered_count += 1
        
        # Example: Flag ratios
        if 'SYN Flag Cnt' in df.columns and 'Tot Fwd Pkts' in df.columns:
            df['syn_ratio'] = df['SYN Flag Cnt'] / (df['Tot Fwd Pkts'] + 1)
            engineered_count += 1
        
        logger.info(f"✓ Created {engineered_count} engineered features")
        return df
    
    def save_features(self, df, features, labels, output_path):
        """
        Save processed features to CSV
        
        Args:
            df: Original DataFrame
            features: Feature DataFrame
            labels: Label series
            output_path: Path to save features
        """
        logger.info(f"Saving features to {output_path}")
        
        # Combine features and labels
        output_df = features.copy()
        output_df['label'] = labels
        
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        output_df.to_csv(output_path, index=False)
        
        logger.info(f"✓ Saved {len(output_df):,} rows, {len(output_df.columns)} columns")
        logger.info(f"  Features: {len(features.columns)}")
        logger.info(f"  File size: {Path(output_path).stat().st_size / 1024**2:.2f} MB")


def main():
    """Main feature engineering pipeline"""
    
    print("\n" + "=" * 70)
    print("WireXplain-IDS: Feature Engineering Pipeline")
    print("=" * 70 + "\n")
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Define paths
    input_path = "data/raw/02-14-2018.csv"  # Using raw data as clean_data.csv
    output_path = "data/processed/features.csv"
    
    try:
        # Step 1: Load data
        df = fe.load_data(input_path)
        
        # Step 2: Validate label column
        fe.validate_label_column(df, label_col='Label')
        
        # Step 3: Encode labels
        df = fe.encode_labels(df, label_col='Label')
        
        # Step 4: Encode categorical features
        df = fe.encode_categorical_features(df)
        
        # Step 5: Engineer features
        df = fe.engineer_features(df)
        
        # Step 6: Select features
        features = fe.select_features(df)
        
        # Step 7: Get labels
        labels = df['label_binary']
        
        # Step 8: Save features
        fe.save_features(df, features, labels, output_path)
        
        print("\n" + "=" * 70)
        print("✅ Feature Engineering Complete!")
        print("=" * 70)
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"Features: {len(features.columns)}")
        print(f"Samples: {len(features):,}")
        print(f"Labels: Binary (0=Normal, 1=Attack)")
        print("=" * 70 + "\n")
        
        return features, labels
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    features, labels = main()
