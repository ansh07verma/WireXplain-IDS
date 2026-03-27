"""
Data Loading and Preprocessing Module
Handles loading network traffic data from various sources
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataLoader:
    """Load and preprocess network traffic data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_csv(self, file_path):
        """
        Load network traffic data from CSV file
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with network traffic data
        """
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"✓ Loaded {len(df)} samples")
        return df
    
    def preprocess(self, df, target_column='label', test_size=0.2):
        """
        Preprocess the data: handle missing values, encode labels, scale features
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            test_size: Proportion of test set
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("\nPreprocessing data...")
        
        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))
        
        # Separate features and target
        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Encode categorical features
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✓ Training samples: {len(X_train)}")
        print(f"✓ Test samples: {len(X_test)}")
        print(f"✓ Number of features: {X_train_scaled.shape[1]}")
        print(f"✓ Number of classes: {len(np.unique(y_encoded))}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_feature_names(self, df, target_column='label'):
        """Get feature names from DataFrame"""
        return [col for col in df.columns if col != target_column]
    
    def get_class_names(self):
        """Get class names after encoding"""
        return self.label_encoder.classes_


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    # df = loader.load_csv("data/processed/train_data.csv")
    # X_train, X_test, y_train, y_test = loader.preprocess(df)
    print("DataLoader module ready!")
