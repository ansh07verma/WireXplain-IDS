"""
Preprocessing Module
Load and preprocess network traffic datasets from CSV or PCAP files
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_csv(file_path, verbose=True):
    """
    Load network traffic data from CSV file
    
    Args:
        file_path: Path to CSV file (str or Path object)
        verbose: Whether to print dataset information (default: True)
        
    Returns:
        pandas.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        pd.errors.EmptyDataError: If the CSV file is empty
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    if verbose:
        print("=" * 70)
        print(f"Loading CSV: {file_path.name}")
        print("=" * 70)
    
    # Load CSV file
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"CSV file is empty: {file_path}")
    
    if verbose:
        print_dataset_info(df)
    
    return df


def print_dataset_info(df):
    """
    Print comprehensive dataset information
    
    Args:
        df: pandas DataFrame to analyze
    """
    print(f"\n📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    print("\n📋 Column Names:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        print(f"  {i:2d}. {col:30s} ({dtype})")
    
    print("\n📈 Basic Statistics:")
    print(f"  • Total columns: {len(df.columns)}")
    print(f"  • Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"  • Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")
    print(f"  • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n🔍 Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✓ No missing values detected")
    else:
        missing_cols = missing[missing > 0]
        for col, count in missing_cols.items():
            percentage = (count / len(df)) * 100
            print(f"  • {col}: {count:,} ({percentage:.2f}%)")
    
    print("\n📊 Data Types Distribution:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  • {dtype}: {count} columns")
    
    print("\n🎯 Sample Data (first 3 rows):")
    print(df.head(3).to_string())
    
    # Check for label/target column
    potential_labels = ['label', 'Label', 'class', 'Class', 'target', 'Target']
    label_col = None
    for col in potential_labels:
        if col in df.columns:
            label_col = col
            break
    
    if label_col:
        print(f"\n🏷️  Label Distribution ('{label_col}'):")
        label_counts = df[label_col].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  • {label}: {count:,} ({percentage:.2f}%)")
    
    print("\n" + "=" * 70)


def pcap_to_csv(pcap_path, output_csv_path=None):
    """
    Convert PCAP file to CSV format using pyshark
    
    This is a placeholder function for future implementation.
    Requires: pip install pyshark
    
    Args:
        pcap_path: Path to PCAP file
        output_csv_path: Path to save CSV file (optional)
        
    Returns:
        pandas.DataFrame: Extracted network traffic features
        
    Note:
        This function is not yet implemented. It serves as a placeholder
        for future PCAP parsing functionality using pyshark.
    """
    print("\n⚠️  PCAP to CSV conversion not yet implemented")
    print("=" * 70)
    print("Future Implementation Plan:")
    print("  1. Install pyshark: pip install pyshark")
    print("  2. Parse PCAP file using pyshark.FileCapture()")
    print("  3. Extract packet features:")
    print("     • Protocol type (TCP, UDP, ICMP)")
    print("     • Source/Destination IP and Port")
    print("     • Packet size and flags")
    print("     • Timestamps and inter-arrival times")
    print("     • Flow-based statistics")
    print("  4. Convert to pandas DataFrame")
    print("  5. Save to CSV if output path provided")
    print("=" * 70)
    
    # Placeholder return
    print("\n💡 For now, please use CSV datasets directly.")
    print("   Recommended datasets: NSL-KDD, CICIDS2017, UNSW-NB15")
    
    return None


def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of required column names (optional)
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if df is None or df.empty:
        raise ValueError("Dataset is empty or None")
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    print("✓ Dataset validation passed")
    return True


def get_dataset_summary(df):
    """
    Get a concise summary of the dataset
    
    Args:
        df: pandas DataFrame
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'column_names': list(df.columns),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    return summary


def main():
    """
    Main function for standalone execution
    Demonstrates CSV loading functionality
    """
    print("\n" + "=" * 70)
    print("WireXplain-IDS: Preprocessing Module")
    print("=" * 70)
    
    # Check for CSV files in data/raw
    raw_data_dir = Path(__file__).parent.parent / "data" / "raw"
    
    if not raw_data_dir.exists():
        print(f"\n⚠️  Raw data directory not found: {raw_data_dir}")
        print("Please create the directory and add your dataset.")
        return
    
    # Find CSV files
    csv_files = list(raw_data_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"\n⚠️  No CSV files found in: {raw_data_dir}")
        print("\nPlease add a CSV dataset to the data/raw/ folder.")
        print("Recommended datasets:")
        print("  • NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html")
        print("  • CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html")
        print("  • UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset")
        return
    
    # Load the first CSV file found
    csv_file = csv_files[0]
    print(f"\n📁 Found {len(csv_files)} CSV file(s) in data/raw/")
    print(f"📂 Loading: {csv_file.name}\n")
    
    try:
        # Load and display dataset info
        df = load_csv(csv_file, verbose=True)
        
        # Get summary
        summary = get_dataset_summary(df)
        
        print("\n✅ Dataset loaded successfully!")
        print(f"   Ready for preprocessing with {summary['num_rows']:,} samples")
        
        # Validate dataset
        validate_dataset(df)
        
        return df
        
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run as standalone module
    dataset = main()
