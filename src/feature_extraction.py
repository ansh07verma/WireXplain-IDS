"""
Feature Extraction Module
Extract features from network traffic data (PCAP files or flow data)
"""

import numpy as np
import pandas as pd
from pathlib import Path


class FeatureExtractor:
    """Extract features from network traffic"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_from_pcap(self, pcap_path):
        """
        Extract features from PCAP file
        
        Args:
            pcap_path: Path to PCAP file
            
        Returns:
            DataFrame with extracted features
        """
        print(f"Extracting features from {pcap_path}...")
        
        # This is a placeholder - in production, you would use scapy or pyshark
        # to parse PCAP files and extract features
        
        # Example features that would be extracted:
        # - Protocol type (TCP, UDP, ICMP)
        # - Packet size statistics
        # - Inter-arrival times
        # - Flow duration
        # - Flags (SYN, ACK, FIN, etc.)
        # - Port numbers
        # - Packet count
        # - Byte count
        
        print("⚠ PCAP parsing not yet implemented. Use CSV data instead.")
        return None
    
    def extract_from_flow(self, flow_data):
        """
        Extract features from flow-based data
        
        Args:
            flow_data: Dictionary or DataFrame with flow information
            
        Returns:
            Feature vector
        """
        features = []
        
        # Example flow features
        feature_list = [
            'duration',
            'protocol_type',
            'src_bytes',
            'dst_bytes',
            'packet_count',
            'flag',
            'src_port',
            'dst_port',
            'land',
            'wrong_fragment',
            'urgent'
        ]
        
        for feature in feature_list:
            if isinstance(flow_data, dict):
                features.append(flow_data.get(feature, 0))
            else:
                features.append(getattr(flow_data, feature, 0))
        
        return np.array(features)
    
    def get_statistical_features(self, packets):
        """
        Calculate statistical features from packet sequence
        
        Args:
            packets: List of packet information
            
        Returns:
            Dictionary of statistical features
        """
        if not packets:
            return {}
        
        sizes = [p.get('size', 0) for p in packets]
        
        features = {
            'mean_packet_size': np.mean(sizes),
            'std_packet_size': np.std(sizes),
            'min_packet_size': np.min(sizes),
            'max_packet_size': np.max(sizes),
            'total_bytes': np.sum(sizes),
            'packet_count': len(packets)
        }
        
        return features


def create_sample_features():
    """
    Create sample feature dataset for testing
    
    Returns:
        DataFrame with sample network traffic features
    """
    print("Creating sample feature dataset...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate network traffic features
    data = {
        'duration': np.random.exponential(10, n_samples),
        'protocol_type': np.random.choice([0, 1, 2], n_samples),  # TCP, UDP, ICMP
        'src_bytes': np.random.randint(0, 10000, n_samples),
        'dst_bytes': np.random.randint(0, 10000, n_samples),
        'packet_count': np.random.randint(1, 100, n_samples),
        'flag': np.random.randint(0, 11, n_samples),
        'src_port': np.random.randint(1024, 65535, n_samples),
        'dst_port': np.random.randint(1, 1024, n_samples),
        'land': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'wrong_fragment': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'urgent': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
        # Add more features as needed
    }
    
    # Create labels (0: Normal, 1: DoS, 2: Probe, 3: R2L, 4: U2R)
    labels = np.random.choice(
        ['Normal', 'DoS', 'Probe', 'R2L', 'U2R'],
        n_samples,
        p=[0.6, 0.2, 0.1, 0.05, 0.05]
    )
    
    df = pd.DataFrame(data)
    df['label'] = labels
    
    print(f"✓ Created {n_samples} sample records")
    return df


if __name__ == "__main__":
    # Example usage
    extractor = FeatureExtractor()
    
    # Create and save sample data
    sample_df = create_sample_features()
    output_path = Path("data/processed/sample_data.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(output_path, index=False)
    print(f"✓ Saved sample data to {output_path}")
