"""
PCAP Feature Extractor
Converts packet batches to CICIDS2018-compatible features
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class FlowKey:
    """Represents a network flow (5-tuple)"""
    
    def __init__(self, src_ip, dst_ip, src_port, dst_port, protocol):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
    
    def __hash__(self):
        return hash((self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol))
    
    def __eq__(self, other):
        return (self.src_ip == other.src_ip and
                self.dst_ip == other.dst_ip and
                self.src_port == other.src_port and
                self.dst_port == other.dst_port and
                self.protocol == other.protocol)
    
    def reverse(self):
        """Get reverse flow key"""
        return FlowKey(self.dst_ip, self.src_ip, self.dst_port, self.src_port, self.protocol)


class PCAPFeatureExtractor:
    """Extract CICIDS2018-style features from packet batches"""
    
    def __init__(self, flow_timeout: float = 120.0):
        """
        Initialize feature extractor
        
        Args:
            flow_timeout: Flow timeout in seconds (default: 120s)
        """
        self.flow_timeout = flow_timeout
        self.flows = {}  # Active flows
        self.flow_features = []  # Completed flow features
        
        logger.info(f"Initialized PCAPFeatureExtractor (timeout: {flow_timeout}s)")
    
    def extract_features_from_batch(self, packets: List) -> pd.DataFrame:
        """
        Extract features from a batch of packets
        
        Args:
            packets: List of pyshark packets
        
        Returns:
            DataFrame with CICIDS2018-style features
        """
        # Update flows with new packets
        for packet in packets:
            self._process_packet(packet)
        
        # Check for expired flows
        self._expire_old_flows()
        
        # Return features for completed flows
        if self.flow_features:
            df = pd.DataFrame(self.flow_features)
            self.flow_features = []  # Clear processed flows
            return df
        else:
            return pd.DataFrame()  # Empty if no completed flows
    
    def _process_packet(self, packet):
        """Process a single packet and update flow statistics"""
        try:
            # Extract basic packet info
            if not hasattr(packet, 'ip'):
                return  # Skip non-IP packets
            
            src_ip = packet.ip.src
            dst_ip = packet.ip.dst
            protocol = int(packet.ip.proto)
            timestamp = float(packet.sniff_timestamp)
            packet_length = int(packet.length)
            
            # Extract ports (TCP/UDP)
            src_port = 0
            dst_port = 0
            if hasattr(packet, 'tcp'):
                src_port = int(packet.tcp.srcport)
                dst_port = int(packet.tcp.dstport)
            elif hasattr(packet, 'udp'):
                src_port = int(packet.udp.srcport)
                dst_port = int(packet.udp.dstport)
            
            # Create flow key
            flow_key = FlowKey(src_ip, dst_ip, src_port, dst_port, protocol)
            reverse_key = flow_key.reverse()
            
            # Determine flow direction
            if flow_key in self.flows:
                flow = self.flows[flow_key]
                direction = 'forward'
            elif reverse_key in self.flows:
                flow = self.flows[reverse_key]
                flow_key = reverse_key
                direction = 'backward'
            else:
                # New flow
                flow = self._create_new_flow(flow_key, timestamp)
                self.flows[flow_key] = flow
                direction = 'forward'
            
            # Update flow statistics
            self._update_flow(flow, packet, direction, timestamp, packet_length)
            
        except Exception as e:
            logger.debug(f"Error processing packet: {e}")
    
    def _create_new_flow(self, flow_key: FlowKey, timestamp: float) -> Dict:
        """Create a new flow entry"""
        return {
            'flow_key': flow_key,
            'start_time': timestamp,
            'last_time': timestamp,
            'fwd_packets': [],
            'bwd_packets': [],
            'fwd_bytes': 0,
            'bwd_bytes': 0,
            'fwd_header_len': 0,
            'bwd_header_len': 0,
            'fwd_iats': [],
            'bwd_iats': [],
            'tcp_flags': defaultdict(int),
            'fwd_pkt_lens': [],
            'bwd_pkt_lens': [],
            'fwd_seg_sizes': [],
            'init_fwd_win_bytes': None,
            'init_bwd_win_bytes': None,
        }
    
    def _update_flow(self, flow: Dict, packet, direction: str, timestamp: float, packet_length: int):
        """Update flow statistics with new packet"""
        # Update last seen time
        flow['last_time'] = timestamp
        
        # Calculate IAT
        if direction == 'forward':
            if flow['fwd_packets']:
                iat = timestamp - flow['fwd_packets'][-1]['time']
                flow['fwd_iats'].append(iat)
            flow['fwd_packets'].append({'time': timestamp, 'length': packet_length})
            flow['fwd_bytes'] += packet_length
            flow['fwd_pkt_lens'].append(packet_length)
            
            # TCP-specific
            if hasattr(packet, 'tcp'):
                flow['fwd_header_len'] += int(packet.tcp.hdr_len) if hasattr(packet.tcp, 'hdr_len') else 20
                
                # Window size (first packet)
                if flow['init_fwd_win_bytes'] is None and hasattr(packet.tcp, 'window_size'):
                    flow['init_fwd_win_bytes'] = int(packet.tcp.window_size)
                
                # Segment size
                if hasattr(packet.tcp, 'len'):
                    flow['fwd_seg_sizes'].append(int(packet.tcp.len))
                
                # TCP flags
                if hasattr(packet.tcp, 'flags'):
                    flags = int(packet.tcp.flags, 16) if isinstance(packet.tcp.flags, str) else int(packet.tcp.flags)
                    if flags & 0x02: flow['tcp_flags']['SYN'] += 1
                    if flags & 0x10: flow['tcp_flags']['ACK'] += 1
                    if flags & 0x01: flow['tcp_flags']['FIN'] += 1
                    if flags & 0x04: flow['tcp_flags']['RST'] += 1
                    if flags & 0x08: flow['tcp_flags']['PSH'] += 1
                    if flags & 0x20: flow['tcp_flags']['URG'] += 1
        
        else:  # backward
            if flow['bwd_packets']:
                iat = timestamp - flow['bwd_packets'][-1]['time']
                flow['bwd_iats'].append(iat)
            flow['bwd_packets'].append({'time': timestamp, 'length': packet_length})
            flow['bwd_bytes'] += packet_length
            flow['bwd_pkt_lens'].append(packet_length)
            
            # TCP-specific
            if hasattr(packet, 'tcp'):
                flow['bwd_header_len'] += int(packet.tcp.hdr_len) if hasattr(packet.tcp, 'hdr_len') else 20
                
                # Window size (first packet)
                if flow['init_bwd_win_bytes'] is None and hasattr(packet.tcp, 'window_size'):
                    flow['init_bwd_win_bytes'] = int(packet.tcp.window_size)
    
    def _expire_old_flows(self):
        """Check for expired flows and compute their features"""
        current_time = max([f['last_time'] for f in self.flows.values()]) if self.flows else 0
        expired_keys = []
        
        for flow_key, flow in self.flows.items():
            if current_time - flow['last_time'] > self.flow_timeout:
                # Flow expired, compute features
                features = self._compute_flow_features(flow)
                self.flow_features.append(features)
                expired_keys.append(flow_key)
        
        # Remove expired flows
        for key in expired_keys:
            del self.flows[key]
        
        if expired_keys:
            logger.debug(f"Expired {len(expired_keys)} flows")
    
    def _compute_flow_features(self, flow: Dict) -> Dict:
        """Compute CICIDS2018 features for a completed flow"""
        features = {}
        
        # Flow duration
        duration = flow['last_time'] - flow['start_time']
        features['Flow Duration'] = int(duration * 1_000_000)  # microseconds
        
        # Packet counts
        fwd_count = len(flow['fwd_packets'])
        bwd_count = len(flow['bwd_packets'])
        total_count = fwd_count + bwd_count
        
        features['Tot Fwd Pkts'] = fwd_count
        features['Tot Bwd Pkts'] = bwd_count
        
        # Byte counts
        features['TotLen Fwd Pkts'] = flow['fwd_bytes']
        features['TotLen Bwd Pkts'] = flow['bwd_bytes']
        
        # Packet length statistics
        if flow['fwd_pkt_lens']:
            features['Fwd Pkt Len Max'] = max(flow['fwd_pkt_lens'])
            features['Fwd Pkt Len Min'] = min(flow['fwd_pkt_lens'])
            features['Fwd Pkt Len Mean'] = np.mean(flow['fwd_pkt_lens'])
            features['Fwd Pkt Len Std'] = np.std(flow['fwd_pkt_lens'])
        else:
            features['Fwd Pkt Len Max'] = 0
            features['Fwd Pkt Len Min'] = 0
            features['Fwd Pkt Len Mean'] = 0
            features['Fwd Pkt Len Std'] = 0
        
        if flow['bwd_pkt_lens']:
            features['Bwd Pkt Len Max'] = max(flow['bwd_pkt_lens'])
            features['Bwd Pkt Len Min'] = min(flow['bwd_pkt_lens'])
            features['Bwd Pkt Len Mean'] = np.mean(flow['bwd_pkt_lens'])
            features['Bwd Pkt Len Std'] = np.std(flow['bwd_pkt_lens'])
        else:
            features['Bwd Pkt Len Max'] = 0
            features['Bwd Pkt Len Min'] = 0
            features['Bwd Pkt Len Mean'] = 0
            features['Bwd Pkt Len Std'] = 0
        
        # Flow IAT statistics
        all_iats = flow['fwd_iats'] + flow['bwd_iats']
        if all_iats:
            features['Flow IAT Mean'] = np.mean(all_iats) * 1_000_000  # microseconds
            features['Flow IAT Std'] = np.std(all_iats) * 1_000_000
            features['Flow IAT Max'] = max(all_iats) * 1_000_000
            features['Flow IAT Min'] = min(all_iats) * 1_000_000
        else:
            features['Flow IAT Mean'] = 0
            features['Flow IAT Std'] = 0
            features['Flow IAT Max'] = 0
            features['Flow IAT Min'] = 0
        
        # Forward IAT
        if flow['fwd_iats']:
            features['Fwd IAT Tot'] = sum(flow['fwd_iats']) * 1_000_000
            features['Fwd IAT Mean'] = np.mean(flow['fwd_iats']) * 1_000_000
            features['Fwd IAT Std'] = np.std(flow['fwd_iats']) * 1_000_000
            features['Fwd IAT Max'] = max(flow['fwd_iats']) * 1_000_000
            features['Fwd IAT Min'] = min(flow['fwd_iats']) * 1_000_000
        else:
            features['Fwd IAT Tot'] = 0
            features['Fwd IAT Mean'] = 0
            features['Fwd IAT Std'] = 0
            features['Fwd IAT Max'] = 0
            features['Fwd IAT Min'] = 0
        
        # Backward IAT
        if flow['bwd_iats']:
            features['Bwd IAT Tot'] = sum(flow['bwd_iats']) * 1_000_000
            features['Bwd IAT Mean'] = np.mean(flow['bwd_iats']) * 1_000_000
            features['Bwd IAT Std'] = np.std(flow['bwd_iats']) * 1_000_000
            features['Bwd IAT Max'] = max(flow['bwd_iats']) * 1_000_000
            features['Bwd IAT Min'] = min(flow['bwd_iats']) * 1_000_000
        else:
            features['Bwd IAT Tot'] = 0
            features['Bwd IAT Mean'] = 0
            features['Bwd IAT Std'] = 0
            features['Bwd IAT Max'] = 0
            features['Bwd IAT Min'] = 0
        
        # TCP flags
        features['FIN Flag Cnt'] = flow['tcp_flags']['FIN']
        features['SYN Flag Cnt'] = flow['tcp_flags']['SYN']
        features['RST Flag Cnt'] = flow['tcp_flags']['RST']
        features['PSH Flag Cnt'] = flow['tcp_flags']['PSH']
        features['ACK Flag Cnt'] = flow['tcp_flags']['ACK']
        features['URG Flag Cnt'] = flow['tcp_flags']['URG']
        
        # Header lengths
        features['Fwd Header Len'] = flow['fwd_header_len']
        features['Bwd Header Len'] = flow['bwd_header_len']
        
        # Packets per second
        if duration > 0:
            features['Fwd Pkts/s'] = fwd_count / duration
            features['Bwd Pkts/s'] = bwd_count / duration
            features['Flow Pkts/s'] = total_count / duration
            features['Flow Byts/s'] = (flow['fwd_bytes'] + flow['bwd_bytes']) / duration
        else:
            features['Fwd Pkts/s'] = 0
            features['Bwd Pkts/s'] = 0
            features['Flow Pkts/s'] = 0
            features['Flow Byts/s'] = 0
        
        # Segment sizes
        if flow['fwd_seg_sizes']:
            features['Fwd Seg Size Min'] = min(flow['fwd_seg_sizes'])
            features['Fwd Seg Size Avg'] = np.mean(flow['fwd_seg_sizes'])
        else:
            features['Fwd Seg Size Min'] = 0
            features['Fwd Seg Size Avg'] = 0
        
        # Window sizes
        features['Init Fwd Win Byts'] = flow['init_fwd_win_bytes'] if flow['init_fwd_win_bytes'] is not None else 0
        features['Init Bwd Win Byts'] = flow['init_bwd_win_bytes'] if flow['init_bwd_win_bytes'] is not None else 0
        
        # Down/Up Ratio
        if flow['fwd_bytes'] > 0:
            features['Down/Up Ratio'] = flow['bwd_bytes'] / flow['fwd_bytes']
        else:
            features['Down/Up Ratio'] = 0
        
        # Average packet size
        if total_count > 0:
            features['Pkt Size Avg'] = (flow['fwd_bytes'] + flow['bwd_bytes']) / total_count
        else:
            features['Pkt Size Avg'] = 0
        
        # Subflow statistics (simplified - treating entire flow as one subflow)
        features['Subflow Fwd Pkts'] = fwd_count
        features['Subflow Fwd Byts'] = flow['fwd_bytes']
        features['Subflow Bwd Pkts'] = bwd_count
        features['Subflow Bwd Byts'] = flow['bwd_bytes']
        
        # Active/Idle times (simplified)
        features['Active Mean'] = duration * 1_000_000 if total_count > 0 else 0
        features['Active Std'] = 0
        features['Active Max'] = duration * 1_000_000 if total_count > 0 else 0
        features['Active Min'] = duration * 1_000_000 if total_count > 0 else 0
        features['Idle Mean'] = 0
        features['Idle Std'] = 0
        features['Idle Max'] = 0
        features['Idle Min'] = 0
        
        return features
    
    def finalize(self) -> pd.DataFrame:
        """
        Finalize all remaining flows and return features
        
        Returns:
            DataFrame with all flow features
        """
        # Process all remaining flows
        for flow in self.flows.values():
            features = self._compute_flow_features(flow)
            self.flow_features.append(features)
        
        self.flows.clear()
        
        if self.flow_features:
            df = pd.DataFrame(self.flow_features)
            self.flow_features = []
            return df
        else:
            return pd.DataFrame()


def main():
    """Test feature extractor"""
    print("PCAP Feature Extractor - Test Mode")
    print("Use with pcap_batch_reader.py for full functionality")


if __name__ == "__main__":
    main()
