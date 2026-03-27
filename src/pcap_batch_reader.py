"""
PCAP Batch Reader Module
Reads PCAP/PCAPNG files and yields batches of packets for processing
"""

import pyshark
import logging
from pathlib import Path
from typing import Generator, List, Dict, Optional
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PCAPBatchReader:
    """
    Read PCAP files in configurable batches for memory-efficient processing
    """
    
    def __init__(
        self,
        pcap_path: str,
        batch_size: int = 100,
        batch_mode: str = 'count',
        time_window: float = 1.0,
        display_filter: Optional[str] = None
    ):
        """
        Initialize PCAP batch reader
        
        Args:
            pcap_path: Path to PCAP/PCAPNG file
            batch_size: Number of packets per batch (for 'count' mode)
            batch_mode: 'count' or 'time_window'
            time_window: Time window in seconds (for 'time_window' mode)
            display_filter: Optional Wireshark display filter
        """
        self.pcap_path = Path(pcap_path)
        self.batch_size = batch_size
        self.batch_mode = batch_mode
        self.time_window = time_window
        self.display_filter = display_filter
        
        # Progress tracking
        self.total_packets = 0
        self.total_batches = 0
        self.current_batch = 0
        
        # Validate file
        if not self.pcap_path.exists():
            raise FileNotFoundError(f"PCAP file not found: {pcap_path}")
        
        logger.info(f"Initialized PCAPBatchReader for {pcap_path}")
        logger.info(f"Batch mode: {batch_mode}, Batch size: {batch_size}")
    
    def read_batches(self) -> Generator[List[pyshark.packet.packet.Packet], None, None]:
        """
        Read PCAP file and yield batches of packets
        
        Yields:
            List of pyshark packets (batch)
        """
        try:
            # Open PCAP file
            logger.info(f"Opening PCAP file: {self.pcap_path}")
            capture = pyshark.FileCapture(
                str(self.pcap_path),
                display_filter=self.display_filter,
                keep_packets=False  # Memory efficient
            )
            
            if self.batch_mode == 'count':
                yield from self._read_by_count(capture)
            elif self.batch_mode == 'time_window':
                yield from self._read_by_time_window(capture)
            else:
                raise ValueError(f"Invalid batch_mode: {self.batch_mode}")
            
            capture.close()
            logger.info(f"Completed reading PCAP: {self.total_packets} packets in {self.total_batches} batches")
            
        except Exception as e:
            logger.error(f"Error reading PCAP file: {e}")
            raise
    
    def _read_by_count(self, capture) -> Generator[List, None, None]:
        """Read packets in batches by count"""
        batch = []
        
        for packet in capture:
            batch.append(packet)
            self.total_packets += 1
            
            if len(batch) >= self.batch_size:
                self.total_batches += 1
                self.current_batch = self.total_batches
                logger.debug(f"Yielding batch {self.total_batches} ({len(batch)} packets)")
                yield batch
                batch = []
        
        # Yield remaining packets
        if batch:
            self.total_batches += 1
            self.current_batch = self.total_batches
            logger.debug(f"Yielding final batch {self.total_batches} ({len(batch)} packets)")
            yield batch
    
    def _read_by_time_window(self, capture) -> Generator[List, None, None]:
        """Read packets in batches by time window"""
        batch = []
        window_start_time = None
        
        for packet in capture:
            try:
                # Get packet timestamp
                packet_time = float(packet.sniff_timestamp)
                
                if window_start_time is None:
                    window_start_time = packet_time
                
                # Check if packet is within current window
                if packet_time - window_start_time <= self.time_window:
                    batch.append(packet)
                    self.total_packets += 1
                else:
                    # Yield current batch and start new window
                    if batch:
                        self.total_batches += 1
                        self.current_batch = self.total_batches
                        logger.debug(f"Yielding batch {self.total_batches} ({len(batch)} packets)")
                        yield batch
                    
                    # Start new batch
                    batch = [packet]
                    window_start_time = packet_time
                    self.total_packets += 1
            
            except AttributeError:
                # Packet doesn't have timestamp, add to current batch
                batch.append(packet)
                self.total_packets += 1
        
        # Yield remaining packets
        if batch:
            self.total_batches += 1
            self.current_batch = self.total_batches
            logger.debug(f"Yielding final batch {self.total_batches} ({len(batch)} packets)")
            yield batch
    
    def get_progress(self) -> Dict[str, int]:
        """
        Get current progress metrics
        
        Returns:
            Dictionary with progress information
        """
        return {
            'total_packets': self.total_packets,
            'total_batches': self.total_batches,
            'current_batch': self.current_batch,
            'batch_size': self.batch_size
        }
    
    def estimate_total_packets(self) -> Optional[int]:
        """
        Estimate total packets in PCAP (requires full scan)
        
        Returns:
            Estimated packet count or None if unavailable
        """
        try:
            logger.info("Estimating total packets (this may take a moment)...")
            capture = pyshark.FileCapture(str(self.pcap_path), keep_packets=False)
            count = sum(1 for _ in capture)
            capture.close()
            logger.info(f"Estimated {count} total packets")
            return count
        except Exception as e:
            logger.warning(f"Could not estimate packet count: {e}")
            return None


def main():
    """Test PCAP batch reader"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PCAP Batch Reader Test")
    parser.add_argument('pcap_file', help='Path to PCAP file')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
    parser.add_argument('--mode', choices=['count', 'time_window'], default='count')
    parser.add_argument('--time-window', type=float, default=1.0, help='Time window in seconds')
    parser.add_argument('--filter', help='Wireshark display filter')
    
    args = parser.parse_args()
    
    # Create reader
    reader = PCAPBatchReader(
        pcap_path=args.pcap_file,
        batch_size=args.batch_size,
        batch_mode=args.mode,
        time_window=args.time_window,
        display_filter=args.filter
    )
    
    # Process batches
    print(f"\n{'='*60}")
    print(f"Reading PCAP: {args.pcap_file}")
    print(f"Batch mode: {args.mode}, Size: {args.batch_size}")
    print(f"{'='*60}\n")
    
    for batch_num, batch in enumerate(reader.read_batches(), 1):
        print(f"Batch {batch_num}: {len(batch)} packets")
        
        # Show first packet info
        if batch:
            pkt = batch[0]
            print(f"  First packet: {pkt.highest_layer}")
            if hasattr(pkt, 'ip'):
                print(f"    IP: {pkt.ip.src} → {pkt.ip.dst}")
    
    # Final stats
    progress = reader.get_progress()
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total packets: {progress['total_packets']}")
    print(f"Total batches: {progress['total_batches']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
