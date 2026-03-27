# PCAP Live Replay - User Guide

## 🎯 Overview

The PCAP Live Replay feature allows you to upload network capture files and watch intrusion detection happen in near real-time, simulating live network monitoring.

---

## 🚀 Quick Start

### 1. Access the Feature
- Open the Streamlit app: `streamlit run app.py`
- Navigate to the **"🔴 Live PCAP Replay"** tab

### 2. Upload PCAP File
- Click "Upload PCAP File"
- Select a `.pcap` or `.pcapng` file
- Supported: Wireshark captures, tcpdump output, etc.

### 3. Configure Batch Size
- **Default**: 100 packets per batch
- **Range**: 10-1000 packets
- **Recommendation**: 
  - Small files (< 10K packets): 50-100
  - Medium files (10K-100K packets): 100-200
  - Large files (> 100K packets): 200-500

### 4. Start Replay
- Click **"🚀 Start Replay"**
- Watch the magic happen!

---

## 📊 What You'll See

### Real-Time Metrics
- **Total Flows**: Number of network flows detected
- **Normal**: Count of benign traffic
- **Attacks**: Count of detected attacks
- **Avg Confidence**: Average prediction confidence

### Live Prediction Table
- Shows last 50 flows processed
- Updates incrementally as batches complete
- Columns:
  - Flow ID
  - Prediction (Normal / 🚨 Attack)
  - Confidence
  - Attack Probability

### Progress Tracking
- Progress bar showing overall completion
- Status text with current batch info
- Packet and flow counters

---

## 🔬 How It Works

### Step 1: Packet Reading
- Pyshark reads packets from PCAP file
- Packets grouped into configurable batches
- Memory-efficient streaming (doesn't load entire file)

### Step 2: Flow Aggregation
- Packets grouped by 5-tuple:
  - Source IP
  - Destination IP
  - Source Port
  - Destination Port
  - Protocol
- Flows tracked with 120-second timeout

### Step 3: Feature Extraction
- Statistical features computed per flow:
  - Packet counts (forward/backward)
  - Byte counts
  - Inter-arrival times (IAT)
  - TCP flags
  - Window sizes
  - Header lengths
  - Packet length statistics
- **50+ CICIDS2018-compatible features**

### Step 4: Feature Engineering
- Same pipeline as CSV mode:
  - Categorical encoding
  - Feature engineering
  - Top 15 feature selection

### Step 5: Prediction
- RandomForest model makes predictions
- Confidence scores computed
- Results displayed in real-time

---

## 💡 Use Cases

### 1. Network Traffic Analysis
- Upload captured traffic from your network
- Identify suspicious flows
- Investigate attack patterns

### 2. Demonstration
- Show IDS capabilities in action
- Visual, engaging presentation
- Real-time feedback

### 3. Testing
- Test model on new PCAP files
- Validate detection accuracy
- Compare with known labels

### 4. Research
- Analyze attack datasets
- Study flow characteristics
- Export results for further analysis

---

## 📥 Export Results

After replay completes:
1. Click **"📥 Download PCAP Replay Results"**
2. CSV file contains:
   - Flow ID
   - Prediction
   - Confidence
   - Attack Probability
3. Use for reporting, analysis, or archiving

---

## 🔍 SHAP Integration

After PCAP replay:
1. Go to **"🔍 SHAP Explanations"** tab
2. Select any flow ID from the replay
3. Generate SHAP explanation
4. See which features influenced the prediction

**This works seamlessly** - PCAP flows are stored in session state just like CSV samples!

---

## ⚡ Performance Tips

### Batch Size Selection
- **Smaller batches (10-50)**:
  - ✅ More frequent updates
  - ✅ Better visual effect
  - ❌ Slower overall processing
  
- **Medium batches (100-200)**:
  - ✅ Balanced performance
  - ✅ Good update frequency
  - ✅ **Recommended for most cases**
  
- **Larger batches (500-1000)**:
  - ✅ Faster processing
  - ❌ Less frequent updates
  - ❌ May miss visual effect

### File Size Considerations
- **Small files (< 1 MB)**: Process quickly, any batch size works
- **Medium files (1-100 MB)**: Use 100-200 packet batches
- **Large files (> 100 MB)**: Use 200-500 packet batches, be patient

### Expected Processing Times
- **1,000 packets**: ~5-10 seconds
- **10,000 packets**: ~30-60 seconds
- **100,000 packets**: ~5-10 minutes
- **1,000,000 packets**: ~30-60 minutes

*Times vary based on flow complexity and system performance*

---

## 🐛 Troubleshooting

### Error: "Failed to load model"
**Solution**: Ensure `models/random_forest_model.pkl` exists
```bash
# Train model first
python main.py
```

### Error: "No module named 'pyshark'"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### No flows detected
**Possible causes**:
- PCAP contains only non-IP traffic
- All packets filtered out
- Corrupted PCAP file

**Solution**: Verify PCAP with Wireshark first

### Slow processing
**Solutions**:
- Increase batch size
- Use smaller PCAP file for testing
- Close other applications

---

## 📝 Example Workflow

```bash
# 1. Capture network traffic
sudo tcpdump -i eth0 -w capture.pcap

# 2. Start Streamlit app
streamlit run app.py

# 3. In browser:
#    - Go to "Live PCAP Replay" tab
#    - Upload capture.pcap
#    - Set batch size: 100
#    - Click "Start Replay"
#    - Watch real-time detection!

# 4. After completion:
#    - Download results CSV
#    - Go to SHAP tab
#    - Explain suspicious flows
```

---

## 🎓 Technical Details

### Supported Protocols
- TCP
- UDP
- ICMP (limited features)
- Other IP protocols (basic features)

### Flow Timeout
- **Default**: 120 seconds
- Flows inactive for > 120s are finalized
- Configurable in `pcap_feature_extractor.py`

### Feature Compatibility
- Extracts 50+ CICIDS2018 features
- Compatible with existing ML pipeline
- Same 15 features selected for prediction

### Memory Usage
- Batch processing keeps memory low
- Doesn't load entire PCAP into RAM
- Suitable for large files

---

## 🔗 Related Features

- **CSV Upload**: For pre-processed datasets
- **Results Analysis**: Detailed metrics and charts
- **SHAP Explanations**: Understand predictions

---

## 📚 Additional Resources

- [Wireshark](https://www.wireshark.org/) - Capture and analyze PCAP files
- [tcpdump](https://www.tcpdump.org/) - Command-line packet capture
- [CICIDS2018 Dataset](https://www.unb.ca/cic/datasets/ids-2018.html) - Reference dataset

---

**Happy network monitoring!** 🛡️
