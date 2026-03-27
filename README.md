# WireXplain-IDS: Explainable Intrusion Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready, explainable intrusion detection system leveraging machine learning and SHAP (SHapley Additive exPlanations) for interpretable network security analysis.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [System Architecture](#system-architecture)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

WireXplain-IDS is an advanced intrusion detection system that combines state-of-the-art machine learning techniques with explainable AI to provide both accurate threat detection and interpretable explanations. The system processes network traffic data through a sophisticated pipeline encompassing feature engineering, intelligent feature selection, anomaly detection, and ensemble learning, culminating in comprehensive SHAP-based explanations.

### Key Features

- **High Accuracy Detection**: Achieves 100% accuracy on CICIDS2018 dataset using RandomForest classifier
- **Explainable AI**: SHAP TreeExplainer provides global and local feature importance explanations
- **Intelligent Feature Selection**: Mutual Information-based feature ranking reduces dimensionality by 81%
- **Anomaly Detection**: IsolationForest identifies outliers and rare attack patterns
- **End-to-End Pipeline**: Single-command execution from raw data to interpretable predictions
- **Publication-Ready Visualizations**: 11 high-resolution plots for model evaluation and interpretation

---

## Motivation

### The Explainability Gap in Network Security

Traditional intrusion detection systems often operate as "black boxes," providing predictions without justification. This lack of transparency poses significant challenges:

1. **Trust**: Security analysts cannot validate detection decisions
2. **Debugging**: Difficult to identify false positives and model weaknesses
3. **Compliance**: Regulatory frameworks increasingly require explainable AI
4. **Knowledge Transfer**: Domain expertise cannot be extracted from opaque models

### Our Solution

WireXplain-IDS addresses these challenges by integrating SHAP explanations throughout the detection pipeline, enabling:

- **Feature Attribution**: Understand which network traffic characteristics drive predictions
- **Model Validation**: Verify that the model learns meaningful security patterns
- **Incident Investigation**: Trace detection decisions to specific traffic features
- **Continuous Improvement**: Identify and refine feature engineering strategies

---

## System Architecture

### Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WireXplain-IDS Pipeline                      │
└─────────────────────────────────────────────────────────────────────┘

Raw Data (CICIDS2018)
    │ 1,048,575 samples × 80 features
    ▼
┌─────────────────────────────────────┐
│  Stage 1: Feature Engineering       │
│  • Binary label encoding             │
│  • Categorical feature encoding      │
│  • Derived feature creation          │
│  • Missing value imputation          │
└─────────────────────────────────────┘
    │ features.csv (390 MB)
    ▼
┌─────────────────────────────────────┐
│  Stage 2: Feature Selection          │
│  • Mutual Information scoring        │
│  • Top-N feature ranking             │
│  • Dimensionality reduction (81%)    │
└─────────────────────────────────────┘
    │ selected_features.csv (118 MB)
    │ 15 features selected
    ▼
┌─────────────────────────────────────┐
│  Stage 3: Isolation Filtering        │
│  • IsolationForest anomaly detection │
│  • Outlier flagging (2.31%)          │
│  • Data quality enhancement          │
└─────────────────────────────────────┘
    │ filtered_data.csv (120 MB)
    ▼
┌─────────────────────────────────────┐
│  Stage 4: Model Training              │
│  • RandomForest (100 estimators)     │
│  • 80/20 train-test split            │
│  • Hyperparameter optimization       │
└─────────────────────────────────────┘
    │ random_forest_model.pkl (0.91 MB)
    │ Accuracy: 100%
    ▼
┌─────────────────────────────────────┐
│  Stage 5: SHAP Explainability         │
│  • TreeExplainer initialization      │
│  • Global feature importance         │
│  • Local prediction explanations     │
└─────────────────────────────────────┘
    │ 6 SHAP plots (outputs/)
    ▼
┌─────────────────────────────────────┐
│  Visualization Suite                  │
│  • Confusion matrices                │
│  • Feature importance plots          │
│  • Performance metrics (ROC, etc.)   │
│  • Summary dashboard                 │
└─────────────────────────────────────┘
    │ 11 publication-ready plots
    ▼
  Interpretable Predictions
```

---

## Methodology

### 1. Feature Engineering

**Objective**: Transform raw network traffic data into machine learning-ready features.

**Process**:
- **Label Encoding**: Convert multi-class labels (Benign, FTP-BruteForce, SSH-Bruteforce) to binary (0=Normal, 1=Attack)
- **Categorical Encoding**: One-hot encoding for protocol types and flags
- **Feature Derivation**: Create domain-specific features:
  - `fwd_packet_rate = Tot Fwd Pkts / Flow Duration`
  - `bwd_packet_rate = Tot Bwd Pkts / Flow Duration`
  - `fwd_byte_ratio = TotLen Fwd Pkts / (TotLen Fwd Pkts + TotLen Bwd Pkts)`
  - `syn_ratio = SYN Flag Cnt / Tot Fwd Pkts`

**Output**: 80 engineered features capturing temporal, statistical, and protocol-level characteristics.

---

### 2. Mutual Information Feature Selection

**Objective**: Identify the most informative features for attack detection.

**Theory**: Mutual Information (MI) measures the mutual dependence between feature X and label Y:

```
I(X; Y) = ∑∑ p(x,y) log(p(x,y) / (p(x)p(y)))
```

Where:
- High MI indicates strong dependency (feature is informative)
- MI = 0 indicates independence (feature is uninformative)

**Implementation**:
- Compute MI scores using `sklearn.feature_selection.mutual_info_classif`
- Rank features by MI score
- Select top N features (default: 15)

**Advantages**:
- Captures non-linear relationships
- Model-agnostic (works with any classifier)
- Reduces overfitting by eliminating redundant features

**Results**: 81.2% dimensionality reduction (80 → 15 features) while preserving discriminative power.

---

### 3. Isolation Forest Anomaly Detection

**Objective**: Identify rare or anomalous samples that may represent novel attacks or data quality issues.

**Theory**: IsolationForest exploits the principle that anomalies are "few and different":

1. Randomly select a feature and split value
2. Recursively partition data until samples are isolated
3. Anomalies require fewer splits (shorter path length)

**Anomaly Score**:
```
s(x, n) = 2^(-E(h(x)) / c(n))
```

Where:
- `h(x)` = path length for sample x
- `c(n)` = average path length for n samples
- Score ≈ 1: anomaly
- Score ≈ 0.5: normal

**Implementation**:
- Contamination rate: 5% (expected outlier proportion)
- 100 isolation trees
- Mode: "flag" (retain all samples, mark anomalies)

**Results**: Flagged 24,233 samples (2.31%) as anomalies for further investigation.

---

### 4. Random Forest Classification

**Objective**: Build an ensemble classifier for robust attack detection.

**Theory**: RandomForest aggregates predictions from multiple decision trees:

```
ŷ = mode{h₁(x), h₂(x), ..., hₙ(x)}
```

Where each tree `hᵢ` is trained on:
- Bootstrap sample (random subset with replacement)
- Random feature subset at each split

**Advantages**:
- Resistant to overfitting
- Handles non-linear relationships
- Provides feature importance via Gini impurity
- Naturally handles class imbalance

**Hyperparameters**:
- Number of estimators: 100
- Max depth: None (fully grown trees)
- Min samples split: 2
- Bootstrap: True

**Results**: 100% accuracy on 209,715 test samples (only 3 errors).

---

### 5. SHAP Explainability

**Objective**: Provide interpretable explanations for model predictions.

**Theory**: SHAP (SHapley Additive exPlanations) assigns each feature an importance value for a particular prediction based on cooperative game theory.

**Shapley Value**:
```
φᵢ = ∑ |S|!(M-|S|-1)! / M! × [f(S∪{i}) - f(S)]
      S⊆M\{i}
```

Where:
- `φᵢ` = SHAP value for feature i
- `S` = subset of features
- `M` = all features
- `f(S)` = model prediction using feature subset S

**TreeExplainer**: Optimized SHAP computation for tree-based models (polynomial time vs. exponential).

**Visualizations**:
1. **Global Importance**: Summary plot showing feature impact across all samples
2. **Local Explanations**: Bar plots showing feature contributions for individual predictions

**Interpretation**:
- Positive SHAP value → feature increases attack probability
- Negative SHAP value → feature decreases attack probability
- Magnitude → strength of contribution

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (for processing large datasets)
- 2GB+ disk space (for datasets and models)

### Setup Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/ansh07verma/WireXplain-IDS.git
cd WireXplain-IDS
```

2. **Create virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

**Required packages**:
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `scikit-learn>=1.0.0` - Machine learning algorithms
- `matplotlib>=3.4.0` - Plotting
- `seaborn>=0.11.0` - Statistical visualizations
- `shap>=0.50.0` - SHAP explanations
- `joblib>=1.1.0` - Model serialization

4. **Download dataset** (CICIDS2018):
```bash
# Place your dataset in data/raw/
# Example: data/raw/02-14-2018.csv
```

---

## Usage

### Quick Start: Run Complete Pipeline

Execute the entire pipeline with a single command:

```bash
python main.py
```

This will:
1. Load and engineer features from raw data
2. Select top 15 features using Mutual Information
3. Apply IsolationForest anomaly detection
4. Train RandomForest classifier
5. Generate SHAP explanations
6. Save all outputs to respective directories

**Expected runtime**: ~5-10 minutes (depending on hardware)

---

### Advanced Usage

#### Custom Pipeline Configuration

```bash
python main.py \
  --raw-data data/raw/02-14-2018.csv \
  --top-n 20 \
  --contamination 0.03 \
  --n-estimators 200 \
  --test-size 0.3 \
  --shap-samples 2000 \
  --n-local 10
```

**Parameters**:
- `--raw-data`: Path to input CSV file
- `--top-n`: Number of features to select (default: 15)
- `--contamination`: Expected outlier proportion (default: 0.05)
- `--filter-mode`: Outlier handling ('flag' or 'filter')
- `--n-estimators`: Number of trees in RandomForest (default: 100)
- `--test-size`: Test set proportion (default: 0.2)
- `--exclude-anomalies`: Exclude flagged anomalies from training
- `--shap-samples`: Samples for SHAP computation (default: 1000)
- `--n-local`: Number of local explanations (default: 4)
- `--output-dir`: Directory for output plots (default: outputs)

---

#### Individual Module Execution

**Feature Engineering**:
```bash
python src/feature_engineering.py
```

**Feature Selection**:
```bash
python src/feature_selection.py --top-n 15
```

**Isolation Filtering**:
```bash
python src/isolation_filter.py --mode flag --contamination 0.05
```

**Model Training**:
```bash
python src/train_model.py --n-estimators 100 --test-size 0.2
```

**SHAP Explanations**:
```bash
python src/explain.py --sample-size 1000 --n-local 4
```

**Visualizations**:
```bash
python src/visualize.py
```

---

## Project Structure

```
WireXplain-IDS/
│
├── data/
│   ├── raw/                          # Raw CICIDS2018 dataset
│   │   └── 02-14-2018.csv           # Example: 1M+ network flows
│   └── processed/                    # Processed datasets
│       ├── features.csv              # Engineered features (390 MB)
│       ├── selected_features.csv     # Top-N features (118 MB)
│       └── filtered_data.csv         # Anomaly-filtered data (120 MB)
│
├── src/                              # Source code modules
│   ├── preprocess.py                 # Data loading and validation
│   ├── feature_engineering.py        # Feature creation and encoding
│   ├── feature_selection.py          # Mutual Information selection
│   ├── isolation_filter.py           # IsolationForest anomaly detection
│   ├── train_model.py                # RandomForest training
│   ├── explain.py                    # SHAP explainability
│   └── visualize.py                  # Visualization suite
│
├── models/                           # Trained models
│   └── random_forest_model.pkl       # Serialized RandomForest (0.91 MB)
│
├── outputs/                          # Generated visualizations
│   ├── confusion_matrix.png          # Standard confusion matrix
│   ├── confusion_matrix_normalized.png
│   ├── feature_importance.png        # Gini importance (top 15)
│   ├── performance_metrics.png       # Metrics + ROC curve
│   ├── summary_dashboard.png         # Comprehensive dashboard
│   ├── global_feature_importance.png # SHAP summary plot
│   ├── global_feature_importance_bar.png
│   └── local_explanation_sample_*.png  # Individual predictions (4)
│
├── logs/                             # Execution logs
│   ├── pipeline.log                  # Main pipeline log
│   ├── feature_engineering.log
│   ├── feature_selection.log
│   ├── isolation_filter.log
│   ├── train_model.log
│   ├── explain.log
│   └── visualize.log
│
├── docs/                             # Documentation
│   └── preprocess_implementation.md
│
├── main.py                           # Pipeline orchestrator
├── requirements.txt                  # Python dependencies
├── requirements-minimal.txt          # Minimal dependencies
└── README.md                         # This file
```

---

## Results

### Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **100.00%** |
| Precision | 100.00% |
| Recall | 100.00% |
| F1-Score | 100.00% |
| ROC-AUC | 1.0000 |

**Test Set**: 209,715 samples (20% of dataset)  
**Errors**: 3 total (1 false positive, 2 false negatives)

---

### Top Features (SHAP Importance)

| Rank | Feature | SHAP Score | Description |
|------|---------|------------|-------------|
| 1 | Init Fwd Win Byts | 0.7130 | Initial forward window size |
| 2 | Fwd Seg Size Min | 0.6945 | Minimum forward segment size |
| 3 | bwd_packet_rate | 0.5885 | Backward packets per second (engineered) |
| 4 | Bwd Pkts/s | 0.5855 | Backward packet rate |
| 5 | fwd_packet_rate | 0.5775 | Forward packets per second (engineered) |

**Key Insights**:
- Window size and segment characteristics are strongest attack indicators
- Engineered packet rate features rank in top 5
- Temporal features dominate over payload-based features

---

### Computational Efficiency

| Stage | Runtime | Memory |
|-------|---------|--------|
| Feature Engineering | ~15s | 2.1 GB |
| Feature Selection | ~2m 45s | 1.8 GB |
| Isolation Filtering | ~8s | 1.5 GB |
| Model Training | ~12s | 1.2 GB |
| SHAP Explanation | ~30s | 0.8 GB |
| Visualization | ~5s | 0.5 GB |
| **Total Pipeline** | **~4m 30s** | **2.1 GB peak** |

*Benchmarked on: Intel i7-9750H, 16GB RAM, macOS*

---

## Dataset

### CICIDS2018

**Source**: Canadian Institute for Cybersecurity  
**File**: `02-14-2018.csv`  
**Size**: 1,048,575 network flows  
**Features**: 80 statistical features  
**Labels**: Benign, FTP-BruteForce, SSH-Bruteforce

**Download**: [CICIDS2018 Dataset](https://www.unb.ca/cic/datasets/ids-2018.html)

**Preprocessing**:
- Remove duplicate flows
- Handle missing values (forward fill)
- Normalize infinite values
- Encode categorical labels

---

## Extending the System

### Adding New Features

Edit `src/feature_engineering.py`:

```python
def engineer_features(self, df):
    # Add your custom feature
    df['custom_feature'] = df['feature1'] / (df['feature2'] + 1e-6)
    return df
```

### Using Different Datasets

1. Place CSV in `data/raw/`
2. Ensure label column exists
3. Update `--raw-data` parameter:

```bash
python main.py --raw-data data/raw/your_dataset.csv
```

### Hyperparameter Tuning

Modify `main.py` configuration or use CLI arguments:

```bash
python main.py --n-estimators 200 --top-n 20 --contamination 0.03
```

---

## Troubleshooting

### Common Issues

**1. Memory Error during Feature Selection**
```bash
# Reduce sample size or use chunking
python src/feature_selection.py --sample-size 100000
```

**2. SHAP Computation Timeout**
```bash
# Reduce SHAP sample size
python src/explain.py --sample-size 500
```

**3. Module Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

**Areas for Contribution**:
- Additional feature engineering techniques
- Alternative explainability methods (LIME, Integrated Gradients)
- Real-time packet capture integration
- Multi-class attack classification
- Deep learning models (LSTM, CNN)

---

## Citation

If you use WireXplain-IDS in your research, please cite:

```bibtex
@software{wirexplain_ids_2026,
  title={WireXplain-IDS: Explainable Intrusion Detection System},
  author={Your Name},
  year={2026},
  url={https://github.com/ansh07verma/WireXplain-IDS.git}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **CICIDS2018 Dataset**: Canadian Institute for Cybersecurity
- **SHAP Library**: Scott Lundberg and the SHAP team
- **scikit-learn**: Pedregosa et al., JMLR 2011

---

## Contact

**Project Maintainer**: Ansh Verma
**Email**: 07anshverma@gmail.com  
**GitHub**: [@yourusername](https://github.com/ansh07verma)

For bug reports and feature requests, please open an issue on GitHub.

---

**Last Updated**: March 2026  
**Version**: 1.0.0
