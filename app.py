"""
WireXplain-IDS: Streamlit Web Application
Interactive interface for intrusion detection with SHAP explanations
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import time

# Page configuration
st.set_page_config(
    page_title="WireXplain-IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    """Load the trained RandomForest model"""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def create_shap_explainer(_model):
    """Create SHAP explainer (cached for performance)"""
    try:
        explainer = shap.TreeExplainer(_model)
        return explainer
    except Exception as e:
        st.error(f"Error creating SHAP explainer: {e}")
        return None


def process_raw_csv(df):
    """Process raw CSV through feature engineering pipeline"""
    try:
        st.info("📊 Processing raw CSV through feature engineering pipeline...")
        
        # Import feature engineering module
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / 'src'))
        from feature_engineering import FeatureEngineer
        
        with st.spinner("Running feature engineering..."):
            fe = FeatureEngineer()
            
            # Validate and encode labels if present
            if 'Label' in df.columns:
                fe.validate_label_column(df)
                df = fe.encode_labels(df)
            
            # Encode categorical features
            df = fe.encode_categorical_features(df)
            
            # Engineer features
            df = fe.engineer_features(df)
            
            # Select features
            features = fe.select_features(df)
            
            st.success("✅ Feature engineering complete!")
            
            return features
            
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None


def process_pcap_file(uploaded_file):
    """Process PCAP file and extract features"""
    try:
        st.info("📦 Processing PCAP file... This may take a few minutes.")
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pcap') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        st.warning("⚠️ PCAP processing requires CICFlowMeter or similar tool.")
        st.info("For now, please convert PCAP to CSV using CICFlowMeter first.")
        st.markdown("""
        **Steps to convert PCAP to CSV:**
        1. Install CICFlowMeter: https://github.com/ahlashkari/CICFlowMeter
        2. Run: `cicflowmeter -f your_file.pcap -c output.csv`
        3. Upload the generated CSV file
        """)
        
        # Clean up
        import os
        os.unlink(tmp_path)
        
        return None
        
    except Exception as e:
        st.error(f"Error processing PCAP: {e}")
        return None


def validate_dataset(df, expected_features):
    """Validate uploaded dataset has correct features"""
    # Remove label columns if present
    feature_cols = [col for col in df.columns if col not in ['label', 'is_anomaly', 'Label', 'label_binary']]
    
    # Check if all expected features are present
    missing_features = set(expected_features) - set(feature_cols)
    extra_features = set(feature_cols) - set(expected_features)
    
    if missing_features:
        st.warning(f"⚠️ Dataset missing required features. Running preprocessing pipeline...")
        return None  # Signal to run preprocessing
    
    if extra_features:
        st.info(f"ℹ️ Extra features will be ignored: {', '.join(list(extra_features)[:5])}...")
    
    # Return only expected features in correct order
    return df[expected_features]


def predict_with_confidence(model, X):
    """Make predictions with confidence scores"""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Get confidence (max probability)
    confidence = np.max(probabilities, axis=1)
    
    # Get attack probability
    attack_prob = probabilities[:, 1]
    
    return predictions, confidence, attack_prob


def plot_prediction_distribution(predictions):
    """Create interactive pie chart of prediction distribution"""
    pred_counts = pd.Series(predictions).value_counts()
    labels = ['Normal' if x == 0 else 'Attack' for x in pred_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=pred_counts.values,
        hole=0.3,
        marker=dict(colors=['#2ecc71', '#e74c3c']),
        textinfo='label+percent+value',
        textfont_size=14
    )])
    
    fig.update_layout(
        title="Prediction Distribution",
        height=400,
        showlegend=True
    )
    
    return fig


def plot_confidence_distribution(confidence, predictions):
    """Create histogram of prediction confidence scores"""
    df_conf = pd.DataFrame({
        'Confidence': confidence,
        'Prediction': ['Normal' if p == 0 else 'Attack' for p in predictions]
    })
    
    fig = px.histogram(
        df_conf,
        x='Confidence',
        color='Prediction',
        nbins=30,
        title='Prediction Confidence Distribution',
        labels={'Confidence': 'Confidence Score', 'count': 'Number of Samples'},
        color_discrete_map={'Normal': '#2ecc71', 'Attack': '#e74c3c'},
        barmode='overlay',
        opacity=0.7
    )
    
    fig.update_layout(height=400)
    
    return fig


def plot_shap_explanation(explainer, X, sample_idx, feature_names):
    """Generate SHAP explanation plot for a specific sample"""
    try:
        # Compute SHAP values for the sample
        shap_values = explainer.shap_values(X.iloc[[sample_idx]])
        
        # Handle binary classification
        # shap_values is a list: [normal_class_values, attack_class_values]
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # Use attack class (index 1) for explanation
            shap_values = shap_values[1]
        
        # Now shap_values should be 2D: (1, n_features) or 1D: (n_features,)
        if isinstance(shap_values, np.ndarray):
            # Flatten to 1D if needed
            shap_values = shap_values.flatten()
            
            # If we still have double the features (concatenated), take first half
            if len(shap_values) == 2 * len(feature_names):
                shap_values = shap_values[:len(feature_names)]
        
        # Final validation
        if len(shap_values) != len(feature_names):
            st.error(f"❌ SHAP values length ({len(shap_values)}) doesn't match features ({len(feature_names)})")
            st.info("This might be due to model/data mismatch. Please ensure you're using the correct model and dataset.")
            return None
        
        # Create DataFrame for plotting
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values
        })
        shap_df['abs_shap'] = shap_df['shap_value'].abs()
        shap_df = shap_df.sort_values('abs_shap', ascending=True).tail(10)
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in shap_df['shap_value']]
        ax.barh(range(len(shap_df)), shap_df['shap_value'], color=colors)
        ax.set_yticks(range(len(shap_df)))
        ax.set_yticklabels(shap_df['feature'])
        ax.set_xlabel('SHAP Value (impact on prediction)', fontsize=12)
        ax.set_title('Top 10 Feature Contributions to Prediction', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Increases Attack Probability'),
            Patch(facecolor='#e74c3c', label='Decreases Attack Probability')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"❌ Error generating SHAP plot: {e}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())
        return None


def main():
    # Header
    st.markdown('<div class="main-header">🛡️ WireXplain-IDS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explainable Intrusion Detection System</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model_path = st.text_input(
            "Model Path",
            value="models/random_forest_model.pkl",
            help="Path to the trained RandomForest model"
        )
        
        st.markdown("---")
        st.markdown("### 📖 Instructions")
        st.markdown("""
        1. **Upload Dataset**: Upload a CSV file with network traffic features
        2. **View Predictions**: See classification results and statistics
        3. **Explore Sample**: Select a sample to view SHAP explanation
        4. **Interpret**: Understand which features influenced the prediction
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **WireXplain-IDS** uses RandomForest and SHAP to provide:
        - Accurate attack detection
        - Interpretable explanations
        - Feature importance insights
        """)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload & Predict", 
        "📊 Results Analysis", 
        "🔍 SHAP Explanations",
        "🔴 Live PCAP Replay"
    ])
    
    # Tab 1: Upload and Predict
    with tab1:
        st.header("Upload Dataset")
        
        st.markdown("""
        **Supported formats:**
        - 📄 **Processed CSV**: CSV with 15 selected features (from pipeline)
        - 📄 **Raw CSV**: CSV with 80 CICIDS2018 features (will be auto-processed)
        - 📦 **PCAP**: Network capture file (requires conversion guide)
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'pcap', 'pcapng'],
            help="Upload CSV with network features or PCAP file"
        )
        
        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            try:
                # Handle PCAP files
                if file_type in ['pcap', 'pcapng']:
                    st.warning("⚠️ PCAP file detected")
                    X = process_pcap_file(uploaded_file)
                    if X is None:
                        st.stop()
                
                # Handle CSV files
                else:
                    # Load dataset
                    with st.spinner("Loading dataset..."):
                        df = pd.read_csv(uploaded_file)
                        st.success(f"✅ Loaded {len(df):,} samples with {len(df.columns)} columns")
                    
                    # Load model
                    with st.spinner("Loading model..."):
                        model = load_model(model_path)
                        
                        if model is None:
                            st.error("❌ Failed to load model. Please check the model path.")
                            st.stop()
                        
                        st.success(f"✅ Model loaded: {type(model).__name__}")
                    
                    # Get expected features from model
                    if hasattr(model, 'feature_names_in_'):
                        expected_features = list(model.feature_names_in_)
                    else:
                        st.warning("⚠️ Model doesn't have feature names. Using default 15 features.")
                        expected_features = [
                            'Flow Duration', 'Fwd Pkt Len Max', 'Fwd Header Len',
                            'Bwd Header Len', 'Flow IAT Mean', 'Flow IAT Min',
                            'Flow IAT Max', 'Fwd Pkts/s', 'Bwd Pkts/s',
                            'Flow Pkts/s', 'Init Fwd Win Byts', 'Init Bwd Win Byts',
                            'Fwd Seg Size Min', 'bwd_packet_rate', 'fwd_packet_rate'
                        ]
                    
                    # Validate and prepare data
                    with st.spinner("Validating dataset..."):
                        X = validate_dataset(df, expected_features)
                        
                        # If validation failed (missing features), try preprocessing
                        if X is None:
                            st.info("🔄 Attempting to process raw CSV through feature engineering pipeline...")
                            X = process_raw_csv(df)
                            
                            if X is None:
                                st.error("❌ Could not process dataset. Please ensure it's in CICIDS2018 format.")
                                st.markdown("""
                                **Required format:**
                                - Either 15 selected features (from pipeline output)
                                - Or 80 raw CICIDS2018 features (will be auto-processed)
                                
                                **Example files you can use:**
                                - `data/processed/selected_features.csv` (15 features)
                                - `data/processed/filtered_data.csv` (15 features)
                                - `data/raw/02-14-2018.csv` (80 features, will be processed)
                                """)
                                st.stop()
                        
                        st.success(f"✅ Dataset ready: {len(X):,} samples, {len(X.columns)} features")
                    
                    # Make predictions
                    if st.button("🚀 Run Predictions", type="primary", use_container_width=True):
                        with st.spinner("Running predictions..."):
                            predictions, confidence, attack_prob = predict_with_confidence(model, X)
                            
                            # Store in session state
                            st.session_state['predictions'] = predictions
                            st.session_state['confidence'] = confidence
                            st.session_state['attack_prob'] = attack_prob
                            st.session_state['X'] = X
                            st.session_state['df'] = df
                            st.session_state['model'] = model
                            
                            st.success("✅ Predictions complete!")
                            st.balloons()
            
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
    
    # Tab 2: Results Analysis
    with tab2:
        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            confidence = st.session_state['confidence']
            attack_prob = st.session_state['attack_prob']
            X = st.session_state['X']
            
            st.header("Prediction Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            n_normal = np.sum(predictions == 0)
            n_attack = np.sum(predictions == 1)
            avg_confidence = np.mean(confidence)
            high_confidence = np.sum(confidence > 0.9)
            
            with col1:
                st.metric("Total Samples", f"{len(predictions):,}")
            with col2:
                st.metric("Normal Traffic", f"{n_normal:,}", 
                         delta=f"{(n_normal/len(predictions)*100):.1f}%")
            with col3:
                st.metric("Attacks Detected", f"{n_attack:,}", 
                         delta=f"{(n_attack/len(predictions)*100):.1f}%",
                         delta_color="inverse")
            with col4:
                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
            
            st.markdown("---")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig_dist = plot_prediction_distribution(predictions)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                fig_conf = plot_confidence_distribution(confidence, predictions)
                st.plotly_chart(fig_conf, use_container_width=True)
            
            st.markdown("---")
            
            # Detailed results table
            st.subheader("Detailed Predictions")
            
            results_df = pd.DataFrame({
                'Sample ID': range(len(predictions)),
                'Prediction': ['Normal' if p == 0 else 'Attack' for p in predictions],
                'Confidence': confidence,
                'Attack Probability': attack_prob
            })
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                filter_pred = st.selectbox(
                    "Filter by Prediction",
                    options=['All', 'Normal', 'Attack']
                )
            with col2:
                min_confidence = st.slider(
                    "Minimum Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05
                )
            
            # Apply filters
            filtered_df = results_df.copy()
            if filter_pred != 'All':
                filtered_df = filtered_df[filtered_df['Prediction'] == filter_pred]
            filtered_df = filtered_df[filtered_df['Confidence'] >= min_confidence]
            
            # Display dataframe (with or without styling based on size)
            # Pandas Styler has a limit of ~262k cells
            max_cells = 262144
            num_cells = len(filtered_df) * len(filtered_df.columns)
            
            if num_cells > max_cells:
                # For large datasets, show without styling
                st.info(f"ℹ️ Showing {len(filtered_df):,} rows (styling disabled for performance)")
                
                # Format the dataframe manually
                display_df = filtered_df.copy()
                display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.2%}")
                display_df['Attack Probability'] = display_df['Attack Probability'].apply(lambda x: f"{x:.2%}")
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )
            else:
                # For smaller datasets, use styling
                st.dataframe(
                    filtered_df.style.format({
                        'Confidence': '{:.2%}',
                        'Attack Probability': '{:.2%}'
                    }).background_gradient(subset=['Attack Probability'], cmap='RdYlGn_r'),
                    use_container_width=True,
                    height=400
                )

            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name="wirexplain_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        else:
            st.info("👆 Please upload a dataset and run predictions first.")
    
    # Tab 3: SHAP Explanations
    with tab3:
        if 'predictions' in st.session_state:
            st.header("SHAP Explanations")
            
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** shows which features contributed most to each prediction.
            - 🟢 **Green bars**: Features that increase attack probability
            - 🔴 **Red bars**: Features that decrease attack probability
            - **Bar length**: Magnitude of contribution
            """)
            
            X = st.session_state['X']
            predictions = st.session_state['predictions']
            confidence = st.session_state['confidence']
            attack_prob = st.session_state['attack_prob']
            model = st.session_state['model']
            
            # Sample selection
            col1, col2 = st.columns([2, 1])
            
            with col1:
                sample_idx = st.number_input(
                    "Select Sample ID",
                    min_value=0,
                    max_value=len(X)-1,
                    value=0,
                    step=1
                )
            
            with col2:
                st.markdown("### Sample Info")
                pred_label = "Normal" if predictions[sample_idx] == 0 else "Attack"
                st.metric("Prediction", pred_label)
                st.metric("Confidence", f"{confidence[sample_idx]:.2%}")
                st.metric("Attack Prob", f"{attack_prob[sample_idx]:.2%}")
            
            st.markdown("---")
            
            # Generate SHAP explanation
            if st.button("🔍 Generate SHAP Explanation", type="primary", use_container_width=True):
                with st.spinner("Computing SHAP values..."):
                    # Create explainer
                    explainer = create_shap_explainer(model)
                    
                    if explainer is None:
                        st.error("❌ Failed to create SHAP explainer")
                        st.stop()
                    
                    # Generate plot
                    fig = plot_shap_explanation(
                        explainer,
                        X,
                        sample_idx,
                        X.columns.tolist()
                    )
                    
                    if fig is not None:
                        st.pyplot(fig)
                        
                        # Show feature values
                        st.markdown("### Feature Values for Selected Sample")
                        feature_values = X.iloc[sample_idx].to_frame(name='Value')
                        feature_values['Feature'] = feature_values.index
                        feature_values = feature_values[['Feature', 'Value']].reset_index(drop=True)
                        
                        st.dataframe(
                            feature_values.style.format({'Value': '{:.4f}'}),
                            use_container_width=True,
                            height=300
                        )
        
        else:
            st.info("👆 Please upload a dataset and run predictions first.")
    
    # Tab 4: Live PCAP Replay
    with tab4:
        st.header("🔴 Live PCAP Replay")
        
        st.markdown("""
        **Near real-time network traffic replay** - Upload a PCAP file and watch predictions appear as packets are processed in batches.
        
        This mode simulates live intrusion detection by:
        1. Reading packets in configurable batches
        2. Extracting flow-based features
        3. Making predictions with the trained model
        4. Displaying results incrementally
        """)
        
        st.markdown("---")
        
        # Configuration
        col1, col2 = st.columns(2)
        
        with col1:
            pcap_file = st.file_uploader(
                "Upload PCAP File",
                type=['pcap', 'pcapng'],
                key='pcap_uploader',
                help="Upload a network capture file for replay"
            )
        
        with col2:
            batch_size = st.number_input(
                "Batch Size (packets)",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                help="Number of packets to process per batch"
            )
        
        if pcap_file is not None:
            # Save uploaded file temporarily
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pcap') as tmp_file:
                tmp_file.write(pcap_file.getvalue())
                tmp_path = tmp_file.name
            
            st.success(f"✅ PCAP file uploaded: {pcap_file.name}")
            
            # Start replay button
            if st.button("🚀 Start Replay", type="primary", use_container_width=True):
                try:
                    # Load model
                    model_path = "models/random_forest_model.pkl"
                    with st.spinner("Loading model..."):
                        model = load_model(model_path)
                        if model is None:
                            st.error("❌ Failed to load model")
                            st.stop()
                    
                    # Import PCAP modules
                    import sys
                    from pathlib import Path
                    sys.path.insert(0, str(Path(__file__).parent / 'src'))
                    
                    from pcap_batch_reader import PCAPBatchReader
                    from pcap_feature_extractor import PCAPFeatureExtractor
                    from feature_engineering import FeatureEngineer
                    
                    # Initialize components
                    reader = PCAPBatchReader(tmp_path, batch_size=batch_size)
                    extractor = PCAPFeatureExtractor()
                    engineer = FeatureEngineer()
                    
                    # Create placeholders for live updates
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    metrics_container = st.container()
                    results_container = st.empty()
                    
                    # Initialize results storage
                    all_predictions = []
                    all_confidence = []
                    all_attack_prob = []
                    all_features = []
                    
                    batch_num = 0
                    total_flows = 0
                    
                    # Process batches
                    status_text.info("🔄 Processing PCAP file...")
                    
                    for batch in reader.read_batches():
                        batch_num += 1
                        
                        # Debug: Show batch info
                        ip_count = sum(1 for p in batch if hasattr(p, 'ip'))
                        st.write(f"**Debug**: Batch {batch_num} - {len(batch)} packets, {ip_count} IP packets")
                        
                        # Extract features from batch
                        batch_features = extractor.extract_features_from_batch(batch)
                        
                        # Debug: Show flow info
                        st.write(f"**Debug**: Active flows: {len(extractor.flows)}, Completed flows: {len(batch_features)}")
                        
                        if not batch_features.empty:
                            # Engineer features
                            batch_features = engineer.encode_categorical_features(batch_features)
                            batch_features = engineer.engineer_features(batch_features)
                            selected_features = engineer.select_features(batch_features)
                            
                            # Make predictions
                            predictions, confidence, attack_prob = predict_with_confidence(model, selected_features)
                            
                            # Store results
                            all_predictions.extend(predictions)
                            all_confidence.extend(confidence)
                            all_attack_prob.extend(attack_prob)
                            all_features.append(selected_features)
                            
                            total_flows += len(predictions)
                        
                        # Update progress
                        progress = reader.get_progress()
                        status_text.info(f"📦 Batch {batch_num} | Packets: {progress['total_packets']:,} | Flows: {total_flows:,}")
                        
                        # Update metrics
                        if all_predictions:
                            with metrics_container:
                                col1, col2, col3, col4 = st.columns(4)
                                
                                n_normal = sum(1 for p in all_predictions if p == 0)
                                n_attack = sum(1 for p in all_predictions if p == 1)
                                avg_conf = np.mean(all_confidence)
                                
                                col1.metric("Total Flows", f"{total_flows:,}")
                                col2.metric("Normal", f"{n_normal:,}", delta=f"{(n_normal/total_flows*100):.1f}%")
                                col3.metric("Attacks", f"{n_attack:,}", delta=f"{(n_attack/total_flows*100):.1f}%", delta_color="inverse")
                                col4.metric("Avg Confidence", f"{avg_conf:.2%}")
                            
                            # Update results table
                            results_df = pd.DataFrame({
                                'Flow ID': range(len(all_predictions)),
                                'Prediction': ['Normal' if p == 0 else '🚨 Attack' for p in all_predictions],
                                'Confidence': all_confidence,
                                'Attack Probability': all_attack_prob
                            })
                            
                            # Show last 50 flows
                            display_df = results_df.tail(50).copy()
                            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.2%}")
                            display_df['Attack Probability'] = display_df['Attack Probability'].apply(lambda x: f"{x:.2%}")
                            
                            results_container.dataframe(
                                display_df,
                                use_container_width=True,
                                height=400
                            )
                        
                        # Small delay for visual effect
                        time.sleep(0.1)
                    
                    # Finalize remaining flows
                    final_features = extractor.finalize()
                    if not final_features.empty:
                        final_features = engineer.encode_categorical_features(final_features)
                        final_features = engineer.engineer_features(final_features)
                        selected_features = engineer.select_features(final_features)
                        predictions, confidence, attack_prob = predict_with_confidence(model, selected_features)
                        
                        all_predictions.extend(predictions)
                        all_confidence.extend(confidence)
                        all_attack_prob.extend(attack_prob)
                        all_features.append(selected_features)
                    
                    # Final results
                    progress_bar.progress(100)
                    status_text.success(f"✅ Replay complete! Processed {reader.get_progress()['total_packets']:,} packets in {batch_num} batches")
                    
                    # Store in session state for SHAP
                    if all_features:
                        combined_features = pd.concat(all_features, ignore_index=True)
                        st.session_state['predictions'] = np.array(all_predictions)
                        st.session_state['confidence'] = np.array(all_confidence)
                        st.session_state['attack_prob'] = np.array(all_attack_prob)
                        st.session_state['X'] = combined_features
                        st.session_state['model'] = model
                    
                    # Download results
                    st.markdown("---")
                    st.subheader("📥 Download Results")
                    
                    final_results = pd.DataFrame({
                        'Flow ID': range(len(all_predictions)),
                        'Prediction': ['Normal' if p == 0 else 'Attack' for p in all_predictions],
                        'Confidence': all_confidence,
                        'Attack Probability': all_attack_prob
                    })
                    
                    csv = final_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download PCAP Replay Results",
                        data=csv,
                        file_name=f"pcap_replay_{pcap_file.name}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error during PCAP replay: {e}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())
                finally:
                    # Cleanup temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        
        else:
            st.info("👆 Please upload a PCAP file to start live replay")
            
            # Show example
            with st.expander("ℹ️ How it works"):
                st.markdown("""
                **PCAP Live Replay Process:**
                
                1. **Upload PCAP**: Select a network capture file
                2. **Configure Batch Size**: Choose how many packets to process at once
                3. **Start Replay**: Click to begin processing
                4. **Watch Live Updates**:
                   - Progress bar shows overall progress
                   - Metrics update in real-time
                   - Prediction table shows latest flows
                5. **Download Results**: Export all predictions to CSV
                6. **View SHAP**: Go to SHAP tab to explain any flow
                
                **Feature Extraction:**
                - Packets are grouped into flows (5-tuple: src IP, dst IP, src port, dst port, protocol)
                - Statistical features are computed per flow (packet counts, byte counts, IAT, etc.)
                - Features are compatible with CICIDS2018 format
                - Flows are processed through the same ML pipeline as CSV data
                
                **Performance:**
                - Batch size affects processing speed
                - Larger batches = faster processing, less frequent updates
                - Smaller batches = slower processing, more frequent updates
                - Recommended: 100-200 packets per batch
                """)
    
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "WireXplain-IDS v1.0 | Built with Streamlit & SHAP | "
        "<a href='https://github.com/Suryooday/WireXplain-IDS'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
