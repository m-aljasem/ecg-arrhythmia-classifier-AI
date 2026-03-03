"""
Streamlit Web Application for ECG Classification

Classification of Life-Threatening Arrhythmia ECG Signals Using Deep Learning

Author: Mohamad AlJasem
Website: https://aljasem.eu.org
GitHub: https://github.com/m-aljasem/ecg-arrhythmia-classifier-AI
Live Demo: https://ecg-classifier.aljasem.eu.org
Contact: mohamad@aljasem.eu.org

This app allows users to upload ECG data and get real-time predictions
for cardiac diagnostic superclasses using trained deep learning models.

Usage:
    streamlit run app.py
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras as keras
from io import StringIO
import wfdb
import tempfile
import pickle

# Page configuration
st.set_page_config(
    page_title="ECG Classification App",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
CLASS_DESCRIPTIONS = {
    'NORM': 'Normal ECG',
    'MI': 'Myocardial Infarction (Heart Attack)',
    'STTC': 'ST/T Change',
    'CD': 'Conduction Disturbance',
    'HYP': 'Hypertrophy (Heart Enlargement)'
}

MODELS_DIR = 'models'
MODEL_PATH = os.path.join(MODELS_DIR, 'model03.keras')
SCALERS_PATH = os.path.join(MODELS_DIR, 'scalers.pkl')
WINDOW_SIZE = 800  # Model03 uses windowed data

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model(model_path):
    """Load trained model from file"""
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_scalers(scalers_path):
    """Load scalers from pickle file"""
    try:
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        return scalers
    except Exception as e:
        st.error(f"Error loading scalers: {e}")
        return None


def preprocess_metadata(age, sex, height, weight, infarction_stadium1, 
                       infarction_stadium2, pacemaker, x_scaler):
    """Preprocess patient metadata into model input format"""
    metadata = pd.DataFrame({
        'age': [float(age) if age else 0.0],
        'sex': [float(sex)],
        'height': [float(height) if height and height >= 50 else 0.0],
        'weight': [float(weight) if weight else 0.0],
        'infarction_stadium1': [float(infarction_stadium1)],
        'infarction_stadium2': [float(infarction_stadium2)],
        'pacemaker': [float(pacemaker)]
    })
    
    # Use the scaler from training
    if x_scaler is not None:
        metadata_scaled = x_scaler.transform(metadata.values)
        return metadata_scaled
    else:
        return metadata.values


def preprocess_ecg_signal(ecg_data, y_scaler, window_size=800):
    """Preprocess ECG signal data"""
    # Apply sliding window if signal is longer than window_size
    if ecg_data.shape[0] > window_size:
        # Take center window
        start_idx = (ecg_data.shape[0] - window_size) // 2
        ecg_data = ecg_data[start_idx:start_idx + window_size, :]
    elif ecg_data.shape[0] < window_size:
        # Pad if too short
        pad_width = ((0, window_size - ecg_data.shape[0]), (0, 0))
        ecg_data = np.pad(ecg_data, pad_width, mode='edge')
    
    # Ensure correct shape (samples, timesteps, channels)
    if len(ecg_data.shape) == 2:
        ecg_data = ecg_data.reshape(1, ecg_data.shape[0], ecg_data.shape[1])
    
    # Use the scaler from training
    if y_scaler is not None:
        original_shape = ecg_data.shape
        ecg_reshaped = ecg_data.reshape(-1, ecg_data.shape[-1])
        ecg_scaled = y_scaler.transform(ecg_reshaped)
        ecg_data = ecg_scaled.reshape(original_shape)
    
    return ecg_data.astype('float32')


def plot_ecg_signal(ecg_data, title="ECG Signal"):
    """Plot ECG signal with all leads"""
    fig, axes = plt.subplots(ecg_data.shape[1], 1, figsize=(12, 8))
    
    lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 
                  'V3', 'V4', 'V5', 'V6']
    
    for i in range(ecg_data.shape[1]):
        lead_name = lead_names[i] if i < len(lead_names) else f'Lead {i+1}'
        axes[i].plot(ecg_data[:, i], linewidth=0.8)
        axes[i].set_ylabel(lead_name, fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, len(ecg_data))
    
    axes[-1].set_xlabel('Sample', fontsize=10)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_predictions(predictions, threshold=0.5):
    """Plot prediction probabilities"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green' if p >= threshold else 'gray' for p in predictions]
    bars = ax.barh(SUPERCLASSES, predictions, color=colors, alpha=0.7)
    
    # Add threshold line
    ax.axvline(x=threshold, color='red', linestyle='--', 
              label=f'Threshold ({threshold})', linewidth=2)
    
    # Add value labels
    for i, (bar, pred) in enumerate(zip(bars, predictions)):
        width = bar.get_width()
        label = f'{pred:.1%}'
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
               label, ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_ylabel('Diagnostic Class', fontsize=12)
    ax.set_title('ECG Classification Predictions', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    # Header
    st.title("❤️ ECG Classification System")
    st.markdown("""
    Upload ECG data and patient metadata to get real-time cardiac diagnostic predictions.
    This system can detect:
    - **NORM**: Normal ECG
    - **MI**: Myocardial Infarction (Heart Attack)
    - **STTC**: ST/T Change
    - **CD**: Conduction Disturbance
    - **HYP**: Hypertrophy (Heart Enlargement)
    """)
    
    st.markdown("---")
    
    # Sidebar - Model Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load model and scalers
        st.markdown("### Model Status")
        
        model = None
        scalers = None
        
        if os.path.exists(MODEL_PATH):
            st.success(f"✓ Model found: model03.keras")
            model = load_model(MODEL_PATH)
        else:
            st.error(f"❌ Model not found at: {MODEL_PATH}")
            st.info("Please add model03.keras to the models/ folder")
        
        if os.path.exists(SCALERS_PATH):
            st.success(f"✓ Scalers found: scalers.pkl")
            scalers = load_scalers(SCALERS_PATH)
            if scalers:
                st.info(f"Loaded scalers for {len(scalers.get('feature_names', []))} features")
        else:
            st.error(f"❌ Scalers not found at: {SCALERS_PATH}")
            st.info("Please add scalers.pkl to the models/ folder")
        
        st.markdown("---")
        
        # Prediction threshold
        threshold = st.slider(
            "Prediction Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Probability threshold for positive diagnosis"
        )
        
        st.markdown("---")
        st.markdown("""
        ### 📖 Instructions
        1. Enter patient metadata
        2. Upload ECG signal data
        3. Click 'Predict' to classify
        """)
    
    # Main content - Two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Patient Metadata")
        
        # Metadata inputs
        age = st.number_input("Age (years)", min_value=0, max_value=120, 
                             value=50, step=1)
        
        sex = st.selectbox("Sex", options=[0.0, 1.0], 
                          format_func=lambda x: "Male" if x == 1.0 else "Female")
        
        height = st.number_input("Height (cm)", min_value=50, max_value=250, 
                                value=170, step=1)
        
        weight = st.number_input("Weight (kg)", min_value=20, max_value=300, 
                                value=70, step=1)
        
        infarction_stadium1 = st.selectbox(
            "Infarction Stadium 1",
            options=[0, 1, 2, 3, 4, 5],
            format_func=lambda x: {
                0: "Unknown/None",
                1: "Stadium I",
                2: "Stadium I-II",
                3: "Stadium II",
                4: "Stadium II-III",
                5: "Stadium III"
            }[x]
        )
        
        infarction_stadium2 = st.selectbox(
            "Infarction Stadium 2",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Unknown/None",
                1: "Stadium I",
                2: "Stadium II",
                3: "Stadium III"
            }[x]
        )
        
        pacemaker = st.checkbox("Pacemaker")
        pacemaker_val = 1.0 if pacemaker else 0.0
        
        # Display metadata summary
        st.markdown("### Summary")
        metadata_summary = pd.DataFrame({
            'Parameter': ['Age', 'Sex', 'Height', 'Weight', 'Infarction I', 
                         'Infarction II', 'Pacemaker'],
            'Value': [
                f"{age} years",
                "Male" if sex == 1.0 else "Female",
                f"{height} cm",
                f"{weight} kg",
                infarction_stadium1,
                infarction_stadium2,
                "Yes" if pacemaker else "No"
            ]
        })
        st.dataframe(metadata_summary, hide_index=True, use_container_width=True)
    
    with col2:
        st.header("📊 ECG Signal Data")
        
        # ECG data upload options
        upload_option = st.radio(
            "ECG Data Input Method",
            ["Upload CSV File", "Upload WFDB File", "Use Sample Data"]
        )
        
        ecg_data = None
        
        if upload_option == "Upload CSV File":
            st.markdown("""
            Upload a CSV file with ECG signal data.
            - Each column represents a lead
            - Each row is a time sample
            - Expected shape: (1000, 12) for 12-lead ECG
            """)
            
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    ecg_data = pd.read_csv(uploaded_file, header=None).values
                    st.success(f"✓ Loaded ECG data: {ecg_data.shape}")
                except Exception as e:
                    st.error(f"Error loading CSV: {e}")
        
        elif upload_option == "Upload WFDB File":
            st.markdown("""
            Upload WFDB format files (.dat and .hea)
            """)
            
            dat_file = st.file_uploader("Upload .dat file", type=['dat'])
            hea_file = st.file_uploader("Upload .hea file", type=['hea'])
            
            if dat_file is not None and hea_file is not None:
                try:
                    # Save to temporary files
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # Save files without extension
                        base_name = "temp_ecg"
                        dat_path = os.path.join(tmpdir, f"{base_name}.dat")
                        hea_path = os.path.join(tmpdir, f"{base_name}.hea")
                        
                        with open(dat_path, 'wb') as f:
                            f.write(dat_file.read())
                        with open(hea_path, 'wb') as f:
                            f.write(hea_file.read())
                        
                        # Read with wfdb
                        record = wfdb.rdrecord(os.path.join(tmpdir, base_name))
                        ecg_data = record.p_signal
                        
                        st.success(f"✓ Loaded WFDB data: {ecg_data.shape}")
                except Exception as e:
                    st.error(f"Error loading WFDB files: {e}")
        
        else:  # Use Sample Data
            st.info("Using randomly generated sample ECG data for demonstration")
            # Generate sample data (12 leads, 1000 samples)
            np.random.seed(42)
            ecg_data = np.random.randn(1000, 12) * 0.5
            # Add some periodic pattern to make it look ECG-like
            t = np.linspace(0, 10, 1000)
            for i in range(12):
                ecg_data[:, i] += np.sin(2 * np.pi * 1.2 * t + i * 0.3)
            
            st.success(f"✓ Sample data generated: {ecg_data.shape}")
        
        # Display ECG preview
        if ecg_data is not None:
            st.markdown("### ECG Signal Preview")
            with st.expander("View ECG Signals", expanded=False):
                fig = plot_ecg_signal(ecg_data, "ECG Signal Preview")
                st.pyplot(fig)
                plt.close()
    
    # Prediction section
    st.markdown("---")
    st.header("🔮 Prediction")
    
    col_pred1, col_pred2, col_pred3 = st.columns([1, 1, 1])
    
    with col_pred2:
        predict_button = st.button("🚀 Predict Diagnosis", use_container_width=True, 
                                   type="primary")
    
    if predict_button:
        if model is None:
            st.error("❌ No model loaded. Please check model path.")
        elif scalers is None:
            st.error("❌ No scalers loaded. Please check scalers.pkl file.")
        elif ecg_data is None:
            st.error("❌ Please upload ECG signal data.")
        else:
            with st.spinner("🔄 Analyzing ECG data..."):
                try:
                    # Extract scalers
                    x_scaler = scalers.get('x_scaler')
                    y_scaler = scalers.get('y_scaler')
                    
                    # Prepare inputs
                    metadata_input = preprocess_metadata(
                        age, sex, height, weight, 
                        infarction_stadium1, infarction_stadium2, pacemaker_val,
                        x_scaler
                    )
                    
                    # Preprocess ECG signal with scaler and window size
                    ecg_input = preprocess_ecg_signal(ecg_data, y_scaler, WINDOW_SIZE)
                    
                    # Make prediction (Model03 requires both metadata and ECG)
                    predictions = model.predict(
                        [metadata_input, ecg_input], verbose=0
                    )[0]
                    
                    # Display results
                    st.success("✅ Prediction completed!")
                    
                    # Create results columns
                    result_col1, result_col2 = st.columns([1, 1])
                    
                    with result_col1:
                        st.markdown("### 📋 Diagnostic Results")
                        
                        results_df = pd.DataFrame({
                            'Condition': [CLASS_DESCRIPTIONS[c] for c in SUPERCLASSES],
                            'Probability': [f"{p:.1%}" for p in predictions],
                            'Diagnosis': ['✓ Positive' if p >= threshold else '− Negative' 
                                        for p in predictions]
                        })
                        
                        # Style the dataframe
                        def highlight_positive(row):
                            if row['Diagnosis'] == '✓ Positive':
                                return ['background-color: #ffcccc'] * len(row)
                            return [''] * len(row)
                        
                        styled_df = results_df.style.apply(highlight_positive, axis=1)
                        st.dataframe(styled_df, hide_index=True, use_container_width=True)
                        
                        # Summary
                        positive_count = sum(predictions >= threshold)
                        st.markdown(f"**Total Positive Diagnoses:** {positive_count}")
                        
                        if positive_count == 0:
                            st.success("✅ No significant abnormalities detected")
                        else:
                            positive_conditions = [
                                CLASS_DESCRIPTIONS[SUPERCLASSES[i]] 
                                for i, p in enumerate(predictions) if p >= threshold
                            ]
                            st.warning(f"⚠️ Detected: {', '.join(positive_conditions)}")
                    
                    with result_col2:
                        st.markdown("### 📊 Prediction Visualization")
                        fig_pred = plot_predictions(predictions, threshold)
                        st.pyplot(fig_pred)
                        plt.close()
                    
                    # Additional info
                    st.markdown("---")
                    st.info("""
                    **⚕️ Medical Disclaimer:**
                    This is a demonstration system for educational purposes only.
                    Results should NOT be used for medical diagnosis without consultation
                    with qualified healthcare professionals.
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
                    st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>ECG Classification System | Built with Streamlit & TensorFlow</p>
        <p>Based on PTB-XL Dataset</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
