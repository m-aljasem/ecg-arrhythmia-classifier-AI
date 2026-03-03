# ECG Classification Web App

A professional Streamlit web application for real-time ECG classification using deep learning models.

## 🚀 Features

- **User-Friendly Interface**: Clean, intuitive UI for medical professionals and researchers
- **Multiple Input Methods**: 
  - Upload CSV files with ECG signals
  - Upload WFDB format files (.dat + .hea)
  - Use sample data for demonstration
- **Patient Metadata**: Complete patient information input
- **Model Selection**: Choose from 3 trained models
- **Real-Time Predictions**: Instant classification results
- **Interactive Visualizations**: 
  - ECG signal plots for all leads
  - Prediction probability charts
  - Diagnostic summaries
- **Adjustable Threshold**: Customize prediction sensitivity

## 📋 Diagnostic Classes

The app detects 5 cardiac diagnostic superclasses:

| Class | Description |
|-------|-------------|
| **NORM** | Normal ECG |
| **MI** | Myocardial Infarction (Heart Attack) |
| **STTC** | ST/T Change |
| **CD** | Conduction Disturbance |
| **HYP** | Hypertrophy (Heart Enlargement) |

## 🛠️ Installation

### Prerequisites

```bash
# Install required packages
pip install -r requirements.txt
```

### Required Dependencies

- streamlit >= 1.28.0
- tensorflow >= 2.9.0
- numpy
- pandas
- matplotlib
- seaborn
- wfdb

## 🎯 Usage

### 1. Train Models First

Before running the app, ensure you have trained models:

```bash
python train.py --data-path /path/to/ptbxl/dataset --model all
```

This creates model files in the `models/` directory:
- `model01.keras` - Metadata-only classifier
- `model02.keras` - Combined metadata + ECG (recommended)
- `model03.keras` - With data augmentation

### 2. Launch the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Using the App

#### Step 1: Configure Settings (Sidebar)
- Select a model (Model 02 recommended for best accuracy)
- Adjust prediction threshold (default: 0.5)

#### Step 2: Enter Patient Metadata
- Age, sex, height, weight
- Infarction stadium information
- Pacemaker status

#### Step 3: Upload ECG Data

**Option A: CSV File**
- Format: Each column = one lead, each row = time sample
- Expected shape: (1000, 12) for 12-lead ECG
- Example:
  ```
  0.1, 0.2, 0.15, ... (12 values)
  0.12, 0.19, 0.14, ...
  ... (1000 rows)
  ```

**Option B: WFDB Files**
- Upload both `.dat` and `.hea` files
- Standard PTB-XL format

**Option C: Sample Data**
- Use auto-generated sample data for testing

#### Step 4: Get Prediction
- Click "Predict Diagnosis"
- View results and visualizations

## 📊 Understanding Results

### Prediction Table
Shows each diagnostic class with:
- **Probability**: Confidence score (0-100%)
- **Diagnosis**: Positive (✓) if above threshold, Negative (−) otherwise

### Visualization
- Bar chart showing probabilities for all classes
- Green bars = positive diagnosis
- Gray bars = negative diagnosis
- Red dashed line = selected threshold

### Summary
- Total number of positive diagnoses
- List of detected conditions
- Medical disclaimer

## 🎨 Screenshots

### Main Interface
```
┌─────────────────────────────────────────────────────────┐
│  ❤️ ECG Classification System                          │
├─────────────────────────────────────────────────────────┤
│  📝 Patient Metadata    │    📊 ECG Signal Data        │
│  ─────────────────────  │    ───────────────────        │
│  Age: [50]              │    [Upload CSV / WFDB]        │
│  Sex: [Male ▼]          │    [Use Sample Data]          │
│  Height: [170] cm       │                                │
│  Weight: [70] kg        │    ECG Preview: [View ▼]      │
│  ...                    │                                │
├─────────────────────────────────────────────────────────┤
│              🚀 Predict Diagnosis                       │
├─────────────────────────────────────────────────────────┤
│  📋 Results             │    📊 Visualization           │
│  ─────────────          │    ────────────────           │
│  [Results Table]        │    [Probability Chart]        │
└─────────────────────────────────────────────────────────┘
```

## 🔧 Customization

### Modify Threshold
Adjust the prediction threshold in the sidebar:
- Lower threshold = more sensitive (more positives)
- Higher threshold = more specific (fewer false positives)
- Default: 0.5 (50%)

### Add Custom Models
Place your trained `.keras` model files in the `models/` directory and update `MODEL_PATHS` in `app.py`:

```python
MODEL_PATHS = {
    'Your Model Name': os.path.join(MODELS_DIR, 'your_model.keras'),
    ...
}
```

### Styling
Modify the Streamlit theme by creating `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

## 🏥 Medical Disclaimer

**IMPORTANT**: This application is for **educational and research purposes only**.

- Results should NOT be used for clinical diagnosis
- Always consult qualified healthcare professionals
- Not FDA approved or certified for medical use
- No warranty or guarantee of accuracy
- Models are trained on PTB-XL dataset and may not generalize to all populations

## 📈 Model Performance

| Model | Type | Binary Accuracy |
|-------|------|----------------|
| Model 01 | Metadata Only | ~0.75 |
| Model 02 | Metadata + ECG | ~0.89 |
| Model 03 | With Augmentation | ~0.87 |

**Recommendation**: Use Model 02 for best overall performance.

## 🐛 Troubleshooting

### Model Not Found
```
❌ Model not found at: models/model02.keras
```
**Solution**: Train models first using `train.py`

### CSV Format Error
```
❌ Error loading CSV
```
**Solution**: Ensure CSV has correct format:
- No headers
- 12 columns (for 12-lead ECG)
- ~1000 rows (samples)

### Memory Error
**Solution**: 
- Use smaller batch sizes
- Close other applications
- Use Model 01 (requires less memory)

### WFDB File Error
**Solution**:
- Upload both .dat AND .hea files
- Ensure files are from same record
- Use PTB-XL format

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from repository
4. Make sure `models/` directory contains trained models

### Deploy to Heroku

```bash
# Add Procfile
echo "web: streamlit run app.py --server.port=$PORT" > Procfile

# Deploy
heroku create your-ecg-app
git push heroku main
```

### Deploy with Docker

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 📚 References

- PTB-XL Dataset: https://physionet.org/content/ptb-xl/
- Streamlit Documentation: https://docs.streamlit.io
- TensorFlow: https://www.tensorflow.org

## 📧 Support

For issues or questions:
- Check existing issues on GitHub
- Review documentation
- Contact project maintainer

---

**Built with ❤️ using Streamlit and TensorFlow**
