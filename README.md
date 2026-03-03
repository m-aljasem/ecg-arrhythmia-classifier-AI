# ECG Classification Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional deep learning system for ECG signal classification using the PTB-XL dataset. This project provides both a command-line training pipeline and an interactive web application for real-time cardiac diagnosis prediction.

## 🎯 Overview

This project implements multiple neural network models for multi-label ECG classification, combining patient metadata and raw ECG signals to predict **5 diagnostic superclasses**:

| Class | Description | Prevalence |
|-------|-------------|------------|
| **NORM** | Normal ECG | 9,528 records |
| **MI** | Myocardial Infarction (Heart Attack) | 5,486 records |
| **STTC** | ST/T Change | 5,250 records |
| **CD** | Conduction Disturbance | 4,907 records |
| **HYP** | Hypertrophy (Heart Enlargement) | 2,655 records |

### Key Features

✅ **Three Trained Models**
- Model 01: Metadata-only classifier (~75% accuracy)
- Model 02: Combined metadata + 1D CNN for ECG signals (~89% accuracy)
- Model 03: Model 02 with data augmentation (~87% accuracy)

✅ **Interactive Web Application**
- Streamlit-based UI for real-time predictions
- Support for CSV and WFDB file formats
- Visual ECG signal plotting
- Patient metadata input forms

✅ **Complete Training Pipeline**
- Automated data preprocessing
- Model training with early stopping
- Comprehensive evaluation metrics
- Cloud training support (Kaggle)

✅ **Professional Code Structure**
- Modular, reusable components
- Comprehensive documentation
- Easy to extend and customize

## 📁 Project Structure

```
project/
├── src/                        # Core package
│   ├── data/                   # Data loading & preprocessing
│   │   ├── loader.py          # ECG dataset loader
│   │   └── preprocessing.py   # Data preprocessing & augmentation
│   ├── models/                 # Model architectures
│   │   └── ecg_classifiers.py # Neural network definitions
│   └── utils/                  # Utility functions
│       ├── visualization.py   # Plotting & visualization
│       └── common.py          # Helper functions
├── scripts/                    # Utility scripts
│   ├── cloud-training.py      # Kaggle/Colab training script
│   ├── generate_sample_data.py # Generate test ECG files
│   ├── run_app.py             # App launcher with checks
│   └── setup.sh               # Quick setup script
├── docs/                       # Documentation
│   └── WEBAPP.md              # Web app user guide
├── tests/                      # Unit tests
├── data/                       # Data directory (created after setup)
├── models/                     # Model checkpoints (created after training)
├── notebooks/                  # Jupyter notebooks
├── config.py                   # Configuration & hyperparameters
├── train.py                    # Training pipeline
├── evaluate.py                 # Model evaluation
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
└── README.md                   # This file
```

## Installation

1. Clone or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install as a package:
```bash
pip install -e .
```

## Data Setup

Download the PTB-XL dataset from Kaggle:
- [PTB-XL Dataset](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1-0-1)

Extract it and note the path for the next steps.

## Usage

### 1. Data Preparation

Prepare the data (loads dataset, preprocesses, and saves to NPZ):

```bash
python train.py --data-path /path/to/ptbxl/dataset --model all --skip-preprocessing false
```

Or just prepare data without training:
```bash
python train.py --data-path /path/to/ptbxl/dataset --skip-preprocessing false
```

### 2. Training Models

Train all models:
```bash
python train.py --data-path /path/to/ptbxl/dataset --model all
```

Train specific model:
```bash
python train.py --data-path /path/to/ptbxl/dataset --model model02
```

Available models:
- **model01**: Metadata-only classifier
- **model02**: Combined metadata + ECG signals (1D CNN)
- **model03**: Model02 with data augmentation

### 3. Evaluation

Evaluate all trained models:
```bash
python evaluate.py --model all
```


Launch the interactive Streamlit web app for real-time ECG classification:

```bash
# Quick start (with automatic checks)
python scripts/run_app.py

# Or directly
streamlit run app.py
```

The app provides:
- 🖥️ User-friendly interface for uploading ECG data
- 📝 Patient metadata input forms
- 📊 Real-time predictions with visualizations
- 📁 Support for CSV and WFDB file formats
- 📈 Interactive ECG signal plotting

Access at: **http://localhost:8501**

See [docs/WEBAPP.md](docs/WEBAPP.md) for detailed app documentation.

## 🧠 Models Architecture

### Model 01: Metadata-Only Classifier
```
Input: Patient metadata (7 features)
├── Dense(32) → Dropout(0.3) → Dense(32) → Dropout(0.3)
├── Dense(64) → ReLU → Dense(64) → ReLU → Dropout(0.5)
└── Output: Dense(5, sigmoid)

Performance: ~75% binary accuracy
Use case: Baseline, quick predictions without ECG data
```

### Model 02: Combined Metadata + ECG (Recommended)
```
Input 1: Metadata (7 features)           Input 2: ECG signals (1000×12)
├── Dense(32) → Dropout(0.3)             ├── Conv1D(64) → BatchNorm → ReLU → MaxPool
├── Dense(32) → Dropout(0.3)             ├── Conv1D(128) → BatchNorm → ReLU → MaxPool
└── Embedding                            ├── Conv1D(256) → BatchNorm → ReLU
                                         └── GlobalAvgPool → Dropout(0.5)
                  ↓
         Concatenate embeddings
                  ↓
    Dense(64) → ReLU → Dense(64) → ReLU → Dropout(0.5)
                  ↓
         Output: Dense(5, sigmoid)

Performance: ~89% binary accuracy ⭐
Use case: Production deployment, best overall performance
```

### Model 03: With Data Augmentation
```
Same as Model 02, but trained with:
- Sliding window (800 samples with random shift)
- Gaussian noise (σ=0.05)

Performance: ~87% binary accuracy
Use case: Experimentation, robust to signal variations
```
# 2. Generate sample data for testing
python scripts/generate_sample_data.py

# 3. Train models (requires PTB-XL dataset)
python train.py --data-path /path/to/ptbxl --model model02

# 4. Launch web app
streamlit run app.py
```

### Option 3: Cloud Training (Kaggle)

For training on Kaggle without local resources:

1. Upload `scripts/cloud-training.py` to Kaggle
2. Add PTB-XL dataset as data source
3. Run the script
4. Download trained models

See the script for detailed instructions
- Patient metadata input forms
- Real-time predictions with visualizations
- S⚙️ Configuration

All settings are centralized in [config.py](config.py):

```python
# Model hyperparameters
MODEL_CONFIGS = {
    'model02': {
        'batch_size': 32,      # Adjust based on GPU memory
        'epochs': 100,         # Maximum epochs
        'patience': 20,        # Early stopping patience
    }
}

# Data paths
DATA_DIR = 'data/'
MODELS_DIR = 'models/'

# Target classes
SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
```

## 📊 Performance Results

| Model | Architecture | Binary Accuracy | Training Time | Best For |
|-------|--------------|-----------------|---------------|----------|
| Model 01 | Metadata Only | 75% | ~2 min | Quick baseline |
| Model 02 | Metadata + ECG | **89%** ⭐ | ~20 min | Production use |
| Model 03 | Model 02 + Aug | 87% | ~25 min | Experimentation |

### Detailed Results (Model 02)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| NORM | 0.91 | 0.92 | 0.91 | 952 |
| MI | 0.87 | 0.84 | 0.86 | 548 |
| S🛠️ Development

### Adding New Models

1. Define model in [src/models/ecg_classifiers.py](src/models/ecg_classifiers.py)
   ```python
   def create_model04(X_shape, Y_shape, Z_shape):
       # Your architecture
       pass
   ```

2. Add to factory class
   ```python
   @staticmethod
   def create_your_model(X_shape, Y_shape, Z_shape):
       return create_model04(X_shape, Y_shape, Z_shape)
   ```

3. Add training function in [train.py](train.py)

4. Update [config.py](config.py) with hyperparameters

### Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_models.py
```

### Code Quality

```bash
# Format code
black src/ *.py

# Lint
flake8 src/ *.py

# Type checking
mypy src/
```

## 📦 Dependencies

Core requirements:
```
python >= 3.8
tensorflow >= 2.9.0
numpy >= 1.20.0
pandas >= 1.3.0
sci🚢 Deployment

### Streamlit Cloud (Free)

1. Push project to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy `app.py`
5. Share the URL ✨

### Docker

```bash
# Build image
docker build -t ecg-classifier .

# Run container
docker run -p 8501:8501 ecg-classifier
```

### Local Server

```bash
# Run on all network interfaces
streamlit run app.py --server.address 0.0.0.0

# Access from other devices: http://YOUR_IP:8501
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- **PTB-XL Dataset**: [PhysioNet](https://physionet.org/content/ptb-xl/1.0.1/)
  ```
  Wagner, P., Strodthoff, N., Bousseljot, R. D., Kreiseler, D., Lunze, F. I., 
  Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available 
  electrocardiography dataset. Scientific Data, 7(1), 154.
  ```

- **Original Kaggle Notebook**: [PTB-XL ECG Classification](https://www.kaggle.com/code/khyeh0719/ptb-xl-dataset-wrangling)

## 🙏 Acknowledgments

- PTB-XL dataset authors and contributors
- Kaggle community for inspiration
- TensorFlow and Streamlit teams

## ⚠️ Medical Disclaimer

**IMPORTANT**: This project is for **educational and research purposes only**.

- NOT intended for clinical diagnosis
- NOT FDA approved or medically certified
- Results should be reviewed by qualified healthcare professionals
- No warranty or guarantee of accuracy
- Authors not liable for any medical decisions

Always consult with licensed medical professionals for health concerns.

## 📧 Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check [docs/WEBAPP.md](docs/WEBAPP.md) for app-specific help

---

**Made with ❤️ for the medical AI community**
See [requirements.txt](requirements.txt) for complete list.
## Key Results

- **Model01**: Limited performance (predicts only NORM and MI)
- **Model02**: Good performance on all classes (89% binary accuracy)
- **Model03**: Alternative approach with augmentation

## Development

### Adding New Models

1. Add model function to `src/models/ecg_classifiers.py`
2. Add factory method to `ECGClassifierFactory`
3. Create training function in `train.py`
4. Update configuration in `config.py`

### Data Processing Pipeline

1. **Loader**: `ECGDataLoader` - loads raw dataset
2. **Preprocessor**: `DataPreprocessor` - creates features and targets
3. **Generator**: `AugmentedDataGenerator` - provides augmented batches

## Requirements

- Python 3.8+
- TensorFlow 2.9+
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- wfdb (for reading ECG files)

## References

PTB-XL: A Large Publicly Available Electrocardiography Dataset:
https://physionet.org/content/ptb-xl/1.0.1/

## License

MIT License

## Notes

- Original notebook (`../main.ipynb`) kept intact as reference
- All models use binary crossentropy loss for multi-label classification
- Early stopping and model checkpointing during training
- Test set evaluation on stratification fold 10 (as per dataset authors)
