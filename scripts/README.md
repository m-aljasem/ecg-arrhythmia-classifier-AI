# Utility Scripts

This directory contains helper scripts for various tasks.

## Scripts Overview

### 🚀 `run_app.py`
Quick launcher for the Streamlit web application with automatic dependency and model checks.

```bash
python scripts/run_app.py
```

**Features:**
- Checks if all dependencies are installed
- Verifies trained models exist
- Auto-generates sample data if needed
- Launches the app with helpful messages

---

### ☁️ `cloud-training.py`
Complete training script designed for Kaggle or Google Colab.

```bash
# On Kaggle: Copy this file to a notebook and run
python cloud-training.py
```

**Features:**
- Self-contained (no external dependencies on project structure)
- Trains all three models automatically
- Saves models and data to `/kaggle/working/`
- Ready for download after training

**Kaggle Setup:**
1. Create new Kaggle notebook
2. Add PTB-XL dataset as data source
3. Upload and run this script

---

### 📊 `generate_sample_data.py`
Generates synthetic ECG data for testing the web application.

```bash
python scripts/generate_sample_data.py
```

**Generates:**
- `data/sample_ecg_normal.csv` - Normal ECG pattern
- `data/sample_ecg_abnormal.csv` - Irregular rhythm
- `data/sample_ecg_mi.csv` - MI indicators (ST elevation)

**Use cases:**
- Testing the web app without real data
- Demonstrating functionality
- Development and debugging

---

### ⚙️ `setup.sh`
Interactive bash script for project setup.

```bash
bash scripts/setup.sh
```

**Features:**
- Checks Python installation
- Installs dependencies
- Generates sample data
- Offers options to train or launch app
- Interactive menu system

**Requirements:**
- Bash shell (Linux/Mac)
- Python 3.8+

---

## Usage Examples

### Quick Start Workflow

```bash
# 1. Run setup
bash scripts/setup.sh

# 2. Generate test data
python scripts/generate_sample_data.py

# 3. Launch app
python scripts/run_app.py
```

### Cloud Training Workflow

```bash
# 1. Go to Kaggle
# 2. Upload cloud-training.py
# 3. Add PTB-XL dataset
# 4. Run script
# 5. Download trained models from /kaggle/working/models/
```

### Development Workflow

```bash
# Generate fresh test data
python scripts/generate_sample_data.py

# Test app locally
python scripts/run_app.py

# Train models locally (requires dataset)
cd ..
python train.py --data-path /path/to/data --model model02
```

---

## Notes

- All scripts are standalone and can be run independently
- Scripts automatically create necessary directories
- Error handling and user-friendly messages included
- Safe to run multiple times (idempotent where applicable)

---

For main project documentation, see [../README.md](../README.md)
