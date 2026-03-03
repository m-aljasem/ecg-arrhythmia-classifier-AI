#!/bin/bash

# ECG Classification Project - Quick Setup Script
# This script helps you get started with the project quickly

echo "=========================================="
echo "ECG Classification Project - Quick Setup"
echo "=========================================="
echo ""

# Check Python installation
echo "[1/5] Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Python found: $PYTHON_VERSION"
else
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Install dependencies
echo ""
echo "[2/5] Installing dependencies..."
read -p "Install required packages? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "⊘ Skipped dependency installation"
fi

# Generate sample data
echo ""
echo "[3/5] Generating sample ECG data for testing..."
read -p "Generate sample data? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python generate_sample_data.py
    echo "✓ Sample data generated in data/ folder"
else
    echo "⊘ Skipped sample data generation"
fi

# Check for trained models
echo ""
echo "[4/5] Checking for trained models..."
if [ -f "models/model02.keras" ]; then
    echo "✓ Model found: models/model02.keras"
else
    echo "⚠ No trained models found"
    echo "  To train models, run:"
    echo "  python train.py --data-path /path/to/ptbxl/dataset --model model02"
fi

# Launch options
echo ""
echo "[5/5] Ready to launch!"
echo ""
echo "Choose an option:"
echo "  1) Launch Streamlit Web App"
echo "  2) Train models (requires PTB-XL dataset)"
echo "  3) Evaluate models"
echo "  4) Exit"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Launching Streamlit app..."
        echo "App will open at http://localhost:8501"
        streamlit run app.py
        ;;
    2)
        echo ""
        read -p "Enter path to PTB-XL dataset: " data_path
        if [ -d "$data_path" ]; then
            python train.py --data-path "$data_path" --model all
        else
            echo "❌ Directory not found: $data_path"
        fi
        ;;
    3)
        if [ -f "models/model02.keras" ]; then
            python evaluate.py --model all
        else
            echo "❌ No trained models found. Please train models first."
        fi
        ;;
    4)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        ;;
esac

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
