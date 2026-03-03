"""
Run the Streamlit web app with sample data

This script:
1. Checks if sample data exists, generates if needed
2. Checks if models exist, warns if not
3. Launches the Streamlit app

Usage:
    python run_app.py
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    required = ['streamlit', 'tensorflow', 'numpy', 'pandas', 'wfdb']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("\nInstall them with:")
        print("pip install -r requirements.txt")
        return False
    
    print("✓ All dependencies installed")
    return True


def generate_sample_data():
    """Generate sample ECG data if it doesn't exist"""
    sample_files = [
        'data/sample_ecg_normal.csv',
        'data/sample_ecg_abnormal.csv',
        'data/sample_ecg_mi.csv'
    ]
    
    if not any(os.path.exists(f) for f in sample_files):
        print("\nGenerating sample ECG data...")
        try:
            subprocess.run([sys.executable, 'generate_sample_data.py'], check=True)
            print("✓ Sample data generated")
        except subprocess.CalledProcessError:
            print("⚠ Failed to generate sample data")
            return False
    else:
        print("✓ Sample data already exists")
    
    return True


def check_models():
    """Check if trained models exist"""
    models = {
        'model01.keras': 'Model 01 (Metadata Only)',
        'model02.keras': 'Model 02 (Best - Metadata + ECG)',
        'model03.keras': 'Model 03 (With Augmentation)'
    }
    
    print("\nChecking for trained models...")
    found_models = []
    
    for model_file, model_name in models.items():
        model_path = os.path.join('models', model_file)
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"  ✓ {model_name} ({size_mb:.1f} MB)")
            found_models.append(model_name)
        else:
            print(f"  ✗ {model_name} - not found")
    
    if not found_models:
        print("\n⚠ WARNING: No trained models found!")
        print("\nTo train models, run:")
        print("  python train.py --data-path /path/to/ptbxl/dataset --model model02")
        print("\nThe app will still run, but predictions won't work without models.")
        return False
    
    print(f"\n✓ Found {len(found_models)} trained model(s)")
    return True


def launch_app():
    """Launch the Streamlit app"""
    print("\n" + "=" * 60)
    print("Launching ECG Classification Web App")
    print("=" * 60)
    print("\nThe app will open in your browser at:")
    print("  👉 http://localhost:8501")
    print("\nPress Ctrl+C to stop the app")
    print("=" * 60 + "\n")
    
    try:
        subprocess.run(['streamlit', 'run', 'app.py'])
    except KeyboardInterrupt:
        print("\n\nApp stopped by user")
    except FileNotFoundError:
        print("❌ Streamlit not found. Install it with:")
        print("pip install streamlit")


def main():
    """Main function"""
    print("=" * 60)
    print("ECG Classification - Web App Launcher")
    print("=" * 60 + "\n")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Generate sample data
    generate_sample_data()
    
    # Check models
    has_models = check_models()
    
    if not has_models:
        response = input("\nContinue without models? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Please train models first.")
            sys.exit(0)
    
    # Launch app
    launch_app()


if __name__ == '__main__':
    main()
