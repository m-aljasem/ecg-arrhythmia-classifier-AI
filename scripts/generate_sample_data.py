"""
Generate sample ECG data for testing the Streamlit app

This script creates sample ECG CSV files that can be uploaded to the app.
"""

import numpy as np
import pandas as pd
import os

def generate_sample_ecg(duration_seconds=10, sampling_rate=100, num_leads=12, 
                       pattern='normal', save_path='data/sample_ecg.csv'):
    """
    Generate synthetic ECG data for testing
    
    Args:
        duration_seconds: Length of ECG in seconds
        sampling_rate: Samples per second
        num_leads: Number of ECG leads (standard is 12)
        pattern: Type of ECG pattern ('normal', 'abnormal', 'mi')
        save_path: Where to save the CSV file
    """
    num_samples = duration_seconds * sampling_rate
    t = np.linspace(0, duration_seconds, num_samples)
    
    # Initialize ECG data
    ecg_data = np.zeros((num_samples, num_leads))
    
    # Generate different patterns for each lead
    for lead in range(num_leads):
        # Base heartbeat pattern (simplified QRS complex)
        heartbeat = np.zeros_like(t)
        heart_rate = 72  # beats per minute
        beat_interval = 60 / heart_rate  # seconds between beats
        
        for beat_time in np.arange(0, duration_seconds, beat_interval):
            # P wave
            p_center = beat_time + 0.1
            p_wave = 0.15 * np.exp(-((t - p_center) ** 2) / (2 * 0.02 ** 2))
            
            # QRS complex
            qrs_center = beat_time + 0.25
            qrs_wave = 0.8 * np.exp(-((t - qrs_center) ** 2) / (2 * 0.015 ** 2))
            
            # T wave
            t_center = beat_time + 0.45
            t_wave = 0.25 * np.exp(-((t - t_center) ** 2) / (2 * 0.04 ** 2))
            
            heartbeat += p_wave + qrs_wave + t_wave
        
        # Add lead-specific variations
        lead_variation = 0.8 + (lead * 0.05)  # Different amplitudes
        phase_shift = lead * 0.1  # Phase differences
        
        ecg_data[:, lead] = heartbeat * lead_variation + \
                           0.05 * np.sin(2 * np.pi * 0.5 * t + phase_shift)
        
        # Add realistic noise
        ecg_data[:, lead] += np.random.normal(0, 0.02, num_samples)
    
    # Apply pattern modifications
    if pattern == 'abnormal':
        # Simulate irregular rhythm
        ecg_data[:, :] *= (1 + 0.2 * np.random.randn(num_samples, 1))
    elif pattern == 'mi':
        # Simulate ST elevation (MI indicator)
        for lead in range(num_leads):
            ecg_data[:, lead] += 0.3 * np.ones(num_samples)
    
    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(ecg_data)
    df.to_csv(save_path, index=False, header=False)
    
    print(f"✓ Sample ECG generated: {save_path}")
    print(f"  Shape: {ecg_data.shape}")
    print(f"  Pattern: {pattern}")
    print(f"  Duration: {duration_seconds}s at {sampling_rate}Hz")
    
    return ecg_data


if __name__ == '__main__':
    # Generate different sample files
    print("Generating sample ECG files for testing...\n")
    
    # Normal ECG
    generate_sample_ecg(
        duration_seconds=10,
        sampling_rate=100,
        pattern='normal',
        save_path='data/sample_ecg_normal.csv'
    )
    
    # Abnormal ECG
    generate_sample_ecg(
        duration_seconds=10,
        sampling_rate=100,
        pattern='abnormal',
        save_path='data/sample_ecg_abnormal.csv'
    )
    
    # MI pattern
    generate_sample_ecg(
        duration_seconds=10,
        sampling_rate=100,
        pattern='mi',
        save_path='data/sample_ecg_mi.csv'
    )
    
    print("\n✅ All sample files generated successfully!")
    print("\nYou can upload these files to the Streamlit app for testing.")
