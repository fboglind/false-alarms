"""
VTaC Dataset - Getting Started Guide
=====================================
This script helps you understand and load the VTaC dataset for the false alarm assignment.
"""

import pandas as pd
import numpy as np
import wfdb  # For reading the waveform files
import os
from pathlib import Path

# ==============================================================================
# 1. SET YOUR DATA PATH
# ==============================================================================
DATA_DIR = Path('raw_data/vtac-a-benchmark-dataset-of-ventricular-tachycardia-alarms-from-icu-monitors-1.0/')

# ==============================================================================
# 2. LOAD THE METADATA
# ==============================================================================

def load_metadata():
    """Load the event labels and data splits."""
    
    # Event labels - contains the TRUE/FALSE alarm labels
    labels_df = pd.read_csv(DATA_DIR / 'event_labels.csv')
    print("=== Event Labels ===")
    print(f"Shape: {labels_df.shape}")
    print(f"Columns: {labels_df.columns.tolist()}")
    print(f"\nFirst few rows:\n{labels_df.head()}")
    print(f"\nLabel distribution:\n{labels_df['decision'].value_counts()}")
    
    # Benchmark splits - train/val/test assignments
    splits_df = pd.read_csv(DATA_DIR / 'benchmark_data_split.csv')
    print("\n=== Data Splits ===")
    print(f"Shape: {splits_df.shape}")
    print(f"Columns: {splits_df.columns.tolist()}")
    print(f"\nSplit distribution:\n{splits_df['split'].value_counts()}")
    
    return labels_df, splits_df

# ==============================================================================
# 3. LOAD A SINGLE WAVEFORM RECORD
# ==============================================================================

def load_waveform(record):
    """
    Load a single waveform record using wfdb library.
    
    The record format is typically: 'waveforms/003c13/003c13_0115'
    """
    record_path = DATA_DIR / record
    
    # Read the record
    record = wfdb.rdrecord(str(record_path))
    
    print(f"=== Waveform Record: {record} ===")
    print(f"Signal names: {record.sig_name}")
    print(f"Signal units: {record.units}")
    print(f"Sampling frequency: {record.fs} Hz")
    print(f"Number of samples: {record.sig_len}")
    print(f"Duration: {record.sig_len / record.fs:.2f} seconds")
    print(f"Signal shape: {record.p_signal.shape}")  # (samples, channels)
    
    return record

# ==============================================================================
# 4. VISUALIZE WAVEFORMS
# ==============================================================================

def plot_waveform(record, time_range=None):
    """
    Plot all signals in a waveform record.
    
    Args:
        record: wfdb record object
        time_range: tuple (start_sec, end_sec) or None for full signal
    """
    import matplotlib.pyplot as plt
    
    signals = record.p_signal
    fs = record.fs
    sig_names = record.sig_name
    
    # Create time axis
    time = np.arange(signals.shape[0]) / fs
    
    # Apply time range if specified
    if time_range:
        start_idx = int(time_range[0] * fs)
        end_idx = int(time_range[1] * fs)
        signals = signals[start_idx:end_idx]
        time = time[start_idx:end_idx]
    
    # Plot each signal
    n_signals = signals.shape[1]
    fig, axes = plt.subplots(n_signals, 1, figsize=(14, 3*n_signals), sharex=True)
    
    if n_signals == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, sig_names)):
        ax.plot(time, signals[:, i], linewidth=0.5)
        ax.set_ylabel(f'{name}\n({record.units[i]})')
        ax.grid(True, alpha=0.3)
        
        # Mark missing data (usually zeros or NaN)
        if np.any(signals[:, i] == 0):
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Potential missing')
    
    axes[-1].set_xlabel('Time (seconds)')
    plt.suptitle(f'Waveform signals (fs={fs} Hz)')
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 5. CHECK DATA AVAILABILITY PER RECORD
# ==============================================================================

def analyze_signal_availability(labels_df, sample_size=100):
    """
    Check which signals are available across records.
    This is important for handling missing data!
    """
    signal_counts = {}
    
    # Sample some records
    sampled = labels_df.sample(min(sample_size, len(labels_df)), random_state=42)
    
    for _, row in sampled.iterrows():
        record = row['record']  # Adjust column name if different
        try:
            record_path = DATA_DIR / 'waveforms' / record[:6] / record
            record = wfdb.rdrecord(str(record_path))
            
            for sig in record.sig_name:
                sig_type = categorize_signal(sig)
                signal_counts[sig_type] = signal_counts.get(sig_type, 0) + 1
        except Exception as e:
            print(f"Could not load {record}: {e}")
    
    print("\n=== Signal Availability (sampled records) ===")
    for sig, count in sorted(signal_counts.items(), key=lambda x: -x[1]):
        print(f"{sig}: {count}/{sample_size} ({100*count/sample_size:.1f}%)")
    
    return signal_counts

def categorize_signal(sig_name):
    """Categorize signal type from its name."""
    sig_name = sig_name.upper()
    if 'ECG' in sig_name or sig_name in ['I', 'II', 'III', 'V', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
        return 'ECG'
    elif 'PLETH' in sig_name or 'PPG' in sig_name:
        return 'PPG'
    elif 'ABP' in sig_name or 'ART' in sig_name:
        return 'ABP'
    else:
        return f'OTHER:{sig_name}'

# ==============================================================================
# 6. PREPARE DATA FOR MODELING
# ==============================================================================

def prepare_dataset(labels_df, splits_df, max_samples=None):
    """
    Load and prepare the full dataset.
    
    Returns a dictionary with train/val/test splits.
    """
    # Merge labels with splits
    df = labels_df.merge(splits_df, on='record')  # Adjust column names as needed
    
    datasets = {}
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        if max_samples:
            split_df = split_df.head(max_samples)
        
        X = []
        y = []
        records = []
        
        for _, row in split_df.iterrows():
            try:
                record_path = DATA_DIR / 'waveforms' / row['record'][:6] / row['record']
                record = wfdb.rdrecord(str(record_path))
                
                X.append(record.p_signal)
                y.append(1 if row['decision'] == 'True' else 0)
                records.append(row['record'])
            except Exception as e:
                print(f"Skipping {row['record']}: {e}")
        
        datasets[split] = {
            'X': X,  # List of arrays (different shapes possible!)
            'y': np.array(y),
            'names': records
        }
        print(f"{split}: {len(X)} samples, {sum(y)} true alarms, {len(y)-sum(y)} false alarms")
    
    return datasets

# ==============================================================================
# 7. FEATURE EXTRACTION IDEAS
# ==============================================================================

def extract_basic_features(signal, fs=250):
    """
    Extract basic statistical features from a single signal.
    
    These are starting points - you should explore more sophisticated features!
    """
    features = {}
    
    # Basic statistics
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['min'] = np.min(signal)
    features['max'] = np.max(signal)
    features['range'] = features['max'] - features['min']
    
    # Higher-order statistics
    features['skewness'] = pd.Series(signal).skew()
    features['kurtosis'] = pd.Series(signal).kurtosis()
    
    # Signal quality indicators
    features['zero_crossings'] = np.sum(np.diff(np.sign(signal)) != 0)
    features['rms'] = np.sqrt(np.mean(signal**2))
    
    # Frequency domain (basic)
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    power = np.abs(fft)**2
    
    # Dominant frequency
    pos_mask = freqs > 0
    features['dominant_freq'] = freqs[pos_mask][np.argmax(power[pos_mask])]
    
    return features

# ==============================================================================
# MAIN - RUN THIS TO EXPLORE YOUR DATA
# ==============================================================================

if __name__ == "__main__":
    print("VTaC Dataset Explorer")
    print("=" * 50)
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        print(f"\nERROR: Data directory not found at {DATA_DIR}")
        print("Please update DATA_DIR to point to your dataset location.")
    else:
        # 1. Load metadata
        labels_df, splits_df = load_metadata()
        
        # 2. Load and visualize a sample waveform
        # Get first record name from labels
        sample_record = labels_df.iloc[0]['record']  # Adjust column name if needed
        
        print(f"\n\nLoading sample record: {sample_record}")
        record = load_waveform(f'waveforms/{sample_record[:6]}/{sample_record}')
        
        # 3. Plot the waveform (first 10 seconds)
        plot_waveform(record, time_range=(0, 10))
        
        print("\n" + "=" * 50)
        print("Next steps:")
        print("1. Explore the data distribution (class balance)")
        print("2. Check for missing signals across records")
        print("3. Decide on your preprocessing strategy")
        print("4. Extract features or prepare for deep learning")
        print("=" * 50)