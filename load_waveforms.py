"""
VTaC Data Loader - Corrected Version
=====================================
Based on actual file structure:
- event_labels.csv: record, event, decision
- benchmark_data_split.csv: split, record, event
"""

import pandas as pd
import numpy as np
import wfdb
from pathlib import Path
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION - UPDATE THIS PATH
# ==============================================================================
DATA_DIR = Path('raw_data/vtac-a-benchmark-dataset-of-ventricular-tachycardia-alarms-from-icu-monitors-1.0/')


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_and_merge_metadata():
    """
    Load event labels and splits, merge them into a single DataFrame.
    
    Returns:
        DataFrame with columns: split, record, event, decision
    """
    # Load both files
    labels_df = pd.read_csv(DATA_DIR / 'event_labels.csv')
    splits_df = pd.read_csv(DATA_DIR / 'benchmark_data_split.csv')
    
    # Merge on record AND event
    df = splits_df.merge(labels_df, on=['record', 'event'], how='inner')
    
    print("=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"\nSplit distribution:")
    print(df['split'].value_counts())
    print(f"\nLabel distribution (decision):")
    print(df['decision'].value_counts())
    print(f"\nClass balance: {df['decision'].mean():.1%} True alarms")
    
    # Cross-tabulation
    print(f"\nLabels per split:")
    print(pd.crosstab(df['split'], df['decision']))
    
    return df


def load_waveform(event_name, record_id=None):
    """
    Load a single waveform.
    
    Args:
        event_name: e.g., '003c13_0115'
        record_id: e.g., '003c13' (optional, extracted from event_name if not provided)
    
    Returns:
        wfdb record object with:
        - record.p_signal: numpy array (n_samples, n_channels)
        - record.sig_name: list of signal names
        - record.fs: sampling frequency (250 Hz)
    """
    if record_id is None:
        record_id = event_name.split('_')[0]
    
    path = DATA_DIR / 'waveforms' / record_id / event_name
    record = wfdb.rdrecord(str(path))
    
    return record


def explore_waveform(event_name):
    """Load and print detailed info about a waveform."""
    record = load_waveform(event_name)
    
    print(f"\n=== Waveform: {event_name} ===")
    print(f"Sampling frequency: {record.fs} Hz")
    print(f"Duration: {record.sig_len / record.fs:.2f} seconds ({record.sig_len} samples)")
    print(f"Number of signals: {len(record.sig_name)}")
    print(f"\nSignals:")
    
    for i, (name, unit) in enumerate(zip(record.sig_name, record.units)):
        signal = record.p_signal[:, i]
        is_missing = np.all(signal == 0) or np.all(np.isnan(signal))
        
        if is_missing:
            status = "⚠️  MISSING (all zeros)"
        else:
            status = f"min={signal.min():.2f}, max={signal.max():.2f}, mean={signal.mean():.2f}"
        
        print(f"  [{i}] {name:10s} ({unit:8s}): {status}")
    
    return record


def analyze_signal_availability(df, sample_size=100):
    """
    Check which signals are available across the dataset.
    Important for understanding missing data!
    """
    print(f"\n=== Signal Availability Analysis (n={sample_size}) ===")
    
    # Sample records
    sampled = df.sample(min(sample_size, len(df)), random_state=42)
    
    signal_stats = {}
    all_signal_names = set()
    
    for _, row in sampled.iterrows():
        try:
            record = load_waveform(row['event'], row['record'])
            
            for i, name in enumerate(record.sig_name):
                all_signal_names.add(name)
                
                if name not in signal_stats:
                    signal_stats[name] = {'present': 0, 'missing': 0}
                
                signal = record.p_signal[:, i]
                if np.all(signal == 0) or np.all(np.isnan(signal)):
                    signal_stats[name]['missing'] += 1
                else:
                    signal_stats[name]['present'] += 1
                    
        except Exception as e:
            print(f"  Error loading {row['event']}: {e}")
    
    print("\nSignal availability:")
    for name in sorted(signal_stats.keys()):
        stats = signal_stats[name]
        total = stats['present'] + stats['missing']
        pct = stats['present'] / total * 100 if total > 0 else 0
        print(f"  {name:12s}: {stats['present']:3d}/{total:3d} available ({pct:.1f}%)")
    
    return signal_stats


def plot_waveform(event_name, time_range=None, save=False):
    """
    Plot all signals from a waveform.
    
    Args:
        event_name: e.g., '003c13_0115'
        time_range: tuple (start_sec, end_sec) or None for full signal
        save: if True, save to file
    """
    record = load_waveform(event_name)
    signals = record.p_signal
    fs = record.fs
    
    # Time axis
    if time_range:
        start_idx = int(time_range[0] * fs)
        end_idx = int(time_range[1] * fs)
    else:
        start_idx = 0
        end_idx = signals.shape[0]
    
    time = np.arange(start_idx, end_idx) / fs
    
    n_signals = signals.shape[1]
    fig, axes = plt.subplots(n_signals, 1, figsize=(14, 2.5*n_signals), sharex=True)
    
    if n_signals == 1:
        axes = [axes]
    
    for i, (ax, name, unit) in enumerate(zip(axes, record.sig_name, record.units)):
        signal_slice = signals[start_idx:end_idx, i]
        
        if np.all(signal_slice == 0):
            ax.text(0.5, 0.5, 'SIGNAL MISSING', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12, color='red')
        else:
            ax.plot(time, signal_slice, linewidth=0.5)
        
        ax.set_ylabel(f'{name}\n({unit})')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (seconds)')
    plt.suptitle(f'{event_name} (fs={fs} Hz)', fontsize=12)
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{event_name}_plot.png', dpi=150, bbox_inches='tight')
        print(f"Saved to {event_name}_plot.png")
    
    plt.show()
    return fig


# ==============================================================================
# DATA PREPARATION FOR MODELING
# ==============================================================================

def get_split_data(df, split):
    """
    Get all events for a specific split.
    
    Args:
        df: merged DataFrame
        split: 'train', 'val', or 'test'
    
    Returns:
        DataFrame filtered to that split
    """
    return df[df['split'] == split].copy()


def load_all_waveforms_for_split(df, split, max_samples=None):
    """
    Load all waveforms for a split.
    
    WARNING: This loads everything into memory. For large datasets,
    consider using a data loader/generator instead.
    
    Returns:
        X: list of numpy arrays (different shapes possible!)
        y: numpy array of labels (1=True alarm, 0=False alarm)
        events: list of event names
    """
    split_df = get_split_data(df, split)
    
    if max_samples:
        split_df = split_df.head(max_samples)
    
    X = []
    y = []
    events = []
    signal_names_list = []
    
    for _, row in split_df.iterrows():
        try:
            record = load_waveform(row['event'], row['record'])
            X.append(record.p_signal)
            y.append(1 if row['decision'] == True else 0)
            events.append(row['event'])
            signal_names_list.append(record.sig_name)
        except Exception as e:
            print(f"Error loading {row['event']}: {e}")
    
    y = np.array(y)
    
    print(f"\n{split} set loaded:")
    print(f"  Samples: {len(X)}")
    print(f"  True alarms: {y.sum()} ({y.mean():.1%})")
    print(f"  False alarms: {len(y) - y.sum()} ({1-y.mean():.1%})")
    
    return X, y, events, signal_names_list


# ==============================================================================
# MAIN - RUN THIS TO EXPLORE YOUR DATA
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VTaC Dataset Explorer")
    print("=" * 60)
    
    # Check data directory exists
    if not DATA_DIR.exists():
        print(f"\n❌ ERROR: Data directory not found!")
        print(f"   Expected: {DATA_DIR}")
        print(f"   Please update DATA_DIR in this script.")
        exit(1)
    
    # 1. Load and merge metadata
    print("\n1. Loading metadata...")
    df = load_and_merge_metadata()
    
    # 2. Explore a sample waveform
    print("\n2. Exploring sample waveforms...")
    sample_events = df['event'].head(3).tolist()
    for event in sample_events:
        explore_waveform(event)
    
    # 3. Analyze signal availability
    print("\n3. Analyzing signal availability...")
    signal_stats = analyze_signal_availability(df, sample_size=50)
    
    # 4. Plot first sample
    print("\n4. Plotting first sample (first 10 seconds)...")
    plot_waveform(sample_events[0], time_range=(0, 10), save=True)
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("1. Check signal availability - some signals may be missing!")
    print("2. Decide how to handle missing signals (imputation vs deletion)")
    print("3. Apply preprocessing (filtering + normalization)")
    print("4. Extract features or prepare for deep learning")
    print("=" * 60)