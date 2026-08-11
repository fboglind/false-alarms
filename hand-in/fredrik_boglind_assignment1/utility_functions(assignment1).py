"""utility_functions.py"""


import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import wfdb

def classify_signal(name, ecg_leads, pleth_names, abp_names):
    """
    Use: Classify a WFDB signal name into 'ecg', 'pleth', 'abp', or 'unknown'.
    Inputs:
        name: str - the signal name from WFDB record.sig_name
        ecg_leads: list of str - known ECG lead names
        pleth_names: list of str - known plethysmography signal names
        abp_names: list of str - known arterial blood pressure signal names
    Outputs: 
        str - one of 'ecg', 'pleth', 'abp', or 'unknown'
    """
    upper = name.upper().strip()
    if upper in ecg_leads:
        return 'ecg'
    elif upper in pleth_names:
        return 'pleth'
    elif upper in abp_names:
        return 'abp'
    return 'unknown'

def load_event(event_id, record_id, data_dir, ecg_leads, pleth_names, abp_names, target_length=90000):
    """
    Use: Load one event into a fixed (4, target_length) array.
    Inputs:
        event_id: str, e.g., '003c13_0115'
        record_id: str, e.g., '003c13'
        data_dir: Path to the directory containing the waveform data
        ecg_leads: list of str, known ECG lead names
        pleth_names: list of str, known plethysmography signal names
        abp_names: list of str, known arterial blood pressure signal names
        target_length: int, number of samples to load (default 90000 for 6 minutes at 250 Hz)

    Outputs:
         output: np.ndarray of shape (4, target_length) with the loaded signals (ECG1, ECG2, Pleth, ABP)
         available: list of 4 booleans indicating which channels were available (True if signal was loaded, False if not)
    """
    rec = wfdb.rdrecord(str(data_dir / record_id / event_id))
    output = np.zeros((4, target_length), dtype=np.float32)
    available = [False, False, False, False]
    
    ecg_count = 0
    for i, name in enumerate(rec.sig_name):
        sig = rec.p_signal[:, i].astype(np.float32)
        # Handle NaNs by replacing with 0
        sig = np.nan_to_num(sig, nan=0.0)
        length = min(len(sig), target_length)
        sig_type = classify_signal(name, ecg_leads, pleth_names, abp_names)
        
        if sig_type == 'ecg' and ecg_count == 0:
            output[0, :length] = sig[:length]
            available[0] = True
            ecg_count += 1
        elif sig_type == 'ecg' and ecg_count == 1:
            output[1, :length] = sig[:length]
            available[1] = True
            ecg_count += 1
        elif sig_type == 'pleth':
            output[2, :length] = sig[:length]
            available[2] = True
        elif sig_type == 'abp':
            output[3, :length] = sig[:length]
            available[3] = True
    
    return output, available

def load_all_events(df, data_dir, ecg_leads, pleth_names, abp_names, target_length=90000):
    """
    Use: Load all events.
    Inputs:
        df: pandas DataFrame with columns 'event', 'record', and 'decision'
        data_dir: Path to the directory containing the waveform data
        ecg_leads: list of str, ECG lead names
        pleth_names: list of str, plethysmography signal names
        abp_names: list of str, arterial blood pressure signal names
        target_length: int, number of samples to load for each event (default 90000 for 6 minutes at 250 Hz)
    Outputs:
        waveforms: np.ndarray of shape (n_events, 4, target_length) with the loaded signals
        labels: np.ndarray of shape (n_events,) with the binary labels (0 or 1)
        availability: np.ndarray of shape (n_events, 4) with booleans indicating which channels were available for each event
        event_ids: list of str with the event IDs corresponding to each loaded waveform
    
    """
    waveforms = []
    labels = []
    availability = []
    event_ids = []
    failed = []
    
    for idx, row in df.iterrows():
        try:
            wf, avail = load_event(row['event'], row['record'], data_dir, ecg_leads, pleth_names, abp_names, target_length)
            waveforms.append(wf)
            labels.append(row['decision'])
            availability.append(avail)
            event_ids.append(row['event'])
        except Exception as e:
            failed.append((row['event'], str(e)))
    
    if failed:
        print(f"WARNING: {len(failed)} events failed to load:")
        for ev, err in failed[:5]:
            print(f"  {ev}: {err}")
    
    return np.array(waveforms), np.array(labels), np.array(availability), event_ids

# FILTERING and NORMALIZATION FROM VTaC repo:
def filter_all(waveforms, availability, sampling_freq=250, powerline_freq=60):
    """Use: Apply channel-specific filters. Only filters channels that are present.
    Inputs:
    - waveforms: np.ndarray of shape (n_events, 4, n_samples)
    - availability: np.ndarray of shape (n_events, 4) with booleans indicating which channels are present
    - sampling_freq: int, sampling frequency in Hz (default 250)
    - powerline_freq: int, powerline frequency in Hz for notch filter (default 60)
    Outputs:
    - filtered: np.ndarray of shape (n_events, 4, n_samples) with"""
    
    
    def butter_highpass(cutoff, fs, order=2):
        nyq = 0.5 * fs
        b, a = butter(order, cutoff / nyq, btype='high', analog=False)
        return b, a

    def butter_lowpass(cutoff, fs, order=2):
        nyq = 0.5 * fs
        b, a = butter(order, cutoff / nyq, btype='low', analog=False)
        return b, a

    def notch_filter(freq, Q, fs):
        b, a = iirnotch(freq, Q, fs)
        return b, a

    def filter_ecg(signal, fs=250):
        """Highpass 1 Hz → Lowpass 30 Hz → Notch 60 Hz"""
        b, a = butter_highpass(1.0, fs)
        out = filtfilt(b, a, signal)
        b, a = butter_lowpass(30.0, fs)
        out = filtfilt(b, a, out)
        b, a = notch_filter(60, 30, fs)
        out = filtfilt(b, a, out)
        return out

    def filter_ppg(signal, fs=250):
        """Notch 60 Hz → Bandpass 0.5–5 Hz"""
        b, a = notch_filter(60, 30, fs)
        out = filtfilt(b, a, signal)
        b, a = butter(1, [0.5, 5], btype='band', analog=False, fs=fs)
        out = filtfilt(b, a, out)
        return out

    def filter_abp(signal, fs=250):
        """Notch 60 Hz → Lowpass 16 Hz"""
        b, a = notch_filter(60, 30, fs)
        out = filtfilt(b, a, signal)
        b, a = butter_lowpass(16.0, fs)
        out = filtfilt(b, a, out)
        return out

    FILTER_MAP = {0: filter_ecg, 1: filter_ecg, 2: filter_ppg, 3: filter_abp}
    filtered = waveforms.copy()
    for ch_idx, filt_fn in FILTER_MAP.items():
        for i in range(len(filtered)):
            if availability[i, ch_idx]:
                filtered[i, ch_idx] = filt_fn(filtered[i, ch_idx])
    return filtered

def normalize_per_sample(waveforms, availability):
    """Z-score normalise each present channel of each event."""
    normalized = waveforms.copy()
    for i in range(len(normalized)):
        for ch in range(4):
            if availability[i, ch]:
                mu = normalized[i, ch].mean()
                sigma = normalized[i, ch].std()
                if sigma > 0:
                    normalized[i, ch] = (normalized[i, ch] - mu) / sigma
                else:
                    normalized[i, ch] = 0.0
    return normalized

#FEATURE EXTRACTION
def extract_features(waveforms, availability, channel_names):
    """
    Use:
    - Extract statistical features per channel for each event.
    Inputs:
    - waveforms: np.ndarray of shape (n_events, 4, n_samples)
    - availability: np.ndarray of shape (n_events, 4) with booleans indicating which channels are present
    - channel_names: list of str, names of the channels in order
    Outputs:
    - DataFrame with shape (n_events, n_features) where n_features = 5 per channel.
    """
    features = []
    for i in range(len(waveforms)):
        row = {}
        for ch, name in enumerate(channel_names):
            sig = waveforms[i, ch]
            if availability[i, ch]:
                row[f'{name}_mean'] = np.mean(sig)
                row[f'{name}_std'] = np.std(sig)
                row[f'{name}_min'] = np.min(sig)
                row[f'{name}_max'] = np.max(sig)
                row[f'{name}_range'] = np.max(sig) - np.min(sig)
            else:
                for feat in ['mean', 'std', 'min', 'max', 'range']:
                    row[f'{name}_{feat}'] = np.nan
        features.append(row)
    return pd.DataFrame(features)