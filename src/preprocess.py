import os
import wfdb
import numpy as np

# Parameters
record_id = '100'
data_dir = 'data/raw'
output_dir = 'data/processed'
window_size_sec = 0.6   # 0.2s before + 0.4s after the beat
fs = 360  # Sampling frequency (Hz)
window_size = int(window_size_sec * fs)

# Common beat types (you can expand this later)
valid_beats = ['N', 'L', 'R', 'A', 'V']  # Normal, LBBB, RBBB, Atrial, PVC

# Make sure output dir exists
os.makedirs(output_dir, exist_ok=True)

# Load ECG record & annotations
record = wfdb.rdrecord(os.path.join(data_dir, record_id))
annotation = wfdb.rdann(os.path.join(data_dir, record_id), 'atr')

signal = record.p_signal[:, 0]  # use channel 0
samples = annotation.sample
labels = annotation.symbol

X = []
y = []

for i, sample in enumerate(samples):
    label = labels[i]
    if label not in valid_beats:
        continue

    start = sample - int(0.2 * fs)
    end = sample + int(0.4 * fs)

    if start < 0 or end > len(signal):
        continue

    segment = signal[start:end]
    if len(segment) == window_size:
        X.append(segment)
        y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Save
np.save(os.path.join(output_dir, 'X.npy'), X)
np.save(os.path.join(output_dir, 'y.npy'), y)

print(f"[✓] Saved {len(X)} segments to '{output_dir}/X.npy' and labels to 'y.npy'")
