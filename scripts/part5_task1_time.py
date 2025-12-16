import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

if 'X_2s' not in locals():
    print("Warning: 'X_2s' (segments) not found. Run previous parts first.")
    # For standalone structure check (mock empty)
    X_2s = np.zeros((10, 100, 3)) # 10 windows, 100 samples, 3 axes
    y_2s = np.array(['mock'] * 10)

print("\n--- Task 5.1: Time-Domain Features ---")

# Input: X_2s (NumPy array of shape [n_windows, window_size, 3])
# y_2s (Labels)

def extract_time_features(X):
    # X shape: (n_windows, window_len, 3) 
    # Axis 0: Windows, Axis 1: Time, Axis 2: Channels (x, y, z)
    
    features = []
    feature_names = []
    
    # Axes names
    axes = ['acc_x', 'acc_y', 'acc_z']
    
    # 1. Basic Stats per Axis
    # Mean
    mean_vals = np.mean(X, axis=1) # Shape: (n_windows, 3)
    features.append(mean_vals)
    feature_names.extend([f'{ax}_mean' for ax in axes])
    
    # Std
    std_vals = np.std(X, axis=1)
    features.append(std_vals)
    feature_names.extend([f'{ax}_std' for ax in axes])
    
    # Min
    min_vals = np.min(X, axis=1)
    features.append(min_vals)
    feature_names.extend([f'{ax}_min' for ax in axes])
    
    # Max
    max_vals = np.max(X, axis=1)
    features.append(max_vals)
    feature_names.extend([f'{ax}_max' for ax in axes])
    
    # Range
    range_vals = max_vals - min_vals
    features.append(range_vals)
    feature_names.extend([f'{ax}_range' for ax in axes])
    
    # 2. Signal Characteristics
    # SMA: 1/N * sum(|x| + |y| + |z|) for each time step -> sum across axes then mean over time
    # Actually usually defined as: 1/T * Integral(|x(t)| + |y(t)| + |z(t)|) dt
    # Here discrete: Mean( Sum(|x|, |y|, |z|) )
    abs_sum = np.sum(np.abs(X), axis=2) # Sum across axes -> (n_windows, window_len)
    sma = np.mean(abs_sum, axis=1).reshape(-1, 1) # Mean over time -> (n_windows, 1)
    features.append(sma)
    feature_names.append('sma')
    
    # Energy: 1/N * sum(x^2)
    energy = np.mean(X**2, axis=1)
    features.append(energy)
    feature_names.extend([f'{ax}_energy' for ax in axes])
    
    # ZCR: Count sign changes
    # (x[i] * x[i-1] < 0)
    # We diff signs
    # Shape: (n_windows, window_len, 3)
    features_zcr = []
    for ax_idx in range(3):
        # Slice: (n_windows, window_len)
        sig = X[:, :, ax_idx]
        # Sign bit
        signs = np.sign(sig)
        # Fix sign(0) issues? sign(0)=0. 
        # Diff of signs != 0 implies cross.
        diffs = np.diff(signs, axis=1)
        zcr = np.sum(np.abs(diffs) > 0, axis=1) / 2 # Each cross creates diff of 2 or 1.
        # Simple count:
        zcr = np.sum(np.diff(np.signbit(sig), axis=1), axis=1)
        features_zcr.append(zcr.reshape(-1, 1))
        
    features.append(np.hstack(features_zcr))
    feature_names.extend([f'{ax}_zcr' for ax in axes])
    
    # RMS: sqrt(1/N * sum(x^2)) = sqrt(Energy)
    rms = np.sqrt(energy)
    features.append(rms)
    feature_names.extend([f'{ax}_rms' for ax in axes])
    
    # 3. Magnitude Features
    # Mag = sqrt(x^2 + y^2 + z^2) at each step
    # Shape: (n_windows, window_len)
    mag = np.sqrt(np.sum(X**2, axis=2))
    
    # Mean Mag
    mag_mean = np.mean(mag, axis=1).reshape(-1, 1)
    features.append(mag_mean)
    feature_names.append('mag_mean')
    
    # Std Mag
    mag_std = np.std(mag, axis=1).reshape(-1, 1)
    features.append(mag_std)
    feature_names.append('mag_std')
    
    # Concatenate all
    # List of (n_windows, k) arrays
    features_all = np.hstack(features)
    
    return features_all, feature_names

# Requirements:
# 1. Create feature matrix
X_features, feat_names = extract_time_features(X_2s)
df_features = pd.DataFrame(X_features, columns=feat_names)
df_features['label'] = y_2s # Add label

# 2. Document total number
n_features = len(feat_names)
print(f"Total features extracted per window: {n_features}")
print("Feature names example:", feat_names[:5])

# 3. Create DataFrame (Done above)

# Questions
print("\n--- Questions to Answer (Task 5.1) ---")

print(f"a) How many total features do you have per window?")
print(f"   {n_features} features.")

print("b) Which time-domain features show best separation?")
# Simple visualization: Boxplot of a few top features (estimated)
# Mag_mean and Energy often separate Activity vs Rest
try:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='label', y='sma', data=df_features)
    plt.title("Separation by Signal Magnitude Area (SMA)")
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='label', y='acc_y_range', data=df_features)
    plt.title("Separation by Y-Axis Range")
    plt.show()
except:
    pass

print("   SMA and Energy typically separate static (Sitting/Standing) from dynamic (Walking/Running) well.")

print("c) Are some features redundant?")
print("   Yes. RMS and Energy are directly related (Energy = RMS^2).")
print("   Mean and Min/Max might be correlated if signal is offset/gravity dominated.")
