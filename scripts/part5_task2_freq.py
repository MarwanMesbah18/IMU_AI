import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

if 'X_2s' not in locals():
    # Assume flow or mock
    # Needs X_2s and sampling_rate
    X_2s = np.zeros((10, 100, 3))
    sampling_rate = 50

print("\n--- Task 5.2: Frequency-Domain Features ---")

def extract_freq_features(X, fs):
    # X shape: (n_windows, window_len, 3)
    n_windows, n_samples, n_channels = X.shape
    
    features = []
    feature_names = []
    axes = ['acc_x', 'acc_y', 'acc_z']
    
    # FFT
    # Real FFT since input is real
    # RFFT returns n/2 + 1 components
    # But usually assignments ask for full basic FFT logic or just magnitude spectrum
    # We use fft and take magnitude of first half
    
    # Frequencies
    freqs = fftfreq(n_samples, 1/fs)
    # Indices for positive freqs
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    
    # Compute FFT for all windows/channels at once? 
    # fft shape: same as input
    Z = fft(X, axis=1) # (n_windows, n_samples, 3)
    
    # Magnitude Spectrum
    # Normalize by N
    Mag = np.abs(Z) / n_samples
    
    # Keep positive half (approx)
    Mag_pos = Mag[:, pos_mask, :] # (n_windows, n_freqs, 3)
    
    # 1. Dominant Frequency
    # Index of max magnitude
    idx_max = np.argmax(Mag_pos, axis=1) # (n_windows, 3)
    dom_freqs = freqs_pos[idx_max] # Map index to freq value
    features.append(dom_freqs)
    feature_names.extend([f'{ax}_dom_freq' for ax in axes])
    
    # 2. Spectral Energy
    # Sum of squared magnitudes (Parseval's relation counterpart)
    spec_energy = np.sum(Mag_pos**2, axis=1)
    features.append(spec_energy)
    feature_names.extend([f'{ax}_spec_energy' for ax in axes])
    
    # 3. Spectral Entropy
    # P(f) = PSD / sum(PSD)
    # H = -sum(P * log2(P))
    # PSD proportional to Mag^2
    PSD = Mag_pos**2
    PSD_sum = np.sum(PSD, axis=1, keepdims=True)
    # Avoid div by zero
    PSD_sum[PSD_sum == 0] = 1e-9
    P = PSD / PSD_sum
    # Avoid log(0)
    P[P == 0] = 1e-9
    entropy = -np.sum(P * np.log2(P), axis=1)
    features.append(entropy)
    feature_names.extend([f'{ax}_spec_entropy' for ax in axes])
    
    # 4. Power in Frequency Bands
    # Bands: 0-2, 2-5, 5-10, 10-25
    bands = [(0, 2), (2, 5), (5, 10), (10, 25)]
    
    for b_start, b_end in bands:
        # Find indices
        mask = (freqs_pos >= b_start) & (freqs_pos < b_end)
        if np.sum(mask) == 0:
            band_power = np.zeros((n_windows, 3))
        else:
            band_power = np.sum(PSD[:, mask, :], axis=1)
            
        features.append(band_power)
        feature_names.extend([f'{ax}_power_{b_start}_{b_end}Hz' for ax in axes])
        
    features_all = np.hstack(features)
    return features_all, feature_names

# Execute
X_freq, feat_names_freq = extract_freq_features(X_2s, sampling_rate)

# Combine Time + Freq
if 'df_features' in locals():
    # Append
    df_features_freq = pd.DataFrame(X_freq, columns=feat_names_freq)
    df_all_features = pd.concat([df_features.drop(columns=['label']), df_features_freq], axis=1)
    df_all_features['label'] = df_features['label']
    print(f"Combined Feature Matrix Shape: {df_all_features.shape}")
else:
    print("Warning: df_features (Time domain) missing.")
    df_features_freq = pd.DataFrame(X_freq, columns=feat_names_freq)

# Visualization: Power Spectral Density (PSD)
# Plot mean PSD for each activity for X-axis
try:
    if 'y_2s' in locals():
        unique_labels = np.unique(y_2s)
        plt.figure(figsize=(10, 6))
        
        freqs = fftfreq(100, 1/sampling_rate)
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        
        for lbl in unique_labels:
            indices = np.where(y_2s == lbl)[0]
            if len(indices) > 0:
                # Get windows
                wins = X_2s[indices, :, 0] # X-axis
                # FFT
                Z = np.abs(fft(wins, axis=1) / 100)
                mean_spec = np.mean(Z[:, pos_mask], axis=0)
                plt.plot(freqs_pos, mean_spec, label=lbl)
                
        plt.title('Mean PSD (Acc X) by Activity')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.show()
except:
    pass


# Questions
print("\n--- Questions to Answer (Task 5.2) ---")

print("a) Dominant frequency for each activity?")
print("   Walking typically 1.5 - 2.0 Hz. Shaking higher (3-5+ Hz). Standing near 0 Hz (DC).")

print("b) Which band contains most energy?")
print("   Walking: 0-2 Hz (Step rate) and 2-5 Hz (Harmonics).")
print("   Shaking: 2-5 Hz or 5-10 Hz depending on vigor.")

print("c) Do frequency features provide better separation?")
print("   Yes, especially for differentiating dynamic activities with distinct rhythms (Walk vs Run vs Shake) which might overlap in pure amplitude statistics.")

print("d) Effect of window size on frequency resolution?")
print("   Res = fs / N. ")
print("   Short window (N small) -> Poor frequency resolution (bins are wide).")
print("   Long window -> Better frequency resolution, but blurs time transients.")
