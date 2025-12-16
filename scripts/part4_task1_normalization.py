import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Check prerequisites
if 'df' not in locals():
    print("Warning: 'df' not found. Run previous parts first.")
    # For standalone testing, we won't mock full loading here as it's complex.
    # We assume notebook flow.

print("\n--- Task 4.1: Normalization Strategies ---")

# We work with numeric axes
axes = ['acc_x', 'acc_y', 'acc_z']

# Make copies for each strategy to keep them separate
df_z = df.copy()
df_mm = df.copy()
df_subj = df.copy()

# Strategy 1: Per-Axis Standardization (Z-score)
# x' = (x - mean) / std (Global)
for col in axes:
    df_z[col] = (df_z[col] - df_z[col].mean()) / df_z[col].std()

print("Strategy 1 (Z-score) applied.")

# Strategy 2: Min-Max Scaling
# x' = (x - min) / (max - min) -> Range [0, 1]
for col in axes:
    min_val = df_mm[col].min()
    max_val = df_mm[col].max()
    df_mm[col] = (df_mm[col] - min_val) / (max_val - min_val)

print("Strategy 2 (Min-Max) applied.")

# Strategy 3: Per-Subject Standardization
# Standardize each axis separately for each subject
# Group by subject, then apply z-score
if 'subject' in df.columns:
    # Use transform to broadcast back to original shape
    for col in axes:
        # lambda x: (x - x.mean()) / x.std()
        df_subj[col] = df_subj.groupby('subject')[col].transform(lambda x: (x - x.mean()) / x.std())
    print("Strategy 3 (Per-Subject) applied.")
else:
    print("Subject column missing for Strategy 3.")

# Visualization: 5-second window
# We pick an arbitrary 5s window (250 samples at 50Hz)
# Avoid the very start/end or gaps
start_idx = 1000
end_idx = 1250 # 5 seconds * 50 Hz = 250 samples

if len(df) > end_idx:
    window_idx = df.index[start_idx:end_idx]
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Original
    axs[0].plot(window_idx, df.iloc[start_idx:end_idx]['acc_x'], label='acc_x')
    axs[0].set_title("Original Data")
    axs[0].legend()
    
    # Z-score
    axs[1].plot(window_idx, df_z.iloc[start_idx:end_idx]['acc_x'], label='acc_x (Z-score)', color='orange')
    axs[1].set_title("Strategy 1: Z-score Normalization")
    
    # Min-Max
    axs[2].plot(window_idx, df_mm.iloc[start_idx:end_idx]['acc_x'], label='acc_x (Min-Max)', color='green')
    axs[2].set_title("Strategy 2: Min-Max Scaling [0,1]")
    
    # Per-Subject
    axs[3].plot(window_idx, df_subj.iloc[start_idx:end_idx]['acc_x'], label='acc_x (Per-Subject)', color='purple')
    axs[3].set_title("Strategy 3: Per-Subject Standardization")
    
    plt.tight_layout()
    plt.show()

# Questions
print("\n--- Questions to Answer (Task 4.1) ---")

print("a) Plot the same 5-second window...")
print("   (See plots above). Z-score centers around 0. Min-Max shifts to 0-1 (offset). Per-subject adapts to user intensity.")

print("b) Which approach preserves relative differences best?")
print("   Z-score preserves the shape and relative distances well without compressing outliers into a tiny range like Min-Max might.")

print("c) Why might per-subject normalization be important?")
print("   It removes inter-subject variability (calibration offsets, sensor placement, physical strength), focusing the model on activity patterns rather than user identity.")

print("d) Risks of min-max scaling with outliers?")
print("   If outliers are present, the max/min range becomes huge, squashing the useful data into a tiny interval (e.g., 0.49 to 0.51), killing variance.")
