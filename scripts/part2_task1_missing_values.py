import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure df is loaded (for standalone testing)
if 'df' not in locals():
    df = pd.read_csv('dataset/imu_messy_data.csv')
    print("Data loaded from CSV.")

print("\n--- Task 2.1: Handling Missing Values ---")

# Setup for comparison
axes = ['acc_x', 'acc_y', 'acc_z']
df_orig = df.copy()

# Strategy A: Forward Fill
df_ffill = df_orig.copy()
df_ffill[axes] = df_ffill[axes].fillna(method='ffill')
missing_ffill = df_ffill[axes].isnull().sum().sum()
print(f"Strategy A (Forward Fill) - Missing values remaining: {missing_ffill}")

# Strategy B: Linear Interpolation
df_interp = df_orig.copy()
df_interp[axes] = df_interp[axes].interpolate(method='linear')
missing_interp = df_interp[axes].isnull().sum().sum()
print(f"Strategy B (Linear Interpolation) - Missing values remaining: {missing_interp}")

# Strategy C: Drop Rows
df_drop = df_orig.copy()
initial_rows = len(df_drop)
df_drop = df_drop.dropna(subset=axes)
dropped_rows = len(df_drop)
percent_lost = ((initial_rows - dropped_rows) / initial_rows) * 100
print(f"Strategy C (Drop Rows) - Data lost: {percent_lost:.2f}%")

# Questions
print("\n--- Questions to Answer (Task 2.1) ---")
print("a) Which strategy preserves the most data?")
if percent_lost > 0:
    print(f"   Forward Fill and Interpolation preserve 100% of rows (if start is valid), while Drop Rows lost {percent_lost:.2f}%.")
else:
    print("   All strategies preserved all data (no missing values found?).")

print("b) Plotting comparison for a segment with missing values...")
# Find a segment with NaNs in original data
nan_indices = df_orig[df_orig[axes].isnull().any(axis=1)].index
if not nan_indices.empty:
    sample_idx = nan_indices[0]
    start = max(0, sample_idx - 10)
    end = min(len(df_orig), sample_idx + 10)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_orig.iloc[start:end]['acc_x'], 'o-', label='Original', color='black', alpha=0.5)
    plt.plot(df_ffill.iloc[start:end]['acc_x'], 'x--', label='Forward Fill', alpha=0.7)
    plt.plot(df_interp.iloc[start:end]['acc_x'], 's--', label='Interpolation', alpha=0.7)
    plt.title(f"Comparison of Missing Value Strategies (Sample around index {sample_idx})")
    plt.legend()
    plt.show()
    print("   (Plot generated)")
else:
    print("   No missing values found to plot.")

print("c) What are the trade-offs?")
print("   - Forward Fill: Simple, causal, but steps values (unrealistic for continuous motion).")
print("   - Interpolation: Smoother, more realistic for physical signals, but requires future data (non-causal if standard).")
print("   - Drop Rows: Preserves true distribution but loses temporal continuity/data quantity.")

print("d) Recommendation:")
print("   Linear Interpolation is recommended for IMU sensor data as physics dictates continuity, and gaps are likely small.")

# Apply chosen strategy for next tasks (Interpolation)
df = df_interp
print("Applied Linear Interpolation for subsequent tasks.")
