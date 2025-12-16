import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Safe check: if df is not loaded, this cell will error naturally in a notebook, 
# or we can print a warning. For clean cells, we assume valid state.
if 'df' not in locals() or 'label' not in df.columns:
    print("Warning: 'df' not found or 'label' missing. Run previous parts first.")

print("\n--- Task 3.1: Time-Series Visualization ---")

# 4. Calculate Acceleration Magnitude
# ||a|| = sqrt(ax^2 + ay^2 + az^2)
# We use .get() or check locally to satisfy linter, but for specific notebook flow:
if 'acc_mag' not in df.columns:
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
print("Calculated Acceleration Magnitude.")

# 1. & 2. Plot 10-second segment for each activity class
activities = df['label'].unique()
print(f"Activities found: {activities}")

fig, axes = plt.subplots(len(activities), 1, figsize=(12, 4 * len(activities)), sharex=False)
if len(activities) == 1: axes = [axes] # Handle single activity case

for i, activity in enumerate(activities):
    # Get data for this activity
    act_data = df[df['label'] == activity]
    
    # Take a 10s segment (50Hz * 10s = 500 samples)
    if len(act_data) > 500:
        start_idx = len(act_data) // 2
        segment = act_data.iloc[start_idx : start_idx + 500]
    else:
        segment = act_data
        
    ax = axes[i]
    ax.plot(segment.index, segment['acc_x'], label='acc_x', alpha=0.7)
    ax.plot(segment.index, segment['acc_y'], label='acc_y', alpha=0.7)
    ax.plot(segment.index, segment['acc_z'], label='acc_z', alpha=0.7)
    ax.plot(segment.index, segment['acc_mag'], label='Magnitude', color='black', linestyle='--', linewidth=1.5)
    
    ax.set_title(f"Activity: {activity} (10s Segment)")
    ax.set_ylabel("Acceleration (m/s^2)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
plt.tight_layout()
plt.show()

# Questions
print("\n--- Questions to Answer (Task 3.1) ---")
print("a) Characteristic patterns:")
print("   - Stationary: Flat lines, near zero (gravity affects z mostly).")
print("   - Walking: Periodic, rhythmic peaks, moderate amplitude.")
print("   - Shaking/Running: High frequency, high amplitude, chaotic or fast periodic.")

print("b) Which activity has the highest variance?")
print("   Typically 'Shake' or 'Run' has the highest variance due to rapid direction changes.")

print("c) Can you visually identify activity transitions?")
print("   Yes, abrupt changes in signal amplitude and frequency usually mark transitions.")

print("d) Does acceleration magnitude provide better separation?")
print("   Magnitude is rotation-invariant, simplifying 3D motion into 1D energy.")
