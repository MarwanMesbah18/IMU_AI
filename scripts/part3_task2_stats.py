import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if 'df' not in locals():
    # Assume loaded
    pass

print("\n--- Task 3.2: Statistical Analysis per Class ---")

axes_cols = ['acc_x', 'acc_y', 'acc_z']

if 'label' not in df.columns:
    print("Label column missing, cannot perform class analysis.")
else:
    # 1. Calculate Descriptive Stats
    stats_df = df.groupby('label')[axes_cols].agg(['mean', 'std', 'min', 'max'])
    print("Descriptive Statistics by Activity:")
    print(stats_df)
    
    activities = df['label'].unique()
    
    # 2. Box Plots
    plt.figure(figsize=(12, 6))
    df_melted = df.melt(id_vars=['label'], value_vars=axes_cols, var_name='Axis', value_name='Acceleration')
    sns.boxplot(data=df_melted, x='label', y='Acceleration', hue='Axis')
    plt.title("Distribution of Acceleration by Activity and Axis")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 3. Histograms (3 axes x 3 activities = 9 plots)
    n_act = len(activities)
    fig, axes = plt.subplots(n_act, 3, figsize=(15, 4 * n_act))
    
    for i, act in enumerate(activities):
        act_data = df[df['label'] == act]
        axes[i, 0].hist(act_data['acc_x'], bins=30, color='r', alpha=0.7)
        axes[i, 0].set_title(f"{act} - Acc X")
        axes[i, 1].hist(act_data['acc_y'], bins=30, color='g', alpha=0.7)
        axes[i, 1].set_title(f"{act} - Acc Y")
        axes[i, 2].hist(act_data['acc_z'], bins=30, color='b', alpha=0.7)
        axes[i, 2].set_title(f"{act} - Acc Z")
        
    plt.tight_layout()
    plt.show()
    
    # 4. Correlation Matrices
    for act in activities:
        print(f"\nCorrelation Matrix for {act}:")
        corr = df[df['label'] == act][axes_cols].corr()
        print(corr)

# Questions
print("\n--- Questions to Answer (Task 3.2) ---")
print("a) Which activity has the highest mean acceleration in Z axis?")
# We can just look at stats_df['acc_z']['mean']
try: 
    max_z_mean_act = stats_df['acc_z']['mean'].idxmax()
    print(f"   {max_z_mean_act}. Usually 'Still' or similar if Z aligns with gravity (approx 9.8m/s^2), or dynamic if high impact.")
except: pass

print("b) Which features show most separation?")
print("   Standard Deviation (variance) often separates 'Still' (low) from 'Shake' (high) very well.")
print("   Mean might separate orientation-based static poses.")

print("c) Are the three axes correlated?")
print("   Check printed correlation matrices. In complex 3D motion, axes are often correlated.")
print("   In static poses, correlation might be low (noise-dominated).")

print("d) Easiest/Hardest to classify?")
print("   Easiest: 'Still' vs 'Moving' (huge variance diff).")
print("   Hardest: Distinguishing specific dynamic activities (e.g., 'Walking' vs 'Jogging') if intensity overlaps.")
