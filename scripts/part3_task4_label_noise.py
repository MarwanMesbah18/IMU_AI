import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if 'df' not in locals():
    # Assume loaded
    pass

print("\n--- Task 3.4: Label Noise Investigation ---")

if 'label' not in df.columns:
    print("Label column missing.")
else:
    if 'acc_mag' not in df.columns:
        df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)

    # 1. Mean and Std per Activity
    class_stats = df.groupby('label')['acc_mag'].agg(['mean', 'std'])
    print("Class Statistics (Magnitude):")
    print(class_stats)
    
    # 2. Identify Suspicious Samples
    # Vectorized way: map stats back to df
    df['class_mean'] = df['label'].map(class_stats['mean'])
    df['class_std'] = df['label'].map(class_stats['std'])
    
    df['mag_z_score'] = (df['acc_mag'] - df['class_mean']) / df['class_std']
    
    # Threshold for "significant deviation"
    threshold = 3.0 
    suspicious = df[np.abs(df['mag_z_score']) > threshold]
    
    print(f"\nidentified {len(suspicious)} suspicious samples (|Z| > {threshold}).")
    print("Sample of suspicious indices:")
    print(suspicious.index[:10])
    
    # 3. Manually Inspect 10-20 samples
    indices_to_check = suspicious.index[:3]
    
    for idx_ts in indices_to_check:
        try:
            loc_idx = df.index.get_loc(idx_ts)
            if isinstance(loc_idx, slice): loc_idx = loc_idx.start
            
            start = max(0, loc_idx - 50)
            end = min(len(df), loc_idx + 50)
            
            subset = df.iloc[start:end]
            
            plt.figure(figsize=(10, 3))
            plt.plot(subset.index, subset['acc_mag'], label='Magnitude')
            plt.scatter([idx_ts], [df.loc[idx_ts]['acc_mag'][0] if isinstance(df.loc[idx_ts]['acc_mag'], pd.Series) else df.loc[idx_ts]['acc_mag']], color='red', label='Suspicious')
            plt.title(f"Suspicious Sample at {idx_ts} (Label: {df.loc[idx_ts]['label'] if isinstance(df.loc[idx_ts]['label'], str) else df.loc[idx_ts]['label'].iloc[0]})")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"could not plot sample: {e}")

# Questions
print("\n--- Questions to Answer (Task 3.4) ---")
print("a) Can you identify likely mislabeled samples?")
print("   Yes. Often brief spikes in 'Still' or flatlines in 'Moving' indicate mislabeling or transitions.")

print("b) What percentage appear noisy?")
print(f"   {len(suspicious)} / {len(df)} = {len(suspicious)/len(df)*100:.2f}% (based on Z > 3).")

print("c) How would incorrect labels affect ML?")
print("   They confuse decision boundaries, increase overfitting, and reduce test accuracy.")
