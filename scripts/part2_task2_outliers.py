import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Ensure df is loaded (for standalone testing)
if 'df' not in locals():
    # Assuming standard flow, load or use previous df state
    # For standalone, we load fresh but typically this runs after Task 2.1
    # We will assume 'df' typically comes from the previous step (interpolated)
    try:
        df = pd.read_csv('dataset/imu_messy_data.csv')
        # Apply Task 2.1 fix (Interpolation) for consistency
        df[['acc_x', 'acc_y', 'acc_z']] = df[['acc_x', 'acc_y', 'acc_z']].interpolate(method='linear')
        print("Data loaded and interpolated (Task 2.1 applied).")
    except:
        print("Error loading data for standalone test.")

print("\n--- Task 2.2: Outlier Detection and Treatment ---")

axes = ['acc_x', 'acc_y', 'acc_z']

# 1. Z-Score Method
z_scores = np.abs(stats.zscore(df[axes]))
z_outliers = (z_scores > 3)
z_outliers_count = z_outliers.sum(axis=0)
print(f"Z-Score Outliers (|z| > 3):\n{z_outliers_count}")

# 2. IQR Method
Q1 = df[axes].quantile(0.25)
Q3 = df[axes].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

iqr_outliers = ((df[axes] < lower_bound) | (df[axes] > upper_bound))
iqr_outliers_count = iqr_outliers.sum(axis=0)
print(f"IQR Outliers:\n{iqr_outliers_count}")

# 3. Visualization
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
df.boxplot(column=axes)
plt.title("Box Plot of Accelerometer Data")

plt.subplot(1, 2, 2)
# Scatter plot of first 1000 points to avoid overplotting
plt.plot(df.index[:1000], df['acc_x'][:1000], '.', label='acc_x', alpha=0.5)
plt.title("Scatter Plot (First 1000 samples)")
plt.legend()
plt.tight_layout()
plt.show()

# 4. Treatment Strategies Comparison
print("\nComparing Treatment Strategies:")

# A. Removal (drop rows with any outlier in axes)
# Using Z-score for this example as it's often cleaner, or IQR depending on preference.
# Let's use IQR for strictness or Z-score for extreme errors. Task asks to compare treatments in general.
# Let's use Z-score outliers for the 'treatment' comparison relative to the original.
mask_outlier = z_outliers.any(axis=1) # Rows with at least one outlier
df_removed = df[~mask_outlier]

# B. Clipping
df_clipped = df.copy()
for col in axes:
    # We clip to min/max observed non-outlier range or theoretical bounds.
    # Usually clipping means capping at threshold.
    # Using IQR thresholds for clipping is common.
    l_bound = lower_bound[col]
    u_bound = upper_bound[col]
    df_clipped[col] = df_clipped[col].clip(lower=l_bound, upper=u_bound)

# C. Interpolation
df_interp_out = df.copy()
df_interp_out[mask_outlier] = np.nan # Set outliers to NaN
df_interp_out[axes] = df_interp_out[axes].interpolate(method='linear') # Re-fill

# Stats Comparison
for name, d in [('Original', df), ('Removal', df_removed), ('Clipping', df_clipped), ('Interpolation', df_interp_out)]:
    print(f"\nStats for {name}:")
    print(d[axes].agg(['mean', 'std']))

# Questions
print("\n--- Questions (Task 2.2) ---")
print("a) How many outliers detected by Z-score vs IQR?")
print(f"   Z-score Total: {z_outliers.sum().sum()}, IQR Total: {iqr_outliers.sum().sum()}")
print("   (IQR typically detects more in non-normal distributions).")

print("b) Which method is more conservative (flags fewer)?")
if z_outliers.sum().sum() < iqr_outliers.sum().sum():
    print("   Z-score is more conservative (flags fewer).")
else:
    print("   IQR is more conservative.")

print("c) Examine 5 specific outlier samples...")
# Let's pick 5 random indices from z_outliers
outlier_indices = df[mask_outlier].index
if not outlier_indices.empty:
    sample_ind = outlier_indices[:5]
    print(f"   Indices: {sample_ind.tolist()}")
    print(df.loc[sample_ind, axes])
    print("   (Check if values > 20m/s^2 or extremely low. Real impacts can be high, errors often typically random large numbers).")
else:
    print("   No outliers found to examine.")

print("d) Mean and Std Dev change:")
print("   (Refer to the printed tables above. Removal/Clipping usually reduces Std Dev).")

print("e) Which treatment strategy would you choose? Why?")
print("   Interpolation is often best for time-series to maintain temporal consistency without losing data, assuming outliers are sparse errors.")
