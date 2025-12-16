import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if 'df' not in locals():
    # Assume flow
    pass

print("\n--- Task 1.2: Missing Value Analysis ---")

# 1. Count missing values
print("Missing Values per Column:")
print(df.isnull().sum())

# 2. Percentage of missing data for each axis
print("\nPercentage of Missing Data per Axis:")
axes = ['acc_x', 'acc_y', 'acc_z']
for axis in axes:
    if axis in df.columns:
        pct_missing = (df[axis].isnull().sum() / len(df)) * 100
        print(f"{axis}: {pct_missing:.2f}%")

# 3. Rows where all three axes are missing simultaneously
missing_all_axes = df[df[axes].isnull().all(axis=1)]
print(f"\nRows with all axes missing: {len(missing_all_axes)}")

# 4. Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.show()

# Questions
print("\n--- Questions to Answer (Task 1.2) ---")

print('a) Which axis has the most missing values?')
most_missing_axis = df[axes].isnull().sum().idxmax()
print(f"   Axis with most missing values: {most_missing_axis}")

print('b) Are missing values randomly distributed or clustered?')
print("   (See Heatmap above. If blocks of yellow appear, they are clustered/bursty.)")

print('c) What percentage of the dataset would be lost if we dropped all rows with any missing value?')
rows_with_any_missing = df.isnull().any(axis=1).sum()
pct_lost = (rows_with_any_missing / len(df)) * 100
print(f"   Percentage of data lost if dropping any missing: {pct_lost:.2f}%")
