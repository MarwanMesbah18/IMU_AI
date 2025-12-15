import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)

# Load the data
file_path = 'dataset/imu_messy_data.csv'
try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"File not found at {file_path}")
    exit()

# Inspect
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Types: \n{df.dtypes}")
# print(df.describe())

# Questions 1.1
print(f"Num samples: {len(df)}")
if 'timestamp' in df.columns:
    print(f"Timestamp Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

if 'label' in df.columns:
    print(f"Class Distribution: \n{df['label'].value_counts()}")

# Task 1.2 Missing
axes = ['acc_x', 'acc_y', 'acc_z']
print("\nMissing per column:")
print(df.isnull().sum())
print("\nMissing % per axis:")
for axis in axes:
    if axis in df.columns:
        print(f"{axis}: {(df[axis].isnull().sum()/len(df))*100:.2f}%")

missing_all = df[df[axes].isnull().all(axis=1)]
print(f"Rows with all axes missing: {len(missing_all)}")

# Task 1.3 Timestamp
if 'timestamp' in df.columns:
    # df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce') # Assuming already float
    is_sorted = df['timestamp'].is_monotonic_increasing
    print(f"\nTimestamps sorted: {is_sorted}")
    
    time_diffs = df['timestamp'].diff()
    large_gaps = time_diffs[time_diffs > 1.0]
    print(f"Gaps > 1s: {len(large_gaps)}")
    
    largest_gap = time_diffs.max()
    print(f"Largest gap: {largest_gap}")
    
    dups = df['timestamp'].duplicated().sum()
    print(f"Duplicate timestamps: {dups}")
