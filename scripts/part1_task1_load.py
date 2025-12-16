import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'dataset/imu_messy_data.csv'
try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"File not found at {file_path}. Please check the path.")

# Display first 20 rows
print("\nFirst 20 rows:")
display(df.head(20))

# Print dataset shape, columns, and types
print("\nDataset Info:")
print(f"Shape: {df.shape}")
print("\nColumn Names:", df.columns.tolist())
print("\nData Types:")
print(df.dtypes)

# Summary statistics
print("\nSummary Statistics:")
display(df.describe(include='all'))

# Questions
print("\n--- Questions to Answer (Task 1.1) ---")

print('a) How many samples are in the dataset?')
num_samples = len(df)
print(f"   Number of samples: {num_samples}")

print('b) What is the range of timestamp values?')
try:
    min_ts = df['timestamp'].min()
    max_ts = df['timestamp'].max()
    print(f"   Timestamp Range: {min_ts} to {max_ts}")
except KeyError:
    print("   Timestamp column not found or formatted incorrectly.")

print('c) What are the min/max values for each acceleration axis? Do they seem physically plausible?')
axes = ['acc_x', 'acc_y', 'acc_z']
for axis in axes:
    if axis in df.columns:
        print(f"   {axis}: Min={df[axis].min()}, Max={df[axis].max()}")
print("   (Physical plausibility depends on unit (g vs m/s^2). If m/s^2, 1g~9.8. Large values might be impacts.)")

print('d) What is the class distribution across activities?')
if 'label' in df.columns:
    print("   Class Distribution:")
    print(df['label'].value_counts())
