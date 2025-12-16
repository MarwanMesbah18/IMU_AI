import pandas as pd

if 'df' not in locals():
    # Load assuming flow
    try:
        df = pd.read_csv('dataset/imu_messy_data.csv')
        # Apply previous steps minimally for flow
        df[axes] = df[axes].interpolate(method='linear') 
        print("Data loaded for Task 2.3")
    except:
        pass

print("\n--- Task 2.3: Duplicate Removal ---")

# 1. Measurement Duplicates (Timestamp + Axes identical)
# The prompt asks for:
# - Exact duplicates (all cols)
# - Measurement duplicates (timestamp, acc_x, acc_y, acc_z) - ignoring potential subject/label variance if any, though usually same.

# a) Exact Duplicates
exact_dupes = df.duplicated().sum()
print(f"Exact duplicates (all columns): {exact_dupes}")

# b) Measurement Duplicates
measure_cols = ['timestamp', 'acc_x', 'acc_y', 'acc_z']
measurement_dupes = df.duplicated(subset=measure_cols).sum()
print(f"Measurement duplicates (ignoring subject/label): {measurement_dupes}")

# Remove Duplicates (keeping first)
initial_len = len(df)
df = df.drop_duplicates(subset=measure_cols, keep='first')
final_len = len(df)
print(f"Removed {initial_len - final_len} duplicates. New shape: {df.shape}")

# Questions
print("\n--- Questions (Task 2.3) ---")
print(f"a) How many exact duplicates exist? {exact_dupes}")
print(f"b) How many measurement duplicates exist? {measurement_dupes}")
print("c) Should duplicate removal happen before or after sorting?")
print("   It should generally happen BEFORE sorting if the order matters for 'keep=first' (though with sorting it becomes deterministic).")
print("   However, usually removing duplicates is a raw data cleaning step done early. Sorting is often required for time-series operations next.")
