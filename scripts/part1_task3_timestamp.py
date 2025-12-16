import pandas as pd
import numpy as np

if 'df' not in locals():
    # Assume flow
    pass

print("\n--- Task 1.3: Timestamp Analysis ---")

# 1. Check if sorted
is_sorted = df['timestamp'].is_monotonic_increasing
print(f"Timestamps sorted: {is_sorted}")

# 2. Calculate time differences
time_diffs = df['timestamp'].diff()

# 3. Identify gaps > expected (50Hz = 0.02s interval)
expected_interval = 1/50 # 0.02s
gap_threshold = 1.0 # 1 second as 'large gap' per question b
large_gaps = time_diffs[time_diffs > gap_threshold]
print(f"\nNumber of gaps > 1s: {len(large_gaps)}")

# 4. Detect timestamp duplicates
duplicate_timestamps = df['timestamp'].duplicated().sum()
print(f"Number of duplicate timestamps: {duplicate_timestamps}")

# Questions
print("\n--- Questions to Answer (Task 1.3) ---")

print('a) What is the expected time interval between samples at 50 Hz?')
print(f"   Expected interval at 50Hz: {expected_interval} s")

print('b) How many samples have timestamp gaps exceeding 1 second?')
print(f"   Samples with gaps > 1s: {len(large_gaps)}")

print('c) What is the largest gap in the data? Where does it occur?')
largest_gap = time_diffs.max()
largest_gap_idx = time_diffs.idxmax()
print(f"   Largest gap: {largest_gap} s at index {largest_gap_idx}")
if largest_gap_idx is not np.nan:
    # Use valid indexing if available
    try:
        t_prev = df.loc[largest_gap_idx-1, "timestamp"]
        t_curr = df.loc[largest_gap_idx, "timestamp"]
        print(f"   Gap occurs between timestamp {t_prev} and {t_curr}")
    except:
        pass

print('d) Are there any duplicate timestamps?')
print(f"   Number of duplicate timestamps: {duplicate_timestamps}")
