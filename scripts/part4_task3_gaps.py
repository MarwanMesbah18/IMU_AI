import pandas as pd
import numpy as np

if 'df' not in locals():
    # Assume flow
    pass

print("\n--- Task 4.3: Handling Gaps and Missing Segments ---")

# 1. Identify the major gap in the data
# We calculate diff of 'timestamp'
# 1. Identify the major gap in the data
# We calculate diff of 'timestamp' which is now the index
df_sorted = df.sort_index()
time_diffs = df_sorted.index.to_series().diff()
max_gap = time_diffs.max().total_seconds()
max_gap_idx = time_diffs.idxmax()

print(f"Largest gap detected: {max_gap} seconds")
print(f"Gap occurs at index: {max_gap_idx}")

# 2. Split the dataset into continuous sessions
# Logic: A new session starts if gap > threshold
gap_threshold = 5.0 # 5 seconds

# Create a session ID
# Mark 1 where diff > threshold, else 0. Cumsum gives session ID.
# Use values to avoid index alignment weirdness in assignment
df_sorted['session_id'] = (time_diffs > pd.Timedelta(seconds=gap_threshold)).cumsum().values

# 3. Report duration of each session
# Group by session_id and aggregate the index (timestamp)
session_stats = df_sorted.groupby('session_id').apply(lambda x: pd.Series({
    'min': x.index.min(),
    'max': x.index.max(),
    'count': len(x)
}))
session_stats['duration_sec'] = session_stats['max'] - session_stats['min']

print("\nIdentified Sessions given gap threshold of {:.1f}s:".format(gap_threshold))
print(session_stats)

for sess_id, row in session_stats.iterrows():
    duration = row['duration_sec'].total_seconds()
    print(f"  Session {sess_id}: {duration:.2f} seconds ({row['count']} samples)")

# Decide strategy
# We typically would process each session independently for segmentation (as done iteratively in Task 4.2).

# Questions
print("\n--- Questions to Answer (Task 4.3) ---")

print("a) How long is the gap? Is interpolation reasonable?")
print(f"   Gap is {max_gap:.2f} seconds.")
if max_gap > 5:
    print("   Interpolation is NOT reasonable (too large, inventing fake data).")
else:
    print("   Interpolation might be passable if short.")

print("b) Should windows spanning the gap be discarded?")
print("   YES. Windows spanning a large gap act like 'teleporting' in time and break the physics continuity assumed by ConvNets/RNNs.")

print("c) How would this gap affect model training?")
print("   If ignored, it creates artifacts where the model learns sudden jumps that don't exist in reality.")
print("   Proper handling (splitting sessions) ensures clean training data.")
