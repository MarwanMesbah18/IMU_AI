import pandas as pd
import numpy as np
from scipy import stats

if 'df' not in locals():
    # Assume flow
    pass

print("\n--- Task 4.2: Sliding Window Segmentation ---")

# We want to segment the normalized data.
# Usually we use Per-Subject normalized data from previous step.
# For this script, let's assume 'df' is the cleaned data. 
# Ideally we should use the output of Task 4.1.
# Let's perform a quick per-subject Z-score again to be safe/standalone-ish.
if 'subject' in df.columns:
    axes = ['acc_x', 'acc_y', 'acc_z']
    # Use transform to broadcast back to original shape
    for col in axes:
        df[col] = df.groupby('subject')[col].transform(lambda x: (x - x.mean()) / x.std())

# Parameters from prompt:
# Window size: 2 seconds (100 samples at 50 Hz)
# Overlap: 50% (50 samples)
sampling_rate = 50
window_size_sec = 2
overlap_pct = 0.50

window_size_samples = int(window_size_sec * sampling_rate)
step_size_samples = int(window_size_samples * (1 - overlap_pct))

print(f"Config: Window={window_size_samples} samples, Step={step_size_samples} samples")

def segment_data(data, window_len, step_len):
    """
    Segments data into windows.
    Returns X (windows) and y (labels).
    """
    X = []
    y = []
    
    # We iterate over the dataframe
    # Logic: Range from 0 to len - window, step by step
    # IMPORTANT: We should usually group by 'subject' and 'activity' (or just subject) 
    # to avoid windows spanning across different subjects or massive time gaps.
    # The prompt asks "How would you handle windows spanning two activities?".
    # For now, let's just slide over the whole sorted dataset for the count experiment,
    # or better, iterate by Subject to respect boundaries.
    
    # Let's iterate by Subject to prevent cross-subject windows.
    subjects = data['subject'].unique() if 'subject' in data.columns else [0]
    
    count = 0
    
    for subj in subjects:
        if 'subject' in data.columns:
            subj_data = data[data['subject'] == subj]
        else:
            subj_data = data
            
        values = subj_data[['acc_x', 'acc_y', 'acc_z']].values
        labels = subj_data['label'].values if 'label' in data.columns else None
        
        for i in range(0, len(values) - window_len, step_len):
            _x = values[i : i + window_len]
            
            # Label Voting: Majority vote
            if labels is not None:
                _y_segment = labels[i : i + window_len]
                # Mode
                # Mode using np.unique (stats.mode fails on strings in new scipy)
                vals, counts = np.unique(_y_segment, return_counts=True)
                _y = vals[np.argmax(counts)]
            else:
                _y = "unknown"
                
            X.append(_x)
            y.append(_y)
            count += 1
            
    return np.array(X), np.array(y)

# 1. Execute Segmentation (2s, 50%)
X_2s, y_2s = segment_data(df, window_size_samples, step_size_samples)
print(f"Total windows generated (2s, 50%): {len(X_2s)}")

# 3. Count total number of windows per activity
values, counts = np.unique(y_2s, return_counts=True)
print("Windows per Activity:")
for v, c in zip(values, counts):
    print(f"  {v}: {c}")

# 4. Experiment with different window sizes: 1s, 2s, 4s
print("\nExperimenting with window sizes (1s, 2s, 4s)...")
sizes = [1, 2, 4]
results = {}

for s in sizes:
    w_len = int(s * sampling_rate)
    s_step = int(w_len * 0.5) # Maintain 50% overlap concept
    _X, _y = segment_data(df, w_len, s_step)
    results[s] = len(_X)
    print(f"  Window {s}s: {len(_X)} windows")


# Questions
print("\n--- Questions to Answer (Task 4.2) ---")

print("a) How many windows generated (2s/50%)?")
print(f"   {len(X_2s)} windows.")

print("b) What happens at activity transitions?")
print("   A window might contain data from two activities (mixed signal).")
print("   Handling: Discard mixed windows (if labels distinct change), or use Majority Voting (as implemented).")

print("c) Trade-offs of window sizes?")
print("   - Short (1s): Faster response, low latency, but less context for complex moves.")
print("   - Long (4s): Better feature capture for slow repetitive motions, but high latency and more mixed windows.")

print("d) What overlap percentage recommended?")
print("   50% is standard. It doubles the training data and ensures edges of windows are also center-stage in next window.")
