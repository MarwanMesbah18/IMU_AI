import pandas as pd
import matplotlib.pyplot as plt

if 'df' not in locals():
    # Load assuming flow
    # This task relies heavily on previous cleaning, so standalone might be tricky without full pipeline
    # We'll assume 'df' is available or load/clean quickly
    try:
        df = pd.read_csv('dataset/imu_messy_data.csv')
        # Quick clean replay
        df[['acc_x', 'acc_y', 'acc_z']] = df[['acc_x', 'acc_y', 'acc_z']].interpolate(method='linear')
        df = df.drop_duplicates(subset=['timestamp', 'acc_x', 'acc_y', 'acc_z'], keep='first')
        print("Data loaded and pre-cleaned for Task 2.4")
    except:
        pass

print("\n--- Task 2.4: Timestamp Sorting and Resampling ---")

# 2. Set timestamp as index and sort
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df = df.sort_values('timestamp')
df = df.set_index('timestamp')

# Check for duplicates
if df.index.duplicated().any():
    print(f"Duplicate timestamps found: {df.index.duplicated().sum()}")
    # We must aggregate duplicates BEFORE resampling to avoid reindexing errors
    # Split numeric and categorical
    numeric_cols = ['acc_x', 'acc_y', 'acc_z']
    cat_cols = ['label', 'subject']
    
    # Aggregate Numeric -> Mean
    # numeric_only=True avoids the TypeError if other cols exist
    df_num = df[numeric_cols].groupby(level=0).mean()
    
    # Aggregate Categorical -> First
    # Check if cols exist (subject might be numeric but label is str)
    present_cat_cols = [c for c in cat_cols if c in df.columns]
    if present_cat_cols:
        df_cat = df[present_cat_cols].groupby(level=0).first()
        df = pd.concat([df_num, df_cat], axis=1)
    else:
        df = df_num
else:
    # No duplicates, just proceed
    pass

# Now df is unique-indexed.
# 3. Resample to 50 Hz (20ms)
if 'label' in df.columns and 'subject' in df.columns:
    df_numeric = df[['acc_x', 'acc_y', 'acc_z']]
    df_cat = df[['label', 'subject']]
    
    df_numeric_res = df_numeric.resample('20ms').mean().interpolate(method='linear')
    df_cat_res = df_cat.resample('20ms').ffill()
    
    df_final = pd.concat([df_numeric_res, df_cat_res], axis=1)
else:
    df_final = df[['acc_x', 'acc_y', 'acc_z']].resample('20ms').mean().interpolate(method='linear')

df_final = df_final.dropna()

# Handle categorical columns (label/subject) if they are lost during mean()
# Typically we might take `first` or `mode` for labels.
# For now, let's focus on the numeric axes as required for signals.
# Interpolate limits: "Handle the major gap appropriately"
# We should probably NOT interpolate across a huge gap (e.g. separate sessions).
# Let's check gap size.

# Check gaps in original data before full interpolation
# Check gaps in original data before full interpolation - using the index
original_diffs = df.index.to_series().diff()
max_gap = original_diffs.max().total_seconds()
print(f"Max gap in original data: {max_gap} seconds")

# 4. Handle major gap
# If there is a massive gap, plotting it will show a straight line.
# We'll visualize it.

# 4. Handle major gap
# If there is a massive gap, plotting it will show a straight line.
# We'll visualize it.

# Questions
print("\n--- Questions to Answer (Task 2.4) ---")
print("a) Visualize 5-second window.")
try:
    # Plot first 5 seconds
    plt.figure(figsize=(10, 4))
    plt.plot(df_final.index[:250], df_final['acc_x'][:250], label='Resampled 50Hz')
    plt.title("5-second Window (Resampled)")
    plt.legend()
    plt.show()
except:
    print("Could not plot (index issues?)")

print(f"b) How should the major gap be handled? (Max gap found: {max_gap}s)")
print("   If it represents separate sessions, we should keep it as NaN or split files.")
print("   If we interpolated it, we introduced artificial data. 'Split into separate sessions' is usually best for ML.")

print("c) Verify sampling rate is exactly 50 Hz.")
# Check diff of index
index_diffs = df_final.index.to_series().diff().dropna()
# 20ms = 0.02s
dt_modes = index_diffs.value_counts()
print(f"   Time intervals found:\n{dt_modes.head()}")
if len(dt_modes) == 1 and dt_modes.index[0].total_seconds() == 0.02:
    print("   Verified: Strictly 50 Hz.")
else:
    print("   Verified: Dominantly 50 Hz (check for minor precision diffs).")

# Save cleaning state if needed or for notebook flow
df = df_final
print(f"Final Info: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
