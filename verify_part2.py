import pandas as pd
import numpy as np
from scipy import stats

pd.set_option('display.max_columns', None)

# Load Data
file_path = 'dataset/imu_messy_data.csv'
df = pd.read_csv(file_path)
print(f"Initial Shape: {df.shape}")

axes = ['acc_x', 'acc_y', 'acc_z']

# Task 2.1 Missing Values (Test Interpolation)
df[axes] = df[axes].interpolate(method='linear')
df[axes] = df[axes].bfill().ffill()
print(f"Missing after interp: {df[axes].isnull().sum().sum()}")

# Task 2.3 Duplicates (Before 2.2 to verify order in actual code flow? Notebook did 2.3 after 2.2 but logic holds)
# Notebook flow: 2.1 (Missing) -> 2.2 (Outlier Analysis) -> 2.3 (Duplicate Removal) -> 2.4 (Sort/Resample)
# Let's follow notebook flow broadly

# Task 2.2 Outliers
z_scores = np.abs(stats.zscore(df[axes]))
z_outliers = (z_scores > 3).sum(axis=0)
print(f"Z-score Outliers: {z_outliers}")

# Task 2.3 Duplicate Removal
subset_cols = ['timestamp', 'acc_x', 'acc_y', 'acc_z']
before_dedup = len(df)
df = df.drop_duplicates(subset=subset_cols, keep='first')
print(f"Duplicates removed: {before_dedup - len(df)}")

# Task 2.4 Sort and Resample
df = df.sort_values('timestamp')
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df = df.set_index('datetime')

# Resample 50Hz
# numeric_only=True to avoid object col error
df_resampled = df.resample('20ms').mean(numeric_only=True)
# Interpolate limited
# Interpolate limited
df_resampled[axes] = df_resampled[axes].interpolate(method='time', limit=50)

df_final = df_resampled.dropna(subset=axes)
print(f"Final Shape: {df_final.shape}")

# Verify continuity / gaps
dt = df_final.index.to_series().diff().dt.total_seconds()
print(f"Max gap in final: {dt.max()}")
print(f"Mean interval: {dt.mean()}")
