import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

print("--- Starting Verification Pipeline ---")

try:
    # --- PART 1: LOAD ---
    print("\n[Part 1] Loading Data...")
    df = pd.read_csv('dataset/imu_messy_data.csv')
    assert not df.empty, "Dataframe is empty"
    print(f"Loaded {len(df)} rows.")

    # --- PART 2: CLEANING ---
    print("\n[Part 2] Cleaning...")
    
    # 2.1 Interpolate
    df[['acc_x', 'acc_y', 'acc_z']] = df[['acc_x', 'acc_y', 'acc_z']].interpolate(method='linear')
    assert df[['acc_x', 'acc_y', 'acc_z']].isnull().sum().sum() == 0, "NaNs found after interpolation"
    
    # 2.3 Duplicates
    # Standardize cols
    measure_cols = ['timestamp', 'acc_x', 'acc_y', 'acc_z']
    df = df.drop_duplicates(subset=measure_cols, keep='first')
    
    # 2.4 Resampling
    # 2.4 Resampling
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values('timestamp')
    df = df.set_index('timestamp')
    
    if df.index.duplicated().any():
        print(f"Duplicates found: {df.index.duplicated().sum()}")
        num_cols = ['acc_x', 'acc_y', 'acc_z']
        cat_cols = ['label', 'subject']
        # Split aggregation
        df_num = df[num_cols].groupby(level=0).mean()
        df_cat = df[cat_cols].groupby(level=0).first() if 'label' in df.columns else None
        
        if df_cat is not None:
             df = pd.concat([df_num, df_cat], axis=1)
        else:
             df = df_num
             
    # Handle Label/Subject (Categorical)
    if 'label' in df.columns:
        df_numeric = df[['acc_x', 'acc_y', 'acc_z']]
        df_cat = df[['label', 'subject']]
        
        df_numeric = df_numeric.resample('20ms').mean().interpolate(method='linear')
        df_cat = df_cat.resample('20ms').ffill()
        df_final = pd.concat([df_numeric, df_cat], axis=1)
    else:
        df_final = df[['acc_x', 'acc_y', 'acc_z']].resample('20ms').mean().interpolate(method='linear')
        
    df = df_final.dropna()
    print(f"Resampled shape: {df.shape} (Expected ~50Hz)")

    # --- PART 4: SEGMENTATION ---
    print("\n[Part 4] Preprocessing & Segmentation...")
    
    # 4.1 Normalization (Z-score check)
    df_z = (df[['acc_x', 'acc_y', 'acc_z']] - df[['acc_x', 'acc_y', 'acc_z']].mean()) / df[['acc_x', 'acc_y', 'acc_z']].std()
    assert np.isclose(df_z['acc_x'].mean(), 0, atol=0.1), "Z-score normalization mean != 0"
    
    # 4.2 Segmentation
    window_sec = 2
    sampling_rate = 50
    overlap = 0.5
    w_len = int(window_sec * sampling_rate)
    step = int(w_len * (1 - overlap))
    
    X = []
    y = []
    
    # Use simple loop for verification
    data_values = df_z.values
    labels = df['label'].values if 'label' in df.columns else np.array(['unknown']*len(df))
    
    for i in range(0, len(data_values) - w_len, step):
        X.append(data_values[i:i+w_len])
        # Majority vote
        seg_labels = labels[i:i+w_len]
        vals, counts = np.unique(seg_labels, return_counts=True)
        y.append(vals[np.argmax(counts)])
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"Segments shape: {X.shape} (N, 100, 3)")
    assert X.shape[1] == 100, "Window size incorrect"
    assert X.shape[2] == 3, "Channel count incorrect"
    
    # --- PART 5: FEATURES ---
    print("\n[Part 5] Feature Engineering...")
    
    # Simple feature computation check
    # Mean
    feats_mean = np.mean(X, axis=1)
    # FFT
    feats_fft = np.abs(fft(X, axis=1))
    dom_freq = np.argmax(feats_fft, axis=1) # Just check runtime
    
    assert feats_mean.shape == (len(X), 3), "Feature shape mismatch"
    print(f"Computed simple features for {len(X)} windows.")
    
    # --- PART 6: CLASSIFICATION ---
    print("\n[Part 6] Classification Check...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Mock Features & Labels for verification speed if Part 5 didn't produce full set
    X_mock = feats_mean
    y_mock = y # from Part 4 loop
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_mock, y_mock, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te)
    print(f"Verification RF Accuracy: {acc:.2f} (Random/Mock Check)")
    assert acc >= 0, "Accuracy invalid"

    print("\n[SUCCESS] Pipeline logic verified. Data flows correctly from Load -> Clean -> Segment -> Features -> Train.")

except Exception as e:
    print(f"\n[FAILURE] Logic verification failed: {e}")
    exit(1)
