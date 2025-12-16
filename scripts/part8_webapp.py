import streamlit as st
import socket
import json
import numpy as np
import pandas as pd
import joblib
import time
from scipy.fft import fft, fftfreq

# --- CONFIGURATION ---
HOST = '0.0.0.0'
PORT = 65432
MODEL_PATH = 'models/best_model.pkl'

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="IMU AI Dashboard",
    page_icon="🏃",
    layout="wide"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #41424b;
        text-align: center;
    }
    .activity-text {
        font-size: 40px;
        font-weight: bold;
        color: #00ff41;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        return None

model = load_model()

# --- SIDEBAR ---
with st.sidebar:
    st.title("Settings")
    st.write(f"**IP Address:**")
    # Try to find local IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except:
        ip = "127.0.0.1"
    
    st.code(f"{ip}")
    st.write(f"**Port:** {PORT}")
    st.caption("Enter these in your phone's UDP Sender App.")
    
    run_server = st.toggle("Start Server", value=False)
    
    st.divider()
    if model:
        st.success("Model Loaded Successfully!")
    else:
        st.error("Model 'best_model.pkl' not found. Run Part 7 in notebook first.")

# --- MAIN LAYOUT ---
st.title("🏃 Real-Time Activity Recognition")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Sensor Data (Acc X, Y, Z)")
    chart_holder = st.empty()

with col2:
    st.subheader("Predicted Activity")
    activity_holder = st.empty()
    conf_holder = st.empty()
    debug_holder = st.empty()

# --- FEATURE EXTRACTION HELPERS (From Part 5) ---
def extract_realtime_features(window_data):
    # window_data shape: (N, 3) -> (100, 3) generally
    X = np.array(window_data)
    
    # 1. Time Domain
    # Mean, Std, Min, Max, Range (3 each = 15)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    rng = maxs - mins
    
    # SMA (1)
    sma = np.mean(np.sum(np.abs(X), axis=1))
    
    # Energy (3)
    energy = np.mean(X**2, axis=0)
    
    # ZCR (3) - explicit to match notebook
    zcr = np.sum(np.diff(np.signbit(X), axis=0), axis=0)
    
    # RMS (3)
    rms = np.sqrt(energy)
    
    # Mag Mean/Std (2)
    mag = np.sqrt(np.sum(X**2, axis=1))
    mag_mean = np.mean(mag)
    mag_std = np.std(mag)
    
    # Total Time Features so far: 15 + 1 + 3 + 3 + 3 + 2 = 27
    
    # 2. Frequency Domain
    sampling_rate = 50
    n_samples = len(X)
    freqs = fftfreq(n_samples, 1/sampling_rate)
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    
    Z = fft(X, axis=0)
    Mag = np.abs(Z) / n_samples
    Mag_pos = Mag[pos_mask, :]
    
    # Dom Freq (3)
    idx_max = np.argmax(Mag_pos, axis=0)
    dom_freq = freqs_pos[idx_max]
    
    # Spec Energy (3)
    spec_energy = np.sum(Mag_pos**2, axis=0)
    
    # Spec Entropy (3)
    psd = Mag_pos**2
    psd_sum = np.sum(psd, axis=0, keepdims=True)
    psd_sum[psd_sum == 0] = 1e-9
    P = psd / psd_sum
    P[P == 0] = 1e-9
    entropy = -np.sum(P * np.log2(P), axis=0)
    
    # Band Powers (4 bands * 3 axes = 12)
    # 0-2, 2-5, 5-10, 10-25
    bands = [(0, 2), (2, 5), (5, 10), (10, 25)]
    band_powers = []
    for b_start, b_end in bands:
        mask = (freqs_pos >= b_start) & (freqs_pos < b_end)
        if np.sum(mask) == 0:
            bp = np.zeros(3)
        else:
            bp = np.sum(psd[mask, :], axis=0)
        band_powers.extend(bp)
        
    # Total Features = 27 (Time) + 3 + 3 + 3 + 12 = 48
    # Match order:
    # Mean(3), Std(3), Min(3), Max(3), Range(3), SMA(1), Energy(3), ZCR(3), RMS(3), MagMean(1), MagStd(1)
    # Dom(3), SpecEn(3), Ent(3), Band(12)
    
    features = np.concatenate([
        mean, std, mins, maxs, rng, [sma], energy, zcr, rms, [mag_mean, mag_std],
        dom_freq, spec_energy, entropy, band_powers
    ])
    
    return features.reshape(1, -1)

# --- SERVER LOGIC ---
if run_server:
    # Initialize UDP Socket
    if 'sock' not in st.session_state:
        st.session_state.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        st.session_state.sock.bind((HOST, PORT))
        st.session_state.sock.setblocking(False) # Non-blocking for UI responsiveness
        
    if 'buffer' not in st.session_state:
        st.session_state.buffer = [] # Store [x, y, z]

    # Initialize Plot Data
    if 'plot_data' not in st.session_state:
        st.session_state.plot_data = pd.DataFrame(columns=['x', 'y', 'z'])

    status_container = st.empty()
    
    # Loop for UI updates
    # Streamlit rerun approach: We use a placeholder loop or st.autorefresh
    # For simple demo, we iterate inside the script execution
    
    MAX_SAMPLES = 200 # Show last 4 seconds
    WINDOW_SIZE = 100 # Inference every 100 samples (2s)
    OVERLAP = 50      # Not strictly needed for live demo, can slide every N
    
    try:
        while True:
            try:
                data, addr = st.session_state.sock.recvfrom(1024)
                msg = data.decode('utf-8')
                
                # Parse
                # Try simple JSON: {"acc_x": ..., } or Array: [x,y,z]
                # Default "Sensor Log" app often sends CSV string or JSON
                # Let's assume JSON first
                try:
                    j = json.loads(msg)
                    # Extract flexible keys
                    x = j.get('acc_x') or j.get('x') or 0
                    y = j.get('acc_y') or j.get('y') or 0
                    z = j.get('acc_z') or j.get('z') or 0
                except:
                    # Parse CSV string fallback? "1.2, 0.5, 9.8"
                    parts = msg.split(',')
                    if len(parts) >= 3:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    else:
                        continue

                # Add to buffer
                st.session_state.buffer.append([x, y, z])
                
                # Update Plot Data
                new_row = pd.DataFrame([[x, y, z]], columns=['x', 'y', 'z'])
                st.session_state.plot_data = pd.concat([st.session_state.plot_data, new_row], ignore_index=True)
                if len(st.session_state.plot_data) > MAX_SAMPLES:
                    st.session_state.plot_data = st.session_state.plot_data.iloc[-MAX_SAMPLES:]
                
                # Update Chart (Every 5 samples to save framerate)
                if len(st.session_state.buffer) % 5 == 0:
                    chart_holder.line_chart(st.session_state.plot_data[-100:])

                # INFERENCE TRIGGER
                if len(st.session_state.buffer) >= WINDOW_SIZE:
                    # Extract last 100
                    window = st.session_state.buffer[-WINDOW_SIZE:]
                    
                    # Convert to features
                    try:
                        feats = extract_realtime_features(window)
                        
                        # Predict
                        if model:
                            pred_class = model.predict(feats)[0]
                            # proba = model.predict_proba(feats) # Optional
                            
                            # Update UI
                            activity_holder.markdown(
                                f"""<div class='metric-card'>
                                    <div style='color: #aaa'>Detected Activity</div>
                                    <div class='activity-text'>{pred_class.upper()}</div>
                                   </div>""", 
                                unsafe_allow_html=True
                            )
                        else:
                            activity_holder.warning("Model missing")
                            
                    except Exception as inf_err:
                        debug_holder.error(f"Inference Error: {inf_err}")
                    
                    # Slide buffer (keep 50% overlap for next)
                    st.session_state.buffer = st.session_state.buffer[OVERLAP:]

            except BlockingIOError:
                # No data derived, sleep briefly
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        st.session_state.sock.close()
        
else:
    st.info("Toggle 'Start Server' in the sidebar to begin.")
    st.write("waiting for connection...")
    
