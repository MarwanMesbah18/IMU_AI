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
    page_title="IMU AI Live Stream",
    page_icon="📡",
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
        margin-bottom: 20px;
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
    
    # Try to find local IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except:
        ip = "127.0.0.1"
    
    st.info(f"**PC IP Address:** `{ip}`")
    st.caption(f"Listening on Port: {PORT}")
    
    st.divider()
    
    if model:
        st.success("✅ Model Loaded")
    else:
        st.error("❌ Model Missing")

    st.divider()
    st.write("### 📲 How to Connect")
    st.markdown("""
    1. Ensure phone & PC are on same Wi-Fi.
    2. Run `web_sensor_bridge.py` in terminal.
    3. Open `https://<IP>:5000` on phone.
    4. Click 'Start Stream'.
    """)

# --- FEATURE EXTRACTION (Simplified for brevity, same logic) ---
def extract_realtime_features(window_data):
    X = np.array(window_data)
    
    # 1. Time Domain
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    rng = maxs - mins
    sma = np.mean(np.sum(np.abs(X), axis=1))
    energy = np.mean(X**2, axis=0)
    zcr = np.sum(np.diff(np.signbit(X), axis=0), axis=0)
    rms = np.sqrt(energy)
    mag = np.sqrt(np.sum(X**2, axis=1))
    mag_mean = np.mean(mag)
    mag_std = np.std(mag)
    
    # 2. Frequency Domain
    sampling_rate = 50
    n_samples = len(X)
    freqs = fftfreq(n_samples, 1/sampling_rate)
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    
    Z = fft(X, axis=0)
    Mag = np.abs(Z) / n_samples
    Mag_pos = Mag[pos_mask, :]
    
    idx_max = np.argmax(Mag_pos, axis=0)
    dom_freq = freqs_pos[idx_max]
    spec_energy = np.sum(Mag_pos**2, axis=0)
    
    psd = Mag_pos**2
    psd_sum = np.sum(psd, axis=0, keepdims=True)
    psd_sum[psd_sum == 0] = 1e-9
    P = psd / psd_sum
    P[P == 0] = 1e-9
    entropy = -np.sum(P * np.log2(P), axis=0)
    
    bands = [(0, 2), (2, 5), (5, 10), (10, 25)]
    band_powers = []
    for b_start, b_end in bands:
        mask = (freqs_pos >= b_start) & (freqs_pos < b_end)
        if np.sum(mask) == 0:
            bp = np.zeros(3)
        else:
            bp = np.sum(psd[mask, :], axis=0)
        band_powers.extend(bp)
        
    features = np.concatenate([
        mean, std, mins, maxs, rng, [sma], energy, zcr, rms, [mag_mean, mag_std],
        dom_freq, spec_energy, entropy, band_powers
    ])
    
    return features.reshape(1, -1)

# --- MAIN PAGE ---
st.title("📡 Live Sensor Stream & AI Recognition")

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Live Accelerometer Data")
    chart_placeholder = st.empty()

with col2:
    st.subheader("Activity Prediction")
    activity_placeholder = st.empty()
    debug_placeholder = st.empty()

# --- SERVER LOGIC ---
if 'buffer' not in st.session_state:
    st.session_state.buffer = []
    
if 'plot_data' not in st.session_state:
    st.session_state.plot_data = pd.DataFrame(columns=['x', 'y', 'z'])

if 'packet_count' not in st.session_state:
    st.session_state.packet_count = 0

# Initialize Socket Once
if 'sock' not in st.session_state:
    st.session_state.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        st.session_state.sock.bind((HOST, PORT))
    except OSError:
        st.error(f"Error: Port {PORT} is occupied.")
        st.stop()
    st.session_state.sock.setblocking(False)

# UI Elements for Loop
stop_button = st.button("Stop Server (Refresh to Restart)")

# Loop
placeholder_status = st.empty()

if not stop_button:
    while True:
        # 1. Drain Queue (Read ALL available packets to fix lag)
        packets = []
        try:
            while True:
                data, addr = st.session_state.sock.recvfrom(1024)
                packets.append(data)
                # Cap burst to avoid freezing if flooding
                if len(packets) > 50: 
                    break 
        except BlockingIOError:
            pass # No more data
        
        # 2. Process Packets
        if packets:
            current_x, current_y, current_z = 0, 0, 0
            
            for data in packets:
                try:
                    msg = data.decode('utf-8')
                    try:
                        j = json.loads(msg)
                        x, y, z = j.get('acc_x', 0), j.get('acc_y', 0), j.get('acc_z', 0)
                        if x==0 and y==0 and z==0:
                             x, y, z = j.get('x', 0), j.get('y', 0), j.get('z', 0)
                    except:
                        parts = msg.split(',')
                        if len(parts) >= 3:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        else:
                            continue
                    
                    # Store latest for display
                    current_x, current_y, current_z = x, y, z
                    
                    # Add to history
                    st.session_state.buffer.append([x, y, z])
                    
                    # Add to plot (maybe downsample? only add latest per batch?)
                    # Adding every point makes plot accurate but potentially slow.
                    # Let's add all points to chart data.
                    # Creating DF row is slow.
                    # append only the last one of the batch to plot_data? 
                    # No, user wants to see the wave.
                    # Compromise: Append via list concat then DF creation once per frame
                except:
                    continue
            
            # Batch update Plot Data
            new_rows = []
            for d in st.session_state.buffer[-len(packets):]:
                new_rows.append(d)
                
            new_df = pd.DataFrame(new_rows, columns=['x', 'y', 'z'])
            st.session_state.plot_data = pd.concat([st.session_state.plot_data, new_df], ignore_index=True)
            
            # Trim
            LIMIT = 200
            if len(st.session_state.plot_data) > LIMIT:
                st.session_state.plot_data = st.session_state.plot_data.iloc[-LIMIT:]
            
            # 3. Update UI (Visuals) ONCE per loop
            chart_placeholder.line_chart(st.session_state.plot_data)
            
            st.session_state.packet_count += len(packets)
            debug_placeholder.markdown(
                f"""
                <div style='background-color: #262730; padding: 10px; border-radius: 5px; font-size: 0.8em;'>
                <b>Status:</b> 🟢 Receiving<br>
                <b>Packets:</b> {st.session_state.packet_count}<br>
                <b>Latest (m/s²):</b><br>X: {current_x:.2f} Y: {current_y:.2f} Z: {current_z:.2f}
                </div>
                """, unsafe_allow_html=True
            )
            
            # 4. Inference
            WINDOW_SIZE = 100
            OVERLAP = 50
            if len(st.session_state.buffer) >= WINDOW_SIZE:
                 window = st.session_state.buffer[-WINDOW_SIZE:]
                 try:
                     feats = extract_realtime_features(window)
                     if model:
                         pred_class = model.predict(feats)[0]
                         activity_placeholder.markdown(
                             f"""<div class='metric-card'>
                                 <div style='color: #aaa'>Detected</div>
                                 <div class='activity-text'>{pred_class.upper()}</div>
                                </div>""", 
                             unsafe_allow_html=True
                         )
                 except:
                     pass
                 st.session_state.buffer = st.session_state.buffer[OVERLAP:]
                 
        else:
            # No data
            time.sleep(0.01) # Short sleep to prevent CPU spin
            
else:
    st.write("Server Stopped.")
