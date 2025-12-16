import streamlit as st
import socket
import json
import numpy as np
import pandas as pd
import joblib
import time
import atexit
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

if 'last_ui_update' not in st.session_state:
    st.session_state.last_ui_update = 0

# Initialize Socket Once
if 'sock' not in st.session_state:
    st.session_state.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Enable socket reuse to prevent port occupation errors
    st.session_state.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        st.session_state.sock.bind((HOST, PORT))
    except OSError as e:
        st.error(f"Error: Port {PORT} is occupied. Try: `lsof -ti:{PORT} | xargs kill -9`")
        st.stop()
    st.session_state.sock.setblocking(False)
    
    # Register cleanup function
    def cleanup_socket():
        try:
            if hasattr(st.session_state, 'sock'):
                st.session_state.sock.close()
        except:
            pass
    atexit.register(cleanup_socket)

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
        
        # 2. Process Packets (use only LAST packet to reduce delay)
        if packets:
            current_x, current_y, current_z = 0, 0, 0
            
            # Use only the LAST packet from the batch for better performance
            data = packets[-1]
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
                        x, y, z = 0, 0, 0
                
                # Store latest for display
                current_x, current_y, current_z = x, y, z
                
                # Add to buffer and plot data
                st.session_state.buffer.append([x, y, z])
                
                # Add to plot data
                new_row = pd.DataFrame([[x, y, z]], columns=['x', 'y', 'z'])
                st.session_state.plot_data = pd.concat([st.session_state.plot_data, new_row], ignore_index=True)
                
                # Trim
                LIMIT = 200
                if len(st.session_state.plot_data) > LIMIT:
                    st.session_state.plot_data = st.session_state.plot_data.iloc[-LIMIT:]
                
            except Exception as e:
                pass
            
            # 3. Update UI (Throttled to max 10 FPS = 100ms)
            current_time = time.time()
            if current_time - st.session_state.last_ui_update >= 0.1:
                chart_placeholder.line_chart(st.session_state.plot_data)
                st.session_state.last_ui_update = current_time
            
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
            
            # 4. Inference with improved threshold logic
            WINDOW_SIZE = 100
            OVERLAP = 50
            if len(st.session_state.buffer) >= WINDOW_SIZE:
                 window = st.session_state.buffer[-WINDOW_SIZE:]
                 try:
                     # Calculate metrics for "still" detection
                     window_arr = np.array(window)
                     mag = np.sqrt(np.sum(window_arr**2, axis=1))
                     mag_std = np.std(mag)
                     
                     # Also check raw acceleration variance (better metric for movement)
                     raw_std = np.std(window_arr, axis=0)  # std per axis
                     raw_std_mean = np.mean(raw_std)
                     
                     # Very conservative threshold: only classify as "still" if BOTH are very low
                     # Lower threshold from 0.5 to 0.12 to reduce false positives
                     if mag_std < 0.12 and raw_std_mean < 0.20:
                         pred_class = "still"
                         confidence = "threshold"
                     elif model:
                         feats = extract_realtime_features(window)
                         pred_class = model.predict(feats)[0]
                         # Try to get prediction probability if available
                         try:
                             proba = model.predict_proba(feats)[0]
                             confidence = f"{max(proba)*100:.0f}%"
                         except:
                             confidence = "model"
                     else:
                         pred_class = "unknown"
                         confidence = "N/A"
                     
                     activity_placeholder.markdown(
                         f"""<div class='metric-card'>
                             <div style='color: #aaa'>Detected</div>
                             <div class='activity-text'>{pred_class.upper()}</div>
                             <div style='color: #666; font-size: 0.7em; margin-top: 5px;'>
                             Conf: {confidence}<br>
                             Mag σ: {mag_std:.3f} | Raw σ: {raw_std_mean:.3f}
                             </div>
                            </div>""", 
                         unsafe_allow_html=True
                     )
                 except Exception as e:
                     activity_placeholder.markdown(f"<div style='color: red'>Error: {str(e)}</div>", unsafe_allow_html=True)
                 st.session_state.buffer = st.session_state.buffer[OVERLAP:]
                 
        else:
            # No data
            time.sleep(0.01) # Short sleep to prevent CPU spin
            
else:
    st.write("Server Stopped.")
