
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

import streamlit as st
import socket
import json
import numpy as np
import pandas as pd
import joblib
import time
import atexit
from datetime import datetime
from scipy.fft import fft, fftfreq

# --- CONFIGURATION ---
HOST = '0.0.0.0'
PORT = 65432
MODEL_PATH = 'models/best_model.pkl'
WINDOW_SIZE = 100
OVERLAP = 90  # Keep 90% (Slide by 10) = predict every 0.2s (5Hz) -> Very Responsive
MAX_CSV_BUFFER = 50000  # Limit CSV buffer to prevent memory issues

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
    .recording-indicator {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 10px;
        background-color: #ff4444;
        border-radius: 8px;
        margin-bottom: 10px;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
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

# --- FEATURE EXTRACTION (Matches Part 5 logic) ---
def extract_realtime_features(window_data):
    """
    Extract features from a 100-sample window.
    NOTE: Training used per-subject global normalization which we can't replicate in real-time.
    Using raw features + threshold approach instead.
    """
    X = np.array(window_data)  # Shape: (100, 3) - RAW sensor data
    
    # 1. Time Domain Features
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
    
    # 2. Frequency Domain Features
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

# --- PREDICTION LOGIC ---
def predict_activity(check_buffer, result_placeholder):
    window = check_buffer[-WINDOW_SIZE:]
    pred_class = "unknown"
    confidence = "N/A"
    
    try:
        # Calculate movement metrics
        window_arr = np.array(window)
        mag = np.sqrt(np.sum(window_arr**2, axis=1))
        mag_std = np.std(mag)
        mag_mean = np.mean(mag)
        
        # Check variance (better metric than just magnitude)
        raw_variance = np.var(window_arr, axis=0)
        total_variance = np.sum(raw_variance)
        
        # --- RAPID STILL DETECTION ---
        # Look at only the last 0.5s (25 samples)
        if len(window_arr) >= 25:
            recent_window = window_arr[-25:]
            recent_var = np.var(recent_window, axis=0)
            recent_total_var = np.sum(recent_var)
            
            if recent_total_var < 0.1:  # Very stable
                pred_class = "still"
                confidence = "99%"
                prob_display = "still: 99% | walk: 1% | shake: 0%"
                
                result_placeholder.markdown(
                    f"""<div class='metric-card'>
                        <div class='activity-text'>{pred_class.upper()}</div>
                        <div style='color: #666; font-size: 0.65em; margin-top: 8px;'>
                        <b>Confidence:</b> {confidence}<br>
                        <b>Probabilities:</b> {prob_display}<br>
                        <b>Metrics:</b> Mag σ={mag_std:.3f} | Var={total_variance:.2f}
                        </div>
                       </div>""", 
                    unsafe_allow_html=True
                )
                return pred_class
        
        # Smart threshold logic (TUNED FOR SENSITIVITY)
        if mag_std < 0.6 and total_variance < 3.0:
            pred_class = "still"
            confidence = "98%"
            prob_display = "still: 98% | walk: 2% | shake: 0%"
        
        # --- SHAKE DETECTION ---
        # Shaking has very high peak accelerations (>15 m/s²) and high variance
        # Walking typically stays below 15 m/s² peak
        elif mag_std > 8.0 or total_variance > 150.0 or np.max(mag) > 25.0:
            pred_class = "shake"
            confidence = "95%"
            prob_display = f"shake: 95% | walk: 5% | still: 0%"
        
        elif model:
            # Trust model for active movements
            feats = extract_realtime_features(window)
            pred_class = model.predict(feats)[0]
            
            # Get probabilities
            try:
                proba = model.predict_proba(feats)[0]
                max_prob = max(proba)
                confidence = f"{max_prob*100:.0f}%"
                class_probs = {model.classes_[i]: f"{proba[i]*100:.0f}%" for i in range(len(model.classes_))}
                prob_display = " | ".join([f"{k}: {v}" for k, v in class_probs.items()])
            except:
                confidence = "N/A"
                prob_display = "N/A"
        else:
            pred_class = "No Model"
            confidence = "N/A"
            prob_display = "N/A"
        
        result_placeholder.markdown(
            f"""<div class='metric-card'>
                <div class='activity-text'>{pred_class.upper()}</div>
                <div style='color: #666; font-size: 0.65em; margin-top: 8px;'>
                <b>Confidence:</b> {confidence}<br>
                <b>Probabilities:</b> {prob_display}<br>
                <b>Metrics:</b> Mag σ={mag_std:.3f} | Var={total_variance:.2f}
                </div>
               </div>""", 
            unsafe_allow_html=True
        )
        return pred_class
    except Exception as e:
        result_placeholder.markdown(f"<div style='color: red'>Error: {str(e)}</div>", unsafe_allow_html=True)
        return "error"


# --- LIVE STREAM MODE ---
def run_live_stream():
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("⚙️ Settings")
        server_running = st.toggle("🔴 Start Live Server", value=False)
        
        st.divider()
        
        # --- CSV RECORDING CONTROLS ---
        st.subheader("📹 Recording")
        recording = st.toggle("Record Data", value=False, key="recording_toggle")
        
        if recording:
            st.markdown("""
                <div style='background-color: #ff4444; padding: 8px; border-radius: 5px; text-align: center;'>
                    🔴 REC
                </div>
            """, unsafe_allow_html=True)
        
        # Show buffer size
        if 'csv_buffer' in st.session_state:
            buffer_size = len(st.session_state.csv_buffer)
            st.caption(f"Buffer: {buffer_size:,} samples")
            
            if buffer_size >= MAX_CSV_BUFFER:
                st.warning("⚠️ Buffer full! Data may be lost.")
        
        # Download button
        if 'csv_buffer' in st.session_state and len(st.session_state.csv_buffer) > 0:
            st.divider()
            st.subheader("💾 Export Data")
            
            # Generate CSV
            df_export = pd.DataFrame(st.session_state.csv_buffer)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"imu_recording_{timestamp}.csv"
            
            csv_data = df_export.to_csv(index=False)
            
            st.download_button(
                label=f"📥 Download CSV ({len(df_export)} rows)",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                key="download_csv"
            )
            
            if st.button("🗑️ Clear Buffer"):
                st.session_state.csv_buffer = []
                st.rerun()
        
        st.divider()
        
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
        st.write("### How to Connect")
        st.markdown("""
        1. Ensure phone & PC are on same Wi-Fi.
        2. Run `web_sensor_bridge.py` in terminal.
        3. Open `https://<IP>:5000` on phone.
        4. Click 'Start Stream'.
        """)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Live Accelerometer Data")
        chart_placeholder = st.empty()
    
    with col2:
        st.subheader("Activity Prediction")
        activity_placeholder = st.empty()
        debug_placeholder = st.empty()

    # Session State Init
    if 'buffer' not in st.session_state: st.session_state.buffer = []
    if 'plot_buffer' not in st.session_state: st.session_state.plot_buffer = []
    if 'csv_buffer' not in st.session_state: st.session_state.csv_buffer = []
    if 'packet_count' not in st.session_state: st.session_state.packet_count = 0
    if 'last_ui_update' not in st.session_state: st.session_state.last_ui_update = 0

    # Initialize Socket
    if 'sock' not in st.session_state:
        st.session_state.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        st.session_state.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            st.session_state.sock.bind((HOST, PORT))
        except OSError as e:
            st.error(f"Error: Port {PORT} is occupied. Please restart the app.")
            st.stop()
        st.session_state.sock.setblocking(False)
        
        def cleanup_socket():
            try:
                if hasattr(st.session_state, 'sock'):
                    st.session_state.sock.close()
            except: pass
        atexit.register(cleanup_socket)

    if server_running:
        current_x, current_y, current_z = 0.0, 0.0, 0.0
        current_prediction = "waiting..."
        
        st.caption("Press 'Stop' in sidebar to pause server.")
        
        while True:
            if not server_running: break

            # 1. Receive - drain the socket queue
            packets_received = 0
            max_packets_per_loop = 10  # Process multiple packets per loop to avoid backlog
            
            for _ in range(max_packets_per_loop):
                try:
                    data, addr = st.session_state.sock.recvfrom(4096)
                    packets_received += 1
                except BlockingIOError:
                    break  # No more data in queue
                except Exception as e:
                    break

                if data:
                    try:
                        msg = data.decode('utf-8')
                        # Parse
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
                        
                        # Validate data
                        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                            x, y, z = 0.0, 0.0, 0.0
                        
                        # Clamp extreme values
                        LIMIT_VAL = 200.0
                        x = max(-LIMIT_VAL, min(LIMIT_VAL, x))
                        y = max(-LIMIT_VAL, min(LIMIT_VAL, y))
                        z = max(-LIMIT_VAL, min(LIMIT_VAL, z))

                        # Buffer for inference
                        st.session_state.buffer.append([x, y, z])
                        
                        # Buffer for Plotting
                        st.session_state.plot_buffer.append({'x': x, 'y': y, 'z': z})
                        if len(st.session_state.plot_buffer) > 200:
                            st.session_state.plot_buffer.pop(0)
                        
                        current_x, current_y, current_z = x, y, z
                        st.session_state.packet_count += 1

                    except Exception as e:
                        pass  # Silently handle parse errors

            # 2. Update UI (rate limited)
            current_time = time.time()
            if current_time - st.session_state.last_ui_update >= 0.1:  # 10Hz UI updates
                if st.session_state.plot_buffer:
                    df_plot = pd.DataFrame(st.session_state.plot_buffer)
                    chart_placeholder.line_chart(df_plot)
                
                st.session_state.last_ui_update = current_time
                
                # Determine recording status
                rec_status = "🔴 RECORDING" if recording else "⚪ Not Recording"
                
                debug_placeholder.markdown(
                    f"""
                    <div style='background-color: #262730; padding: 10px; border-radius: 5px; font-size: 0.8em;'>
                    <b>Status:</b> 🟢 Receiving<br>
                    <b>Packets:</b> {st.session_state.packet_count}<br>
                    <b>Latest:</b> X: {current_x:.2f} Y: {current_y:.2f} Z: {current_z:.2f}<br>
                    <b>Recording:</b> {rec_status}
                    </div>
                    """, unsafe_allow_html=True
                )
            
            # 3. Predict
            if len(st.session_state.buffer) >= WINDOW_SIZE:
                current_prediction = predict_activity(st.session_state.buffer, activity_placeholder)
                
                # Record to CSV if enabled
                if recording and len(st.session_state.csv_buffer) < MAX_CSV_BUFFER:
                    # Record the last data point with prediction
                    st.session_state.csv_buffer.append({
                        'timestamp': datetime.now().isoformat(),
                        'acc_x': current_x,
                        'acc_y': current_y,
                        'acc_z': current_z,
                        'prediction': current_prediction
                    })
                
                # Slide Window
                st.session_state.buffer = st.session_state.buffer[-OVERLAP:]
            
            # Small sleep to prevent CPU overload
            time.sleep(0.01)
    else:
        st.warning("Server is STOPPED. Toggle 'Start Live Server' in the sidebar to begin.")


# --- MAIN ---
st.title("IMU Activity Recognition System 🧠")

# Run Live Stream Directly
run_live_stream()
