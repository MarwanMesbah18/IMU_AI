
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
WINDOW_SIZE = 100
OVERLAP = 50

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

# --- PREDICTION LOGIC (Used by both modes) ---
def predict_activity(check_buffer, result_placeholder):
    window = check_buffer[-WINDOW_SIZE:]
    try:
        # Calculate movement metrics
        window_arr = np.array(window)
        mag = np.sqrt(np.sum(window_arr**2, axis=1))
        mag_std = np.std(mag)
        mag_mean = np.mean(mag)
        
        # Check variance (better metric than just magnitude)
        raw_variance = np.var(window_arr, axis=0)
        total_variance = np.sum(raw_variance)
        
        # Smart threshold logic
        if mag_std < 0.8 and total_variance < 1.5:
            pred_class = "still"
            confidence = f"threshold"
            prob_display = "still: 95% (rule)"
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
    except Exception as e:
        result_placeholder.markdown(f"<div style='color: red'>Error: {str(e)}</div>", unsafe_allow_html=True)


# --- LIVE STREAM MODE ---
def run_live_stream():
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("Settings")
        server_running = st.toggle("🔴 Start Live Server", value=False)
        
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
    if 'packet_count' not in st.session_state: st.session_state.packet_count = 0
    if 'last_ui_update' not in st.session_state: st.session_state.last_ui_update = 0

    # Initialize Socket
    if 'sock' not in st.session_state:
        st.session_state.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        st.session_state.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            st.session_state.sock.bind((HOST, PORT))
        except OSError as e:
            st.error(f"Error: Port {PORT} is occupied.")
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
        # print("🚀 Server Loop Running")  # Commented to reduce log spam
        
        # Non-blocking loop simulation using Streamlit's rerun capability is tricky.
        # But for "Simple Mode", a while Loop is acceptable IF user knows it blocks UI navigation.
        # To make it stoppable, we check server_running every iter.
        
        # Better: Use a placeholder for the "Stop" warning
        st.caption("Press 'Stop' in sidebar to pause server.")
        
        while True:
            if not server_running: break

            # 1. Receive
            try:
                data, addr = st.session_state.sock.recvfrom(4096)
                has_data = True
            except BlockingIOError:
                has_data = False
                time.sleep(0.01)
            except Exception as e:
                # print(f"Socket error: {e}")
                has_data = False
                time.sleep(0.1)

            if has_data:
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
                    
                    # Sanitize
                    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                        x, y, z = 0.0, 0.0, 0.0
                    
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
                    print(f"Process Error: {e}")

            # 2. Update UI
            current_time = time.time()
            if current_time - st.session_state.last_ui_update >= 0.1:
                if st.session_state.plot_buffer:
                    df_plot = pd.DataFrame(st.session_state.plot_buffer)
                    chart_placeholder.line_chart(df_plot)
                
                st.session_state.last_ui_update = current_time
                
                debug_placeholder.markdown(
                    f"""
                    <div style='background-color: #262730; padding: 10px; border-radius: 5px; font-size: 0.8em;'>
                    <b>Status:</b> 🟢 Receiving<br>
                    <b>Packets:</b> {st.session_state.packet_count}<br>
                    <b>Latest:</b> X: {current_x:.2f} Y: {current_y:.2f} Z: {current_z:.2f}
                    </div>
                    """, unsafe_allow_html=True
                )
            
            # 3. Predict
            if len(st.session_state.buffer) >= WINDOW_SIZE:
                predict_activity(st.session_state.buffer, activity_placeholder)
                # Slide Window
                st.session_state.buffer = st.session_state.buffer[-OVERLAP:]
            
            time.sleep(0.02)
    else:
        st.warning("Server is STOPPED. Toggle 'Start Live Server' in the sidebar to begin.")

# --- FILE REPLAY MODE ---
def run_file_replay():
    st.markdown("### 📂 Drag & Drop Sensor Data")
    st.markdown("Upload a CSV with columns: `x, y, z` OR `acc_x, acc_y, acc_z`")
    
    uploaded_file = st.file_uploader("Drop CSV here (max 200MB)", type=['csv', 'json'])
    
    speed = st.slider("Playback Speed", 0.1, 10.0, 1.0, 0.1)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Replay Stream")
        replay_chart = st.empty()
    with col2:
        st.subheader("Prediction Replay")
        replay_activity = st.empty()
    
    if uploaded_file is not None:
        try:
            # Load Data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
            
            # Show Preview
            with st.expander("📄 Data Preview (First 5 rows)", expanded=True):
                st.dataframe(df.head())
            
            # Robust Column Mapping
            df.columns = [c.lower().strip() for c in df.columns]
            
            # Try to find x, y, z
            x_col, y_col, z_col = None, None, None
            
            possible_x = ['x', 'acc_x', 'acceleration_x', 'ax']
            possible_y = ['y', 'acc_y', 'acceleration_y', 'ay']
            possible_z = ['z', 'acc_z', 'acceleration_z', 'az']
            
            for c in df.columns:
                if c in possible_x: x_col = c
                if c in possible_y: y_col = c
                if c in possible_z: z_col = c
            
            if x_col and y_col and z_col:
                st.success(f"✅ Found columns: `{x_col}`, `{y_col}`, `{z_col}`. Ready to play!")
                data = df[[x_col, y_col, z_col]].values
                
                if st.button("▶️ CLICK HERE TO START REPLAY", type="primary"):
                    
                    # --- PHASE 1: PRE-COMPILE ---
                    st.info("Compiling results... (Please wait)")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    replay_buffer = []
                    plot_data = []
                    playback_frames = [] # Store pre-calculated frames
                    
                    total_rows = len(data)
                    
                    # 1. Compile Loop (No Sleep = Fast)
                    for i, row in enumerate(data):
                        x, y, z = row
                        
                        replay_buffer.append([x, y, z])
                        plot_data.append({'x': x, 'y': y, 'z': z})
                        if len(plot_data) > 200: plot_data.pop(0)
                        
                        # Only capture keyframes for playback (e.g. 10Hz)
                        if i % 5 == 0:
                            # Snapshot current state
                            frame = {
                                'pct': i / total_rows,
                                'chart': pd.DataFrame(plot_data),
                                'pred_html': None
                            }
                            
                            # Run Prediction logic if window full
                            if len(replay_buffer) >= WINDOW_SIZE:
                                # We need to capture the HTML output of predict_activity
                                # Refactor predict_activity to RETURN string instead of direct update?
                                # For now, let's just duplicate logic briefly or assume we invoke it during playback?
                                # BETTER: Invoke it here and save Result String.
                                pass 
                            
                            playback_frames.append(frame)

                        # Slide Buffer
                        if len(replay_buffer) >= WINDOW_SIZE:
                            # Perform Actual Prediction Here
                            window = replay_buffer[-WINDOW_SIZE:]
                            # ... (Prediction Logic Copy) ...
                            # To avoid massive code duplication, let's keep it simple: 
                            # Pre-calculation of just the MODEL PREDICTIONS is enough. Chart can be live.
                            
                            # Actually, user wants "Delay" gone. 
                            # If we pre-calc ALL predictions, we can just look them up.
                            pass
                            
                            replay_buffer = replay_buffer[-OVERLAP:]
                    
                    # Okay, full pre-calc is complex due to the HTML generation. 
                    # Let's do a meaningful optimization: 
                    # Just RUN the loop but DO NOT SLEEP, save the "Changes" to a list, then play the list.
                    
                    results_list = []
                    temp_buffer = []
                    
                    # Re-Running Logic Cleanly
                    for i, row in enumerate(data):
                        temp_buffer.append(row)
                        
                        # Time to predict?
                        pred_result = None
                        if len(temp_buffer) >= WINDOW_SIZE:
                            # Predict
                            window = temp_buffer[-WINDOW_SIZE:]
                            # Extract Features
                            try:
                                # Reuse existing functions...
                                # We'll just run the logic live but pre-calculate "Is it time to predict?"
                                pass
                            except: pass
                            
                            temp_buffer = temp_buffer[-OVERLAP:] # Slide

                    # RE-STRATEGY: The user just wants it to be faster/smoother.
                    # The current implementation calculates AND sleeps.
                    # Let's stick to the Live Loop but Optimize the PREDICT call.
                    
                    # BETTER: Just remove the prediction logic from the rendering loop? No, that defeats the purpose.
                    
                    # Let's Implement: FAST PRE-SCAN
                    # We will predict for EVERY window index in advance.
                    
                    predictions_map = {} # Index -> HTML String
                    
                    sim_buffer = []
                    steps = []
                    
                    # 1. Fast Scan
                    for i, row in enumerate(data):
                        sim_buffer.append(row)
                        if len(sim_buffer) >= WINDOW_SIZE:
                            # Predict!
                            try:
                                window = sim_buffer[-WINDOW_SIZE:]
                                # LOGIC START
                                window_arr = np.array(window)
                                mag = np.sqrt(np.sum(window_arr**2, axis=1))
                                mag_std = np.std(mag)
                                total_variance = np.sum(np.var(window_arr, axis=0))
                                
                                if mag_std < 0.8 and total_variance < 1.5:
                                    res = ("STILL", "threshold", "still: 95%", mag_std, total_variance)
                                elif model:
                                    feats = extract_realtime_features(window)
                                    p_cls = model.predict(feats)[0]
                                    proba = model.predict_proba(feats)[0]
                                    conf = f"{max(proba)*100:.0f}%"
                                    # ... simplified for storage
                                    res = (p_cls, conf, "Model", mag_std, total_variance)
                                else:
                                    res = ("No Model", "", "", 0, 0)
                                
                                predictions_map[i] = res
                                # LOGIC END
                            except: pass
                            sim_buffer = sim_buffer[-OVERLAP:]
                    
                    st.success(f"✅ Pre-compiled {len(predictions_map)} predictions!")
                    
                    # 2. Playback Loop (Graphics Only)
                    replay_buffer = []
                    plot_data = []
                    
                    for i, row in enumerate(data):
                         x, y, z = row
                         plot_data.append({'x': x, 'y': y, 'z': z})
                         if len(plot_data) > 200: plot_data.pop(0)
                         
                         # Render Frame
                         if i % 5 == 0:
                             status_text.text(f"▶️ Playing: {i}/{total_rows}")
                             replay_chart.line_chart(pd.DataFrame(plot_data))
                             progress_bar.progress(i / total_rows)
                             
                         # Check if we have a pre-calc result for this index
                         # MOVED OUTSIDE the render block to ensure we catch indices like 99, 149
                         if i in predictions_map:
                             p_cls, conf, prob, m_std, m_var = predictions_map[i]
                             replay_activity.markdown(
                                f"""<div class='metric-card'>
                                    <div class='activity-text'>{str(p_cls).upper()}</div>
                                    <div style='color: #666; font-size: 0.65em; margin-top: 8px;'>
                                    <b>Confidence:</b> {conf}<br>
                                    <b>Metrics:</b> Mag σ={m_std:.3f} | Var={m_var:.2f}
                                    </div>
                                   </div>""", 
                                unsafe_allow_html=True
                             )

                         time.sleep(0.02 / speed)
                         
                    status_text.text("✅ Done!")
                    st.balloons()
            else:
                st.error("❌ Could not find X, Y, Z columns! Please rename columns in your CSV.")
                st.write("Available columns:", list(df.columns))
                
        except Exception as e:
            st.error(f"Error parsing file: {e}")

# --- MAIN ---
st.title("IMU Activity Recognition System 🧠")

tab1, tab2 = st.tabs(["📡 Live Stream", "📂 File Replay"])

with tab1:
    run_live_stream()

with tab2:
    run_file_replay()
