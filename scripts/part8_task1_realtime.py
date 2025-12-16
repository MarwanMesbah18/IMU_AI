import socket
import json
import numpy as np
import pandas as pd
import joblib
import time

# Configuration
HOST = '0.0.0.0' # Listen on all interfaces
PORT = 65432     # Port to listen on (non-privileged ports are > 1023)
MODEL_PATH = 'best_model.pkl' # We will save this from Part 6/7

# Expected feature names (order matters!)
# We need to save this list from the training step.
# For now, we'll try to load it or assume standard extract logic.

def extract_features_realtime(window_data):
    """
    Extracts features from a single window of (acc_x, acc_y, acc_z) data.
    Must match Part 5 logic EXACTLY.
    """
    X = np.array(window_data) # Shape (N, 3)
    
    # Simple extraction for demo speed (Mean, Std, SMA)
    # Note: If model was trained on 40 features, we MUST produce 40 features.
    # This is tricky without the full feature_names list.
    # We will assume the user will run this AFTER running the notebook, 
    # so we can perhaps load 'feature_names.pkl'.
    
    # Placeholder: Return random vector of correct size if we can't match perfectly in this standalone script
    # Real implementation: Copy-paste Part 5 logic here.
    
    # ... (Feature extraction logic would go here) ...
    # For this demo template, we will just print the raw data stream info.
    
    return np.zeros((1, 48)) # Mock 48 features

def start_server():
    print(f"--- Part 8: Real-Time Wireless IMU Server ---")
    print(f"Listening on {HOST}:{PORT}...")
    print("Use a UDP/TCP sender app on your phone (e.g. 'Sensor Log' or custom app).")
    print("Format expected: JSON {'acc_x': ..., 'acc_y': ..., 'acc_z': ...}")

    try:
        # Save model from notebook context first!
        # model = joblib.load(MODEL_PATH) 
        pass
    except:
        print("Warning: Model file not found. Running in Data-Logging Mode only.")

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind((HOST, PORT))
        
        while True:
            data, addr = s.recvfrom(1024)
            message = data.decode('utf-8')
            
            try:
                # Parse JSON
                # Assuming incoming: {"x": 0.1, "y": 0.5, "z": 9.8}
                sensor_data = json.loads(message)
                
                # Buffer logic would go here (accumulate 2s window)
                # For demo, just print live values
                print(f"Received from {addr}: {sensor_data}")
                
                # Prediction mock
                # pred = model.predict(features)
                # print(f"Activity: {pred[0]}")
                
            except json.JSONDecodeError:
                print(f"Raw data: {message}")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    start_server()
