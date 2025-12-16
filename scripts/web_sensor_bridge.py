import socket
import json
import logging
import atexit
from flask import Flask, render_template_string, request, jsonify

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- CONFIGURATION ---
UDP_TARGET_IP = "127.0.0.1"
UDP_TARGET_PORT = 65432

# Setup UDP Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Register cleanup function
def cleanup_socket():
    try:
        sock.close()
        logger.info("UDP socket closed")
    except:
        pass

atexit.register(cleanup_socket)

# --- HTML TEMPLATE ---
# Mobile-friendly generic sensor sender
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>IMU Sensor Bridge</title>
    <style>
        body {
            background-color: #1a1a1a;
            color: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
            padding: 20px;
            box-sizing: border-box;
            text-align: center;
        }
        h1 { margin-bottom: 10px; font-size: 1.5rem; }
        .status { 
            font-size: 1.2rem; 
            margin-bottom: 20px; 
            padding: 10px; 
            border-radius: 8px; 
            width: 100%;
            max-width: 300px;
        }
        .status.connected { background-color: #2e7d32; }
        .status.stopped { background-color: #c62828; }
        
        .data-display {
            background-color: #333;
            padding: 15px;
            border-radius: 10px;
            width: 100%;
            max-width: 300px;
            margin-bottom: 20px;
            font-family: monospace;
            font-size: 1rem;
        }
        
        button {
            background-color: #2196F3;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 50px;
            font-size: 1.2rem;
            cursor: pointer;
            width: 100%;
            max-width: 300px;
            touch-action: manipulation;
            transition: background 0.2s;
        }
        button:active { transform: scale(0.98); }
        button.stop { background-color: #c62828; }
        
        .debug-info {
            margin-top: 20px;
            font-size: 0.8rem;
            color: #888;
        }
    </style>
</head>
<body>

    <h1>📱 Sensor Bridge</h1>
    
    <div id="statusBox" class="status stopped">STOPPED</div>
    
    <div class="data-display">
        X: <span id="valX">0.00</span><br>
        Y: <span id="valY">0.00</span><br>
        Z: <span id="valZ">0.00</span>
    </div>

    <button id="toggleBtn" onclick="toggleSensors()">Start Stream 🚀</button>

    <div class="debug-info">
        Ensure you are on HTTPS.<br>
        Keep screen on.
    </div>

    <script>
        let isRunning = false;
        let lastSendTime = 0;
        let lastUiUpdate = 0;
        const SEND_INTERVAL = 20; // ms (Target ~50Hz for Model)
        
        const statusBox = document.getElementById('statusBox');
        const toggleBtn = document.getElementById('toggleBtn');
        const valX = document.getElementById('valX');
        const valY = document.getElementById('valY');
        const valZ = document.getElementById('valZ');

        let isSendingData = false;

        function handleMotion(event) {
            if (!isRunning) return;

            const acc = event.accelerationIncludingGravity;
            const now = Date.now();
            const x = acc.x || 0;
            const y = acc.y || 0;
            const z = acc.z || 0;

            // 1. Update UI (Low Priority - 2Hz)
            if (now - lastUiUpdate > 500) {
                valX.innerText = x.toFixed(2);
                valY.innerText = y.toFixed(2);
                valZ.innerText = z.toFixed(2);
                lastUiUpdate = now;
            }

            // 2. Send Data (Network Safe Mode)
            // Only send if the previous request is DONE.
            // This prevents "Connection queue" freezes in mobile browsers.
            if (!isSendingData && (now - lastSendTime > SEND_INTERVAL)) {
                sendData(x, y, z);
                lastSendTime = now;
            }
        }

        async function sendData(x, y, z) {
            isSendingData = true;
            try {
                await fetch('/data', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        acc_x: x,
                        acc_y: y,
                        acc_z: z
                    })
                });
            } catch (e) {
                console.error("Send failed", e);
            } finally {
                isSendingData = false;
            }
        }

        async function requestPermission() {
            // iOS 13+ permission request
            if (typeof DeviceMotionEvent.requestPermission === 'function') {
                try {
                    const response = await DeviceMotionEvent.requestPermission();
                    if (response === 'granted') {
                        return true;
                    } else {
                        alert('Permission denied');
                        return false;
                    }
                } catch (e) {
                    console.error(e);
                    return false;
                }
            }
            return true;
        }

        async function toggleSensors() {
            if (isRunning) {
                // Stop
                window.removeEventListener('devicemotion', handleMotion);
                isRunning = false;
                statusBox.textContent = "STOPPED";
                statusBox.className = "status stopped";
                toggleBtn.textContent = "Start Stream 🚀";
                toggleBtn.className = "";
            } else {
                // Start
                const granted = await requestPermission();
                if (granted) {
                    window.addEventListener('devicemotion', handleMotion);
                    isRunning = true;
                    statusBox.textContent = "STREAMING...";
                    statusBox.className = "status connected";
                    toggleBtn.textContent = "Stop Stream 🛑";
                    toggleBtn.className = "stop";
                }
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/data', methods=['POST'])
def receive_data():
    try:
        data = request.json
        if data:
            # Create JSON payload for UDP
            # Main app expects: {'acc_x': float, ...} or similiar
            msg = json.dumps(data)
            sock.sendto(msg.encode('utf-8'), (UDP_TARGET_IP, UDP_TARGET_PORT))
            return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.error(f"Error forwarding data: {e}")
        return jsonify({"status": "error"}), 500

    return jsonify({"status": "no data"}), 400

if __name__ == '__main__':
    # Get local IP for convenience print
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "127.0.0.1"

    print("="*40)
    print(f"🚀 SENSOR BRIDGE STARTED")
    print(f"📱 Open this URL on your phone:")
    print(f"👉 https://{local_ip}:5000")
    print("="*40)
    print("NOTE: You will see a 'Not Secure' warning.")
    print("Click 'Advanced' -> 'Proceed' to access the site.")
    print("="*40)
    
    # Run with adhoc SSL (required for sensors on mobile)
    app.run(host='0.0.0.0', port=5000, ssl_context='adhoc', debug=False)
