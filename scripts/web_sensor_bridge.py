import socket
import json
import logging
import atexit
import tempfile
import os
import ssl
import sys

from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit

# Suppress Flask's default logging for cleaner output
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Configure our logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'imu_sensor_secret'

# Use threading mode (most compatible)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

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
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>IMU Sensor Bridge</title>
    <style>
        * { box-sizing: border-box; }
        body {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
            text-align: center;
        }
        h1 { 
            margin-bottom: 5px; 
            font-size: 1.8rem; 
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { color: #888; margin-bottom: 20px; font-size: 0.9rem; }
        .status { 
            font-size: 1.1rem; 
            margin-bottom: 15px; 
            padding: 12px 20px; 
            border-radius: 25px; 
            width: 100%;
            max-width: 280px;
            font-weight: 600;
        }
        .status.connected { 
            background: linear-gradient(90deg, #00c853, #00e676);
            box-shadow: 0 4px 15px rgba(0, 200, 83, 0.4);
        }
        .status.stopped { 
            background: linear-gradient(90deg, #ff5252, #ff1744);
            box-shadow: 0 4px 15px rgba(255, 23, 68, 0.4);
        }
        .data-display {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 15px;
            width: 100%;
            max-width: 280px;
            margin-bottom: 20px;
            font-family: monospace;
            font-size: 1.1rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .data-row { display: flex; justify-content: space-between; margin: 8px 0; }
        .data-label { color: #888; }
        .data-value { color: #00d4ff; font-weight: 600; }
        .stats {
            background: rgba(0, 200, 83, 0.2);
            padding: 10px 15px;
            border-radius: 10px;
            width: 100%;
            max-width: 280px;
            margin-bottom: 20px;
            font-size: 0.85rem;
        }
        button {
            background: linear-gradient(90deg, #2196F3, #00bcd4);
            color: white;
            border: none;
            padding: 18px 35px;
            border-radius: 50px;
            font-size: 1.2rem;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            max-width: 280px;
            touch-action: manipulation;
            box-shadow: 0 4px 15px rgba(33, 150, 243, 0.4);
        }
        button:active { transform: scale(0.98); }
        button.stop { 
            background: linear-gradient(90deg, #ff5252, #ff1744);
            box-shadow: 0 4px 15px rgba(255, 23, 68, 0.4);
        }
        .debug-info { margin-top: 20px; font-size: 0.75rem; color: #666; max-width: 280px; }
        .connection-dot {
            display: inline-block;
            width: 10px; height: 10px;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 1.5s infinite;
        }
        .connection-dot.connected { background: #00e676; }
        .connection-dot.disconnected { background: #ff1744; animation: none; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    </style>
</head>
<body>
    <h1>📱 IMU Sensor Bridge</h1>
    <p class="subtitle">Real-time sensor streaming</p>
    
    <div id="statusBox" class="status stopped">
        <span id="connDot" class="connection-dot disconnected"></span>
        <span id="statusText">STOPPED</span>
    </div>
    
    <div class="data-display">
        <div class="data-row"><span class="data-label">X:</span><span class="data-value" id="valX">0.000</span></div>
        <div class="data-row"><span class="data-label">Y:</span><span class="data-value" id="valY">0.000</span></div>
        <div class="data-row"><span class="data-label">Z:</span><span class="data-value" id="valZ">0.000</span></div>
    </div>
    
    <div class="stats">
        <strong>Packets:</strong> <span id="packetCount">0</span> | 
        <strong>Rate:</strong> <span id="sendRate">0</span> Hz
    </div>

    <button id="toggleBtn" onclick="toggleSensors()">Start Stream 🚀</button>
    <div class="debug-info">Keep screen on during streaming.</div>

    <script src="https://cdn.socket.io/4.6.0/socket.io.min.js"></script>
    <script>
        let isRunning = false;
        let packetCount = 0;
        let lastRateCalc = Date.now();
        let packetsInWindow = 0;
        
        const statusBox = document.getElementById('statusBox');
        const statusText = document.getElementById('statusText');
        const connDot = document.getElementById('connDot');
        const toggleBtn = document.getElementById('toggleBtn');
        const valX = document.getElementById('valX');
        const valY = document.getElementById('valY');
        const valZ = document.getElementById('valZ');
        const packetCountEl = document.getElementById('packetCount');
        const sendRateEl = document.getElementById('sendRate');

        // Socket.IO connection - use polling as fallback for compatibility
        const socket = io({
            transports: ['polling', 'websocket']
        });
        
        socket.on('connect', () => {
            console.log('Connected to server');
        });
        
        socket.on('disconnect', () => {
            console.log('Disconnected');
            if (isRunning) statusText.textContent = "RECONNECTING...";
        });

        function handleMotion(event) {
            if (!isRunning) return;

            const acc = event.accelerationIncludingGravity;
            const x = acc.x || 0;
            const y = acc.y || 0;
            const z = acc.z || 0;

            valX.innerText = x.toFixed(3);
            valY.innerText = y.toFixed(3);
            valZ.innerText = z.toFixed(3);

            // Send via Socket.IO
            socket.emit('sensor_data', { acc_x: x, acc_y: y, acc_z: z, t: Date.now() });
            
            packetCount++;
            packetsInWindow++;
            packetCountEl.innerText = packetCount;
            
            const now = Date.now();
            if (now - lastRateCalc >= 1000) {
                sendRateEl.innerText = packetsInWindow;
                packetsInWindow = 0;
                lastRateCalc = now;
            }
        }

        async function requestPermission() {
            if (typeof DeviceMotionEvent.requestPermission === 'function') {
                try {
                    const response = await DeviceMotionEvent.requestPermission();
                    return response === 'granted';
                } catch (e) {
                    console.error(e);
                    return false;
                }
            }
            return true;
        }

        async function toggleSensors() {
            if (isRunning) {
                window.removeEventListener('devicemotion', handleMotion);
                isRunning = false;
                statusText.textContent = "STOPPED";
                statusBox.className = "status stopped";
                connDot.className = "connection-dot disconnected";
                toggleBtn.textContent = "Start Stream 🚀";
                toggleBtn.className = "";
            } else {
                const granted = await requestPermission();
                if (granted) {
                    window.addEventListener('devicemotion', handleMotion);
                    isRunning = true;
                    statusText.textContent = "STREAMING";
                    statusBox.className = "status connected";
                    connDot.className = "connection-dot connected";
                    toggleBtn.textContent = "Stop Stream 🛑";
                    toggleBtn.className = "stop";
                    packetCount = 0;
                    packetsInWindow = 0;
                    lastRateCalc = Date.now();
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

@socketio.on('sensor_data')
def handle_sensor_data(data):
    """Receive sensor data via Socket.IO and forward via UDP"""
    try:
        if data:
            msg = json.dumps(data)
            sock.sendto(msg.encode('utf-8'), (UDP_TARGET_IP, UDP_TARGET_PORT))
    except Exception as e:
        pass  # Silently handle to avoid log spam

@socketio.on('connect')
def handle_connect():
    logger.info("📱 Mobile client connected")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("📱 Mobile client disconnected")

def generate_ssl_cert():
    """Generate a self-signed SSL certificate"""
    from OpenSSL import crypto
    
    key = crypto.PKey()
    key.generate_key(crypto.TYPE_RSA, 2048)
    
    cert = crypto.X509()
    cert.get_subject().CN = "localhost"
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365*24*60*60)
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(key)
    cert.sign(key, 'sha256')
    
    cert_dir = tempfile.mkdtemp()
    cert_path = os.path.join(cert_dir, 'server.crt')
    key_path = os.path.join(cert_dir, 'server.key')
    
    with open(cert_path, 'wb') as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    with open(key_path, 'wb') as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))
    
    return cert_path, key_path

if __name__ == '__main__':
    # Get local IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "127.0.0.1"

    print("="*50)
    print("🚀 IMU SENSOR BRIDGE")
    print("="*50)
    print(f"📱 Open on phone: https://{local_ip}:5000")
    print("="*50)
    print("Accept the certificate warning to proceed.")
    print(f"📡 Forwarding to: {UDP_TARGET_IP}:{UDP_TARGET_PORT}")
    print("="*50)
    
    # Generate SSL certificate
    try:
        cert_path, key_path = generate_ssl_cert()
        print("✅ SSL certificate ready")
        
        # Run with Flask-SocketIO's built-in server (most compatible)
        socketio.run(
            app, 
            host='0.0.0.0', 
            port=5000, 
            ssl_context=(cert_path, key_path),
            debug=False,
            log_output=False,
            allow_unsafe_werkzeug=True
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
