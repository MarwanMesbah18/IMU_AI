# 📱 Mobile Sensor Setup Guide

This guide explains how to stream Acceleroemter data from your phone to your PC for the AI project.

## ✅ Prerequisites
1.  **Wi-Fi**: Your Phone and PC must be connected to the **same Wi-Fi network**.
2.  **Dependencies**: You must have the project dependencies installed (Flask, Streamlit).

## 🚀 Quick Start

### 1. Start the Bridge Server
Open a terminal in the project folder and run:
```bash
venv/bin/python scripts/web_sensor_bridge.py
```
*Leave this terminal open.*

### 2. Connect Your Phone
1.  Look at the terminal output from Step 1. It will show a URL, for example:
    > 👉 `https://192.168.1.11:5000`
2.  Open **Chrome** or **Safari** on your phone.
3.  Type that URL into the address bar.
4.  **Security Warning**: You will see a "Not Secure" warning (because we generate a self-signed certificate).
    *   **Chrome**: Click `Advanced` -> `Proceed to ... (unsafe)`
    *   **Safari**: Click `Show Details` -> `visit this website`

### 3. Start Streaming
1.  On the phone webpage, tap the **Start Stream 🚀** button.
    *   Click "Allow" if asked for Motion Permissions.
2.  **Keep the phone screen ON** (data may stop if the screen locks).

### 4. Run the AI Dashboard
Open a **new terminal** tab/window and run:
```bash
streamlit run scripts/part8_webapp.py
```
You should see the data flowing immediately!

## ❓ Troubleshooting

| Issue | Fix |
| :--- | :--- |
| **"Site can't be reached"** | Check if Phone/PC are on different Wi-Fi networks (e.g., 2.4GHz vs 5GHz or Guest network). |
| **Data Lag / Slow** | Refresh the phone page. Ensure you are close to the router. |
| **No Data on PC** | Make sure you clicked "Start Stream" on the phone. Check if "Stop Server" button in Streamlit was clicked. |
| **Values look wrong** | The app expects m/s². Ensure your phone is sending reasonable gravity values (~9.8 on Z when flat). |
