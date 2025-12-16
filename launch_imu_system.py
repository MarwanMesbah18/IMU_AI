#!/usr/bin/env python3
"""
IMU Activity Recognition - Project Launcher
Starts both the web sensor bridge (for mobile) and Streamlit webapp.
"""

import subprocess
import socket
import sys
import time
import os
from pathlib import Path

def get_local_ip():
    """Get the local IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def print_banner():
    """Print a nice banner with project info."""
    print("\n" + "="*70)
    print("  🚀 IMU Activity Recognition System")
    print("="*70)

def print_connection_info(ip):
    """Print connection instructions."""
    print("\n📡 CONNECTION INFORMATION:")
    print("-" * 70)
    print(f"\n🖥️  PC (Streamlit Webapp):")
    print(f"   └─ Local:  http://localhost:8501")
    print(f"   └─ Network: http://{ip}:8501")
    
    print(f"\n📱 Mobile (Sensor Bridge):")
    print(f"   └─ Open on phone: https://{ip}:5000")
    print(f"   └─ Accept SSL certificate warning")
    print(f"   └─ Click 'Start Stream' to send data")
    
    print("\n" + "-" * 70)
    print("⚠️  IMPORTANT: Ensure phone & PC are on the SAME Wi-Fi network")
    print("-" * 70 + "\n")

def check_venv():
    """Check if running in virtual environment."""
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  WARNING: Not running in virtual environment!")
        print("   It's recommended to activate venv first:")
        print("   $ source venv/bin/activate")
        print()
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)

def main():
    """Main launcher function."""
    # Get project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print_banner()
    check_venv()
    
    # Get IP address
    local_ip = get_local_ip()
    
    print("\n🔧 Starting services...\n")
    
    try:
        # Start web sensor bridge (mobile interface)
        print("📱 Launching Mobile Sensor Bridge...")
        bridge_process = subprocess.Popen(
            [sys.executable, "scripts/web_sensor_bridge.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(2)  # Give it time to start
        
        # Start Streamlit webapp
        print("🖥️  Launching Streamlit Webapp...")
        streamlit_process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "scripts/part8_webapp.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(3)  # Give Streamlit time to start
        
        # Print connection info
        print_connection_info(local_ip)
        
        print("✅ Both services are running!")
        print("\n💡 TIP: Use Ctrl+C to stop all services\n")
        
        # Keep running and handle shutdown
        try:
            bridge_process.wait()
            streamlit_process.wait()
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping services...")
            bridge_process.terminate()
            streamlit_process.terminate()
            
            # Wait for graceful shutdown
            time.sleep(2)
            
            # Force kill if still running
            try:
                bridge_process.kill()
                streamlit_process.kill()
            except:
                pass
            
            print("✅ All services stopped. Goodbye!\n")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you're in the project directory")
        print("  2. Activate virtual environment: source venv/bin/activate")
        print("  3. Install dependencies: pip install -r requirements.txt")
        print("  4. Check if ports 5000 and 8501 are available")
        sys.exit(1)

if __name__ == "__main__":
    main()
