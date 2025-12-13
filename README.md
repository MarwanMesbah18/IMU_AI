# IMU Activity Recognition Project

Complete machine learning pipeline for human activity recognition using smartphone IMU (accelerometer) data.

## 📋 Project Overview

This project analyzes IMU sensor data to classify human activities (still, walking, shaking) using machine learning. It includes:
- **Data analysis** and quality assessment
- **Feature engineering** (time and frequency domain)
- **Model training** (Random Forest, SVM, Gradient Boosting)
- **Deployment** ready for mobile devices

## 🎯 Activities Recognized

1. **Still** - Person is stationary
2. **Walk** - Person is walking
3. **Shake** - Device is being shaken

## 📁 Project Structure

```
IMU_AI/
├── dataset/
│   └── imu_messy_data.csv          # Training data
├── IMU_Analysis_Complete.ipynb      # Main analysis notebook
├── activity_model.pkl               # Trained model (generated)
├── feature_scaler.pkl               # Feature scaler (generated)
├── feature_names.pkl                # Feature list (generated)
└── README.md                        # This file
```

## 🚀 Getting Started

### Prerequisites

**Quick Setup (Recommended):**

```bash
# Install all dependencies at once
pip install -r requirements.txt
```

**Or install manually:**
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
```

**On Windows, you can also run:**
```bash
setup.bat
```

### Running the Analysis

1. **Open the notebook:**
   ```bash
   jupyter notebook IMU_Analysis_Complete.ipynb
   ```

2. **Run all cells** (Cell → Run All)
   - This will process the data, extract features, and train models
   - Generates 3 model files: `activity_model.pkl`, `feature_scaler.pkl`, `feature_names.pkl`

3. **Check results:**
   - Model accuracy and confusion matrix will be displayed
   - Models saved automatically for deployment

## 📊 Model Performance

The notebook trains and compares 3 classifiers:
- **Random Forest** - Ensemble of decision trees
- **SVM** - Support Vector Machine with RBF kernel
- **Gradient Boosting** - Boosted decision trees

Expected accuracy: **>90%** on test set

## 📱 Testing on Mobile Device

### Option 1: Using Python on Mobile (Termux/Pydroid)

1. **Install Python app** on your phone (Pydroid 3 for Android)

2. **Copy files** to your phone:
   - `activity_model.pkl`
   - `feature_scaler.pkl`
   - `feature_names.pkl`
   - `mobile_test.py` (create using script below)

3. **Create `mobile_test.py`:**

```python
import pickle
import numpy as np
from scipy.fft import fft, fftfreq

# Load model
with open('activity_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

def extract_features(acc_x, acc_y, acc_z):
    """Extract features from 2-second window (100 samples at 50Hz)"""
    features = {}
    
    for i, (axis, data) in enumerate([('x', acc_x), ('y', acc_y), ('z', acc_z)]):
        # Time-domain
        features[f'mean_{axis}'] = np.mean(data)
        features[f'std_{axis}'] = np.std(data)
        features[f'min_{axis}'] = np.min(data)
        features[f'max_{axis}'] = np.max(data)
        features[f'range_{axis}'] = np.max(data) - np.min(data)
        features[f'sma_{axis}'] = np.mean(np.abs(data))
        features[f'energy_{axis}'] = np.mean(data**2)
        features[f'rms_{axis}'] = np.sqrt(np.mean(data**2))
        features[f'zcr_{axis}'] = np.sum(np.diff(np.sign(data)) != 0) / len(data)
        
        # Frequency-domain
        fft_vals = fft(data)
        fft_mag = np.abs(fft_vals[:len(data)//2])
        freqs = fftfreq(len(data), 1/50)[:len(data)//2]
        
        features[f'dom_freq_{axis}'] = freqs[np.argmax(fft_mag)]
        features[f'spectral_energy_{axis}'] = np.sum(fft_mag**2)
        
        # Power bands
        for low, high in [(0, 2), (2, 5), (5, 10), (10, 25)]:
            band_mask = (freqs >= low) & (freqs < high)
            features[f'power_{low}-{high}Hz_{axis}'] = np.sum(fft_mag[band_mask]**2)
    
    # Magnitude
    mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    features['mag_mean'] = np.mean(mag)
    features['mag_std'] = np.std(mag)
    
    return features

def predict_activity(acc_x, acc_y, acc_z):
    """Predict activity from sensor data"""
    import pandas as pd
    
    features = extract_features(acc_x, acc_y, acc_z)
    feature_vector = pd.DataFrame([features])
    feature_vector = feature_vector.reindex(columns=feature_names, fill_value=0)
    feature_vector_scaled = scaler.transform(feature_vector)
    
    prediction = model.predict(feature_vector_scaled)[0]
    return prediction

# Test with sample data (2 seconds = 100 samples)
if __name__ == "__main__":
    # Example: still activity (near gravity on z-axis)
    test_x = np.random.normal(0, 0.1, 100)
    test_y = np.random.normal(0, 0.1, 100)
    test_z = np.random.normal(9.81, 0.1, 100)
    
    result = predict_activity(test_x, test_y, test_z)
    print(f"Predicted activity: {result}")
```

4. **Collect real sensor data:**

```python
# Using Android sensors (e.g., with kivy or plyer)
from plyer import accelerometer

accelerometer.enable()
# Collect for 2 seconds at ~50Hz
# Process with predict_activity()
```

### Option 2: Web-based Testing

1. **Create Flask API** (`app.py`):

```python
from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open('activity_model.pkl', 'rb'))
scaler = pickle.load(open('feature_scaler.pkl', 'rb'))
feature_names = pickle.load(open('feature_names.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    acc_x = data['acc_x']  # List of 100 values
    acc_y = data['acc_y']
    acc_z = data['acc_z']
    
    # Extract features (use function from mobile_test.py)
   features = extract_features(acc_x, acc_y, acc_z)
    feature_vector = pd.DataFrame([features])
    feature_vector = feature_vector.reindex(columns=feature_names, fill_value=0)
    feature_vector_scaled = scaler.transform(feature_vector)
    
    prediction = model.predict(feature_vector_scaled)[0]
    
    return jsonify({'activity': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

2. **Run server** on your computer:
   ```bash
   python app.py
   ```

3. **Send data from phone** using JavaScript:

```javascript
// In your mobile web app
async function sendSensorData(accX, accY, accZ) {
    const response = await fetch('http://YOUR_PC_IP:5000/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            acc_x: accX,
            acc_y: accY,
            acc_z: accZ
        })
    });
    const result = await response.json();
    console.log('Activity:', result.activity);
}
```

### Option 3: TensorFlow Lite (Android)

1. **Convert model to TFLite:**

```python
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

# Note: Requires converting sklearn to TF first
# Or use tf.keras model instead
```

2. **Deploy to Android app** using TensorFlow Lite Android library

## 📈 Data Collection from Mobile

To collect your own IMU data:

```python
# Example using Android
from kivy.app import App
from plyer import accelerometer
import time

data = {'x': [], 'y': [], 'z': [], 'time': []}

def on_acceleration(x, y, z):
    data['x'].append(x)
    data['y'].append(y)
    data['z'].append(z)
    data['time'].append(time.time())

accelerometer.enable()
# Collect for desired duration
# Save to CSV format matching training data
```

## 🔬 Feature Engineering

**Time-Domain (33 features):**
- Mean, Std, Min, Max, Range per axis
- SMA, Energy, RMS, Zero-crossing rate per axis
- Magnitude mean and std

**Frequency-Domain:**
- Dominant frequency per axis
- Spectral energy and entropy per axis
- Power in frequency bands (0-2Hz, 2-5Hz, 5-10Hz, 10-25Hz) per axis

**Total: ~60 features**

## 📝 Notes

- **Sampling Rate**: 50 Hz (50 samples per second)
- **Window Size**: 2 seconds (100 samples)
- **Window Overlap**: 50% for training

## 🛠️ Troubleshooting

**Model not loading on mobile:**
- Ensure scikit-learn version matches on both devices
- Try using `protocol=4` when saving pickle files

**Low accuracy on real data:**
- Check sensor sampling rate matches training (50Hz)
- Ensure phone orientation is similar to training data
- Collect more calibration data

**Import errors:**
- Install missing packages: `pip install package-name`

## 📚 References

- Assignment PDF: `dataset/imu_assignment.pdf`
- Training data: `dataset/imu_messy_data.csv`

## 👤 Author

Created for Mathematical Foundations for AI course - IMU Data Processing Assignment

## 📄 License

Educational project for academic purposes.
