# Sales Predictor ML Backend

A Scikit-Learn Random Forest regression model served via FastAPI.

## 🚀 Local Windows Setup

1. Install Dependencies:
   ```bash
   pip install fastapi uvicorn pandas scikit-learn joblib pytest apscheduler
   ```

2. Train the Initial Model:
   ```bash
   python train_model.py
   ```

3. Start the Server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

Interactive Docs available at: http://localhost:8000/docs

## 📱 Android Connection

When running the Android App, ensure it points to your machine's local IP address.

API Base URL (Current): `http://192.168.0.101:8000/`

Cleartext Traffic: Ensure your Android app allows HTTP connections during local testing.
In `AndroidManifest.xml`:

```xml
<application
    android:usesCleartextTraffic="true"
    ...>
</application>
```

You now have a fully container-ready, self-documenting, and automatically retraining machine learning backend.
