# Driver Drowsiness & Identity Monitoring System

Real-time driver monitoring system using computer vision and deep learning.

## Features

- **Face Recognition** — Verifies driver identity using facial embeddings
- **Drowsiness Detection** — Monitors Eye Aspect Ratio (EAR) to detect microsleep
- **Yawning Detection** — Tracks Mouth Aspect Ratio (MAR) for fatigue signs
- **Phone Usage Detection** — YOLOv11-based cell phone detection
- **Real-time Alerts** — WebSocket-pushed alerts displayed on the React dashboard
- **Emergency Notification** — Triggers SMS/email alerts if driver becomes unresponsive

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend   | FastAPI, OpenCV, MediaPipe, YOLOv11 |
| Frontend  | React 18, Recharts, Vite |
| Models    | MediaPipe Face Mesh, YOLOv11 (phone), custom face embeddings |
| Transport | WebSocket (bi-directional video + telemetry) |

## Project Structure

```
driver-monitor/
├── backend/
│   ├── main.py                 # FastAPI server (REST + WebSocket)
│   ├── requirements.txt
│   ├── .env.example
│   ├── models/
│   │   ├── face_recognition.py # Face enrollment & verification
│   │   ├── drowsiness.py       # EAR-based drowsiness detection
│   │   ├── yawning.py          # MAR-based yawning detection
│   │   └── phone_usage.py      # YOLOv11 phone detection
│   └── utils/
│       ├── config.py           # Thresholds & configuration
│       ├── notifications.py    # Alert system (email/SMS)
│       └── video_processor.py  # Frame processing pipeline
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── hooks/useWebSocket.js
│       ├── components/
│       │   ├── Dashboard.jsx
│       │   ├── VideoFeed.jsx
│       │   ├── StatusPanel.jsx
│       │   ├── AlertPanel.jsx
│       │   └── MetricsChart.jsx
│       └── styles/dashboard.css
└── README.md
```

## Quick Start

### Backend

```bash
cd driver-monitor/backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt

# Register a known face (place images in known_faces/ or use API)
# Run the server:
python main.py
```

The server starts at **http://localhost:8000**.

### Frontend

```bash
cd driver-monitor/frontend
npm install
npm run dev
```

The dashboard is served at **http://localhost:3000**.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/status` | Current driver status |
| GET | `/known-faces` | List registered faces |
| POST | `/register-face/{name}` | Register a new driver face |
| GET | `/alerts?limit=20` | Recent alert history |
| WS | `/ws/video` | Real-time video + telemetry stream |
| WS | `/ws/telemetry` | Telemetry-only stream |

## Architecture

### Detection Pipeline

1. Frame captured from webcam
2. MediaPipe Face Mesh extracts 468 facial landmarks
3. EAR (Eye Aspect Ratio) calculated from eye landmarks
4. MAR (Mouth Aspect Ratio) calculated from lip landmarks
5. YOLOv11 runs object detection for cell phone (class 67)
6. Face recognition compares embeddings against known drivers
7. Alert system evaluates thresholds and triggers notifications

### Alert Thresholds

| Condition | Threshold | Consecutive Frames |
|-----------|-----------|-------------------|
| Drowsiness (EAR) | < 0.25 | 48 (~1.6s at 30fps) |
| Yawning (MAR) | > 0.60 | 30 (~1.0s) |
| Phone Usage | conf > 0.50 | 15 (~0.5s) |
| Unresponsive | — | 150 (~5.0s) |

## Emergency Notification

Configure `.env` with Twilio credentials for SMS alerts or SMTP settings for email:

- **SMS**: Activated via Twilio when driver is unresponsive for 5+ seconds
- **Email**: Gmail SMTP with app password (configurable in `.env`)

## Customization

Edit thresholds in `backend/utils/config.py`:

```python
EYE_AR_THRESHOLD = 0.25       # Lower = less sensitive to eye closure
EYE_AR_CONSEC_FRAMES = 48     # Higher = longer time before alert
YAWN_AR_THRESHOLD = 0.6       # Lower = more sensitive to yawns
PHONE_CONFIDENCE_THRESHOLD = 0.5  # YOLO confidence threshold
```
