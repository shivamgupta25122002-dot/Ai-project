# Object Detection System in Python 👁️

A production-ready, modular **Object Detection & Analytics System** built in Python. Features multi-backend model engine support (Ultralytics YOLOv8/v10/v11, OpenCV DNN ONNX, OpenCV Fallback), real-time object centroid tracking, dynamic HUD visualization, JSON/CSV exports, CLI tool, and an interactive **Flask Web Dashboard**.

---

## 🌟 Key Features

- **Multi-Backend Detection Core**:
  - **Ultralytics YOLOv8/v10/v11** (80 COCO object classes: person, vehicle, animals, electronics, furniture, etc.).
  - **OpenCV DNN & ONNX** acceleration.
  - **Fail-safe Fallback Detector** to guarantee detection capability on any system.
- **Real-Time Object Tracking**:
  - Centroid tracking algorithm with persistent Object IDs (`#1`, `#2`).
  - Movement trajectory trail rendering.
- **Interactive Web Dashboard**:
  - **Live Webcam Stream**: Real-time video detection directly in web browser (`/video_feed`).
  - **Photo Studio**: Drag & drop image analysis with confidence sliders and class filters.
  - **Analytics Center**: Chart.js doughnut chart visualizing object class distribution and detection metrics.
- **Command-Line Interface (CLI)**:
  - Process images, videos, or webcam streams directly from terminal.
  - Export analytics to **JSON** and **CSV** logs.

---

## 📁 System Architecture

```
minor/
├── app.py                  # Flask Web Dashboard server & REST API endpoints
├── main.py                 # Primary entry point launcher
├── cli.py                  # Command-line interface parser
├── requirements.txt        # Python dependencies
├── README.md               # Complete setup & usage guide
├── models/                 # Model files & auto-downloader
│   └── download_models.py # Model weights downloader script
├── core/                   # Core Python logic modules
│   ├── __init__.py
│   ├── detector.py         # Multi-backend Object Detection Engine
│   ├── tracker.py          # Centroid Object Tracker
│   ├── visualizer.py       # HUD overlay & bounding box drawer
│   └── utils.py            # FPS counter, JSON/CSV exporter, Base64 converter
├── static/                 # Web assets (CSS & JS)
│   ├── css/
│   │   └── style.css       # Premium Dark Theme & Glassmorphism styles
│   └── js/
│       └── main.js         # Interactive Web UI & Chart.js controller
├── templates/              # HTML Templates
│   └── index.html          # Web Dashboard UI
└── uploads/                # Output directory for processed media
```

---

## 🚀 Quick Start & Setup

### 1. Install Dependencies
Run the following command to install required Python packages:

```bash
py -m pip install -r requirements.txt
```

*(Optionally install Ultralytics YOLO for 80-class SOTA detection)*:
```bash
py -m pip install ultralytics
```

### 2. Download Pre-trained Models
Download pre-trained model weights automatically:

```bash
py models/download_models.py
```

---

## 💻 Usage Guide

### Launch Interactive Web Dashboard
Run the web application server:

```bash
py main.py --web
```
Open **`http://127.0.0.1:5000`** in your browser to access the Web UI.

---

### Command-Line Interface (CLI)

#### 1. Synthetic Quick Test
```bash
py main.py
```

#### 2. Detect Objects in an Image
```bash
py main.py --source input.jpg --output result.jpg --conf 0.4
```

#### 3. Detect Specific Object Classes
Detect only `person` and `car`:
```bash
py main.py --source photo.jpg --classes person,car --conf 0.35
```

#### 4. Process Video File & Export JSON/CSV Logs
```bash
py main.py --source video.mp4 --output annotated_video.mp4 --save-json results.json --save-csv results.csv
```

#### 5. Live Webcam Detection
```bash
py main.py --source 0
```

---

## 📡 REST API Documentation

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /` | `GET` | Renders Web Dashboard UI |
| `POST /api/detect-image` | `POST` | Upload image file (`image`), returns base64 annotated image & detection JSON |
| `GET /api/detect-sample` | `GET` | Runs instant test detection on synthetic test image |
| `GET /video_feed` | `GET` | Multipart video stream endpoint for live webcam feed |
| `POST /api/config` | `POST` | Updates confidence threshold dynamically |

---

## 📊 Sample Output Schema (JSON)

```json
{
  "timestamp": "2026-09-29T18:09:54.258734",
  "total_objects": 2,
  "fps": 30.0,
  "processing_time_ms": 24.5,
  "detections": [
    {
      "label": "car",
      "confidence": 0.92,
      "bbox": [100, 200, 320, 360],
      "object_id": 1
    },
    {
      "label": "person",
      "confidence": 0.88,
      "bbox": [455, 205, 505, 330],
      "object_id": 2
    }
  ]
}
```

---

## ⚙️ License & Author
Built for educational, research, and industrial computer vision applications.
