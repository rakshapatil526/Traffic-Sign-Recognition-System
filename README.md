# 🚦 Traffic Sign Recognition System (TSR ADAS)

An intelligent, multi-module computer vision and deep learning system for automated road sign detection, classification, driving compliance advisory, and road survey analytics.

Built with **Python**, **Streamlit**, **OpenCV**, **TensorFlow / Keras CNN**, and **YOLO candidate localization**.

---

## 🌟 Key Highlights & Technologies

- **Python 3.10+**: Core programming language.
- **Streamlit**: Responsive, modern Dark Navy-Blue road-themed dashboard UI.
- **OpenCV**: Computer vision engine for image preprocessing, color-space filtering, morphological operations, polygon vertex approximation, and ADAS HUD annotation.
- **TensorFlow & Keras**: Deep Convolutional Neural Network (CNN) multi-class classifier for 43 GTSRB international traffic sign categories.
- **YOLO Spatial Localization**: Candidate region-of-interest (RoI) proposals, multiscale bounding box extraction, and Non-Maximum Suppression (NMS).
- **SQLite Database**: Persistent audit logging for detection records, search, multi-parameter filtering, and CSV/JSON export.
- **Plotly & Matplotlib**: Interactive data analytics, category breakdowns, confidence distributions, and trend charts.

---

## 🏗️ System Architecture & 5 Modules

```
Traffic Sign Recognition System
├── Module 1: Home / Dashboard
│   ├── KPI metric cards (Total Images, Total Signs, Avg Confidence, Feeds)
│   ├── Recently detected traffic signs carousel
│   ├── Quick access action buttons
│   └── 5 Core categories overview (Speed Limit, Stop & Yield, No Entry, Turn, Warning)
│
├── Module 2: Image Upload & Detection
│   ├── JPG, JPEG, PNG upload with drag-and-drop
│   ├── 6 Preloaded road test scenes (Highways, Intersections, One-Way, Roundabouts)
│   ├── OpenCV RoI + Deep CNN classification + NMS
│   └── Side-by-side original vs detected HUD bounding box comparison
│
├── Module 3: Video & Live Camera Detection
│   ├── Video file upload (.mp4, .avi, .mov)
│   ├── Preloaded 120-frame simulated highway drive video
│   ├── Live camera streaming & snapshot analysis
│   └── Frame-by-frame HUD bounding boxes, live FPS telemetry & rolling logs
│
├── Module 4: Prediction Results & Sign Information
│   ├── Deep diagnostic inspector for detected traffic signs
│   ├── Sign meaning, safe driving instructions, and legal penalty risks
│   └── Full GTSRB 43-Class Traffic Sign Encyclopedia
│
└── Module 5: Detection History & Analytics
    ├── Search & multi-parameter filter (Category, Confidence, Source, Date)
    ├── Plotly Donut, Bar, and Histogram analytical charts
    ├── Detailed audit history table
    └── 1-Click CSV and JSON export + Clear history
```

---

## 📁 Directory Structure

```
Traffic Sign Recognition System/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Dependency specification
├── run_app.bat                 # 1-Click Windows execution script
├── README.md                   # System documentation
├── src/
│   ├── __init__.py
│   ├── config.py               # 43 GTSRB classes, categories, driving rules
│   ├── database.py             # SQLite persistence, queries, KPI metrics
│   ├── detector.py             # YOLO/OpenCV RoI detection, NMS, HUD rendering
│   ├── model_cnn.py            # Keras CNN architecture & inference engine
│   ├── sample_data_generator.py# Realistic road scene & dashcam video generator
│   └── utils.py                # Visual styling, base64 helpers, Plotly charts
├── models/
│   └── traffic_sign_cnn.keras  # Trained TensorFlow CNN classifier
└── data/
    ├── traffic_signs.db        # SQLite database
    ├── samples/                # Preloaded test road images & sample dashcam video
    └── uploads/                # User uploaded media
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your computer.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
Run via Streamlit:
```bash
streamlit run app.py
```
Or on Windows, simply double-click `run_app.bat`.

The application will launch in your default web browser at `http://localhost:8501`.

---

## 🚸 Supported Traffic Sign Categories

1. **Speed Limit Signs** (20, 30, 50, 60, 70, 80, 100, 120 km/h)
2. **Stop & Yield Signs** (Stop Octagon, Yield Inverted Triangle, Priority Road)
3. **No Entry & Prohibitory** (No Entry, No Passing, Road Closed, Truck Restrictions)
4. **Turn & Mandatory Signs** (Turn Right Ahead, Turn Left Ahead, Ahead Only, Keep Right, Roundabout Mandatory)
5. **Danger & Warning Signs** (General Caution, Sharp Curves, Pedestrian Crossing, Road Work, Traffic Signals)

---

## 🛡️ License & Acknowledgements
Developed with open-source technologies adhering to GTSRB international road signage standards for ADAS and Computer Vision research.
