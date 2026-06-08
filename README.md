# 🚦 Smart Traffic Violation Detector

![Python](https://img.shields.io/badge/Python-3.12-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)

## 📌 Overview
A real-time traffic violation detection system using custom trained YOLOv8 models that automatically detects traffic violations from video footage. The system detects helmet violations, red light jumping, and triple riding violations.

## 🎯 Violations Detected
- 🪖 **Helmet Violation** — Detects riders without helmets
- 🔴 **Red Light Jumping** — Detects vehicles crossing during red signal
- 🏍️ **Triple Riding** — Detects 3 or more people on a motorcycle

## 🧠 Model Performance

| Model | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| Helmet Detection | 53.7% | 34.4% | 33.9% |
| Red Light Detection | 83.6% | 78.6% | 85.6% |
| Triple Riding Detection | 91.1% | 86.8% | 92.4% |

## 🛠️ Tech Stack
- **YOLOv8** (Ultralytics) — Object Detection
- **OpenCV** — Video Processing
- **Python 3.12** — Core Language
- **Google Colab + T4 GPU** — Model Training
- **Roboflow** — Dataset Management
- **Pandas** — Violation Logging

## 📂 Project Structure
```
Smart_Traffic_Violation_Detector/
│
├── models/
│   ├── helmet_detection.pt
│   ├── traffic_light.pt
│   └── triple_riding.pt
│
├── violations/
│   ├── no_helmet.py
│   ├── red_light.py
│   └── triple_riding.py
│
├── train.ipynb
├── train_redlight.ipynb
├── train_tripleriding.ipynb
├── main.py
├── test_main.py
└── requirements.txt
```

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/shyam-260706/Smart_Traffic_Violation_Detector.git
cd Smart_Traffic_Violation_Detector
```

### 2. Install dependencies
```bash
pip install ultralytics opencv-python pandas
```

### 3. Run the detector
```bash
python main.py videos/your_video.mp4
```

## 📊 Output
- `output_violations.mp4` — Annotated video with bounding boxes
- `violations_log.csv` — Detailed log with timestamps and confidence scores

## 📈 Training Details

| Model | Dataset Size | Epochs | Platform |
|-------|-------------|--------|----------|
| Helmet | 3,276 images | 100 | Google Colab T4 GPU |
| Red Light | 9,768 images | 100 | Google Colab T4 GPU |
| Triple Riding | 1,505 images | 100 | Google Colab T4 GPU |

## 🔮 Future Enhancements
- 📩 SMS/Email notifications for violations
- 🔢 Automatic Number Plate Recognition (ANPR)
- 📱 Web dashboard for violation monitoring
- ☁️ Cloud-based violation storage

## 🧑‍💻 Use Cases
- Smart city traffic management
- Automated traffic surveillance
- Law enforcement assistance
- Traffic analytics and research
