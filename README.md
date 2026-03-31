# 🚦 Smart Traffic Violation Detector

## 📌 Overview

The Smart Traffic Violation Detector is a computer vision–based system designed to automatically detect traffic rule violations from video streams or CCTV footage.
It focuses on identifying violations such as signal jumping, helmet violations, lane violations, and vehicle detection using deep learning and image processing techniques.

This project aims to demonstrate how AI can assist traffic authorities by reducing manual monitoring, improving road safety, and enabling scalable automated surveillance systems.

## 🎯 Objectives

Detect traffic violations in real-time or recorded video

Identify vehicles and traffic signals accurately

Reduce dependency on manual traffic monitoring

Provide a foundation for future automation such as alerts and fines

## 🧠 Key Features

🚗 Vehicle Detection using deep learning models

🚦 Traffic Signal Recognition

❌ Violation Detection Logic (e.g., crossing during red signal)

🎥 Supports video input and live camera feed

📊 Logs detected violations for analysis

⚠️ Currently, the system does not use any external APIs (SMS, email, or cloud services). Detected violations are displayed and stored locally.

## 🛠️ Tech Stack

Programming Language: Python

Libraries & Tools:

OpenCV

NumPy

Deep Learning Model (YOLO / CNN-based detector)

Matplotlib (for visualization)

## ⚙️ System Architecture

Video stream is captured from CCTV / camera

Frames are processed using OpenCV

Vehicles and traffic signals are detected using ML models

Violation rules are applied

Violations are logged and displayed

## 📂 Project Structure
Smart-Traffic-Violation-Detector/
│
├── models/              # Trained ML/DL models
├── videos/              # Sample test videos
├── src/
│   ├── detection.py     # Vehicle & signal detection
│   ├── violation.py    # Violation logic
│   └── utils.py        # Helper functions
│
├── outputs/             # Detected violation outputs
├── requirements.txt
└── README.md

## 🚀 How to Run

Clone the repository

git clone https://github.com/your-username/smart-traffic-violation-detector.git


Install dependencies

pip install -r requirements.txt


Run the detector

python src/detection.py

## 🧪 Sample Violations Detected

Red light signal jump

Vehicle crossing stop line

Lane misuse (if lane marking is visible)

Helmet violation (for two-wheelers)

## 📈 Results

The system successfully detects traffic violations with good accuracy under proper lighting and camera positioning. Performance may vary based on video quality and environmental conditions.

## 🔮 Future Enhancements

📩 SMS / Email notification using APIs

☁️ Cloud-based violation storage

🔢 Automatic number plate recognition (ANPR)

📱 Web or mobile dashboard

🧾 Fine generation and reporting system

## 🧑‍💻 Use Cases

Smart city traffic management

Automated traffic surveillance

Law enforcement assistance

Traffic analytics and research

## 📜 Conclusion

This project demonstrates the practical application of computer vision and machine learning in real-world traffic systems. It serves as a strong foundation for building fully automated, scalable smart traffic monitoring solutions.