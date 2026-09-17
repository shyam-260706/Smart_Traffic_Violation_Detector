# Smart Traffic Violation Detector

A computer vision system that detects traffic rule violations from video footage using custom-trained YOLOv8 models. It identifies three types of violations: riders without helmets, vehicles running red lights, and triple riding on motorcycles.

## Overview

This project processes traffic video input and flags violations frame by frame. Three separate YOLOv8 models were trained for the three violation types, and the system runs them together, overlays detections on the video, and logs every violation with a timestamp and confidence score.

## Features

- Helmet violation detection (riders without helmets)
- Red light jumping detection
- Triple riding detection (3+ people on a motorcycle)
- Annotated output video with bounding boxes and live violation counts
- CSV log of all detected violations with timestamps and confidence scores

## Model Performance

| Model                   | Precision | Recall | mAP50 |
|--------------------------|-----------|--------|-------|
| Helmet Detection         | 53.7%     | 34.4%  | 33.9% |
| Red Light Detection      | 83.6%     | 78.6%  | 85.6% |
| Triple Riding Detection  | 91.1%     | 86.8%  | 92.4% |

## Tech Stack

- Python 3.12
- YOLOv8 (Ultralytics) for object detection
- OpenCV for video processing
- Pandas for violation logging
- Trained on Google Colab (T4 GPU)
- Datasets managed via Roboflow

## Project Structure
