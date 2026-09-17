#Smart Traffic Violation Detector

A computer vision system for detecting traffic rule violations from video footage using YOLOv8 models trained for custom purposes. It identifies three types of violations: riding without helmets, vehicles running red lights and triple riding on motorcycles.

##Overview

The project accepts traffic video as input and detects violations in each frame. Three different YOLOv8 models were trained for the three violation types. The system runs them simultaneously and overlays the detections to the video and logs every violation with a timestamp and confidence score.

##Feature

- Helmet Violation Detection (no helmet riders)- Red light violation detection
- Triple riding detection (3 or more people on a motorcycle)- Annotated output video with bounding boxes and real-time counts of violations
- CSV log of all detected violations with timestamps and confidence scores .


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


