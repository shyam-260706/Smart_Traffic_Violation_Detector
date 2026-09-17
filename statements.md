# Problem Statement

## Problem Statement

Traffic rule violations — riding without a helmet, jumping red lights, and triple riding
on two-wheelers — are widespread on Indian roads and contribute significantly to road
accidents and fatalities. Manual enforcement by traffic police is limited by manpower,
cost, and the sheer number of intersections and roads that need monitoring at any
given time. Most violations go unnoticed or unrecorded simply because there aren't
enough personnel to watch every stretch of road continuously.

This project addresses that gap by using computer vision to automatically detect these
three violation types from video footage, without requiring a human to watch the
footage in real time.

## Scope

The system processes a video file (recorded or live feed converted to file input) and:
- Detects motorcycle riders not wearing a helmet
- Detects vehicles crossing an intersection while the signal is red
- Detects motorcycles carrying three or more riders

It outputs an annotated video with bounding boxes and labels on detected violations,
along with a CSV log recording each violation's type, frame number, timestamp, and
detection confidence.

The scope is limited to detection and logging — it does not include automatic number
plate recognition, fine generation, or a web dashboard (these are listed as future
enhancements). The system is designed to run on pre-recorded video files via the
command line, not as a live-monitoring deployment.

## Target Users

- Traffic police departments and smart city traffic management authorities looking to
  automate violation monitoring at intersections
- Researchers and students studying automated traffic surveillance using computer vision
- Municipal bodies evaluating low-cost ways to supplement manual traffic enforcement

## High-Level Features

- Three independently trained YOLOv8 models, one per violation type (helmet, red light,
  triple riding), run together on the same video input
- Real-time-style processing with an on-screen HUD showing live violation counts and
  signal status
- Automatic export of an annotated output video and a structured CSV violation log
- Modular detector design — each violation type is implemented as its own class, making
  it straightforward to add a new violation type in the future