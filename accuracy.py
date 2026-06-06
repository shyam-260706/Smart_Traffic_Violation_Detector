from ultralytics import YOLO
import cv2
import pandas as pd

helmet_model = YOLO('models/helmet_best.pt')
redlight_model = YOLO('models/red_light_best.pt')

print("=" * 40)
print("HELMET MODEL INFO")
print("=" * 40)
print("Classes:", helmet_model.names)

print("\n" + "=" * 40)
print("RED LIGHT MODEL INFO")
print("=" * 40)
print("Classes:", redlight_model.names)

# Test on both videos and collect confidence scores
videos = [
    ("videos/Bangalore police officer fined for traffic violation.mp4", "HELMET"),
    ("videos/Follow These Traffic Light Rules to Avoid Accidents 🚦.mp4", "REDLIGHT")
]

for video_path, vtype in videos:
    print(f"\n{'='*40}")
    print(f"Testing {vtype} on: {video_path}")
    print("="*40)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    all_confs = []
    detection_frames = 0
    frame_count = 0

    model = helmet_model if vtype == "HELMET" else redlight_model

    while frame_count < min(100, total_frames):  # test first 100 frames
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        results = model(frame, conf=0.25, verbose=False)[0]
        if len(results.boxes) > 0:
            detection_frames += 1
            for box in results.boxes:
                all_confs.append(float(box.conf[0]))

    cap.release()

    if all_confs:
        print(f"Frames tested:         {frame_count}")
        print(f"Frames with detections:{detection_frames}")
        print(f"Detection rate:        {detection_frames/frame_count*100:.1f}%")
        print(f"Avg confidence:        {sum(all_confs)/len(all_confs):.3f}")
        print(f"Max confidence:        {max(all_confs):.3f}")
        print(f"Min confidence:        {min(all_confs):.3f}")
        print(f"Total detections:      {len(all_confs)}")
    else:
        print("No detections found in first 100 frames!")