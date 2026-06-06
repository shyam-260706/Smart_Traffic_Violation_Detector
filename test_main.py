from ultralytics import YOLO
import cv2

helmet_model = YOLO('models/helmet_best.pt')
redlight_model = YOLO('models/red_light_best.pt')

print("Helmet model classes:", helmet_model.names)
print("Redlight model classes:", redlight_model.names)

# Test on first frame of video
cap = cv2.VideoCapture('videos/Bangalore police officer fined for traffic violation.mp4')
ret, frame = cap.read()
cap.release()

if ret:
    print("\n--- Helmet Model Results ---")
    results = helmet_model(frame, conf=0.1)[0]  # low conf to see anything
    print(f"Detections found: {len(results.boxes)}")
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = helmet_model.names[cls_id]
        conf = float(box.conf[0])
        print(f"  {label}: {conf:.2f}")

    print("\n--- RedLight Model Results ---")
    results = redlight_model(frame, conf=0.1)[0]
    print(f"Detections found: {len(results.boxes)}")
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = redlight_model.names[cls_id]
        conf = float(box.conf[0])
        print(f"  {label}: {conf:.2f}")
else:
    print("ERROR: Could not read video file!")