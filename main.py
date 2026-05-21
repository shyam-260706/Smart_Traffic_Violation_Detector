import cv2
from ultralytics import YOLO
import pandas as pd
import os
from datetime import datetime


class TrafficSystem:
    def __init__(self, video_path='videos/traffic.mp4'):
        print("Initializing Traffic Violation System...")

        self.helmet_model = YOLO('models/helmet_best.pt')
        self.redlight_model = YOLO('models/red_light_best.pt')
        print("Both models loaded!")

        self.cap = cv2.VideoCapture(video_path)
        self.violations_log = []
        print("Ready!")

    def run(self):
        print("Starting video processing...")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('output_violations.mp4', fourcc, 20.0, (width, height))
        print("Output will be saved to: output_violations.mp4")

        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Video ended")
                break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}...")

            # Run both models
            helmet_results = self.helmet_model(frame)[0]
            redlight_results = self.redlight_model(frame)[0]

            violations = []

            # Helmet detections
            for box in helmet_results.boxes:
                cls_id = int(box.cls[0])
                label  = self.helmet_model.names[cls_id]
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                violation = None
                if label in ["No-Helmet", "Invalid"]:
                    violation = "no_helmet"

                color = (0, 0, 255) if violation else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if violation:
                    violations.append(violation)
                    self.violations_log.append({
                        'type': violation,
                        'frame': frame_count,
                        'confidence': round(conf, 2),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    print(f"VIOLATION: {violation} | Conf: {conf:.2f} | Frame: {frame_count}")

            # Red light detections
            for box in redlight_results.boxes:
                cls_id = int(box.cls[0])
                label  = self.redlight_model.names[cls_id]
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                violation = None
                if label == "red":
                    violation = "red_light"

                color = (0, 0, 255) if violation else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if violation:
                    violations.append(violation)
                    self.violations_log.append({
                        'type': violation,
                        'frame': frame_count,
                        'confidence': round(conf, 2),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    print(f"VIOLATION: {violation} | Conf: {conf:.2f} | Frame: {frame_count}")

            # HUD
            cv2.putText(frame, f"Violations: {len(violations)}",
                        (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            out.write(frame)

        out.release()
        self.cap.release()
        self.save_log()
        print("Done! Output: output_violations.mp4")

    def save_log(self):
        if not self.violations_log:
            print("No violations detected")
            return
        df = pd.DataFrame(self.violations_log)
        df.to_csv('violations_log.csv', index=False)
        print(f"\nSUMMARY:")
        print(df['type'].value_counts())


if __name__ == "__main__":
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'videos/traffic.mp4'
    print("=" * 60)
    print("SMART TRAFFIC VIOLATION DETECTOR")
    print("=" * 60)
    try:
        system = TrafficSystem(video_path)
        system.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
