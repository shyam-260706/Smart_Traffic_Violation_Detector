import cv2
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
from violations.no_helmet import HelmetDetector
from violations.red_light import RedLightDetector
from violations.triple_riding import TripleRidingDetector


class TrafficSystem:
    def __init__(self, video_path='videos/sample_video2.mp4'):
        print("Initializing Traffic Violation System...")

        helmet_model = YOLO('models/helmet_detection.pt')
        redlight_model = YOLO('models/traffic_light.pt')
        triple_model = YOLO('models/triple_riding.pt')

        self.helmet_detector   = HelmetDetector(helmet_model)
        self.redlight_detector = RedLightDetector(redlight_model)
        self.triple_detector   = TripleRidingDetector(triple_model)

        self.cap = cv2.VideoCapture(video_path)
        self.violations_log = []
        print("All detectors ready!")

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

            # Run all detectors
            helmet_violations            = self.helmet_detector.detect(frame)
            redlight_violations, red_active = self.redlight_detector.detect(frame)
            triple_violations            = self.triple_detector.detect(frame)

            all_violations = helmet_violations + redlight_violations + triple_violations

            # Red light status
            status_color = (0, 0, 255) if red_active else (0, 255, 0)
            status_text  = "RED LIGHT ACTIVE" if red_active else "SIGNAL OK"
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # HUD
            cv2.putText(frame,
                        f"Helmet:{self.helmet_detector.get_count()} | "
                        f"RedLight:{self.redlight_detector.get_count()} | "
                        f"Triple:{self.triple_detector.get_count()}",
                        (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Log violations
            for v in all_violations:
                self.violations_log.append({
                    'type': v['type'],
                    'frame': frame_count,
                    'confidence': round(v['conf'], 2),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                print(f"VIOLATION: {v['type']} | Conf: {v['conf']:.2f} | Frame: {frame_count}")

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
        print(f"\nTotal violations: {len(self.violations_log)}")


if __name__ == "__main__":
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'videos/sample_video2.mp4'
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