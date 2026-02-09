"""
Smart Traffic Violation Detector - Fixed for headless mode
"""

import cv2
from ultralytics import YOLO

# Import utils
from utils.video_reader import VideoReader
from utils.tracker import Tracker
from utils.plate_recognition import PlateRecognizer
from utils.notification_system import NotificationSystem

import easyocr
import pandas as pd
import os
from datetime import datetime

# Import violation detectors
from violations.red_light import RedLightDetector
from violations.no_helmet import HelmetDetector
from violations.triple_riding import TripleRidingDetector
from violations.zebra_crossing import ZebraCrossingDetector
from violations.no_plate import NoPlateDetector
from violations.one_way import OneWayDetector


# Simple PlateRecognizer (inline)
class PlateRecognizer:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.db = self.load_db()
    
    def load_db(self):
        if os.path.exists('vehicle_database.csv'):
            return pd.read_csv('vehicle_database.csv')
        else:
            df = pd.DataFrame({
                'plate_number': ['MH12AB1234', 'DL01CD5678'],
                'owner_name': ['Rajesh Kumar', 'Priya Sharma'],
                'phone_number': ['+919876543210', '+919123456780']
            })
            df.to_csv('vehicle_database.csv', index=False)
            return df
    
    def read_plate(self, img):
        if img is None or img.size == 0:
            return None
        try:
            results = self.reader.readtext(img)
            if results:
                text = ''.join([r[1] for r in results])
                import re
                text = re.sub(r'[^A-Z0-9]', '', text.upper())
                if 8 <= len(text) <= 10:
                    return text
        except:
            pass
        return None
    
    def get_owner(self, plate):
        match = self.db[self.db['plate_number'] == plate]
        if not match.empty:
            return match.iloc[0].to_dict()
        return None


# Simple NotificationSystem (inline)
class NotificationSystem:
    def __init__(self):
        self.sent = set()
        self.plate_reader = PlateRecognizer()
    
    def send_alert(self, plate, violation_type, frame):
        key = f"{plate}_{violation_type}"
        if key in self.sent:
            return
        
        owner = self.plate_reader.get_owner(plate)
        if not owner:
            print(f"⚠️ No owner for {plate}")
            return
        
        msg = f"VIOLATION: {violation_type} | Plate: {plate} | Owner: {owner['owner_name']}"
        print(f"📱 SMS to {owner['phone_number']}: {msg}")
        
        os.makedirs('evidence', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cv2.imwrite(f"evidence/{plate}_{violation_type}_{timestamp}.jpg", frame)
        
        self.sent.add(key)


class TrafficSystem:
    def __init__(self, video_path='videos/traffic.mp4'):
        print("🚀 Initializing Traffic Violation System...")
        
        # Create directories
        os.makedirs('evidence', exist_ok=True)
        os.makedirs('snapshots', exist_ok=True)
        
        self.model = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded")
        
        self.video = VideoReader(video_path)
        self.tracker = Tracker()
        self.plate_reader = PlateRecognizer()
        self.notifier = NotificationSystem()
        
        self.red_light_detector = RedLightDetector()
        self.helmet_detector = HelmetDetector()
        self.triple_riding_detector = TripleRidingDetector()
        self.zebra_detector = ZebraCrossingDetector()
        self.no_plate_detector = NoPlateDetector()
        self.one_way_detector = OneWayDetector(allowed_direction='down')
        
        print("✅ All detectors initialized")
        self.violations_log = []
        
    def run(self):
        print("▶️ Starting video processing...")
        print("💡 Running in headless mode (no display window)")
        frame_count = 0
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_violations.mp4', fourcc, 20.0, 
                              (self.video.width, self.video.height))
        print("📹 Output will be saved to: output_violations.mp4")
        
        while True:
            ret, frame = self.video.read()
            if not ret:
                print("📹 Video ended")
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}...")
            
            results = self.model(frame)[0]
            tracked = self.tracker.update(results)
            
            all_violations = []
            
            for vehicle in tracked:
                violations = []
                v_class = vehicle['class']
                v_id = vehicle['id']
                
                if self.red_light_detector.check_violation(frame, vehicle, results):
                    violations.append('red_light')
                
                if v_class == 'motorcycle':
                    if self.helmet_detector.check_violation(frame, vehicle, results):
                        violations.append('no_helmet')
                    if self.triple_riding_detector.check_violation(frame, vehicle, results):
                        violations.append('triple_riding')
                
                if self.zebra_detector.check_violation(frame, vehicle):
                    violations.append('zebra_crossing')
                
                if self.one_way_detector.check_violation(vehicle):
                    violations.append('one_way')
                
                if self.no_plate_detector.check_violation(frame, vehicle):
                    violations.append('no_plate')
                
                if violations:
                    all_violations.append({'vehicle': vehicle, 'types': violations})
                    plate = self.get_plate(frame, vehicle['bbox'])
                    
                    if plate:
                        for v_type in violations:
                            self.notifier.send_alert(plate, v_type, frame)
                            self.violations_log.append({
                                'plate': plate,
                                'type': v_type,
                                'frame': frame_count,
                                'vehicle_id': v_id
                            })
                            print(f"⚠️ VIOLATION: {v_type} - Plate: {plate}")
            
            # Draw annotations
            display_frame = self.draw_detections(frame, tracked, all_violations)
            display_frame = self.red_light_detector.draw_debug(display_frame)
            display_frame = self.zebra_detector.draw_zone(display_frame)
            display_frame = self.one_way_detector.draw_arrow(display_frame)
            
            # Write to output video
            out.write(display_frame)
            
            # Save snapshots every 100 frames
            if frame_count % 100 == 0:
                cv2.imwrite(f'snapshots/frame_{frame_count}.jpg', display_frame)
                print(f"📸 Saved snapshot at frame {frame_count}")
        
        # Cleanup
        out.release()
        self.save_log()
        self.video.release()
        print("✅ Processing complete!")
        print(f"📹 Output video: output_violations.mp4")
        print(f"📊 Violations log: violations_log.csv")
    
    def get_plate(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        vehicle_roi = frame[y1:y2, x1:x2]
        roi_h, roi_w = vehicle_roi.shape[:2]
        
        if roi_h == 0 or roi_w == 0:
            return None
        
        plate_roi = vehicle_roi[int(roi_h*0.6):roi_h, :]
        return self.plate_reader.read_plate(plate_roi)
    
    def draw_detections(self, frame, tracked, violations):
        display = frame.copy()
        violation_dict = {}
        for v in violations:
            violation_dict[v['vehicle']['id']] = v['types']
        
        for vehicle in tracked:
            v_id = vehicle['id']
            x1, y1, x2, y2 = map(int, vehicle['bbox'])
            has_violation = v_id in violation_dict
            color = (0, 0, 255) if has_violation else (0, 255, 0)
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, f"ID:{v_id} ({vehicle['class']})", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if has_violation:
                violation_text = ", ".join(violation_dict[v_id])
                cv2.putText(display, f"⚠️ {violation_text}", 
                           (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.putText(display, f"Tracked: {len(tracked)} | Violations: {len(violations)}", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display
    
    def save_log(self):
        if len(self.violations_log) == 0:
            print("ℹ️ No violations detected")
            return
        
        df = pd.DataFrame(self.violations_log)
        df.to_csv('violations_log.csv', index=False)
        print(f"💾 Saved {len(self.violations_log)} violations")
        print("\n📊 SUMMARY:")
        print(df['type'].value_counts())


if __name__ == "__main__":
    import sys
    
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'videos/traffic.mp4'
    
    print("="*60)
    print("🚦 SMART TRAFFIC VIOLATION DETECTOR")
    print("="*60)
    
    try:
        system = TrafficSystem(video_path)
        system.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()