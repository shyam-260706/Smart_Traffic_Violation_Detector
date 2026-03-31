import cv2
import numpy as np

class RedLightDetector:
    def __init__(self):
        self.stop_line_y = 300
        self.traffic_light_roi = (50, 50, 150, 200)
        self.crossed_vehicles = set()
        self.light_state = 'green'
    
    def detect_light_state(self, frame):
        x1, y1, x2, y2 = self.traffic_light_roi
        light_roi = frame[y1:y2, x1:x2]
        
        if light_roi.size == 0:
            return self.light_state
        
        hsv = cv2.cvtColor(light_roi, cv2.COLOR_BGR2HSV)
        
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = red_mask1 + red_mask2
        
        red_pixels = np.sum(red_mask > 0)
        
        if red_pixels > 100:
            self.light_state = 'red'
        else:
            self.light_state = 'green'
        
        return self.light_state
    
    def check_violation(self, frame, vehicle, results):
        current_light = self.detect_light_state(frame)
        
        if current_light != 'red':
            return False
        
        v_id = vehicle['id']
        x1, y1, x2, y2 = vehicle['bbox']
        vehicle_front = y2
        
        if vehicle_front > self.stop_line_y:
            if v_id not in self.crossed_vehicles:
                self.crossed_vehicles.add(v_id)
                return True
        
        return False
    
    def draw_debug(self, frame):
        cv2.line(frame, (0, self.stop_line_y), 
                (frame.shape[1], self.stop_line_y), 
                (0, 0, 255), 3)
        
        x1, y1, x2, y2 = self.traffic_light_roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        color = (0, 0, 255) if self.light_state == 'red' else (0, 255, 0)
        cv2.putText(frame, f"Light: {self.light_state.upper()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return frame