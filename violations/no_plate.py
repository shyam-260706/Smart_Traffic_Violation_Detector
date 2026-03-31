import cv2
import numpy as np

class NoPlateDetector:
    def __init__(self):
        self.violation_cache = {}
        self.cache_frames = 20
    
    def check_violation(self, frame, vehicle):
        v_id = vehicle['id']
        x1, y1, x2, y2 = map(int, vehicle['bbox'])
        
        vehicle_height = y2 - y1
        plate_region_start = int(y1 + vehicle_height * 0.65)
        plate_roi = frame[plate_region_start:y2, x1:x2]
        
        if plate_roi.size == 0:
            return False
        
        has_plate = self.detect_plate_presence(plate_roi)
        
        if v_id not in self.violation_cache:
            self.violation_cache[v_id] = []
        
        self.violation_cache[v_id].append(not has_plate)
        
        if len(self.violation_cache[v_id]) > self.cache_frames:
            self.violation_cache[v_id].pop(0)
        
        if len(self.violation_cache[v_id]) >= self.cache_frames:
            no_plate_ratio = sum(self.violation_cache[v_id]) / len(self.violation_cache[v_id])
            if no_plate_ratio > 0.7:
                return True
        
        return False
    
    def detect_plate_presence(self, plate_roi):
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < 100:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            if 1.5 < aspect_ratio < 5:
                roi = gray[y:y+h, x:x+w]
                mean_brightness = np.mean(roi)
                
                if mean_brightness > 120:
                    return True
        
        return False