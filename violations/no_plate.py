"""
No Number Plate Detection
Detects vehicles without visible number plates
"""

import cv2
import numpy as np

class NoPlateDetector:
    def __init__(self):
        self.violation_cache = {}
        self.cache_frames = 20  # Check over 20 frames
    
    def check_violation(self, frame, vehicle):
        """
        Check if vehicle has no visible number plate
        
        Logic:
        1. Extract plate region (bottom 30% of vehicle)
        2. Look for bright rectangular regions (plates are usually white/yellow)
        3. Confirm over multiple frames
        """
        v_id = vehicle['id']
        x1, y1, x2, y2 = map(int, vehicle['bbox'])
        
        # Get plate region (bottom portion of vehicle)
        vehicle_height = y2 - y1
        plate_region_start = int(y1 + vehicle_height * 0.65)
        plate_roi = frame[plate_region_start:y2, x1:x2]
        
        if plate_roi.size == 0:
            return False
        
        # Detect if plate is visible
        has_plate = self.detect_plate_presence(plate_roi)
        
        # Update cache
        if v_id not in self.violation_cache:
            self.violation_cache[v_id] = []
        
        self.violation_cache[v_id].append(not has_plate)
        
        # Keep recent frames only
        if len(self.violation_cache[v_id]) > self.cache_frames:
            self.violation_cache[v_id].pop(0)
        
        # Confirm violation if no plate in majority of frames
        if len(self.violation_cache[v_id]) >= self.cache_frames:
            no_plate_ratio = sum(self.violation_cache[v_id]) / len(self.violation_cache[v_id])
            if no_plate_ratio > 0.7:  # 70% of frames show no plate
                return True
        
        return False
    
    def detect_plate_presence(self, plate_roi):
        """
        Detect if number plate is present in ROI
        
        Method:
        1. Convert to grayscale
        2. Look for bright regions (plates are white/yellow)
        3. Check for rectangular contours
        """
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold for bright regions
        _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Plate should have reasonable size
            if area < 100:
                continue
            
            # Check aspect ratio (plates are rectangular)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Indian plates: aspect ratio ~2:1 to 4:1
            if 1.5 < aspect_ratio < 5:
                # Check if bright enough
                roi = gray[y:y+h, x:x+w]
                mean_brightness = np.mean(roi)
                
                if mean_brightness > 120:
                    return True  # Plate detected
        
        return False  # No plate found