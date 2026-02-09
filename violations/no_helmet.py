"""
Helmet Detection for Motorcycles
Detects riders without helmets
"""

import cv2
import numpy as np

class HelmetDetector:
    def __init__(self):
        # Minimum person confidence
        self.person_conf_threshold = 0.5
        
        # Track violations
        self.violation_cache = {}
        self.cache_frames = 15  # Confirm over 15 frames
    
    def check_violation(self, frame, vehicle, results):
        """
        Check if motorcycle rider is without helmet
        
        Logic:
        1. Find persons on motorcycle
        2. Check head region for helmet (dark object detection)
        3. Confirm over multiple frames
        """
        # Only check motorcycles
        if vehicle['class'] != 'motorcycle':
            return False
        
        v_id = vehicle['id']
        x1, y1, x2, y2 = map(int, vehicle['bbox'])
        
        # Find persons on this motorcycle
        persons = self.get_persons_on_vehicle(vehicle, results)
        
        if len(persons) == 0:
            return False
        
        # Check each person for helmet
        no_helmet_count = 0
        
        for person_bbox in persons:
            has_helmet = self.detect_helmet(frame, person_bbox, vehicle['bbox'])
            if not has_helmet:
                no_helmet_count += 1
        
        # Update cache
        if v_id not in self.violation_cache:
            self.violation_cache[v_id] = []
        
        self.violation_cache[v_id].append(no_helmet_count > 0)
        
        # Keep only recent frames
        if len(self.violation_cache[v_id]) > self.cache_frames:
            self.violation_cache[v_id].pop(0)
        
        # Confirm violation if detected in majority of frames
        if len(self.violation_cache[v_id]) >= self.cache_frames:
            violation_ratio = sum(self.violation_cache[v_id]) / len(self.violation_cache[v_id])
            if violation_ratio > 0.6:  # 60% of frames
                return True
        
        return False
    
    def get_persons_on_vehicle(self, vehicle, results):
        """Find all persons on this vehicle"""
        vx1, vy1, vx2, vy2 = vehicle['bbox']
        
        persons = []
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        names = results.names
        
        for box, conf, cls in zip(boxes, confs, classes):
            if names[int(cls)] == 'person' and conf > self.person_conf_threshold:
                px1, py1, px2, py2 = box
                
                # Check if person overlaps with vehicle
                # Calculate IoU (Intersection over Union)
                x_overlap = max(0, min(vx2, px2) - max(vx1, px1))
                y_overlap = max(0, min(vy2, py2) - max(vy1, py1))
                overlap_area = x_overlap * y_overlap
                
                person_area = (px2 - px1) * (py2 - py1)
                
                if overlap_area / person_area > 0.3:  # 30% overlap
                    persons.append(box)
        
        return persons
    
    def detect_helmet(self, frame, person_bbox, vehicle_bbox):
        """
        Detect if person has helmet
        
        Method: Check head region for dark/colored object
        Better method: Train custom YOLO model for helmet detection
        """
        px1, py1, px2, py2 = map(int, person_bbox)
        
        # Get head region (top 25% of person bbox)
        head_height = int((py2 - py1) * 0.25)
        head_roi = frame[py1:py1+head_height, px1:px2]
        
        if head_roi.size == 0:
            return True  # Assume helmet if can't detect
        
        # Convert to HSV
        hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
        
        # Helmets are usually:
        # 1. Dark colored (black, dark blue, etc.)
        # 2. Bright colored (red, yellow, white)
        # 3. Have consistent color in head region
        
        # Dark helmet detection
        dark_lower = np.array([0, 0, 0])
        dark_upper = np.array([180, 255, 80])
        dark_mask = cv2.inRange(hsv, dark_lower, dark_upper)
        dark_ratio = np.sum(dark_mask > 0) / dark_mask.size
        
        # Bright helmet detection
        bright_lower = np.array([0, 0, 150])
        bright_upper = np.array([180, 100, 255])
        bright_mask = cv2.inRange(hsv, bright_lower, bright_upper)
        bright_ratio = np.sum(bright_mask > 0) / bright_mask.size
        
        # If significant dark or bright area in head = helmet
        if dark_ratio > 0.3 or bright_ratio > 0.3:
            return True
        
        # No helmet detected
        return False