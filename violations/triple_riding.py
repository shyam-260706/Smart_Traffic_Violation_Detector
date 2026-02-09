"""
Triple Riding Detection
Detects more than 2 persons on motorcycle
"""

import cv2
import numpy as np

class TripleRidingDetector:
    def __init__(self):
        self.person_conf_threshold = 0.5
        self.violation_cache = {}
        self.cache_frames = 10
    
    def check_violation(self, frame, vehicle, results):
        """
        Check if more than 2 persons on motorcycle
        
        Logic:
        1. Count persons overlapping with motorcycle bbox
        2. Confirm over multiple frames to avoid false positives
        """
        # Only check motorcycles
        if vehicle['class'] != 'motorcycle':
            return False
        
        v_id = vehicle['id']
        vx1, vy1, vx2, vy2 = vehicle['bbox']
        
        # Count persons on bike
        person_count = 0
        
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        names = results.names
        
        for box, conf, cls in zip(boxes, confs, classes):
            if names[int(cls)] == 'person' and conf > self.person_conf_threshold:
                px1, py1, px2, py2 = box
                
                # Check overlap with motorcycle
                x_overlap = max(0, min(vx2, px2) - max(vx1, px1))
                y_overlap = max(0, min(vy2, py2) - max(vy1, py1))
                overlap_area = x_overlap * y_overlap
                
                person_area = (px2 - px1) * (py2 - py1)
                
                # If person significantly overlaps with bike
                if overlap_area / person_area > 0.3:
                    person_count += 1
        
        # Update cache
        if v_id not in self.violation_cache:
            self.violation_cache[v_id] = []
        
        self.violation_cache[v_id].append(person_count > 2)
        
        # Keep recent frames only
        if len(self.violation_cache[v_id]) > self.cache_frames:
            self.violation_cache[v_id].pop(0)
        
        # Confirm if violation in majority of frames
        if len(self.violation_cache[v_id]) >= self.cache_frames:
            violation_ratio = sum(self.violation_cache[v_id]) / len(self.violation_cache[v_id])
            if violation_ratio > 0.6:  # 60% confirmation
                return True
        
        return False