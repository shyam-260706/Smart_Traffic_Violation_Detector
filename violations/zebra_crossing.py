"""
Zebra Crossing Violation Detection
Detects vehicles stopped on pedestrian crossing
"""

import cv2
import numpy as np

class ZebraCrossingDetector:
    def __init__(self):
        # Define zebra crossing zone (calibrate for your video)
        # Points should be in clockwise order: top-left, top-right, bottom-right, bottom-left
        self.zebra_zone = np.array([
            [200, 350],  # top-left
            [600, 350],  # top-right
            [600, 420],  # bottom-right
            [200, 420]   # bottom-left
        ], dtype=np.int32)
        
        # Track vehicles in zone
        self.vehicles_in_zone = {}
        self.stationary_threshold = 20  # frames
    
    def set_zone(self, points):
        """Manually set zebra crossing zone"""
        self.zebra_zone = np.array(points, dtype=np.int32)
    
    def check_violation(self, frame, vehicle):
        """
        Check if vehicle is stopped on zebra crossing
        
        Logic:
        1. Check if vehicle is in zebra crossing zone
        2. Track if vehicle is stationary (not moving)
        3. Violation if stationary for threshold frames
        """
        v_id = vehicle['id']
        x1, y1, x2, y2 = vehicle['bbox']
        
        # Get vehicle center point
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Check if center is in zebra crossing zone
        result = cv2.pointPolygonTest(self.zebra_zone, (cx, cy), False)
        
        if result >= 0:  # Inside zone
            # Track this vehicle
            if v_id not in self.vehicles_in_zone:
                self.vehicles_in_zone[v_id] = {
                    'frames': 0,
                    'last_pos': (cx, cy)
                }
            else:
                # Check if stationary
                last_cx, last_cy = self.vehicles_in_zone[v_id]['last_pos']
                distance_moved = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                
                if distance_moved < 5:  # Not moving much
                    self.vehicles_in_zone[v_id]['frames'] += 1
                else:
                    self.vehicles_in_zone[v_id]['frames'] = 0
                
                self.vehicles_in_zone[v_id]['last_pos'] = (cx, cy)
                
                # Check violation
                if self.vehicles_in_zone[v_id]['frames'] > self.stationary_threshold:
                    return True
        else:
            # Vehicle left zone
            if v_id in self.vehicles_in_zone:
                del self.vehicles_in_zone[v_id]
        
        return False
    
    def draw_zone(self, frame):
        """Draw zebra crossing zone for visualization"""
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.zebra_zone], (0, 255, 255))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.polylines(frame, [self.zebra_zone], True, (0, 255, 255), 2)
        cv2.putText(frame, "ZEBRA CROSSING", 
                   tuple(self.zebra_zone[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame