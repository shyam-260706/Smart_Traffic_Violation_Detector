import cv2
import numpy as np

class ZebraCrossingDetector:
    def __init__(self):
        self.zebra_zone = np.array([
            [200, 350],
            [600, 350],
            [600, 420],
            [200, 420]
        ], dtype=np.int32)
        
        self.vehicles_in_zone = {}
        self.stationary_threshold = 20
    
    def set_zone(self, points):
        self.zebra_zone = np.array(points, dtype=np.int32)
    
    def check_violation(self, frame, vehicle):
        v_id = vehicle['id']
        x1, y1, x2, y2 = vehicle['bbox']
        
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        result = cv2.pointPolygonTest(self.zebra_zone, (cx, cy), False)
        
        if result >= 0:
            if v_id not in self.vehicles_in_zone:
                self.vehicles_in_zone[v_id] = {
                    'frames': 0,
                    'last_pos': (cx, cy)
                }
            else:
                last_cx, last_cy = self.vehicles_in_zone[v_id]['last_pos']
                distance_moved = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                
                if distance_moved < 5:
                    self.vehicles_in_zone[v_id]['frames'] += 1
                else:
                    self.vehicles_in_zone[v_id]['frames'] = 0
                
                self.vehicles_in_zone[v_id]['last_pos'] = (cx, cy)
                
                if self.vehicles_in_zone[v_id]['frames'] > self.stationary_threshold:
                    return True
        else:
            if v_id in self.vehicles_in_zone:
                del self.vehicles_in_zone[v_id]
        
        return False
    
    def draw_zone(self, frame):
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.zebra_zone], (0, 255, 255))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.polylines(frame, [self.zebra_zone], True, (0, 255, 255), 2)
        cv2.putText(frame, "ZEBRA CROSSING", 
                   tuple(self.zebra_zone[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame