import cv2
import numpy as np

class OneWayDetector:
    def __init__(self, allowed_direction='down'):
        self.allowed_direction = allowed_direction
        self.vehicle_tracks = {}
        self.min_track_length = 10
        self.movement_threshold = 15
    
    def set_direction(self, direction):
        self.allowed_direction = direction
    
    def check_violation(self, vehicle):
        v_id = vehicle['id']
        x1, y1, x2, y2 = vehicle['bbox']
        
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        if v_id not in self.vehicle_tracks:
            self.vehicle_tracks[v_id] = []
        
        self.vehicle_tracks[v_id].append((cx, cy))
        
        if len(self.vehicle_tracks[v_id]) > 30:
            self.vehicle_tracks[v_id].pop(0)
        
        if len(self.vehicle_tracks[v_id]) < self.min_track_length:
            return False
        
        actual_direction = self.calculate_direction(self.vehicle_tracks[v_id])
        
        if actual_direction and actual_direction != self.allowed_direction:
            return True
        
        return False
    
    def calculate_direction(self, track):
        if len(track) < 2:
            return None
        
        start_x, start_y = track[0]
        end_x, end_y = track[-1]
        
        dx = end_x - start_x
        dy = end_y - start_y
        
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < self.movement_threshold:
            return None
        
        if abs(dx) > abs(dy):
            if dx > 0:
                return 'right'
            else:
                return 'left'
        else:
            if dy > 0:
                return 'down'
            else:
                return 'up'
    
    def draw_arrow(self, frame):
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        
        arrow_length = 100
        
        if self.allowed_direction == 'up':
            end_point = (center[0], center[1] - arrow_length)
        elif self.allowed_direction == 'down':
            end_point = (center[0], center[1] + arrow_length)
        elif self.allowed_direction == 'left':
            end_point = (center[0] - arrow_length, center[1])
        else:
            end_point = (center[0] + arrow_length, center[1])
        
        cv2.arrowedLine(frame, center, end_point, (0, 255, 0), 5, tipLength=0.3)
        cv2.putText(frame, f"ONE WAY: {self.allowed_direction.upper()}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame