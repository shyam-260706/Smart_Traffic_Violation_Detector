"""
One-Way Violation Detection
Detects vehicles moving in wrong direction
"""

import cv2
import numpy as np

class OneWayDetector:
    def __init__(self, allowed_direction='down'):
        """
        Initialize one-way detector
        
        Args:
            allowed_direction: 'up', 'down', 'left', 'right'
        """
        self.allowed_direction = allowed_direction
        self.vehicle_tracks = {}
        self.min_track_length = 10  # Minimum frames to confirm direction
        self.movement_threshold = 15  # Pixels moved to consider as movement
    
    def set_direction(self, direction):
        """Set allowed direction"""
        self.allowed_direction = direction
    
    def check_violation(self, vehicle):
        """
        Check if vehicle is moving in wrong direction
        
        Logic:
        1. Track vehicle position over frames
        2. Calculate movement direction
        3. Compare with allowed direction
        """
        v_id = vehicle['id']
        x1, y1, x2, y2 = vehicle['bbox']
        
        # Get center point
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Initialize or update track
        if v_id not in self.vehicle_tracks:
            self.vehicle_tracks[v_id] = []
        
        self.vehicle_tracks[v_id].append((cx, cy))
        
        # Keep only recent positions
        if len(self.vehicle_tracks[v_id]) > 30:
            self.vehicle_tracks[v_id].pop(0)
        
        # Need minimum track length to determine direction
        if len(self.vehicle_tracks[v_id]) < self.min_track_length:
            return False
        
        # Calculate movement direction
        actual_direction = self.calculate_direction(self.vehicle_tracks[v_id])
        
        # Check if moving in wrong direction
        if actual_direction and actual_direction != self.allowed_direction:
            return True
        
        return False
    
    def calculate_direction(self, track):
        """
        Calculate primary movement direction from track
        
        Returns: 'up', 'down', 'left', 'right', or None
        """
        if len(track) < 2:
            return None
        
        # Get start and end points
        start_x, start_y = track[0]
        end_x, end_y = track[-1]
        
        # Calculate displacement
        dx = end_x - start_x
        dy = end_y - start_y
        
        # Check if moved enough
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < self.movement_threshold:
            return None  # Not enough movement
        
        # Determine primary direction
        if abs(dx) > abs(dy):
            # Horizontal movement
            if dx > 0:
                return 'right'
            else:
                return 'left'
        else:
            # Vertical movement
            if dy > 0:
                return 'down'
            else:
                return 'up'
    
    def draw_arrow(self, frame):
        """Draw allowed direction arrow"""
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        
        arrow_length = 100
        
        if self.allowed_direction == 'up':
            end_point = (center[0], center[1] - arrow_length)
        elif self.allowed_direction == 'down':
            end_point = (center[0], center[1] + arrow_length)
        elif self.allowed_direction == 'left':
            end_point = (center[0] - arrow_length, center[1])
        else:  # right
            end_point = (center[0] + arrow_length, center[1])
        
        cv2.arrowedLine(frame, center, end_point, (0, 255, 0), 5, tipLength=0.3)
        cv2.putText(frame, f"ONE WAY: {self.allowed_direction.upper()}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame