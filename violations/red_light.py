"""
Red Light Violation Detection
Detects vehicles crossing stop line during red signal
"""

import cv2
import numpy as np

class RedLightDetector:
    def __init__(self):
        # Stop line configuration (calibrate for your camera)
        self.stop_line_y = 300  # Y-coordinate of stop line
        
        # Traffic light ROI (region where traffic light is visible)
        self.traffic_light_roi = (50, 50, 150, 200)  # (x1, y1, x2, y2)
        
        # Track which vehicles have crossed
        self.crossed_vehicles = set()
        
        # Current light state
        self.light_state = 'green'  # 'red', 'yellow', 'green'
    
    def detect_light_state(self, frame):
        """
        Detect traffic light color from frame
        Returns: 'red', 'yellow', or 'green'
        """
        x1, y1, x2, y2 = self.traffic_light_roi
        light_roi = frame[y1:y2, x1:x2]
        
        if light_roi.size == 0:
            return self.light_state
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(light_roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges
        # Red (two ranges because red wraps around in HSV)
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        
        # Yellow
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        
        # Green
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        
        # Create masks
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = red_mask1 + red_mask2
        
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Count pixels
        red_pixels = np.sum(red_mask > 0)
        yellow_pixels = np.sum(yellow_mask > 0)
        green_pixels = np.sum(green_mask > 0)
        
        # Determine state
        max_pixels = max(red_pixels, yellow_pixels, green_pixels)
        
        if max_pixels == red_pixels and red_pixels > 100:
            self.light_state = 'red'
        elif max_pixels == yellow_pixels and yellow_pixels > 100:
            self.light_state = 'yellow'
        elif max_pixels == green_pixels and green_pixels > 100:
            self.light_state = 'green'
        
        return self.light_state
    
    def check_violation(self, frame, vehicle, results):
        """
        Check if vehicle violated red light
        
        Args:
            frame: Current frame
            vehicle: Vehicle dict with 'id', 'bbox', 'class'
            results: YOLO results
        
        Returns:
            True if violation detected
        """
        # Update light state
        current_light = self.detect_light_state(frame)
        
        # Only check during red light
        if current_light != 'red':
            return False
        
        v_id = vehicle['id']
        x1, y1, x2, y2 = vehicle['bbox']
        
        # Get vehicle bottom (front of vehicle)
        vehicle_front = y2
        
        # Check if crossed stop line
        if vehicle_front > self.stop_line_y:
            # Check if we haven't already logged this vehicle
            if v_id not in self.crossed_vehicles:
                self.crossed_vehicles.add(v_id)
                return True
        
        return False
    
    def draw_debug(self, frame):
        """Draw stop line and traffic light ROI for debugging"""
        # Draw stop line
        cv2.line(frame, (0, self.stop_line_y), 
                (frame.shape[1], self.stop_line_y), 
                (0, 0, 255), 3)
        
        # Draw traffic light ROI
        x1, y1, x2, y2 = self.traffic_light_roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Show light state
        color = (0, 0, 255) if self.light_state == 'red' else \
                (0, 255, 255) if self.light_state == 'yellow' else (0, 255, 0)
        cv2.putText(frame, f"Light: {self.light_state.upper()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return frame