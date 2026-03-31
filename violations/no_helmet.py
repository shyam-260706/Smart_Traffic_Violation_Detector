import cv2
import numpy as np

class HelmetDetector:
    def __init__(self):
        self.person_conf_threshold = 0.5
        self.violation_cache = {}
        self.cache_frames = 15
    
    def check_violation(self, frame, vehicle, results):
        if vehicle['class'] != 'motorcycle':
            return False
        
        v_id = vehicle['id']
        x1, y1, x2, y2 = map(int, vehicle['bbox'])
        
        persons = self.get_persons_on_vehicle(vehicle, results)
        
        if len(persons) == 0:
            return False
        
        no_helmet_count = 0
        
        for person_bbox in persons:
            has_helmet = self.detect_helmet(frame, person_bbox, vehicle['bbox'])
            if not has_helmet:
                no_helmet_count += 1
        
        if v_id not in self.violation_cache:
            self.violation_cache[v_id] = []
        
        self.violation_cache[v_id].append(no_helmet_count > 0)
        
        if len(self.violation_cache[v_id]) > self.cache_frames:
            self.violation_cache[v_id].pop(0)
        
        if len(self.violation_cache[v_id]) >= self.cache_frames:
            violation_ratio = sum(self.violation_cache[v_id]) / len(self.violation_cache[v_id])
            if violation_ratio > 0.6:
                return True
        
        return False
    
    def get_persons_on_vehicle(self, vehicle, results):
        vx1, vy1, vx2, vy2 = vehicle['bbox']
        
        persons = []
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        names = results.names
        
        for box, conf, cls in zip(boxes, confs, classes):
            if names[int(cls)] == 'person' and conf > self.person_conf_threshold:
                px1, py1, px2, py2 = box
                
                x_overlap = max(0, min(vx2, px2) - max(vx1, px1))
                y_overlap = max(0, min(vy2, py2) - max(vy1, py1))
                overlap_area = x_overlap * y_overlap
                
                person_area = (px2 - px1) * (py2 - py1)
                
                if overlap_area / person_area > 0.3:
                    persons.append(box)
        
        return persons
    
    def detect_helmet(self, frame, person_bbox, vehicle_bbox):
        px1, py1, px2, py2 = map(int, person_bbox)
        
        head_height = int((py2 - py1) * 0.25)
        head_roi = frame[py1:py1+head_height, px1:px2]
        
        if head_roi.size == 0:
            return True
        
        hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
        
        dark_lower = np.array([0, 0, 0])
        dark_upper = np.array([180, 255, 80])
        dark_mask = cv2.inRange(hsv, dark_lower, dark_upper)
        dark_ratio = np.sum(dark_mask > 0) / dark_mask.size
        
        bright_lower = np.array([0, 0, 150])
        bright_upper = np.array([180, 100, 255])
        bright_mask = cv2.inRange(hsv, bright_lower, bright_upper)
        bright_ratio = np.sum(bright_mask > 0) / bright_mask.size
        
        if dark_ratio > 0.3 or bright_ratio > 0.3:
            return True
        
        return False