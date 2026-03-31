"""
Vehicle Tracker
"""

import numpy as np

class Tracker:
    def __init__(self):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_distance = 100
        self.max_disappeared = 30
    
    def update(self, detections):
        boxes = detections.boxes.xyxy.cpu().numpy()
        confs = detections.boxes.conf.cpu().numpy()
        classes = detections.boxes.cls.cpu().numpy()
        names = detections.names
        
        vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']
        current_vehicles = []
        
        for box, conf, cls in zip(boxes, confs, classes):
            class_name = names[int(cls)]
            if conf > 0.5 and class_name in vehicle_classes:
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                current_vehicles.append({
                    'bbox': box,
                    'centroid': (cx, cy),
                    'class': class_name
                })
        
        if len(self.objects) == 0:
            for vehicle in current_vehicles:
                self.register(vehicle)
        else:
            for vehicle in current_vehicles:
                matched = False
                min_dist = float('inf')
                match_id = None
                
                for obj_id, obj in self.objects.items():
                    dist = np.linalg.norm(np.array(vehicle['centroid']) - np.array(obj['centroid']))
                    if dist < min_dist and dist < self.max_distance:
                        min_dist = dist
                        match_id = obj_id
                        matched = True
                
                if matched:
                    self.objects[match_id] = vehicle
                    self.disappeared[match_id] = 0
                else:
                    self.register(vehicle)
            
            to_remove = []
            for obj_id in list(self.disappeared.keys()):
                if obj_id not in self.objects:
                    continue
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    to_remove.append(obj_id)
            
            for obj_id in to_remove:
                if obj_id in self.objects:
                    del self.objects[obj_id]
                if obj_id in self.disappeared:
                    del self.disappeared[obj_id]
        
        tracked = []
        for obj_id, obj in self.objects.items():
            vehicle = obj.copy()
            vehicle['id'] = obj_id
            tracked.append(vehicle)
        return tracked
    
    def register(self, vehicle):
        self.objects[self.next_id] = vehicle
        self.disappeared[self.next_id] = 0
        self.next_id += 1