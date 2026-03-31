import cv2
from datetime import datetime
import os

class NotificationSystem:
    def __init__(self):
        self.sent = set()
        self.messages = {
            'red_light': '🚦 Red Light Violation',
            'no_helmet': '🪖 No Helmet Violation',
            'triple_riding': '🏍️ Triple Riding Violation',
            'zebra_crossing': '🚶 Zebra Crossing Violation',
            'no_plate': '🚗 No Number Plate',
            'one_way': '⚠️ Wrong Way Violation'
        }
        
        from utils.plate_recognition import PlateRecognizer
        self.plate_reader = PlateRecognizer()
    
    def send_alert(self, plate, violation_type, frame):
        key = f"{plate}_{violation_type}"
        if key in self.sent:
            return
        
        # Get owner
        owner = self.plate_reader.get_owner(plate)
        if not owner:
            print(f"No owner found for {plate}")
            return
        
        # Create message
        msg = f"Dear {owner['owner_name']},\n"
        msg += self.messages.get(violation_type, 'Traffic Violation')
        msg += f"\nVehicle: {plate}"
        msg += f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        msg += "\nFine details will follow.\n- Traffic Police"
        
        # Send SMS (simulated)
        print(f"\nSMS to {owner['phone_number']}:")
        print(msg)
        
        # Save evidence
        self.save_evidence(plate, violation_type, frame)
        
        self.sent.add(key)
    
    def save_evidence(self, plate, violation_type, frame):
        os.makedirs('evidence', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"evidence/{plate}_{violation_type}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Evidence saved: {filename}")