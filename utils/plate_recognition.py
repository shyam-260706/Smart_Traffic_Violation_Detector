import cv2
import easyocr
import re
import pandas as pd
import os

class PlateRecognizer:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.load_database()
    
    def load_database(self):
        db_path = 'vehicle_database.csv'
        if os.path.exists(db_path):
            self.db = pd.read_csv(db_path)
        else:
            # Create sample database
            self.db = pd.DataFrame({
                'plate_number': ['MH12AB1234', 'DL01CD5678', 'KA05EF9012'],
                'owner_name': ['Rajesh Kumar', 'Priya Sharma', 'Amit Patel'],
                'phone_number': ['+919876543210', '+919123456780', '+919988776655']
            })
            self.db.to_csv(db_path, index=False)
        print(f"Loaded {len(self.db)} vehicles in database")
    
    def read_plate(self, plate_img):
        if plate_img is None or plate_img.size == 0:
            return None
        
        # Preprocess
        plate_img = cv2.resize(plate_img, None, fx=2, fy=2)
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # OCR
        try:
            results = self.reader.readtext(gray)
            if not results:
                return None
            
            # Get text
            text = ''.join([r[1] for r in results if r[2] > 0.5])
            plate = self.clean_text(text)
            
            return plate
        except:
            return None
    
    def clean_text(self, text):
        # Remove special chars
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Indian plate: 8-10 chars
        if 8 <= len(text) <= 10:
            return text
        return None
    
    def get_owner(self, plate):
        match = self.db[self.db['plate_number'] == plate]
        if not match.empty:
            return match.iloc[0].to_dict()
        return None