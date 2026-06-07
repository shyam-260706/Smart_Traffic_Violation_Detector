import cv2

class RedLightDetector:
    def __init__(self, model):
        self.model = model
        self.violation_count = 0
        self.red_light_active = False

    def detect(self, frame):
        violations = []
        results = self.model(frame, conf=0.15)[0]
        self.red_light_active = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "red":
                self.red_light_active = True
                color = (0, 0, 255)
            elif label == "green":
                color = (0, 255, 0)
            elif label == "yellow":
                color = (0, 255, 255)
            else:
                color = (255, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

            if label == "red":
                self.violation_count += 1
                cv2.putText(frame, "⚠ RED LIGHT",
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
                violations.append({
                    'type': 'red_light',
                    'label': label,
                    'conf': conf,
                    'bbox': (x1, y1, x2, y2)
                })

        return violations, self.red_light_active

    def get_count(self):
        return self.violation_count