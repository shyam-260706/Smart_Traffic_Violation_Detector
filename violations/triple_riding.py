import cv2

class TripleRidingDetector:
    def __init__(self, model):
        self.model = model
        self.violation_count = 0

    def detect(self, frame):
        violations = []
        results = self.model(frame, conf=0.25)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            is_violation = label == "Triple Riding"

            color = (0, 0, 255) if is_violation else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

            if is_violation:
                self.violation_count += 1
                cv2.putText(frame, "⚠ TRIPLE RIDING",
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
                violations.append({
                    'type': 'triple_riding',
                    'label': label,
                    'conf': conf,
                    'bbox': (x1, y1, x2, y2)
                })

        return violations

    def get_count(self):
        return self.violation_count