import cv2

# ──────────────────────────────────────────────────────────
# Edit these coordinates to match your camera/video frame.
# ──────────────────────────────────────────────────────────
ZONES = {
    "stop_line_y"   : 400,                   # y-coordinate of red-light stop line
    "crossing"      : (0, 350, 1920, 450),   # zebra crossing rectangle
    "entry_zone"    : (0, 0, 300, 1080),     # one-way: expected entry side
    "exit_zone"     : (1600, 0, 1920, 1080), # one-way: expected exit side
}

COLORS = {
    "crossing"   : (255, 255, 0),
    "entry_zone" : (0, 255, 0),
    "exit_zone"  : (0, 0, 255),
}


def in_box(point, zone_name):
    """Check if (x, y) is inside a rectangular zone"""
    box = ZONES.get(zone_name)
    if not box or len(box) != 4:
        return False
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def draw_zones(frame):
    """Draw all zones on the frame"""
    for name, val in ZONES.items():
        if name == "stop_line_y":
            cv2.line(frame, (0, val), (frame.shape[1], val), (0, 255, 255), 2)
            cv2.putText(frame, "STOP LINE", (10, val - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        elif isinstance(val, tuple) and len(val) == 4:
            c = COLORS.get(name, (200, 200, 200))
            cv2.rectangle(frame, (val[0], val[1]), (val[2], val[3]), c, 2)
            cv2.putText(frame, name.upper(), (val[0] + 4, val[1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)