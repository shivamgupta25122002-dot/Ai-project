import time
import json
import csv
import os
import base64
import cv2
import numpy as np
from datetime import datetime

class FPSCounter:
    """Calculates and smooths Frames Per Second (FPS)."""
    def __init__(self, avg_count=10):
        self.avg_count = avg_count
        self.prev_time = time.time()
        self.fps_list = []
        self.fps = 0.0

    def update(self):
        curr_time = time.time()
        delta = curr_time - self.prev_time
        self.prev_time = curr_time
        if delta > 0:
            current_fps = 1.0 / delta
            self.fps_list.append(current_fps)
            if len(self.fps_list) > self.avg_count:
                self.fps_list.pop(0)
            self.fps = sum(self.fps_list) / len(self.fps_list)
        return self.fps

    def get_fps(self):
        return round(self.fps, 1)


class ColorPalette:
    """Generates consistent RGB/BGR colors for class IDs."""
    _colors = {}

    @classmethod
    def get_color(cls, class_id_or_label):
        key = str(class_id_or_label)
        if key not in cls._colors:
            # Deterministic color generation based on hash
            h = hash(key) & 0xFFFFFF
            r = (h & 0xFF0000) >> 16
            g = (h & 0x00FF00) >> 8
            b = (h & 0x0000FF)
            # Boost brightness/vibrancy
            r = int(100 + (r % 155))
            g = int(100 + (g % 155))
            b = int(100 + (b % 155))
            cls._colors[key] = (b, g, r)  # BGR for OpenCV
        return cls._colors[key]


class Exporter:
    """Exports detection analytics to JSON or CSV format."""
    
    @staticmethod
    def to_json(detections, fps=None, extra_meta=None):
        timestamp = datetime.now().isoformat()
        results = []
        for det in detections:
            results.append({
                "label": det.label,
                "confidence": round(float(det.confidence), 4),
                "bbox": [int(x) for x in det.bbox], # [xmin, ymin, xmax, ymax]
                "object_id": getattr(det, 'object_id', None)
            })
        
        output = {
            "timestamp": timestamp,
            "total_objects": len(results),
            "fps": fps,
            "detections": results
        }
        if extra_meta:
            output.update(extra_meta)
        return output

    @staticmethod
    def save_json(detections, filepath, fps=None):
        data = Exporter.to_json(detections, fps)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return filepath

    @staticmethod
    def save_csv(detections, filepath):
        timestamp = datetime.now().isoformat()
        file_exists = os.path.exists(filepath)
        
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "object_id", "label", "confidence", "xmin", "ymin", "xmax", "ymax"])
            
            for det in detections:
                obj_id = getattr(det, 'object_id', '')
                writer.writerow([
                    timestamp,
                    obj_id,
                    det.label,
                    round(float(det.confidence), 4),
                    int(det.bbox[0]),
                    int(det.bbox[1]),
                    int(det.bbox[2]),
                    int(det.bbox[3])
                ])
        return filepath


def frame_to_base64(frame):
    """Converts a BGR OpenCV frame to a base64 encoded JPEG string."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    encoded = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded}"


def generate_sample_image(width=640, height=480):
    """Generates a synthetic test image with shapes for detection testing."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Background gradient
    for y in range(height):
        r = int(30 + (y / height) * 40)
        g = int(30 + (y / height) * 30)
        b = int(50 + (y / height) * 60)
        img[y, :] = (b, g, r)
        
    # Draw sample synthetic objects (Car, Person, Bottle)
    # Car box
    cv2.rectangle(img, (100, 200), (320, 360), (40, 100, 220), -1)
    cv2.circle(img, (150, 360), 30, (20, 20, 20), -1)
    cv2.circle(img, (270, 360), 30, (20, 20, 20), -1)
    cv2.putText(img, "SAMPLE VEHICLE", (110, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Person shape
    cv2.circle(img, (480, 180), 25, (200, 180, 150), -1) # Head
    cv2.rectangle(img, (455, 205), (505, 330), (180, 60, 50), -1) # Body
    cv2.putText(img, "PERSON", (460, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img
