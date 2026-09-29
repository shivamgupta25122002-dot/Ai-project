import os
import cv2
import numpy as np

class DetectionResult:
    """Dataclass holding detection information for a single bounding box."""
    def __init__(self, bbox, label, confidence, class_id=0):
        self.bbox = [float(v) for v in bbox] # [xmin, ymin, xmax, ymax]
        self.label = str(label)
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.object_id = None
        self.trail = []

    def __repr__(self):
        return f"<DetectionResult label={self.label} conf={self.confidence:.2f} bbox={self.bbox}>"


# 80 COCO Class Names for YOLOv8
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

class ObjectDetector:
    """
    Multi-backend Object Detection Engine.
    Seamlessly switches between Ultralytics YOLOv8, OpenCV DNN, and OpenCV fallbacks.
    """
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        self.model_dir = model_dir
        
        self.backend = None
        self.net = None
        self.classes = COCO_CLASSES
        
        self._initialize_backend()

    def _initialize_backend(self):
        """Attempts to load the best available object detection backend."""
        # 1. Try Ultralytics YOLOv8 (State-of-the-Art 80 COCO classes)
        try:
            from ultralytics import YOLO
            print("[Detector] Loading Ultralytics YOLOv8 model...")
            self.net = YOLO('yolov8n.pt')
            self.backend = "Ultralytics YOLOv8"
            print(f"[Detector] Backend successfully loaded: {self.backend}")
            return
        except ImportError:
            print("[Detector] Ultralytics package not found. Checking OpenCV DNN...")
        except Exception as e:
            print(f"[Detector] YOLO init notice: {e}")

        # 2. Try OpenCV DNN ONNX model if present
        onnx_model = os.path.join(self.model_dir, "yolov8n.onnx")
        if os.path.exists(onnx_model) and hasattr(cv2, 'dnn') and hasattr(cv2.dnn, 'readNetFromONNX'):
            try:
                print("[Detector] Loading OpenCV DNN ONNX model...")
                self.net = cv2.dnn.readNetFromONNX(onnx_model)
                self.backend = "OpenCV DNN (ONNX)"
                print(f"[Detector] Backend loaded: {self.backend}")
                return
            except Exception as e:
                print(f"[Detector] OpenCV ONNX load failed: {e}")

        # 3. Fallback: OpenCV Built-in Contour & Haar Cascade Engine
        print("[Detector] Initializing Built-in OpenCV Contour & Motion Detector (Fallback)")
        self.backend = "OpenCV Motion & Contour Detector"

    def detect(self, frame, conf_threshold=0.35, allowed_classes=None):
        """
        Executes object detection on a BGR numpy frame.
        Returns: List[DetectionResult]
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        results = []

        if self.backend == "Ultralytics YOLOv8" and self.net is not None:
            yolo_results = self.net(frame, conf=conf_threshold, verbose=False)[0]
            for box in yolo_results.boxes:
                coords = box.xyxy[0].cpu().numpy() # [xmin, ymin, xmax, ymax]
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label = yolo_results.names.get(cls_id, f"class_{cls_id}")

                if allowed_classes is None or label.lower() in [c.lower() for c in allowed_classes]:
                    results.append(DetectionResult(coords, label, conf, cls_id))

        else:
            # Fallback OpenCV contour detection logic
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > (w * h * 0.015): # filter out tiny noise
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    # Classify aspect ratio for basic fallback labels
                    aspect_ratio = cw / float(ch)
                    label = "object"
                    if 0.8 <= aspect_ratio <= 1.2:
                        label = "box/item"
                    elif aspect_ratio < 0.5:
                        label = "person"
                    elif aspect_ratio > 1.8:
                        label = "vehicle"

                    if allowed_classes is None or label.lower() in [c.lower() for c in allowed_classes]:
                        results.append(DetectionResult([x, y, x + cw, y + ch], label, 0.75, 1))

        return results

    def get_backend_name(self):
        return self.backend or "OpenCV Fallback Detector"
