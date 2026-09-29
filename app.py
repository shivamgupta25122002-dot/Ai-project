import os
import cv2
import json
import time
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename

from core import ObjectDetector, CentroidTracker, Visualizer, Exporter, FPSCounter
from core.utils import frame_to_base64, generate_sample_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Detector and Tracker
detector = ObjectDetector()
tracker = CentroidTracker()
fps_counter = FPSCounter()

# Global state for webcam streaming control
WEBCAM_ACTIVE = False
CURRENT_CONF_THRES = 0.35
SELECTED_CLASSES = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render Web Dashboard homepage."""
    backend_name = detector.get_backend_name()
    return render_template('index.html', backend_name=backend_name)

@app.route('/api/detect-image', methods=['POST'])
def detect_image_api():
    """Endpoint for processing uploaded images."""
    if 'image' not in request.files and 'file' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
    
    file = request.files.get('image') or request.files.get('file')
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    conf_threshold = float(request.form.get('conf_threshold', 0.35))
    classes_str = request.form.get('allowed_classes', '')
    allowed_classes = [c.strip() for c in classes_str.split(',')] if classes_str else None

    # Read image file into numpy array
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Invalid image format'}), 400

    start_time = time.time()
    detections = detector.detect(frame, conf_threshold=conf_threshold, allowed_classes=allowed_classes)
    proc_time = round((time.time() - start_time) * 1000, 2)

    # Annotate frame
    annotated_frame = Visualizer.draw_detections(
        frame, detections,
        fps=None,
        backend_name=detector.get_backend_name(),
        show_hud=True
    )

    # Save output file
    output_filename = f"annotated_{secure_filename(file.filename)}"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    cv2.imwrite(output_path, annotated_frame)

    base64_image = frame_to_base64(annotated_frame)
    json_data = Exporter.to_json(detections, extra_meta={"processing_time_ms": proc_time})

    return jsonify({
        'status': 'success',
        'image': base64_image,
        'detections': json_data['detections'],
        'total_objects': len(detections),
        'processing_time_ms': proc_time,
        'backend': detector.get_backend_name()
    })

@app.route('/api/detect-sample', methods=['GET'])
def detect_sample_api():
    """Generates and processes a synthetic test image."""
    sample_img = generate_sample_image()
    detections = detector.detect(sample_img, conf_threshold=0.3)
    tracked_dets = tracker.update(detections)
    annotated = Visualizer.draw_detections(sample_img, tracked_dets, fps=30, backend_name=detector.get_backend_name())
    
    base64_image = frame_to_base64(annotated)
    json_data = Exporter.to_json(tracked_dets)
    
    return jsonify({
        'status': 'success',
        'image': base64_image,
        'detections': json_data['detections'],
        'total_objects': len(tracked_dets),
        'backend': detector.get_backend_name()
    })

def generate_video_stream():
    """Generator function for live webcam object detection streaming."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Yield error frame if webcam is not available
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "WEBCAM NOT AVAILABLE", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', blank)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        fps = fps_counter.update()
        detections = detector.detect(frame, conf_threshold=CURRENT_CONF_THRES, allowed_classes=SELECTED_CLASSES)
        tracked_dets = tracker.update(detections)
        annotated = Visualizer.draw_detections(
            frame, tracked_dets,
            fps=fps_counter.get_fps(),
            backend_name=detector.get_backend_name()
        )

        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.01)

    cap.release()

@app.route('/video_feed')
def video_feed():
    """Multipart video stream route for live webcam."""
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update active detection thresholds or settings."""
    global CURRENT_CONF_THRES, SELECTED_CLASSES
    data = request.json or {}
    if 'conf_threshold' in data:
        CURRENT_CONF_THRES = float(data['conf_threshold'])
    if 'classes' in data:
        SELECTED_CLASSES = data['classes'] if data['classes'] else None
    return jsonify({'status': 'success', 'conf_threshold': CURRENT_CONF_THRES})

if __name__ == '__main__':
    print(f"Starting Object Detection Web Server on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False)
