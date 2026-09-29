import argparse
import os
import cv2
import time
from core import ObjectDetector, CentroidTracker, Visualizer, Exporter, FPSCounter
from core.utils import generate_sample_image

def run_cli():
    parser = argparse.ArgumentParser(description="Object Detection & Analytics System (CLI)")
    parser.add_argument("--source", type=str, default=None, help="Path to input image, video file, or '0' for webcam feed")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated image or video file")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated target classes (e.g., person,car)")
    parser.add_argument("--save-json", type=str, default=None, help="Path to save JSON detection results")
    parser.add_argument("--save-csv", type=str, default=None, help="Path to append CSV detection results")
    parser.add_argument("--no-hud", action="store_true", help="Disable HUD header bar on annotated output")
    parser.add_argument("--web", action="store_true", help="Launch Flask Web Dashboard")

    args = parser.parse_args()

    if args.web:
        from app import app
        print("Launching Object Detection Web Server on http://127.0.0.1:5000 ...")
        app.run(host='0.0.0.0', port=5000, debug=False)
        return

    if args.source is None:
        print("[CLI] No --source provided. Running quick synthetic test...")
        detector = ObjectDetector()
        tracker = CentroidTracker()
        sample_img = generate_sample_image()
        
        detections = detector.detect(sample_img, conf_threshold=args.conf)
        tracked = tracker.update(detections)
        annotated = Visualizer.draw_detections(sample_img, tracked, fps=30, backend_name=detector.get_backend_name(), show_hud=not args.no_hud)
        
        out_path = args.output or "sample_output.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"[CLI] Synthetic test complete! Saved output to: {os.path.abspath(out_path)}")
        print(f"[CLI] Total Detections: {len(tracked)}")
        print(Exporter.to_json(tracked, fps=30))
        return

    # Parse target classes if provided
    allowed_classes = [c.strip() for c in args.classes.split(',')] if args.classes else None

    detector = ObjectDetector()
    tracker = CentroidTracker()
    fps_counter = FPSCounter()

    # Image source
    if args.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        print(f"[CLI] Processing image file: {args.source}")
        frame = cv2.imread(args.source)
        if frame is None:
            print(f"Error: Unable to load image from {args.source}")
            return

        start_time = time.time()
        detections = detector.detect(frame, conf_threshold=args.conf, allowed_classes=allowed_classes)
        proc_time = round((time.time() - start_time) * 1000, 2)
        tracked = tracker.update(detections)
        
        annotated = Visualizer.draw_detections(
            frame, tracked, fps=None,
            backend_name=detector.get_backend_name(),
            show_hud=not args.no_hud
        )

        out_path = args.output or f"annotated_{os.path.basename(args.source)}"
        cv2.imwrite(out_path, annotated)
        print(f"[CLI] Processing finished in {proc_time} ms. Saved output to: {os.path.abspath(out_path)}")
        print(f"[CLI] Total Detections: {len(tracked)}")
        
        if args.save_json:
            Exporter.save_json(tracked, args.save_json)
            print(f"[CLI] Saved JSON results to: {args.save_json}")
        if args.save_csv:
            Exporter.save_csv(tracked, args.save_csv)
            print(f"[CLI] Saved CSV results to: {args.save_csv}")

    # Video or Webcam source
    else:
        src = 0 if args.source == "0" or args.source.lower() == "webcam" else args.source
        print(f"[CLI] Opening video stream source: {src}")
        cap = cv2.VideoCapture(src)
        
        if not cap.isOpened():
            print(f"Error: Unable to open video source {src}")
            return

        out_writer = None
        if args.output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_writer = cv2.VideoWriter(args.output, fourcc, fps_val, (width, height))
            print(f"[CLI] Saving annotated video to: {args.output}")

        print("[CLI] Processing stream. Press 'q' or Ctrl+C to stop...")
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                fps = fps_counter.update()
                detections = detector.detect(frame, conf_threshold=args.conf, allowed_classes=allowed_classes)
                tracked = tracker.update(detections)
                
                annotated = Visualizer.draw_detections(
                    frame, tracked, fps=fps_counter.get_fps(),
                    backend_name=detector.get_backend_name(),
                    show_hud=not args.no_hud
                )

                if out_writer:
                    out_writer.write(annotated)

                if src == 0:
                    cv2.imshow("Object Detection System (Press 'q' to quit)", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            cv2.destroyAllWindows()
            print("[CLI] Processing finished.")

if __name__ == "__main__":
    run_cli()
