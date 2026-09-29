import cv2
import numpy as np
from .utils import ColorPalette

class Visualizer:
    """Renders sleek bounding boxes, label badges, HUD overlay, and tracking trails on video frames."""

    @staticmethod
    def draw_detections(frame, detections, fps=None, backend_name="Detector", show_hud=True, show_trails=True):
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Summary counter per label
        class_counts = {}

        # 1. Draw Tracking Trails & Bounding Boxes
        for det in detections:
            xmin, ymin, xmax, ymax = [int(v) for v in det.bbox]
            label = det.label
            conf = det.confidence
            obj_id = getattr(det, 'object_id', None)
            trail = getattr(det, 'trail', [])

            class_counts[label] = class_counts.get(label, 0) + 1
            color = ColorPalette.get_color(label)

            # Draw trajectory trail line if enabled
            if show_trails and len(trail) > 1:
                for i in range(1, len(trail)):
                    pt1 = trail[i - 1]
                    pt2 = trail[i]
                    thickness = int(np.sqrt(20 / float(len(trail) - i + 1)) * 1.5)
                    cv2.line(annotated, pt1, pt2, color, max(1, thickness))

            # Corner Reticle Bounding Box
            box_thickness = 2
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, box_thickness)

            # Fancy corner notches for futuristic aesthetic
            corner_len = min(int((xmax - xmin) * 0.2), 15)
            # Top-left
            cv2.line(annotated, (xmin, ymin), (xmin + corner_len, ymin), (255, 255, 255), 3)
            cv2.line(annotated, (xmin, ymin), (xmin, ymin + corner_len), (255, 255, 255), 3)
            # Top-right
            cv2.line(annotated, (xmax, ymin), (xmax - corner_len, ymin), (255, 255, 255), 3)
            cv2.line(annotated, (xmax, ymin), (xmax, ymin + corner_len), (255, 255, 255), 3)
            # Bottom-left
            cv2.line(annotated, (xmin, ymax), (xmin + corner_len, ymax), (255, 255, 255), 3)
            cv2.line(annotated, (xmin, ymax), (xmin, ymax - corner_len), (255, 255, 255), 3)
            # Bottom-right
            cv2.line(annotated, (xmax, ymax), (xmax - corner_len, ymax), (255, 255, 255), 3)
            cv2.line(annotated, (xmax, ymax), (xmax, ymax - corner_len), (255, 255, 255), 3)

            # Label text content
            if obj_id:
                label_text = f"#{obj_id} {label.upper()} {int(conf * 100)}%"
            else:
                label_text = f"{label.upper()} {int(conf * 100)}%"

            # Calculate label background pill
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thick = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thick)

            label_bg_y1 = max(ymin - text_h - 10, 0)
            label_bg_y2 = ymin if ymin - text_h - 10 >= 0 else ymin + text_h + 10

            # Semi-transparent background tag for readable text
            overlay = annotated.copy()
            cv2.rectangle(overlay, (xmin, label_bg_y1), (xmin + text_w + 12, label_bg_y2), color, -1)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

            # Text
            text_y = label_bg_y2 - 4 if label_bg_y1 < ymin else label_bg_y1 + text_h + 2
            cv2.putText(annotated, label_text, (xmin + 6, text_y), font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

        # 2. Heads-Up Display (HUD) Header Bar
        if show_hud:
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, 40), (15, 15, 20), -1)
            cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
            
            # Left side: System / Engine Info
            hud_info = f"ENGINE: {backend_name} | DETECTED: {len(detections)}"
            cv2.putText(annotated, hud_info, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)

            # Right side: FPS
            if fps is not None:
                fps_text = f"FPS: {fps}"
                cv2.putText(annotated, fps_text, (w - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2, cv2.LINE_AA)

        return annotated
