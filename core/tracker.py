import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

class CentroidTracker:
    """
    Centroid Object Tracker.
    Associates bounding boxes across frames to assign unique IDs and track trajectories.
    """
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_object_id = 1
        self.objects = OrderedDict()        # ID -> centroid (x, y)
        self.bboxes = OrderedDict()         # ID -> bbox [xmin, ymin, xmax, ymax]
        self.labels = OrderedDict()         # ID -> label
        self.disappeared = OrderedDict()    # ID -> frames disappeared count
        self.history = OrderedDict()        # ID -> list of past centroids [(x, y), ...]
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox, label):
        """Register a new detected object."""
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.labels[self.next_object_id] = label
        self.disappeared[self.next_object_id] = 0
        self.history[self.next_object_id] = [centroid]
        self.next_object_id += 1

    def deregister(self, object_id):
        """Remove a tracked object when it disappears."""
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.labels[object_id]
        del self.disappeared[object_id]
        del self.history[object_id]

    def update(self, detections):
        """
        Update tracker with list of DetectionResult objects.
        Returns a list of updated DetectionResult objects populated with object_id and trail.
        """
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return []

        # Extract input centroids & bboxes
        input_centroids = []
        input_bboxes = []
        input_labels = []
        
        for det in detections:
            xmin, ymin, xmax, ymax = det.bbox
            cX = int((xmin + xmax) / 2.0)
            cY = int((ymin + ymax) / 2.0)
            input_centroids.append((cX, cY))
            input_bboxes.append(det.bbox)
            input_labels.append(det.label)

        input_centroids = np.array(input_centroids)

        # If no objects are currently tracked, register all input detections
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i], input_labels[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            # Compute Euclidean distance matrix between tracked objects & input centroids
            D = dist.cdist(object_centroids, input_centroids)

            # Sort rows based on minimum values, then columns
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.bboxes[object_id] = input_bboxes[col]
                self.labels[object_id] = input_labels[col]
                self.disappeared[object_id] = 0
                
                # Append to centroid history trail
                self.history[object_id].append(tuple(input_centroids[col]))
                if len(self.history[object_id]) > 20:
                    self.history[object_id].pop(0)

                used_rows.add(row)
                used_cols.add(col)

            # Check for unmatched rows (disappeared tracked objects)
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Check for unmatched cols (new objects)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col], input_bboxes[col], input_labels[col])

        # Attach tracking data back to detections
        updated_detections = []
        for det in detections:
            xmin, ymin, xmax, ymax = det.bbox
            cX = int((xmin + xmax) / 2.0)
            cY = int((ymin + ymax) / 2.0)
            
            best_id = None
            min_d = float('inf')
            for obj_id, centroid in self.objects.items():
                d = np.hypot(cX - centroid[0], cY - centroid[1])
                if d < min_d and d < 30:
                    min_d = d
                    best_id = obj_id
            
            if best_id is not None:
                det.object_id = best_id
                det.trail = list(self.history.get(best_id, []))
            updated_detections.append(det)

        return updated_detections
