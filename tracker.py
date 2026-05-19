"""
SORT: Simple Online and Realtime Tracking
Reference: https://arxiv.org/abs/1602.00763
Original implementation: https://github.com/abewley/sort
"""
import threading
import numpy as np
from scipy.optimize import linear_sum_assignment

class TrackIDCounter:
    _lock = threading.Lock()
    _counter = None

    @classmethod
    def initialize(cls, last_id):
        cls._counter = last_id
        cls._initialized = True

    @classmethod
    def get_next(cls):
        with cls._lock:
            cls._counter += 1
            return cls._counter

    @classmethod
    def get_current(cls):
        with cls._lock:
            return cls._counter

class FaceSortTracker:
    def __init__(self, max_age=5, min_hits=1, iou_threshold=0.15):
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    def update(self, detections):
        if not isinstance(detections, np.ndarray):
            detections = np.array(detections)
            
        if detections.shape[0] > 0:
            # Append a dummy score of 1.0 to each detection using efficient numpy operations
            detections_with_score = np.hstack((detections, np.ones((detections.shape[0], 1))))
        else:
            # Sort tracker expects a numpy array even if it's empty
            detections_with_score = np.empty((0, 5))
        
        return self.tracker.update(detections_with_score)

class KalmanBoxTracker:
    def __init__(self, bbox):
        # bbox: [x1, y1, x2, y2]
        self.bbox = np.array(bbox, dtype=np.float32)
        self.velocity = np.zeros(4, dtype=np.float32)
        self.id = TrackIDCounter.get_next()
        self.hits = 1
        self.no_losses = 0

    def update(self, bbox):
        new_bbox = np.array(bbox, dtype=np.float32)
        delta = new_bbox - self.bbox
        # Exponential smoothing: blend new delta with previous velocity
        self.velocity = 0.7 * delta + 0.3 * self.velocity
        self.bbox = new_bbox
        self.hits += 1
        self.no_losses = 0

    def predict(self):
        self.no_losses += 1
        # Project forward by velocity × frames since last match
        return self.bbox + self.velocity * self.no_losses

    def get_state(self):
        return self.bbox

class Sort:
    def __init__(self, max_age=3, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
    def update(self, dets):
        self.frame_count += 1

        # Predict current state of all trackers
        if len(self.trackers) > 0:
            trks = np.array([trk.predict() for trk in self.trackers])
        else:
            trks = np.empty((0, 4))

        # Associate detections with existing trackers
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :4]))

        # Remove old trackers that have not been seen for max_age frames
        self.trackers = [
            trk for trk in self.trackers
            if trk.no_losses <= self.max_age
        ]

        # Construct the output array of active trackers
        ret = []
        for trk in self.trackers:
            if (trk.hits >= self.min_hits) or (self.frame_count <= self.min_hits):
                ret.append(np.concatenate((trk.get_state(), [trk.id])).reshape(1, -1))
        
        if len(ret) > 0:
            return np.concatenate(ret)
            
        return np.empty((0, 5))

def iou_vectorized(boxes1, boxes2):
    """
    Calculate IoU between two sets of boxes in a vectorized way.
    boxes1: (N, 4)
    boxes2: (M, 4)
    Returns: (N, M) matrix of IoUs.
    """
    # Expand dimensions to broadcast operations
    boxes1 = np.expand_dims(boxes1, axis=1)  # (N, 1, 4)
    boxes2 = np.expand_dims(boxes2, axis=0)  # (1, M, 4)

    # Intersection
    xx1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    yy1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    xx2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    yy2 = np.minimum(boxes1[..., 3], boxes2[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersection = w * h

    # Union
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - intersection

    # IoU
    iou = intersection / (union + 1e-6) # Add epsilon to avoid division by zero
    return iou

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0),dtype=int)
    
    # Use the vectorized IoU calculation
    if len(detections) > 0:
        detection_boxes = detections[:, :4]
        iou_matrix = iou_vectorized(detection_boxes, trackers)
    else:
        iou_matrix = np.empty((0, len(trackers)), dtype=np.float32)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(list(zip(row_ind, col_ind)))

    if len(matched_indices) == 0:
        matched_indices = np.empty((0, 2), dtype=int)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
            
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
        
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
