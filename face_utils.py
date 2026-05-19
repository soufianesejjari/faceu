import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class FaceDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    left_eye: Tuple[float, float]   # pixel coords in frame
    right_eye: Tuple[float, float]  # pixel coords in frame
    score: float


class FaceDetector:
    """Thin wrapper around MediaPipe Face Detection.

    Returns direct face bboxes + 5-point landmarks in a single forward pass,
    replacing the prior MediaPipe Pose (full-body) approach and eliminating
    the need to run a second detector (dlib) just to get eye coordinates.
    """

    def __init__(self, model_selection: int = 1, min_confidence: float = 0.5,
                 padding: float = 0.10):
        """
        model_selection: 0 = short-range (≤2m), 1 = full-range (≤5m, better for doorways)
        padding: fractional padding applied to the returned bbox (adds forehead/chin room)
        """
        self.padding = padding
        self._fd = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence
        )

    def detect(self, frame_rgb: np.ndarray) -> List[FaceDetection]:
        h, w = frame_rgb.shape[:2]
        results = self._fd.process(frame_rgb)
        if not results.detections:
            return []

        detections = []
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            kps = det.location_data.relative_keypoints

            # Raw bbox
            x1 = bb.xmin * w
            y1 = bb.ymin * h
            x2 = (bb.xmin + bb.width) * w
            y2 = (bb.ymin + bb.height) * h

            # Add symmetric padding for forehead/chin
            pad_x = (x2 - x1) * self.padding
            pad_y = (y2 - y1) * self.padding
            x1 = max(0, int(x1 - pad_x))
            y1 = max(0, int(y1 - pad_y))
            x2 = min(w, int(x2 + pad_x))
            y2 = min(h, int(y2 + pad_y))

            # MediaPipe uses the SUBJECT's perspective:
            #   kps[0] = subject's right eye → appears on the LEFT in the camera image
            #   kps[1] = subject's left eye  → appears on the RIGHT in the camera image
            # align_face() expects (left_in_image, right_in_image) so the diff vector
            # points left→right and the rotation angle is near 0° for a level face.
            left_eye = (kps[0].x * w, kps[0].y * h)   # camera-left
            right_eye = (kps[1].x * w, kps[1].y * h)  # camera-right

            detections.append(FaceDetection(x1, y1, x2, y2, left_eye, right_eye,
                                            float(det.score[0])))
        return detections

    def close(self):
        self._fd.close()


def align_face(face_rgb: np.ndarray,
               left_eye: Tuple[float, float],
               right_eye: Tuple[float, float]) -> np.ndarray:
    """Rotate face_rgb so eyes are level.

    left_eye / right_eye must be in the coordinate space of face_rgb
    (i.e., already offset by the crop origin).
    """
    le = np.array(left_eye, dtype=np.float32)
    re = np.array(right_eye, dtype=np.float32)
    diff = re - le
    angle = np.degrees(np.arctan2(diff[1], diff[0]))
    center = tuple(((le + re) * 0.5).astype(np.float32))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(face_rgb, M, (face_rgb.shape[1], face_rgb.shape[0]),
                          flags=cv2.INTER_LINEAR)
