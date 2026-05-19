"""Face Image Assessment Quality (FIAQ)

Accepts landmarks as a simple dict produced by MediaPipe Face Detection
(no dlib dependency):
    {
        'left_eye':  (x, y),   # pixel coords inside the face ROI
        'right_eye': (x, y),
        'nose':      (x, y),   # optional
        'mouth':     (x, y),   # optional
    }
All pose metrics (roll/yaw) are derived geometrically from these 4 points,
replacing the prior 68-point dlib + solvePnP approach.
"""

import cv2
import numpy as np
import math

cfg = {
    'sharpness_sigma': 0.8,
    'laplacian_ideal': 200.0,
    'tenengrad_ideal': 1_500_000.0,

    'max_acceptable_roll_abs': 15.0,   # degrees
    'max_acceptable_yaw_proxy': 0.25,  # eye-distance-normalised nose offset
    'pose_roll_weight': 0.5,
    'pose_yaw_weight': 0.5,

    'brightness_ideal_low': 90.0,
    'brightness_ideal_high': 180.0,
    'min_acceptable_contrast_std': 30.0,

    'min_face_size_pixels': 80,

    'weights': {
        'sharpness':   0.35,
        'pose':        0.25,
        'brightness':  0.10,
        'contrast':    0.10,
        'resolution':  0.20,
    },
}


def _sharpness(gray):
    blurred = cv2.GaussianBlur(gray, (0, 0), cfg['sharpness_sigma'])
    laplacian_var = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3).var()
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float((sobelx ** 2 + sobely ** 2).sum())
    nl = min(laplacian_var / cfg['laplacian_ideal'], 1.0)
    nt = min(tenengrad / cfg['tenengrad_ideal'], 1.0)
    return (nl + nt) / 2.0


def _pose_score(landmarks):
    """Geometric pose from eye / nose keypoints. Returns 0-1."""
    le = landmarks.get('left_eye')
    re = landmarks.get('right_eye')
    if le is None or re is None:
        return 0.5  # unknown — neutral penalty

    le = np.array(le, dtype=np.float32)
    re = np.array(re, dtype=np.float32)
    diff = re - le
    eye_dist = float(np.linalg.norm(diff)) + 1e-6

    # Roll: angle of inter-eye vector (should be ~0)
    roll = abs(math.degrees(math.atan2(float(diff[1]), float(diff[0]))))
    roll_score = max(0.0, 1.0 - roll / cfg['max_acceptable_roll_abs'])

    # Yaw proxy: normalised horizontal offset of nose from eye midpoint
    nose = landmarks.get('nose')
    if nose is not None:
        mid_x = (le[0] + re[0]) / 2.0
        yaw_proxy = abs(float(nose[0]) - mid_x) / eye_dist
        yaw_score = max(0.0, 1.0 - yaw_proxy / cfg['max_acceptable_yaw_proxy'])
    else:
        yaw_score = 0.5

    return (roll_score * cfg['pose_roll_weight'] +
            yaw_score * cfg['pose_yaw_weight'])


def assess_face_quality(face_roi_rgb: np.ndarray, landmarks: dict = None) -> float:
    """Return a 0-1 quality score for a face ROI.

    Parameters
    ----------
    face_roi_rgb : H×W×3 uint8 RGB image (the cropped, aligned face).
    landmarks    : dict with optional keys 'left_eye', 'right_eye', 'nose',
                   'mouth' as (x, y) tuples in ROI pixel space.
                   Pass None to skip pose/occlusion checks.
    """
    if landmarks is None:
        landmarks = {}

    gray = cv2.cvtColor(face_roi_rgb, cv2.COLOR_RGB2GRAY)
    img_h, img_w = face_roi_rgb.shape[:2]

    sharpness_score = _sharpness(gray)
    pose_score = _pose_score(landmarks)

    brightness = float(np.mean(gray))
    if cfg['brightness_ideal_low'] <= brightness <= cfg['brightness_ideal_high']:
        brightness_score = 1.0
    elif brightness < cfg['brightness_ideal_low']:
        brightness_score = brightness / cfg['brightness_ideal_low']
    else:
        brightness_score = (255.0 - brightness) / (255.0 - cfg['brightness_ideal_high'])
    brightness_score = max(0.0, min(brightness_score, 1.0))

    contrast_score = min(float(np.std(gray)) / cfg['min_acceptable_contrast_std'], 1.0)

    min_px = cfg['min_face_size_pixels']
    if img_w < min_px or img_h < min_px:
        resolution_score = 0.0
    else:
        resolution_score = min((img_w + img_h) / (2.0 * min_px), 1.0)

    w = cfg['weights']
    return (sharpness_score   * w['sharpness'] +
            pose_score        * w['pose'] +
            brightness_score  * w['brightness'] +
            contrast_score    * w['contrast'] +
            resolution_score  * w['resolution'])
