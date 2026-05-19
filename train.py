import cv2
import numpy as np
import os
import pickle
import logging
import threading
from init import initialize_shared, get_recognizer, reload_embeddings, get_model_hash
from face_utils import FaceDetector, align_face

DATASET_PATH = "dataset"
EMBEDDINGS_FILE = "known_faces_embeddings.pkl"
_retrain_lock = threading.Lock()

logging.basicConfig(level=logging.INFO)

# Module-level detector reused across all training images
_detector = None
_detector_lock = threading.Lock()


def _get_detector():
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = FaceDetector(model_selection=1, min_confidence=0.5, padding=0.10)
    return _detector


def preprocess_for_recognizer(image_rgb: np.ndarray, input_shape: tuple) -> np.ndarray:
    """Detect face, align using eye landmarks, resize, and normalise to [-1, 1].

    Falls back to centre-crop if no face is detected.
    """
    detector = _get_detector()
    detections = detector.detect(image_rgb)

    if detections:
        # Use highest-confidence detection
        d = max(detections, key=lambda x: x.score)
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = d.x1, d.y1, d.x2, d.y2
        face_crop = image_rgb[y1:y2, x1:x2]
        if face_crop.size > 0:
            le_crop = (d.left_eye[0] - x1, d.left_eye[1] - y1)
            re_crop = (d.right_eye[0] - x1, d.right_eye[1] - y1)
            face_crop = align_face(face_crop, le_crop, re_crop)
    else:
        face_crop = image_rgb  # no face found — use full image

    face = cv2.resize(face_crop, input_shape)
    face = face.astype(np.float32) * (2.0 / 255.0) - 1.0
    face = np.transpose(face, (2, 0, 1))
    return np.expand_dims(face, axis=0)


def retrain_and_save_embeddings():
    """Scan dataset/, generate embeddings, save to pickle.

    Skips people already present in the saved file.
    Stamps the model hash so mismatches can be detected at load time.
    """
    with _retrain_lock:
        try:
            initialize_shared("models", EMBEDDINGS_FILE)
            recognizer_session, in_name, out_name = get_recognizer()
            input_shape = tuple(recognizer_session.get_inputs()[0].shape[-2:][::-1])
        except Exception as e:
            logging.error(f"Error loading ONNX model: {e}")
            return

        existing_names: set = set()
        existing_embeddings: list = []
        existing_name_list: list = []

        if os.path.exists(EMBEDDINGS_FILE):
            try:
                with open(EMBEDDINGS_FILE, "rb") as f:
                    saved = pickle.load(f)
                existing_names = set(saved.get("names", []))
                existing_embeddings = saved.get("embeddings", [])
                existing_name_list = saved.get("names", [])
                logging.info(f"Existing embeddings loaded; skipping: {existing_names}")
            except Exception as e:
                logging.warning(f"Could not load existing embeddings: {e}")

        new_embeddings = []
        new_names = []

        logging.info("Scanning dataset for new persons …")
        for person_name in sorted(os.listdir(DATASET_PATH)):
            person_dir = os.path.join(DATASET_PATH, person_name)
            if not os.path.isdir(person_dir):
                continue
            if person_name in existing_names:
                logging.info(f"  skip {person_name} (already embedded)")
                continue

            count = 0
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    logging.warning(f"  cannot read {img_path}")
                    continue
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                inp = preprocess_for_recognizer(rgb, input_shape)
                emb = recognizer_session.run([out_name], {in_name: inp})[0].flatten()
                new_embeddings.append(emb)
                new_names.append(person_name)
                count += 1
            logging.info(f"  {person_name}: {count} images processed")

        if not new_embeddings:
            logging.info("No new embeddings to add.")
            return

        all_embeddings = existing_embeddings + new_embeddings
        all_names = existing_name_list + new_names

        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump({
                "embeddings":  all_embeddings,
                "names":       all_names,
                "model_hash":  get_model_hash(),
            }, f)

        total = len(all_embeddings)
        logging.info(f"Saved {total} embeddings to {EMBEDDINGS_FILE}")
        reload_embeddings()


if __name__ == '__main__':
    retrain_and_save_embeddings()
