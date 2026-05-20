import cv2
import numpy as np
import os
import pickle
import logging
import threading
from pathlib import Path
from dotenv import load_dotenv
from init import initialize_shared, get_recognizer, reload_embeddings, get_model_hash
from face_utils import FaceDetector, align_face

# Read .env so train.py uses the same model as the live server
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
RECOGNITION_MODEL = os.getenv("RECOGNITION_MODEL", "w600k_r50.onnx")

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
    """Scan dataset/, generate embeddings with the current model, save to pickle.

    Always regenerates ALL embeddings to ensure they match the active model.
    If the model changed (e.g. w600k_r50 → edgeface_s), stale embeddings
    from the old model will produce wrong similarities and cause all-Unknown.
    """
    with _retrain_lock:
        try:
            initialize_shared("models", EMBEDDINGS_FILE,
                              recognizer_model_name=RECOGNITION_MODEL)
            recognizer_session, in_name, out_name = get_recognizer()
            input_shape = tuple(recognizer_session.get_inputs()[0].shape[-2:][::-1])
            logging.info(f"Training with model: {RECOGNITION_MODEL}  input: {input_shape}")
        except Exception as e:
            logging.error(f"Error loading ONNX model: {e}")
            return

        new_embeddings = []
        new_names = []
        quality_stats = {}   # person → {total, kept, outliers, cohesion, spread}

        logging.info("Scanning dataset/ and regenerating all embeddings …")
        for person_name in sorted(os.listdir(DATASET_PATH)):
            person_dir = os.path.join(DATASET_PATH, person_name)
            if not os.path.isdir(person_dir):
                continue

            person_embs = []
            for img_name in sorted(os.listdir(person_dir)):
                img_path = os.path.join(person_dir, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    logging.warning(f"  cannot read {img_path}")
                    continue
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                inp = preprocess_for_recognizer(rgb, input_shape)
                emb = recognizer_session.run([out_name], {in_name: inp})[0].flatten()
                person_embs.append(emb)

            if not person_embs:
                logging.warning(f"  {person_name}: no usable images, skipped")
                continue

            total_images = len(person_embs)
            emb_array = np.array(person_embs, dtype=np.float32)

            # 1. Normalise each embedding
            norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
            emb_array = emb_array / np.where(norms > 0, norms, 1.0)

            # 2. Compute internal consistency (pairwise cosine similarities)
            n_dropped = 0
            peer_sims_all = None
            if len(emb_array) >= 4:
                sim_matrix = emb_array @ emb_array.T          # (N, N)
                np.fill_diagonal(sim_matrix, 0.0)
                peer_sims_all = sim_matrix.sum(axis=1) / (len(emb_array) - 1)

                # 3. Remove outliers: drop the bottom 20% by peer similarity
                cutoff = np.percentile(peer_sims_all, 20)
                mask = peer_sims_all >= cutoff
                n_dropped = int((~mask).sum())
                emb_array = emb_array[mask]
                if n_dropped:
                    logging.info(f"    dropped {n_dropped} outlier image(s) "
                                 f"(low peer-similarity)")

            # 4. Average only the clean embeddings
            avg = emb_array.mean(axis=0)

            # 5. Normalise the averaged vector
            norm = np.linalg.norm(avg)
            if norm > 0:
                avg /= norm

            # 6. Store quality stats: cohesion = mean pairwise sim of kept images,
            #    spread = std of pairwise sims (low spread = consistent captures)
            kept = len(emb_array)
            if kept >= 2:
                kept_sim_matrix = emb_array @ emb_array.T
                np.fill_diagonal(kept_sim_matrix, 0.0)
                kept_peer_sims = kept_sim_matrix.sum(axis=1) / (kept - 1)
                cohesion = float(np.mean(kept_peer_sims))
                spread   = float(np.std(kept_peer_sims))
            else:
                cohesion = 1.0
                spread   = 0.0

            quality_stats[person_name] = {
                "total_images": total_images,
                "kept_images":  kept,
                "outliers":     n_dropped,
                "cohesion":     round(cohesion, 4),  # higher = more consistent training set
                "spread":       round(spread, 4),    # lower  = less variance between images
            }

            new_embeddings.append(avg)
            new_names.append(person_name)
            logging.info(
                f"  {person_name}: {kept}/{total_images} images kept  "
                f"cohesion={cohesion:.3f}  spread={spread:.3f}"
            )

        if not new_embeddings:
            logging.info("No images found in dataset/.")
            return

        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump({
                "embeddings":    new_embeddings,
                "names":         new_names,
                "model_hash":    get_model_hash(),
                "quality_stats": quality_stats,
            }, f)

        for name, s in quality_stats.items():
            flag = " ⚠ LOW COHESION" if s["cohesion"] < 0.30 else ""
            logging.info(
                f"  {name}: {s['kept_images']}/{s['total_images']} kept, "
                f"outliers={s['outliers']}, cohesion={s['cohesion']:.3f}{flag}"
            )
        logging.info(f"Saved {len(new_embeddings)} person embeddings to {EMBEDDINGS_FILE}")
        reload_embeddings()


if __name__ == '__main__':
    retrain_and_save_embeddings()
