import numpy as np
import multiprocessing as mp
import os

from init import get_recognizer, get_embeddings, reload_embeddings


try:
    mp.set_start_method("fork")
except RuntimeError:
    pass

_RELOAD_SENTINEL = "__RELOAD__"


def _recognition_worker(input_queue, output_queue):
    recognizer_session, in_name, out_name = get_recognizer()
    known_embeddings, known_names = get_embeddings()

    while True:
        item = input_queue.get()
        if item is None:
            break

        job_id, face_rgb = item

        if job_id == _RELOAD_SENTINEL:
            reload_embeddings()
            known_embeddings, known_names = get_embeddings()
            continue

        # face_rgb is already 112×112 RGB uint8 (aligned, quality-checked)
        face = np.ascontiguousarray(face_rgb, dtype=np.float32)
        face = face * (2.0 / 255.0) - 1.0
        recognizer_input = np.expand_dims(np.transpose(face, (2, 0, 1)), axis=0)

        curr_emb = recognizer_session.run([out_name], {in_name: recognizer_input})[0].flatten()
        norm = np.linalg.norm(curr_emb)
        if norm > 0:
            curr_emb /= norm

        if known_embeddings is not None and len(known_embeddings) > 0:
            similarities = np.dot(known_embeddings, curr_emb)
            
            # Group similarities by person name
            person_scores = {}
            for name, sim in zip(known_names, similarities):
                person_scores.setdefault(name, []).append(float(sim))
            
            # Compute consolidated score for each person using Top-K average (K=3)
            consolidated_scores = {}
            K = 3
            for name, sims in person_scores.items():
                sims_sorted = sorted(sims, reverse=True)
                k_eff = min(len(sims_sorted), K)
                consolidated_scores[name] = sum(sims_sorted[:k_eff]) / k_eff
            
            # Find the best match among all consolidated scores
            best_name = max(consolidated_scores, key=consolidated_scores.get)
            best_sim = consolidated_scores[best_name]
            name = best_name
        else:
            best_sim = 0.0
            name = "Unknown"

        output_queue.put((job_id, name, best_sim))


class FaceRecognitionWorker:
    def __init__(self, num_workers=None):
        if num_workers is None:
            # On a Pi 5 (4 cores) use 2 workers: leaves headroom for capture+tracking.
            num_workers = max(1, min(2, (os.cpu_count() or 2) // 2))
        self.num_workers = num_workers
        self.input_queue = mp.Queue(maxsize=num_workers * 4)  # bounded back-pressure
        self.output_queue = mp.Queue()
        self.procs = [
            mp.Process(target=_recognition_worker,
                       args=(self.input_queue, self.output_queue), daemon=True)
            for _ in range(self.num_workers)
        ]
        for p in self.procs:
            p.start()

    def recognize_async(self, job_id, face_rgb):
        try:
            self.input_queue.put_nowait((job_id, np.ascontiguousarray(face_rgb)))
        except Exception:
            pass  # queue full — drop this frame, a better one will come

    def get_result(self, block=False, timeout=None):
        try:
            return self.output_queue.get(block=block, timeout=timeout)
        except Exception:
            return None

    def reload_embeddings(self):
        """Signal all workers to reload embeddings from disk."""
        for _ in range(self.num_workers):
            try:
                self.input_queue.put_nowait((_RELOAD_SENTINEL, None))
            except Exception:
                pass

    def close(self):
        for _ in range(self.num_workers):
            self.input_queue.put(None)
        for p in self.procs:
            p.join(timeout=5)
