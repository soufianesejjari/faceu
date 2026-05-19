import os
import pickle
import hashlib
import threading
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass, field
from typing import Optional, Any, List


@dataclass
class SharedState:
    recognizer_session: Optional[Any] = None
    recognizer_input_name: Optional[str] = None
    recognizer_output_name: Optional[str] = None
    known_embeddings: Optional[np.ndarray] = None
    known_names: Optional[List[str]] = None
    embeddings_file: Optional[str] = None
    model_dir: Optional[str] = None
    model_hash: Optional[str] = None


embeddings_lock = threading.Lock()
shared_state = SharedState()


def _model_hash(model_path: str) -> str:
    h = hashlib.md5()
    with open(model_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


def _create_session(model_path: str):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Leave half the cores for capture/tracking; recognition workers run in parallel anyway.
    sess_options.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
    sess_options.inter_op_num_threads = 1

    # Prefer XNNPACK on ARM (Pi 5); fall back silently to default CPU MLAS.
    for providers in (["XnnpackExecutionProvider", "CPUExecutionProvider"],
                      ["CPUExecutionProvider"]):
        try:
            session = ort.InferenceSession(model_path, sess_options=sess_options,
                                           providers=providers)
            used = session.get_providers()
            if "XnnpackExecutionProvider" in used:
                import logging
                logging.info("ONNX Runtime using XnnpackExecutionProvider")
            return session
        except Exception:
            continue
    raise RuntimeError(f"Could not load ONNX session for {model_path}")


def initialize_shared(model_dir: str, embeddings_file: str,
                      recognizer_model_name: str = 'w600k_r50.onnx'):
    if shared_state.recognizer_session is None:
        model_path = os.path.join(model_dir, recognizer_model_name)
        shared_state.recognizer_session = _create_session(model_path)
        shared_state.recognizer_input_name = shared_state.recognizer_session.get_inputs()[0].name
        shared_state.recognizer_output_name = shared_state.recognizer_session.get_outputs()[0].name
        shared_state.model_dir = model_dir
        shared_state.embeddings_file = embeddings_file
        shared_state.model_hash = _model_hash(model_path)
    reload_embeddings()


def reload_embeddings():
    with embeddings_lock:
        embeddings_file = shared_state.embeddings_file
        if not os.path.exists(embeddings_file):
            return
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)

        saved_hash = data.get('model_hash')
        if saved_hash and saved_hash != shared_state.model_hash:
            import logging
            logging.warning(
                "Embeddings were generated with a different model "
                f"(saved={saved_hash}, current={shared_state.model_hash}). "
                "Re-run train.py to regenerate embeddings."
            )

        shared_state.known_embeddings = np.array([
            emb / np.linalg.norm(emb) if np.linalg.norm(emb) > 0 else emb
            for emb in data['embeddings']
        ], dtype=np.float32)
        shared_state.known_names = data['names']


def get_recognizer():
    return (shared_state.recognizer_session,
            shared_state.recognizer_input_name,
            shared_state.recognizer_output_name)


def get_embeddings():
    return shared_state.known_embeddings, shared_state.known_names


def get_model_hash() -> Optional[str]:
    return shared_state.model_hash
