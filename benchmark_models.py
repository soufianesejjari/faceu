#!/usr/bin/env python3
"""
benchmark_models.py — FaceU Multi-Model Benchmark
===================================================
Reads config from .env, auto-downloads missing models, then runs a full
quality + latency comparison across all configured models.

Data sources:
  dataset/  — enrolled faces (ground truth: subfolder name = person identity)
  log/      — real captured face crops from the Pi camera (no ground truth)

Run:
    python benchmark_models.py
"""

import os
import sys
import time
import glob
import hashlib
import logging
import urllib.request
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

def _env(key: str, default):
    val = os.getenv(key, "")
    if val == "":
        return default
    try:
        return type(default)(val)
    except (ValueError, TypeError):
        return val

# ── Config from .env ───────────────────────────────────────────────────────────
MODELS_DIR          = _env("MODELS_DIR", "models")
DATASET_DIR         = _env("DATASET_DIR", "dataset")
LOG_DIR             = _env("LOG_DIR", "log")
BENCH_THRESHOLD     = _env("BENCH_THRESHOLD", 0.40)
BENCH_LATENCY_RUNS  = _env("BENCH_LATENCY_RUNS", 100)
BENCH_WINDOW_SEC    = _env("BENCH_WINDOW_SECONDS", 10.0)
CONSECUTIVE_FRAMES  = _env("CONSECUTIVE_FRAMES", 3)

# Models to benchmark: read from BENCH_MODELS env, split by comma
_bench_models_raw = os.getenv("BENCH_MODELS", "w600k_r50.onnx")
BENCH_MODEL_FILES  = [m.strip() for m in _bench_models_raw.split(",") if m.strip()]

# ── Model registry: filename → download URL + friendly label ──────────────────
# All URLs are official/verified direct links.
MODEL_REGISTRY = {
    "w600k_r50.onnx": {
        "label": "w600k_r50 (InsightFace R50)",
        "url":   "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "zip":   True,
        "zip_member": "w600k_r50.onnx",
        "size_hint": "~97MB zip",
    },
    "edgeface_s_gamma_05.onnx": {
        "label": "EdgeFace-S (γ=0.5)",
        "url":   "https://github.com/yakhyo/edgeface-onnx/releases/download/weights/edgeface_s_gamma_05.onnx",
        "zip":   False,
        "size_hint": "~14MB",
    },
    "edgeface_xs_gamma_06.onnx": {
        "label": "EdgeFace-XS (γ=0.6)",
        "url":   "https://github.com/yakhyo/edgeface-onnx/releases/download/weights/edgeface_xs_gamma_06.onnx",
        "zip":   False,
        "size_hint": "~7MB",
    },
    "edgeface_xxs.onnx": {
        "label": "EdgeFace-XXS",
        "url":   "https://github.com/yakhyo/edgeface-onnx/releases/download/weights/edgeface_xxs.onnx",
        "zip":   False,
        "size_hint": "~3MB",
    },
    "mobilefacenet_int8.onnx": {
        "label": "MobileFaceNet INT8",
        "url":   None,  # no single canonical URL; user must place manually
        "zip":   False,
        "size_hint": "~4MB (place manually in models/)",
    },
}

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("bench")

sys.path.insert(0, str(Path(__file__).parent))
from face_utils import FaceDetector, align_face


# ══════════════════════════════════════════════════════════════════════════════
#  Auto-download
# ══════════════════════════════════════════════════════════════════════════════

def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"\r    [{bar}] {pct:5.1f}%  {downloaded//1024//1024}MB", end="", flush=True)


def ensure_model(filename: str) -> bool:
    """Download model if not already present. Returns True if ready to use."""
    dest = os.path.join(MODELS_DIR, filename)
    if os.path.exists(dest):
        log.info(f"  ✓ {filename} already present")
        return True

    info = MODEL_REGISTRY.get(filename)
    if info is None:
        log.warning(f"  ✗ {filename} not in registry — place it manually in {MODELS_DIR}/")
        return False

    if info["url"] is None:
        log.warning(f"  ✗ {filename}: {info['size_hint']}")
        return False

    os.makedirs(MODELS_DIR, exist_ok=True)
    log.info(f"  ↓ Downloading {filename} ({info['size_hint']}) ...")

    if info["zip"]:
        import zipfile, tempfile
        tmp_zip = os.path.join(MODELS_DIR, "_tmp_download.zip")
        try:
            urllib.request.urlretrieve(info["url"], tmp_zip, _progress_hook)
            print()
            with zipfile.ZipFile(tmp_zip, "r") as z:
                # Find the target file inside zip (may be in a subfolder)
                target = info["zip_member"]
                matches = [n for n in z.namelist() if n.endswith(target)]
                if not matches:
                    log.error(f"    {target} not found in zip. Contents: {z.namelist()}")
                    return False
                with z.open(matches[0]) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
            log.info(f"    Extracted → {dest}")
        finally:
            if os.path.exists(tmp_zip):
                os.remove(tmp_zip)
    else:
        try:
            urllib.request.urlretrieve(info["url"], dest, _progress_hook)
            print()
            log.info(f"    Saved → {dest}")
        except Exception as e:
            log.error(f"    Download failed: {e}")
            if os.path.exists(dest):
                os.remove(dest)
            return False

    return os.path.exists(dest)


# ══════════════════════════════════════════════════════════════════════════════
#  ONNX session
# ══════════════════════════════════════════════════════════════════════════════

def load_session(model_path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
    opts.inter_op_num_threads = 1
    for providers in (
        ["XnnpackExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ):
        try:
            sess = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
            ep = sess.get_providers()[0]
            log.info(f"    Loaded [{ep}]")
            return sess
        except Exception:
            continue
    raise RuntimeError(f"Cannot load {model_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Preprocessing — detect → align → normalise
# ══════════════════════════════════════════════════════════════════════════════

_detector: FaceDetector | None = None

def get_detector() -> FaceDetector:
    global _detector
    if _detector is None:
        _detector = FaceDetector(model_selection=1, min_confidence=0.4, padding=0.10)
    return _detector


def preprocess(image_rgb: np.ndarray,
               input_size: tuple = (112, 112)) -> np.ndarray | None:
    detector = get_detector()
    detections = detector.detect(image_rgb)

    if detections:
        d = max(detections, key=lambda x: x.score)
        face_crop = image_rgb[d.y1:d.y2, d.x1:d.x2]
        if face_crop.size == 0:
            return None
        le = (d.left_eye[0] - d.x1, d.left_eye[1] - d.y1)
        re = (d.right_eye[0] - d.x1, d.right_eye[1] - d.y1)
        face_crop = align_face(face_crop, le, re)
    else:
        # Image may already be a tight crop (log/ images) — use as-is
        face_crop = image_rgb

    face = cv2.resize(face_crop, input_size).astype(np.float32)
    face = face * (2.0 / 255.0) - 1.0
    return np.expand_dims(np.transpose(face, (2, 0, 1)), 0)   # (1,3,H,W)


def embed(sess: ort.InferenceSession, inp: np.ndarray) -> np.ndarray:
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    emb = sess.run([out_name], {in_name: inp})[0].flatten().astype(np.float32)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def get_input_size(sess: ort.InferenceSession) -> tuple:
    shape = sess.get_inputs()[0].shape  # e.g. [1, 3, 112, 112]
    if len(shape) == 4:
        return (int(shape[3]), int(shape[2]))  # (W, H)
    return (112, 112)


# ══════════════════════════════════════════════════════════════════════════════
#  Latency benchmark
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_latency(sess: ort.InferenceSession,
                      input_size: tuple,
                      n_runs: int) -> dict:
    dummy = np.random.randn(1, 3, input_size[1], input_size[0]).astype(np.float32)
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    for _ in range(10):   # warmup
        sess.run([out_name], {in_name: dummy})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run([out_name], {in_name: dummy})
        times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000
    fps = 1000.0 / np.median(times_ms)
    frames_in_window = fps * BENCH_WINDOW_SEC
    vote_cycles = frames_in_window / max(CONSECUTIVE_FRAMES, 1)

    return {
        "median_ms":        float(np.median(times_ms)),
        "p95_ms":           float(np.percentile(times_ms, 95)),
        "fps":              float(fps),
        "frames_in_window": float(frames_in_window),
        "vote_cycles":      float(vote_cycles),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Gallery build from dataset/
# ══════════════════════════════════════════════════════════════════════════════

def build_gallery(sess: ort.InferenceSession,
                  input_size: tuple) -> tuple[np.ndarray, list]:
    embeddings, names = [], []
    if not os.path.isdir(DATASET_DIR):
        log.warning(f"    dataset/ not found at {os.path.abspath(DATASET_DIR)}")
        return np.empty((0, 512), dtype=np.float32), []

    for person in sorted(os.listdir(DATASET_DIR)):
        person_dir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_dir):
            continue
        imgs = glob.glob(os.path.join(person_dir, "*.jpg")) + \
               glob.glob(os.path.join(person_dir, "*.png"))
        count = 0
        for img_path in imgs:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            inp = preprocess(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), input_size)
            if inp is None:
                continue
            embeddings.append(embed(sess, inp))
            names.append(person)
            count += 1
        log.info(f"      {person}: {count} images enrolled")

    if not embeddings:
        return np.empty((0, 512), dtype=np.float32), []
    return np.array(embeddings, dtype=np.float32), names


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset evaluation (ground truth)
# ══════════════════════════════════════════════════════════════════════════════

def eval_dataset(sess, gallery_emb, gallery_names, input_size) -> dict:
    if not os.path.isdir(DATASET_DIR) or len(gallery_emb) == 0:
        return {}

    true_pos = false_neg = 0
    sims_ok, sims_fail = [], []
    per_person = {}

    for person in sorted(os.listdir(DATASET_DIR)):
        person_dir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_dir):
            continue
        imgs = glob.glob(os.path.join(person_dir, "*.jpg")) + \
               glob.glob(os.path.join(person_dir, "*.png"))
        per_person[person] = {"correct": 0, "total": 0, "mean_sim": 0.0, "sims": []}

        for img_path in imgs:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            inp = preprocess(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), input_size)
            if inp is None:
                continue
            emb  = embed(sess, inp)
            sims = np.dot(gallery_emb, emb)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            predicted = gallery_names[best_idx] if best_sim >= BENCH_THRESHOLD else "Unknown"
            per_person[person]["total"] += 1
            per_person[person]["sims"].append(best_sim)
            if predicted == person:
                true_pos += 1
                sims_ok.append(best_sim)
                per_person[person]["correct"] += 1
            else:
                false_neg += 1
                sims_fail.append(best_sim)

    for p, d in per_person.items():
        d["mean_sim"] = float(np.mean(d["sims"])) if d["sims"] else 0.0

    total = true_pos + false_neg
    return {
        "tar":              true_pos / total if total > 0 else 0.0,
        "true_pos":         true_pos,
        "false_neg":        false_neg,
        "total":            total,
        "mean_sim_correct": float(np.mean(sims_ok))   if sims_ok   else 0.0,
        "mean_sim_wrong":   float(np.mean(sims_fail))  if sims_fail else 0.0,
        "per_person":       per_person,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Log evaluation (real Pi captures — the "Unknown" problem)
# ══════════════════════════════════════════════════════════════════════════════

def eval_log(sess, gallery_emb, gallery_names, input_size) -> dict:
    if not os.path.isdir(LOG_DIR) or len(gallery_emb) == 0:
        return {}

    imgs = glob.glob(os.path.join(LOG_DIR, "face_*.jpg"))
    if not imgs:
        log.warning(f"    No face_*.jpg found in {LOG_DIR}/")
        return {}

    identified = 0
    unknown    = 0
    sims_all   = []
    id_names: dict[str, int] = {}

    for img_path in imgs:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        inp = preprocess(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), input_size)
        if inp is None:
            unknown += 1
            continue
        emb  = embed(sess, inp)
        sims = np.dot(gallery_emb, emb)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        sims_all.append(best_sim)

        if best_sim >= BENCH_THRESHOLD:
            name = gallery_names[best_idx]
            identified += 1
            id_names[name] = id_names.get(name, 0) + 1
        else:
            unknown += 1

    total = identified + unknown
    return {
        "total":       total,
        "identified":  identified,
        "unknown":     unknown,
        "id_rate":     identified / total if total > 0 else 0.0,
        "mean_sim":    float(np.mean(sims_all))          if sims_all else 0.0,
        "median_sim":  float(np.median(sims_all))        if sims_all else 0.0,
        "p10_sim":     float(np.percentile(sims_all, 10)) if sims_all else 0.0,
        "id_names":    id_names,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║         FaceU — Multi-Model Benchmark                       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nConfig:  threshold={BENCH_THRESHOLD}  window={BENCH_WINDOW_SEC}s"
          f"  consecutive_frames={CONSECUTIVE_FRAMES}  latency_runs={BENCH_LATENCY_RUNS}")
    print(f"Models:  {BENCH_MODEL_FILES}\n")

    # ── Step 1: ensure all models are downloaded ───────────────────────────────
    print("─── Downloading missing models ───────────────────────────────────")
    ready = []
    for filename in BENCH_MODEL_FILES:
        if ensure_model(filename):
            ready.append(filename)
    if not ready:
        print("\n✗ No models available. Exiting.")
        return

    # ── Step 2: run benchmark per model ───────────────────────────────────────
    results = []
    for filename in ready:
        model_path = os.path.join(MODELS_DIR, filename)
        info  = MODEL_REGISTRY.get(filename, {})
        label = info.get("label", filename)

        print(f"\n{'═'*64}")
        print(f"  MODEL: {label}")
        print(f"{'═'*64}")

        sess       = load_session(model_path)
        input_size = get_input_size(sess)
        print(f"  Input size: {input_size}")

        print("  ▶ Latency …")
        lat = benchmark_latency(sess, input_size, BENCH_LATENCY_RUNS)
        print(f"    Median {lat['median_ms']:.1f}ms | P95 {lat['p95_ms']:.1f}ms"
              f" | {lat['fps']:.0f} FPS")
        print(f"    In {BENCH_WINDOW_SEC}s window → ~{lat['frames_in_window']:.0f} frames"
              f" → {lat['vote_cycles']:.0f} majority-vote cycles")

        print("  ▶ Building gallery from dataset/ …")
        gallery_emb, gallery_names = build_gallery(sess, input_size)
        n_persons = len(set(gallery_names))
        print(f"    {n_persons} persons, {len(gallery_names)} vectors")

        if len(gallery_emb) == 0:
            log.warning("  Empty gallery — skipping evaluation")
            continue

        print("  ▶ Evaluating on dataset/ (ground truth) …")
        ds = eval_dataset(sess, gallery_emb, gallery_names, input_size)
        if ds:
            print(f"    TAR @ {BENCH_THRESHOLD}: {ds['tar']:.1%}"
                  f"  ({ds['true_pos']}/{ds['total']})")
            print(f"    Mean sim [correct]={ds['mean_sim_correct']:.4f}"
                  f"  [wrong]={ds['mean_sim_wrong']:.4f}")
            print("    Per person:")
            for person, pd in ds["per_person"].items():
                acc = pd["correct"] / pd["total"] if pd["total"] > 0 else 0
                print(f"      {person:<20}  {acc:.0%}  mean_sim={pd['mean_sim']:.4f}"
                      f"  ({pd['correct']}/{pd['total']})")

        print("  ▶ Evaluating on log/ (real Pi captures) …")
        lg = eval_log(sess, gallery_emb, gallery_names, input_size)
        if lg:
            print(f"    Total images : {lg['total']}")
            print(f"    Identified   : {lg['identified']}  ({lg['id_rate']:.1%})")
            print(f"    Unknown      : {lg['unknown']}")
            print(f"    Median sim   : {lg['median_sim']:.4f}  "
                  f"(P10={lg['p10_sim']:.4f}, mean={lg['mean_sim']:.4f})")
            if lg["id_names"]:
                breakdown = ", ".join(f"{n}×{c}"
                    for n, c in sorted(lg["id_names"].items(), key=lambda x: -x[1]))
                print(f"    Identified as: {breakdown}")

        results.append({"label": label, "lat": lat, "ds": ds, "lg": lg})

    # ── Summary table ──────────────────────────────────────────────────────────
    if not results:
        return

    W = 94
    print(f"\n{'═'*W}")
    print("  SUMMARY")
    print(f"{'═'*W}")
    hdr = (f"{'Model':<26} {'Lat(ms)':>8} {'FPS':>6} {'Frames/win':>11}"
           f" {'VoteCycles':>11} {'TAR(ds)':>8} {'ID%(log)':>9} {'MedSim(log)':>12}")
    print(hdr)
    print(f"{'─'*W}")
    for r in results:
        l, ds, lg = r["lat"], r.get("ds", {}), r.get("lg", {})
        print(
            f"{r['label']:<26}"
            f" {l['median_ms']:>8.1f}"
            f" {l['fps']:>6.0f}"
            f" {l['frames_in_window']:>11.0f}"
            f" {l['vote_cycles']:>11.0f}"
            f" {ds.get('tar', 0):>8.1%}"
            f" {lg.get('id_rate', 0):>9.1%}"
            f" {lg.get('median_sim', 0):>12.4f}"
        )
    print(f"{'═'*W}")
    print(f"\nNote: FPS = recognition-only throughput. Faster model → more vote cycles")
    print(f"      in the {BENCH_WINDOW_SEC}s window → better majority-vote accuracy.")
    print(f"\nDecision guide:")
    print(f"  TAR ≥ 95% AND ID% ≥ 80%  → use as primary model")
    print(f"  TAR 88–95%               → use as primary + w600k as verifier for uncertain hits")
    print(f"  TAR < 88%                → discard this model\n")

    # Save summary
    summary_path = "benchmark_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"FaceU Benchmark — threshold={BENCH_THRESHOLD} window={BENCH_WINDOW_SEC}s\n\n")
        f.write(f"{'Model':<26} {'Lat(ms)':>8} {'FPS':>6} {'Frames/win':>11}"
                f" {'VoteCycles':>11} {'TAR(ds)':>8} {'ID%(log)':>9} {'MedSim(log)':>12}\n")
        for r in results:
            l, ds, lg = r["lat"], r.get("ds", {}), r.get("lg", {})
            f.write(
                f"{r['label']:<26}"
                f" {l['median_ms']:>8.1f}"
                f" {l['fps']:>6.0f}"
                f" {l['frames_in_window']:>11.0f}"
                f" {l['vote_cycles']:>11.0f}"
                f" {ds.get('tar', 0):>8.1%}"
                f" {lg.get('id_rate', 0):>9.1%}"
                f" {lg.get('median_sim', 0):>12.4f}\n"
            )
    print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
