# FaceU — Facial Recognition Entry/Exit Detection System

FaceU is a real-time entry/exit detection system that uses facial recognition to monitor and log people entering and leaving a monitored area. It combines multi-pose face capture for enrollment, MediaPipe Face Detection, an ArcFace recognition model, and a SORT-style multi-object tracker to identify individuals and record directional crossing events against user-defined virtual lines or camera-type designations.

The system exposes a Flask-based web interface for camera management, face enrollment with pose-guided capture, live video feeds with overlaid detections, a paginated entry/exit log with snapshot review, and advanced tunable parameters for recognition, tracking, and motion detection.

![Overview](./assets/1.png)
![Camera](./assets/2.png)

## How It Works

FaceU operates in a two-phase pipeline per camera feed:

1. **Motion Detection Phase** — The system runs a lightweight frame-differencing motion detector at a low configurable FPS (default: 2 fps). When the movement score exceeds a threshold, it transitions to the face detection phase.

2. **Face Detection & Recognition Phase** — MediaPipe Face Detection (single-pass, returns bbox + 5 keypoints) locates faces and provides eye landmarks for alignment. The SORT tracker assigns a persistent track ID to each face across frames using IoU matching with velocity-based prediction — the tracker extrapolates each face's position forward using a smoothed velocity estimate, so fast-moving subjects maintain their ID instead of being assigned a new one. Face crops are aligned using eye-landmark rotation, assessed for quality by the FIAQ module, and submitted asynchronously to a pool of multiprocessing recognition workers. Each worker runs the ArcFace model (w600k_r50) to produce a 512-dimensional embedding, compared against known per-person embeddings via cosine similarity. The last 5 recognition results per track are majority-voted to determine the final identity, making the system robust to single bad-angle frames. The best match above the configured threshold (default: 0.5) identifies the person; otherwise the face is labeled "Unknown".

Entry and exit events are detected by tracking whether a face's center point crosses a user-defined virtual line (horizontal or vertical) in a specific direction, or by designating the entire camera as an "entry-only" or "exit-only" source. Events are persisted to an SQLite database along with full-frame snapshots.

---

## Architecture Overview

```
                          ┌──────────────────────────────────────┐
                          │              Web Server              │
                          │  (Camera config, feeds, enrollment,  │
                          │   log UI, face capture, datasets)    │
                          └──────────────────┬───────────────────┘
                                             │
                  ┌──────────────────────────┼──────────────────────────┐
                  │                          │                          │
                  ▼                          ▼                          ▼
          ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
          │  Camera 1    │          │  Camera 2    │          │  Camera N    │
          │  (Thread)    │          │  (Thread)    │          │  (Thread)    │
          └──────┬───────┘          └──────┬───────┘          └──────┬───────┘
                 │                         │                         │
                 ▼                         ▼                         ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                     FaceRecognizer (one per camera)                     │
  │                                                                         │
  │   ┌────────────┐    ┌────────────┐    ┌────────────┐                    │
  │   │  Motion    │───▶│   Face     │───▶│   SORT     │                    │
  │   │  Detection │    │  Detection │    │  Tracker   │                    │
  │   └────────────┘    └─────┬──────┘    └──────┬─────┘                    │
  │                           │                  │                          │
  │                           ▼                  ▼                          │
  │                   ┌────────────┐      ┌──────────────┐                  │
  │                   │   FIAQ     │      │  Entry/Exit  │                  │
  │                   │  Quality   │      │  Line Logic  │                  │
  │                   │  Check     │      │              │                  │
  │                   └──────┬─────┘      └──────┬───────┘                  │
  │                          │                   │                          │
  └──────────────────────────┼───────────────────┼──────────────────────────┘
                             │                   │
                             ▼                   ▼
                 ┌────────────────────┐  ┌────────────────────────┐
                 │  Recognition       │  │  EntryExitPersistence  │
                 │  Worker Pool       │  │  Thread                │
                 │  (multiprocessing) │  │  (SQLite + snapshots)  │
                 └────────────────────┘  └────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.8+
- A webcam, Raspberry Pi Camera Module, or RTSP-capable IP camera
- (Optional) NVIDIA GPU with CUDA for faster ONNX inference — the default configuration runs on CPU

### Step 1 — Clone the repository

```bash
git clone https://github.com/m1tk/faceu.git
cd faceu
```

### Step 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Download the recognition model

FaceU uses the **ArcFace w600k_r50** model (`w600k_r50.onnx`) for face embedding generation. Place it in the `models/` directory:

```bash
mkdir -p models
# Download the model from the InsightFace model zoo
# See: https://github.com/deepinsight/insightface/tree/master/model_zoo
# And: https://github.com/SthPhoenix/InsightFace-REST
# The file should be placed at: models/w600k_r50.onnx
```

### Step 4 — Run the server

```bash
python webserver.py
```

The web interface will be available at **http://0.0.0.0:8080**.

---

## Usage Guide

### Adding a Camera

Open the web interface at `http://<host>:8080/` to add new camera(s). Entry and exit lines can also be configred for each camera.

### Enrolling a Face

Face enrollment captures 10 images of a person at different head poses.

A user can enroll their face by accessing the URL:
```
http://<host>:8080/face_capture/<user_id>
```

where `user_id` is any valid unique username.


### Tuning Advanced Parameters

Open **Advanced Settings** for a camera to fine-tune:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recognition_threshold` | 0.5 | Cosine similarity threshold for accepting a match. Raise to reduce false positives; lower if known people are missed. |
| `min_face_size` | 50 px | Minimum bounding box dimension to consider a face valid. |
| `movement_threshold` | 3000 | Pixel-change count to trigger transition from motion to face detection. |
| `motion_fps` | 2 | FPS during motion detection phase (power-saving). |
| `no_face_timeout` | 5.0 s | Seconds without a face before reverting to motion detection. |
| `consecutive_frames` | 3 | Minimum quality-passing frames before first recognition submission. |
| `fiaq_threshold` | 0.4 | Minimum face quality score (0–1) to accept a crop for recognition. |
| `tracker_max_age` | 8 | Frames a track survives without a matching detection. At 5 fps this is ~1.6 s — increase if tracks break during brief occlusions. |
| `tracker_min_hits` | 1 | Minimum detections before a track is reported. |
| `tracker_iou_threshold` | 0.2 | IoU threshold for associating detections with existing tracks. |
| `detector_conf_thresh` | 0.6 | MediaPipe Face Detection minimum confidence. |

### API & Webhooks

FaceU exposes a REST API to query attendance logs and a webhook system to automatically push entry/exit events to external services in real-time.

**1. Configuration**
In your `.env` file, configure the following:
```env
API_KEY=your_secret_api_key_here
WEBHOOK_URL=http://your-external-service.com/webhook
WEBHOOK_API_KEY=your_webhook_api_key_here
```

**2. GET Attendance API**
Retrieve the entry/exit log for known users.
*   **Endpoint:** `GET /api/attendance`
*   **Authentication:** Pass your API key via the `x-api-key` header or as a query parameter (`?api_key=your_secret_api_key_here`).
*   **Parameters:** `date` (Optional, format: `YYYY-MM-DD`) - filters the results for a specific day.
*   **Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "user": "Soufiane",
      "direction": "entered",
      "timestamp": 1684345200.0,
      "datetime": "2023-05-17T17:40:00"
    }
  ]
}
```

**3. Webhook Integration**
When a known person's entry/exit event is fully resolved, FaceU immediately `POST`s the event data to your configured `WEBHOOK_URL`. The webhook fires exactly once per event — purely event-driven, with no background polling or historical backfill.

An event is considered resolved when **both** conditions are met:
- The person has been identified (cosine similarity ≥ threshold)
- A line-crossing direction has been recorded (`entered` or `exited`)

Whichever condition is satisfied last triggers the delivery. Events present in the database before the server started are never resent.

*   **Headers sent:**
    *   `Content-Type: application/json`
    *   `x-api-key: <WEBHOOK_API_KEY>` (if configured)
*   **Payload sent:**
```json
{
  "id": 1,
  "user": "Soufiane",
  "action": "entered",
  "timestamp": 1684345200.0
}
```
If delivery fails (network error or non-2xx response), the event remains in the database with `synced = 0` for manual inspection. Unknown persons and events without a direction are never sent.

## Recent Updates

### Recognition Accuracy

- **Per-person averaged embeddings** — Training no longer stores one vector per image. All images for a person are embedded, outliers are filtered using mutual consistency (each image's mean cosine similarity to its peers), and the remaining clean embeddings are averaged into one robust vector per person. This eliminates the false-positive problem where a single outlier training image could pull a wrong identity.
- **Mutual consistency outlier filter** — During training, the bottom 20% of images by peer similarity are automatically dropped before averaging. An accidentally-captured frame of the wrong person or a heavily-blurred image no longer contaminates the embedding.
- **Training quality stats** — `train.py` logs and stores per-person cohesion and spread scores. The Face Datasets page displays a quality badge (Good / Fair / Weak) for each enrolled person so you can identify who needs re-enrollment.
- **Multi-frame majority voting** — The last 5 recognition results per track are voted before showing an identity. One bad-angle frame returning "Unknown" no longer wipes a correctly-identified person.
- **Per-camera threshold now works** — The `recognition_threshold` Advanced setting was previously ignored (the worker hardcoded 0.4). The threshold is now applied in the camera pipeline so the slider in Advanced Settings takes effect.
- **Default threshold raised to 0.5** — The previous default of 0.4 caused a high false-accept rate. 0.5 is the recommended open-set boundary for w600k_r50.

### Tracking Stability

- **Velocity-based prediction** — `KalmanBoxTracker` now tracks a smoothed velocity (exponential moving average). `predict()` extrapolates the bounding box forward by `velocity × missed_frames` instead of returning the last known position. Walking subjects maintain their track ID across frames instead of being re-assigned on movement.
- **`tracker_max_age` default raised to 8** — At 5 fps this gives ~1.6 s of tolerance for brief detection failures (face turns sideways, lighting change).

### Webhook

- **Event-driven delivery** — The webhook fires immediately when an event becomes fully resolved (known identity + real direction). There is no background polling loop.
- **No historical backfill** — Events already in the database when the server starts are marked synced and never sent. Only events created in the current live session trigger a delivery.
- **Precise trigger** — The webhook fires from `_log_entry_exit_event` (if identity was already known) or from `_update_user` (if the line crossing happened first) — whichever completes the event last.

### Infrastructure

- **dlib / face_recognition removed** — All dlib dependencies eliminated. Detection and alignment use MediaPipe Face Detection exclusively.
- **MediaPipe Face Detection** replaces MediaPipe Pose — single-pass, returns bbox + 5 keypoints (including eye positions for alignment) with lower CPU cost.
- **ONNX Runtime optimised for ARM** — XNNPACK execution provider enabled on Raspberry Pi 5 with `ORT_ENABLE_ALL` graph optimisations and half-core thread allocation.
- **HTTPS support** — The server reads `cert.pem` / `key.pem` from the project directory and serves over TLS.

---

## Future Work

### Better Tracking Algorithm

- **DeepSORT integration**: Augmenting the IoU-based assignment with appearance descriptors (e.g., the face embeddings already computed by the system) would dramatically reduce ID switches. The DeepSORT paper [1] shows that combining motion and appearance cues improves MOTA by 10–20% on standard benchmarks.
- **ByteTrack**: The ByteTrack algorithm [2] associates high-confidence detections first, then rescues low-confidence ones, achieving state-of-the-art performance on intermittent detections.

### Multi-Camera Object Tracking

- **Global re-identification**: Using face embeddings as a person re-ID feature across cameras to track an individual's path through a building with multiple entry/exit points.
- **Cross-camera entry/exit correlation**: An "entered" event on Camera A followed by an "exited" event on Camera B could be automatically linked.

### Liveness Detection

- **Anti-spoofing**: Integrating a liveness detection module (e.g., SilentFace [5]) would prevent photo and video replay attacks, which is critical for security applications.

### Production Hardening

- **WSGI server**: Replace the Flask development server with Gunicorn or uWSGI for production deployment.
- **Authentication & authorization**: Add user authentication, role-based access control, and API token management.
- **Database migration**: Move from SQLite to PostgreSQL for concurrent access and better scalability.
- **Containerization**: Provide a Dockerfile and docker-compose.yml for reproducible deployment.
- **GPU support**: Add optional CUDA execution provider for ONNX Runtime to leverage NVIDIA GPUs.

### Enhanced Enrollment

- **Quality-gated enrollment**: Enforce FIAQ quality thresholds during capture, rejecting low-quality images before they enter the dataset.
- **Template update**: Periodically update a user's embeddings using high-quality recognition-time captures, allowing the system to adapt to appearance changes over time.

---

## References

1. **DeepSORT** — Wojke, N., Bewley, A., & Paulus, D. (2017). *Simple Online and Realtime Tracking with a Deep Association Metric.* IEEE International Conference on Image Processing (ICIP). [arXiv:1703.07402](https://arxiv.org/abs/1703.07402)

2. **ByteTrack** — Zhang, Y., Sun, P., Jiang, Y., et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* European Conference on Computer Vision (ECCV). [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)

3. **SCRFD** — Guo, J., Zhu, X., Lei, Z., & Li, S. Z. (2021). *Sample and Computation Redistribution for Efficient Face Detection.* arXiv. [arXiv:2105.04714](https://arxiv.org/abs/2105.04714)

4. **RetinaFace** — Deng, J., Guo, J., Ververas, E., Kotsia, I., & Zafeiriou, S. (2020). *RetinaFace: Single-shot Multi-level Face Localisation in the Wild.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). [arXiv:1905.00641](https://arxiv.org/abs/1905.00641)

5. **SilentFace Anti-Spoofing** — Liu, H., Li, Y., & Cao, J. (2020). *A Study on Face Presentation Attack Detection Using Adaptive Pixel-Level Feature Extraction.* arXiv. [GitHub](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)

6. **ArcFace** — Deng, J., Guo, J., & Zafeiriou, S. (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). [arXiv:1801.07698](https://arxiv.org/abs/1801.07698)

7. **SORT** — Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). *Simple Online and Realtime Tracking.* IEEE International Conference on Image Processing (ICIP). [arXiv:1602.00763](https://arxiv.org/abs/1602.00763)

8. **MediaPipe** — Lugaresi, C., et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines.* arXiv. [arXiv:1906.08172](https://arxiv.org/abs/1906.08172)

9. **InsightFace** — Deng, J., Guo, J., et al. (2022). *InsightFace: 2D and 3D Face Analysis Project.* [GitHub](https://github.com/deepinsight/insightface)

10. **Chokepoint Dataset** — ARMA. *Chokepoint: A Dataset for People Flow Counting and Re-identification.* [https://arma.sourceforge.net/chokepoint/](https://arma.sourceforge.net/chokepoint/) — Used for demo.
