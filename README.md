# FaceU — Facial Recognition Entry/Exit Detection System

FaceU is a real-time entry/exit detection system that uses facial recognition to monitor and log people entering and leaving a monitored area. It combines multi-pose face capture for enrollment, MediaPipe-based face detection, an ArcFace recognition model, and a SORT-style multi-object tracker to identify individuals and record directional crossing events against user-defined virtual lines or camera-type designations.

The system exposes a Flask-based web interface for camera management, face enrollment with pose-guided capture, live video feeds with overlaid detections, a paginated entry/exit log with snapshot review, and advanced tunable parameters for recognition, tracking, and motion detection.

![Overview](./assets/1.png)
![Camera](./assets/2.png)

## How It Works

FaceU operates in a two-phase pipeline per camera feed:

1. **Motion Detection Phase** — The system runs a lightweight frame-differencing motion detector at a low configurable FPS (default: 2 fps). When the movement score exceeds a threshold, it transitions to the face detection phase.

2. **Face Detection & Recognition Phase** — MediaPipe Pose is used to extract facial landmarks (nose, eyes, mouth corners) and derive a face bounding box with generous padding. The SORT (Simple Online and Realtime Tracking) algorithm assigns a persistent track ID to each detected face across frames. Face crops are aligned using eye-landmark-based rotation, assessed for quality using the FIAQ (Face Image Assessment Quality) module, and submitted asynchronously to a pool of multiprocessing recognition workers. Each worker runs the ArcFace model to produce a 512-dimensional embedding, which is compared against known embeddings via cosine similarity. The best match above a threshold (default: 0.4) identifies the person; otherwise the face is labeled "Unknown".

Entry and exit events are detected by tracking whether a face's center point crosses a user-defined virtual line (horizontal or vertical) in a specific direction, or by designating the entire camera as an "entry-only" or "exit-only" source. Events are persisted to an SQLite database along with raw frame snapshots.

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
| `recognition_threshold` | 0.4 | Cosine similarity threshold for face matching. Lower = stricter. |
| `min_face_size` | 50 px | Minimum bounding box dimension to consider a face valid. |
| `movement_threshold` | 3000 | Pixel-change count to trigger transition from motion to face detection. |
| `motion_fps` | 2 | FPS during motion detection phase (power-saving). |
| `no_face_timeout` | 5.0 s | Seconds without a face before reverting to motion detection. |
| `consecutive_frames` | 3 | Number of face crops submitted for majority-vote recognition. |
| `fiaq_threshold` | 0.4 | Minimum face quality score (0–1) to accept a crop for recognition. |
| `tracker_max_age` | 3 | Frames a track survives without a matching detection. |
| `tracker_min_hits` | 1 | Minimum detections before a track is reported. |
| `tracker_iou_threshold` | 0.2 | IoU threshold for associating detections with existing tracks. |
| `detector_conf_thresh` | 0.6 | MediaPipe Pose minimum detection/tracking confidence. |

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
When a known person's entry/exit event is fully resolved, FaceU will automatically `POST` the event data to your configured `WEBHOOK_URL`.
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
If the webhook endpoint is unreachable or returns an error, FaceU safely queues the event locally (via SQLite `synced = 0`) and will automatically retry delivery in the background until a `200`, `201`, or `204` success status is received.

## Future Work

### Better Tracking Algorithm

The current SORT implementation is minimal and IOU-only. Significant improvements could be achieved by:

- **DeepSORT integration**: Augmenting the IoU-based assignment with appearance descriptors (e.g., the face embeddings already computed by the system) would dramatically reduce ID switches. The DeepSORT paper [1] shows that combining motion and appearance cues improves MOTA (Multiple Object Tracking Accuracy) by 10–20% on standard benchmarks.
- **Proper Kalman Filter**: Implementing full Kalman filter state prediction (position + velocity) would allow the tracker to predict positions during brief occlusions, reducing track fragmentation.
- **ByteTrack**: The ByteTrack algorithm [2] associates high-confidence detections first, then rescues low-confidence ones, achieving state-of-the-art performance. This would be particularly useful for the motion-detection phase where detections may be intermittent.

### Multi-Camera Object Tracking

The current system treats each camera as an independent silo. Cross-camera tracking would enable:

- **Global re-identification**: Using face embeddings as a person re-ID feature across cameras, allowing the system to track an individual's path through a building with multiple entry/exit points.
- **Overlapping camera fusion**: When two cameras have overlapping fields of view, detections from both can be merged to reduce occlusion-related misses and improve 3D position estimation.
- **Cross-camera entry/exit correlation**: An "entered" event on Camera A followed by an "exited" event on Camera B could be automatically linked, providing a complete movement trace.

### Improved Face Detection

- **Dedicated face detector**: Replacing MediaPipe Pose with a purpose-built face detector like SCRFD [3] or RetinaFace [4] would reduce computational overhead by 3–5x and improve detection accuracy, especially for small or partially occluded faces.
- **Face detection + tracking decoupling**: Running face detection at a lower frequency (e.g., every 3–5 frames) and relying on the tracker to interpolate between detections would reduce CPU usage while maintaining temporal coherence.

### Liveness Detection

- **Anti-spoofing**: Integrating a liveness detection module (e.g., SilentFace [5] or a challenge-response approach) would prevent photo and video replay attacks, which is critical for security applications.

### Production Hardening

- **WSGI server**: Replace the Flask development server with Gunicorn or uWSGI for production deployment.
- **Authentication & authorization**: Add user authentication, role-based access control, and API token management.
- **Database migration**: Move from SQLite to PostgreSQL for concurrent access and better scalability.
- **Containerization**: Provide a Dockerfile and docker-compose.yml for reproducible deployment.
- **GPU support**: Add optional CUDA execution provider for ONNX Runtime to leverage NVIDIA GPUs.
- **Incremental embedding updates**: Instead of regenerating the entire embeddings pickle on each enrollment, append new embeddings incrementally and rebuild periodically.

### Enhanced Enrollment

- **Quality-gated enrollment**: Enforce FIAQ quality thresholds during the capture process, rejecting low-quality images before they enter the dataset.
- **Adaptive pose selection**: Instead of fixed yaw/pitch angles, dynamically determine the most informative poses based on the user's face geometry.
- **Template update**: Periodically update a user's embeddings using high-quality recognition-time captures, allowing the system to adapt to aging, hairstyle changes, and other gradual appearance variations.

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

10. **Face Recognition (Python library)** — Geitgey, A. (2017). *face_recognition: Recognize and manipulate faces from Python.* [GitHub](https://github.com/ageitgey/face_recognition)
11. **Chokepoint Dataset** — ARMA. *Chokepoint: A Dataset for People Flow Counting and Re-identification.* [https://arma.sourceforge.net/chokepoint/](https://arma.sourceforge.net/chokepoint/) — Used for demo.
