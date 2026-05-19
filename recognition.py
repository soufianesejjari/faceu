import cv2
import numpy as np
import time
import fiaq
from face_utils import FaceDetector, align_face
from init import get_recognizer, get_embeddings
from tracker import FaceSortTracker


class FaceRecognizer:
    def __init__(self, settings, log_store, recognition_worker):
        self.settings = settings
        self.fps = settings.get('fps', 5)
        self.recognizer_input_shape = tuple(settings.get('recognizer_input_shape', (112, 112)))
        self.tracker = FaceSortTracker(
            max_age=settings.get('tracker_max_age', 5),
            min_hits=settings.get('tracker_min_hits', 1),
            iou_threshold=settings.get('tracker_iou_threshold', 0.15)
        )
        self.track_id_to_face = {}
        self.prev_frame_gray = None
        self.current_state = 1  # 1: DETECTING_MOVEMENT, 2: DETECTING_FACES
        self.last_face_detected_time = None
        self.last_motion_check_time = 0
        self.entry_exit_persistence = log_store
        self.recognition_worker = recognition_worker
        # job_id → track_id mapping for async result collection
        self._pending_jobs = {}

        self.motion_detection_interval = settings.get('motion_detection_interval', 0.5)
        self.motion_fps = settings.get('motion_fps', 2)
        self.movement_threshold = settings.get('movement_threshold', 1500)
        self.no_face_timeout = settings.get('no_face_timeout', 10.0)
        self.consecutive_frames = settings.get('consecutive_frames', 3)
        self.recognition_threshold = settings.get('recognition_threshold', 0.4)
        self.fiaq_threshold = settings.get('fiaq_threshold', 0.4)
        self.min_face_size = settings.get('min_face_size', 50)
        self.movement_blur_size = tuple(settings.get('movement_blur_size', (7, 7)))

        self.frame_time = 1.0 / max(self.fps, 1)
        self.motion_frame_time = 1.0 / max(self.motion_fps, 1)

        self._recognizer = get_recognizer()
        self._embeddings = get_embeddings()

        self.face_detector = FaceDetector(
            model_selection=1,
            min_confidence=settings.get('detector_conf_thresh', 0.5),
            padding=0.10
        )

        self.detection_mode = settings.get('detection_mode', 'line')
        self.camera_type = settings.get('camera_type', 'entry')

        self.recognizer_input_w = self.recognizer_input_shape[1]
        self.recognizer_input_h = self.recognizer_input_shape[0]

        self._fps_last_time = 0.0
        self._fps_counter = 0
        self._fps_value = 0.0

    def process_frame(self, frame, now):
        if self._fps_last_time == 0.0:
            self._fps_last_time = now
        self._fps_counter += 1
        if now - self._fps_last_time >= 1.0:
            self._fps_value = self._fps_counter / (now - self._fps_last_time)
            self._fps_last_time = now
            self._fps_counter = 0

        # Drain completed recognition results every frame (no-block)
        self._collect_recognition_results()

        if self.current_state == 1:
            if now - self.last_motion_check_time >= self.motion_detection_interval:
                gray, movement_score = self.detect_movement(frame, self.prev_frame_gray)
                self.last_motion_check_time = now
                if self.prev_frame_gray is not None and movement_score > self.movement_threshold:
                    self.current_state = 2
                    self.last_face_detected_time = now
                    self.track_id_to_face.clear()
                self.prev_frame_gray = gray
            cv2.putText(frame, "Mode: Detecting Movement", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"FPS: {self._fps_value:.1f}", (frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            self.track_id_to_face, self.last_face_detected_time = self.detect_faces(
                frame, now, self.track_id_to_face, self.last_face_detected_time
            )
            if self.last_face_detected_time and now - self.last_face_detected_time > self.no_face_timeout:
                self.current_state = 1
                self.prev_frame_gray = None
            cv2.putText(frame, "Mode: Detecting Faces", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Tracked: {len(self.track_id_to_face)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if self.last_face_detected_time:
                remaining = max(0, self.no_face_timeout - (now - self.last_face_detected_time))
                cv2.putText(frame, f"Timeout: {remaining:.1f}s", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"FPS: {self._fps_value:.1f}", (frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        elapsed = time.time() - now
        if self.current_state == 1:
            time_to_sleep = max(0, self.motion_frame_time - elapsed)
        else:
            time_to_sleep = max(0, self.frame_time - elapsed)
        return frame, time_to_sleep

    # ------------------------------------------------------------------ #
    #  Motion detection                                                    #
    # ------------------------------------------------------------------ #

    def detect_movement(self, frame_rgb, prev_frame_gray):
        small = cv2.resize(frame_rgb, (320, 240), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, self.movement_blur_size, 0)
        movement_score = 0
        if prev_frame_gray is not None:
            delta = cv2.absdiff(prev_frame_gray, gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            movement_score = int(np.count_nonzero(thresh))
        return gray, movement_score

    # ------------------------------------------------------------------ #
    #  Async recognition result collection (non-blocking)                 #
    # ------------------------------------------------------------------ #

    def _collect_recognition_results(self):
        while True:
            result = self.recognition_worker.get_result(block=False)
            if result is None:
                break
            job_id, name, sim = result
            track_id = self._pending_jobs.pop(job_id, None)
            if track_id is None:
                continue
            # Always write to DB — face may have left frame before result arrived
            self.entry_exit_persistence.update_user(track_id, name)
            # Update live overlay only if track is still visible
            tface = self.track_id_to_face.get(track_id)
            if tface is not None:
                tface.recognition_pending = False
                tface.add_recognition_result(name, float(sim))

    # ------------------------------------------------------------------ #
    #  Face detection + tracking + recognition submission                 #
    # ------------------------------------------------------------------ #

    def detect_faces(self, frame_rgb, loop_start_time, track_id_to_face, last_face_detected_time):
        raw_frame = frame_rgb.copy()  # full rotated frame — saved to log UI
        frame_height, frame_width = frame_rgb.shape[:2]

        raw_detections = self.face_detector.detect(frame_rgb)

        # Build bbox list for SORT (filter by min_face_size)
        det_boxes = []
        det_meta = []  # (left_eye, right_eye) in frame coords
        for d in raw_detections:
            if (d.x2 - d.x1) >= self.min_face_size and (d.y2 - d.y1) >= self.min_face_size:
                det_boxes.append([d.x1, d.y1, d.x2, d.y2])
                det_meta.append((d.left_eye, d.right_eye))

        tracks = self.tracker.update(
            np.array(det_boxes) if det_boxes else np.empty((0, 5))
        )

        # Remove lost tracks
        current_ids = {int(t[4]) for t in tracks}
        for lost_id in set(track_id_to_face) - current_ids:
            self.entry_exit_persistence.cleanup_old_pending_faces(lost_id)
            track_id_to_face.pop(lost_id, None)

        # Precompute entry/exit line config once
        entry_line = self.settings.get('entry_line')
        exit_line = self.settings.get('exit_line')

        if self.detection_mode == 'line':
            if entry_line:
                self._draw_line(frame_rgb, entry_line, (40, 200, 40), "ENTRY", frame_width, frame_height)
            if exit_line:
                self._draw_line(frame_rgb, exit_line, (200, 40, 120), "EXIT", frame_width, frame_height)

        for trk in tracks:
            x1, y1, x2, y2, track_id = map(int, trk)
            cx, cy = (x1 + x2) >> 1, (y1 + y2) >> 1

            # Init or update TrackedFace
            if track_id not in track_id_to_face:
                # Try to find matching detection for eye landmarks
                le, re = self._match_eyes(x1, y1, x2, y2, det_boxes, det_meta)
                track_id_to_face[track_id] = TrackedFace(track_id, (x1, y1, x2, y2),
                                                          loop_start_time, le, re)
            else:
                le, re = self._match_eyes(x1, y1, x2, y2, det_boxes, det_meta)
                track_id_to_face[track_id].update((x1, y1, x2, y2), loop_start_time, le, re)

            tface = track_id_to_face[track_id]

            # Entry/exit logic
            if self.detection_mode == 'line':
                self._check_line_crossing(tface, cx, cy, track_id, loop_start_time,
                                          entry_line, exit_line, frame_rgb)
            elif self.detection_mode == 'camera' and not tface.entry_exit_logged:
                direction = 'entered' if self.camera_type == 'entry' else 'exited'
                self.entry_exit_persistence.log_entry_exit_event(
                    track_id=track_id, direction=direction, timestamp=loop_start_time)
                tface.entry_exit_logged = True

        if det_boxes:
            last_face_detected_time = loop_start_time

        # Face quality assessment + async recognition submission
        for track_id, tface in track_id_to_face.items():
            x1, y1, x2, y2 = tface.bbox
            face_crop = frame_rgb[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # Align using stored eye coords (translate to crop space)
            aligned = self._align_crop(face_crop, tface, x1, y1)

            # Resize to model input
            aligned_112 = cv2.resize(aligned, (self.recognizer_input_w, self.recognizer_input_h))

            # Build FIAQ landmark dict in 112×112 space
            fq_landmarks = self._fiaq_landmarks(tface, x1, y1,
                                                  face_crop.shape[1], face_crop.shape[0])
            quality = fiaq.assess_face_quality(aligned_112, fq_landmarks)

            label = f"ID {track_id} | {tface.identity}"
            if quality < self.fiaq_threshold:
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 140, 255), 2)
                cv2.putText(frame_rgb, f"{label} | LQ", (x1 + 5, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
                continue

            # Save the full rotated frame once per track (first quality-passing frame)
            if not tface.raw_image_saved:
                self.entry_exit_persistence.add_raw_face_image(
                    track_id=track_id, image=raw_frame, timestamp=loop_start_time)
                tface.raw_image_saved = True

            # ── Recognition submission logic ──────────────────────────────
            # Two modes:
            #   Still Unknown  → retry every quality-passing frame (with a
            #                    short cooldown) so a clearer face is grabbed.
            #   Already named  → only upgrade if a significantly better frame
            #                    arrives (avoids noisy re-submissions).
            _RECO_COOLDOWN = 0.5   # min seconds between jobs for the same track
            time_since_last = loop_start_time - tface.last_reco_time

            if tface.identity == 'Unknown':
                # Always retry while unidentified, subject to cooldown and
                # worker not already processing a job for this track.
                should_submit = (
                    not tface.recognition_pending
                    and time_since_last >= _RECO_COOLDOWN
                )
            else:
                # Already identified — only re-submit if quality improved >5%
                should_submit = (
                    not tface.recognition_pending
                    and quality > tface.best_quality * 1.05 + 0.01
                )

            if should_submit:
                tface.best_quality = max(tface.best_quality, quality)
                tface.last_reco_time = loop_start_time
                job_id = f"reco_{track_id}_{int(loop_start_time * 1000)}"
                self._pending_jobs[job_id] = track_id
                self.recognition_worker.recognize_async(job_id, aligned_112)
                tface.recognition_pending = True

            if tface.recognition_pending:
                color = (0, 200, 255)   # cyan = queued, waiting for worker
                status = "Queued"
            elif tface.identity != 'Unknown':
                color = (0, 200, 80)    # green = recognised
                status = f"{tface.identity} ({tface.identity_sim:.2f})"
            else:
                color = (255, 165, 0)   # orange = unknown
                status = "Unknown"
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_rgb, f"ID {track_id} | {status}", (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return track_id_to_face, last_face_detected_time

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _match_eyes(self, x1, y1, x2, y2, det_boxes, det_meta):
        """Return (left_eye, right_eye) in frame coords for the closest detection."""
        best_iou = -1
        best = (None, None)
        for box, meta in zip(det_boxes, det_meta):
            iou = self._iou([x1, y1, x2, y2], box)
            if iou > best_iou:
                best_iou = iou
                best = meta
        return best

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def _align_crop(self, face_crop, tface, x1, y1):
        """Align face_crop if eye coordinates are available; else return as-is."""
        if tface.left_eye is None or tface.right_eye is None:
            return face_crop
        le_crop = (tface.left_eye[0] - x1, tface.left_eye[1] - y1)
        re_crop = (tface.right_eye[0] - x1, tface.right_eye[1] - y1)
        return align_face(face_crop, le_crop, re_crop)

    def _fiaq_landmarks(self, tface, x1, y1, crop_w, crop_h):
        """Convert frame-space eye coords to 112×112 space for FIAQ."""
        if tface.left_eye is None or tface.right_eye is None:
            return {}
        sx = self.recognizer_input_w / max(crop_w, 1)
        sy = self.recognizer_input_h / max(crop_h, 1)
        le = ((tface.left_eye[0] - x1) * sx, (tface.left_eye[1] - y1) * sy)
        re = ((tface.right_eye[0] - x1) * sx, (tface.right_eye[1] - y1) * sy)
        return {'left_eye': le, 'right_eye': re}

    def _draw_line(self, frame, line_cfg, color, label, fw, fh):
        orient = line_cfg.get('orientation', 'horizontal')
        if orient == 'horizontal':
            y = line_cfg.get('y', 100)
            cv2.line(frame, (0, y), (fw, y), color, 2)
            cv2.putText(frame, label, (10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            x = line_cfg.get('x', 100)
            cv2.line(frame, (x, 0), (x, fh), color, 2)
            cv2.putText(frame, label, (x + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _check_line_crossing(self, tface, cx, cy, track_id, timestamp,
                              entry_line, exit_line, frame_rgb):
        def zone(lc, cx, cy):
            if lc is None:
                return None
            orient = lc.get('orientation', 'horizontal')
            direction = lc.get('direction', 'top')
            if orient == 'horizontal':
                y = lc.get('y', 100)
                before = cy > y if direction == 'top' else cy < y
            else:
                x = lc.get('x', 100)
                before = cx < x if direction == 'left' else cx > x
            return 'before' if before else 'after'

        entry_zone = zone(entry_line, cx, cy)
        exit_zone = zone(exit_line, cx, cy)
        new_zone = entry_zone or exit_zone
        kind = 'entry' if entry_zone else ('exit' if exit_zone else None)

        if tface.current_zone is not None and new_zone != tface.current_zone:
            prev_before = tface.current_zone == 'before'
            now_after = new_zone == 'after'
            if prev_before and now_after and kind == 'entry':
                self.entry_exit_persistence.log_entry_exit_event(
                    track_id=track_id, direction='entered', timestamp=timestamp)
                cv2.circle(frame_rgb, (cx, cy), 10, (0, 255, 255), 3)
            elif prev_before and now_after and kind == 'exit':
                self.entry_exit_persistence.log_entry_exit_event(
                    track_id=track_id, direction='exited', timestamp=timestamp)
                cv2.circle(frame_rgb, (cx, cy), 10, (0, 255, 255), 3)

        if new_zone is not None:
            tface.current_zone = new_zone


_VOTE_WINDOW = 5  # number of recent recognition results kept per track


class TrackedFace:
    def __init__(self, face_id, bbox, timestamp, left_eye=None, right_eye=None):
        self.face_id = face_id
        self.bbox = bbox
        self.last_update = timestamp
        self.frame_count = 1
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.entry_exit_logged = False
        self.current_zone = None
        self.raw_image_saved = False
        # Recognition state
        self.identity = 'Unknown'
        self.identity_sim = 0.0
        self.best_quality = 0.0
        self.recognition_pending = False
        self.last_reco_time = 0.0   # timestamp of last recognition job submitted
        # Voting: list of (name, sim) from the last _VOTE_WINDOW results
        self._votes: list = []

    def add_recognition_result(self, name: str, sim: float):
        """Record a new result and recompute voted identity."""
        self._votes.append((name, sim))
        if len(self._votes) > _VOTE_WINDOW:
            self._votes.pop(0)

        # Count votes per name
        counts: dict = {}
        sims: dict = {}
        for n, s in self._votes:
            counts[n] = counts.get(n, 0) + 1
            sims.setdefault(n, []).append(s)

        # Pick the name with the most votes; tie-break by average similarity
        winner = max(counts, key=lambda n: (counts[n], sum(sims[n]) / len(sims[n])))
        avg_sim = sum(sims[winner]) / len(sims[winner])

        self.identity = winner
        self.identity_sim = avg_sim

    def update(self, bbox, timestamp, left_eye=None, right_eye=None):
        self.bbox = bbox
        self.last_update = timestamp
        self.frame_count += 1
        if left_eye is not None:
            self.left_eye = left_eye
        if right_eye is not None:
            self.right_eye = right_eye

    def reset(self, bbox, timestamp):
        self.bbox = bbox
        self.last_update = timestamp
        self.frame_count = 1
        self.identity = 'Unknown'
        self.identity_sim = 0.0
        self.best_quality = 0.0
        self.recognition_pending = False
        self.last_reco_time = 0.0
        self._votes = []
