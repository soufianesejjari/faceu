import cv2
import threading
import atexit
import signal
import logging
import time
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

from recognition import FaceRecognizer
from worker import FaceRecognitionWorker
from entry_exit_persistence import entry_exit_persistence

# Global camera dictionary to manage multiple camera instances
cameras = {}
# Face recognition worker
face_recognition_worker = FaceRecognitionWorker()

# Camera abstraction
class CameraBase:
    def __init__(self, settings):
        self.configure(settings)
    
    def configure(self, settings):
        self.source = settings.get('source', 0)
        self.width = settings.get('width', 640)
        self.height = settings.get('height', 480)
        self.cap = None
        self.running = False
        self.frame = None
        self.thread = None
        self._stop_event = threading.Event()
        self.last_read_success = False
        self.raw_frame = None  # Store the raw frame for cropping
        self.face_reco = FaceRecognizer(settings, entry_exit_persistence, face_recognition_worker)
        self.crop = (
            settings.get('x', 50),
            settings.get('y', 50),
            settings.get('w', 50),
            settings.get('h', 50)
        ) if settings.get('crop', False) else None
        self.rotate = settings.get('rotate', None)
        self.sleep_time = 1
        self.target_fps = settings.get('fps', 5)

    def start(self):
        if not self.running:
            self.cap = cv2.VideoCapture(int(self.source))
            if not self.cap.isOpened():
                logging.error(f"Failed to open camera source: {self.source}")
                return
            
            # Optimize for real-time streaming
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for real-time
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)  # Set target FPS
            
            # Additional optimizations for USB cameras
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            self.running = True
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()
            logging.info(f"Camera {self.source} started with real-time optimizations")

    def update(self):
        while self.running and not self._stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                logging.error(f"Camera {self.source} not opened.")
                time.sleep(1)
                continue
            
            current_time = time.time()
            # Clear buffer to get the latest frame (drop buffered frames)
            # This helps ensure we're processing the most recent frame
            buffer_size = self.cap.get(cv2.CAP_PROP_BUFFERSIZE)
            if buffer_size > 1:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            ret, frame = self.cap.read()
            self.last_read_success = ret

            if ret and frame is not None:
                # VideoCapture always returns BGR; convert to RGB once here so
                # every downstream consumer (recognition, webserver) sees RGB.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.handle_frame(frame, current_time)
            else:
                # No frame available small sleep to prevent busy waiting
                time.sleep(0.01)

    def handle_frame(self, frame, now):
        
        self.raw_frame = frame.copy()
        # Apply transformations
        if self.rotate:
            if self.rotate == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotate == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotate == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.crop:
            x, y, w, h = self.crop
            frame = frame[y:y+h, x:x+w]
        
        # Process frame with face recognition
        self.frame, self.sleep_time = self.face_reco.process_frame(frame, now)
        # For real-time playback, minimize artificial delays
        # Only use processing sleep time if it's very small
        if self.sleep_time > 0.01:
            sleep_time = max(0.001, self.sleep_time)
            time.sleep(sleep_time)

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def __del__(self):
        self.stop()
        if self.thread and self.thread.is_alive():
            # Graceful shutdown
            self._stop_event.set()
            self.thread.join(timeout=5)
        logging.info(f"Camera {self.__class__.__name__}:{self.source} stopped and resources released.")

class PiCameraStream(CameraBase):
    def __init__(self, settings):
        super().__init__(settings)
        self.picam2 = None

    def start(self):
        if not self.running:
            self.picam2 = Picamera2()
            camera_config = self.picam2.create_preview_configuration(main={"size": (self.width, self.height), "format": "RGB888"})
            self.picam2.configure(camera_config)
            self.picam2.start()
            self.running = True
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()

    def update(self):
        while self.running and not self._stop_event.is_set():
            current_time = time.time()
            frame = self.picam2.capture_array()
            if frame is not None:
                self.handle_frame(frame, current_time)
            else:
                # No frame available, small sleep to prevent busy waiting
                time.sleep(0.01)
    
    def stop(self):
        self.running = False
        if self.picam2:
            try:
                if hasattr(self.picam2, 'started') and self.picam2.started:
                    self.picam2.stop()
                self.picam2.close()
                logging.info("PiCamera2 stopped and closed successfully")
            except Exception as e:
                logging.error(f"Error stopping PiCamera2: {e}")
            finally:
                self.picam2 = None

class RTSPCamera(CameraBase):
    def __init__(self, settings):
        super().__init__(settings)

    def start(self):
        if not self.running:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                logging.error(f"Failed to open RTSP stream: {self.source}")
                return
            
            # Optimize for real-time streaming
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for real-time
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)  # Set target FPS
            
            self.running = True
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()
            logging.info(f"RTSP camera {self.source} started with real-time optimizations")

def cleanup_cameras():
    """Clean up all cameras on shutdown"""
    logging.info("Cleaning up cameras on shutdown...")
    for name, camera in cameras.items():
        try:
            camera.stop()
            logging.info(f"Stopped camera {name}")
        except Exception as e:
            logging.error(f"Error stopping camera {name}: {e}")
    cameras.clear()

# Register cleanup function for normal exit
atexit.register(cleanup_cameras)
atexit.register(face_recognition_worker.close)

# Register cleanup for SIGINT and SIGTERM
signal.signal(signal.SIGINT, lambda signum, frame: (cleanup_cameras(), exit(0)))
signal.signal(signal.SIGTERM, lambda signum, frame: (cleanup_cameras(), exit(0)))