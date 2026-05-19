import threading
import queue
import os
import cv2
import time
import glob
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), 'entry_exit_log.db')
entry_exit_persistence = None


class EntryExitPersistenceThread(threading.Thread):
    @staticmethod
    def init_global():
        global entry_exit_persistence
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        entry_exit_persistence = EntryExitPersistenceThread(conn)
        entry_exit_persistence.start()
        return entry_exit_persistence

    def __init__(self, conn, save_dir="log", queue_size=200):
        super().__init__(daemon=True)
        self.save_dir = save_dir
        self.conn = conn
        self.cursor = conn.cursor()
        self.q = queue.Queue(maxsize=queue_size)
        os.makedirs(self.save_dir, exist_ok=True)
        self.running = True
        self._setup_db()

    def run(self):
        dispatch = {
            'add_raw_face_image':   self._add_raw_face_image,
            'log_entry_exit_event': self._log_entry_exit_event,
            'update_user':          self._update_user,
            'cleanup':              self._cleanup_pending,
        }
        while self.running:
            try:
                task = self.q.get(timeout=0.5)
            except queue.Empty:
                self._sync_pending_events()
                continue
            handler = dispatch.get(task[0])
            if handler:
                handler(*task[1:])
            self.q.task_done()

    def stop(self):
        self.running = False
        self.conn.close()

    # ------------------------------------------------------------------ #
    #  Schema                                                              #
    # ------------------------------------------------------------------ #

    def _setup_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                employee_id TEXT PRIMARY KEY
            );
            INSERT OR IGNORE INTO users (employee_id) VALUES ('Unknown');
            CREATE TABLE IF NOT EXISTS entry_exit (
                id        INTEGER PRIMARY KEY,
                user      TEXT,
                direction TEXT,
                timestamp REAL,
                synced    INTEGER DEFAULT 0,
                FOREIGN KEY(user) REFERENCES users(employee_id)
            );
        """)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    #  Public API (all thread-safe — put to queue)                        #
    # ------------------------------------------------------------------ #

    def get_next_track_id(self):
        self.cursor.execute("SELECT MAX(id) FROM entry_exit")
        row = self.cursor.fetchone()
        return row[0] if row[0] is not None else 0

    def add_raw_face_image(self, track_id, image, timestamp):
        self.q.put(('add_raw_face_image', track_id, image, timestamp))

    def log_entry_exit_event(self, track_id, direction, timestamp):
        self.q.put(('log_entry_exit_event', track_id, direction, timestamp))

    def update_user(self, track_id, name):
        """Called by FaceRecognizer when async recognition completes."""
        self.q.put(('update_user', track_id, name))

    def cleanup_old_pending_faces(self, lost_id):
        self.q.put(('cleanup', lost_id))

    # ------------------------------------------------------------------ #
    #  Private handlers (run on persistence thread only)                  #
    # ------------------------------------------------------------------ #

    def _add_raw_face_image(self, track_id, image, timestamp):
        ts = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp))
        img_path = os.path.join(self.save_dir, f"face_{track_id}_{ts}.jpg")
        cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        self.cursor.execute(
            "INSERT OR IGNORE INTO entry_exit (id, user, direction, timestamp) "
            "VALUES (?, 'Unknown', 'pending', ?)",
            (track_id, timestamp)
        )
        self.conn.commit()

    def _log_entry_exit_event(self, track_id, direction, timestamp):
        self.cursor.execute(
            "INSERT INTO entry_exit (id, user, direction, timestamp) VALUES (?, 'Unknown', ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET direction = excluded.direction, timestamp = excluded.timestamp",
            (track_id, direction, timestamp)
        )
        self.conn.commit()

    def _update_user(self, track_id, name):
        self.cursor.execute(
            "UPDATE entry_exit SET user = ? WHERE id = ?",
            (name, track_id)
        )
        self.conn.commit()

    def _cleanup_pending(self, lost_id):
        self.cursor.execute(
            "SELECT direction FROM entry_exit WHERE id = ?", (lost_id,)
        )
        row = self.cursor.fetchone()
        direction = row[0] if row else 'pending'

        if direction == 'pending':
            # Track never crossed a line — safe to purge record and images
            self.cursor.execute("DELETE FROM entry_exit WHERE id = ?", (lost_id,))
            self.conn.commit()
            for img_path in glob.glob(os.path.join(self.save_dir, f"face_{lost_id}_*.jpg")):
                try:
                    os.remove(img_path)
                except OSError:
                    pass
        # If direction is 'entered'/'exited' keep both the DB row and the images

    def _sync_pending_events(self):
        webhook_url = os.environ.get('WEBHOOK_URL', '').strip()
        if not webhook_url:
            return

        api_key = os.environ.get('WEBHOOK_API_KEY', '').strip()

        # Select events that are ready to sync
        self.cursor.execute(
            "SELECT id, user, direction, timestamp FROM entry_exit "
            "WHERE user != 'Unknown' AND direction != 'pending' AND synced = 0"
        )
        rows = self.cursor.fetchall()
        if not rows:
            return

        import json
        from urllib import request, error

        for row in rows:
            track_id, user, direction, ts = row
            data = {
                "id": track_id,
                "user": user,
                "action": direction,
                "timestamp": ts
            }
            
            req = request.Request(webhook_url, method="POST")
            req.add_header('Content-Type', 'application/json')
            if api_key:
                req.add_header('x-api-key', api_key)
            
            try:
                with request.urlopen(req, data=json.dumps(data).encode('utf-8'), timeout=5) as response:
                    if response.status in (200, 201, 204):
                        self.cursor.execute("UPDATE entry_exit SET synced = 1 WHERE id = ?", (track_id,))
                        self.conn.commit()
            except Exception as e:
                print(f"[Webhook Error] Failed to sync event {track_id}: {e}")

