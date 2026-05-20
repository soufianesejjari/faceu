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
        # Webhook is real-time only. Mark every pre-existing unsynced row as
        # already synced so historical data is never pushed to the webhook.
        # Only events created during this session (synced=0 from now on) fire.
        self.conn.execute("UPDATE entry_exit SET synced = 1 WHERE synced = 0")
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
        # If recognition already completed before the line was crossed, fire now
        self._try_fire_webhook(track_id)

    def _update_user(self, track_id, name):
        self.cursor.execute(
            "UPDATE entry_exit SET user = ? WHERE id = ?",
            (name, track_id)
        )
        self.conn.commit()
        # If the line was already crossed before recognition completed, fire now
        if name != 'Unknown':
            self._try_fire_webhook(track_id)

    def _try_fire_webhook(self, track_id):
        """Fire webhook for this event only if it is fully resolved: known user + real direction."""
        self.cursor.execute(
            "SELECT user, direction, timestamp FROM entry_exit "
            "WHERE id = ? AND synced = 0",
            (track_id,)
        )
        row = self.cursor.fetchone()
        if not row:
            return
        user, direction, ts = row
        # Both conditions must be true — only then is the event actionable
        if user == 'Unknown' or direction == 'pending':
            return
        self._send_webhook(track_id, user, direction, ts)

    def _send_webhook(self, track_id, user, direction, ts):
        import json
        from urllib import request as urequest, error as uerror
        webhook_url = os.environ.get('WEBHOOK_URL', '').strip()
        if not webhook_url:
            return
        api_key = os.environ.get('WEBHOOK_API_KEY', '').strip()
        payload = json.dumps({
            "id": track_id,
            "user": user,
            "action": direction,
            "timestamp": ts,
        }).encode('utf-8')
        req = urequest.Request(webhook_url, method="POST")
        req.add_header('Content-Type', 'application/json')
        if api_key:
            req.add_header('x-api-key', api_key)
        try:
            with urequest.urlopen(req, data=payload, timeout=5) as resp:
                if resp.status in (200, 201, 204):
                    self.cursor.execute(
                        "UPDATE entry_exit SET synced = 1 WHERE id = ?", (track_id,)
                    )
                    self.conn.commit()
        except Exception as e:
            print(f"[Webhook] Failed for event {track_id}: {e}")

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


