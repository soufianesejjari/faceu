import time
import cv2
import logging
import os
import base64
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, send_from_directory
import json
from datetime import datetime
import sqlite3
import numpy as np
logging.basicConfig(level=logging.INFO)

from entry_exit_persistence import EntryExitPersistenceThread, DB_PATH
entry_exit_persistence = EntryExitPersistenceThread.init_global()
# Initialize atomic track id counter from DB (only once)
last_id = entry_exit_persistence.get_next_track_id()
from init import initialize_shared
from train import retrain_and_save_embeddings
import threading
initialize_shared(
    os.path.join(os.path.dirname(__file__), 'models'),
    os.path.join(os.path.dirname(__file__), 'known_faces_embeddings.pkl')
)
from tracker import TrackIDCounter
TrackIDCounter.initialize(last_id)
from camera import CameraBase, PiCameraStream, RTSPCamera, cameras, PICAMERA_AVAILABLE

# Flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')
settings_store = {}  # {camera_name: {settings}}
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'camera_config.json')

entry_exit_conn = sqlite3.connect(DB_PATH, check_same_thread=False)

def restore_cameras():
    for name, settings in settings_store.items():
        cam_type = settings.get('type', 'picamera')  # Default to picamera if not set
        if name in cameras:
            cameras[name].stop()
        if cam_type == 'picamera' and PICAMERA_AVAILABLE:
            cameras[name] = PiCameraStream(settings)
        elif cam_type == 'rtsp':
            cameras[name] = RTSPCamera(settings)
        else:
            cameras[name] = CameraBase(settings)
        cameras[name].start()

def restart_camera(name, settings):
    if name in cameras:
        cameras[name].stop()
        del cameras[name]
        cam_type = settings.get('type', 'picamera')
        if cam_type == 'picamera' and PICAMERA_AVAILABLE:
            cameras[name] = PiCameraStream(settings)
        elif cam_type == 'rtsp':
            cameras[name] = RTSPCamera(settings)
        else:
            cameras[name] = CameraBase(settings)
        cameras[name].start()

def save_config():
    with open(CONFIG_PATH, 'w') as f:
        json.dump({
            'settings_store': settings_store
        }, f)

def load_config():
    global settings_store
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            data = json.load(f)
            settings_store.update(data.get('settings_store', {}))

@app.route('/')
def index():
    return render_template('index.html', cameras=cameras.keys(), PICAMERA_AVAILABLE=PICAMERA_AVAILABLE)

@app.route('/configure', methods=['POST'])
def configure():
    name = request.form['name']
    cam_type = request.form['type']
    source = request.form['source']
    width = int(request.form.get('width', 640))
    height = int(request.form.get('height', 480))
    # Save type and source in settings_store for persistence
    if name not in settings_store:
        settings_store[name] = {}
    settings_store[name]['type'] = cam_type
    settings_store[name]['source'] = source
    settings_store[name]['width'] = width
    settings_store[name]['height'] = height
    if name in cameras:
        cameras[name].stop()
    if cam_type == 'picamera' and PICAMERA_AVAILABLE:
        cameras[name] = PiCameraStream(settings_store[name])
    elif cam_type == 'rtsp':
        cameras[name] = RTSPCamera(settings_store[name])
    else:
        cameras[name] = CameraBase(settings_store[name])
    cameras[name].start()
    save_config()
    return redirect(url_for('index'))

@app.route('/feed/<name>')
def feed_page(name):
    return render_template('feed.html', name=name)

@app.route('/feed/<name>/stream')
def feed_stream(name):
    raw = request.args.get('raw') == '1'
    def gen():
        last_frame_time = 0
        frame_interval = 1.0 / 30.0  # 30 FPS target for streaming
        
        while True:
            current_time = time.time()
            
            if name not in cameras:
                blank = (255 * np.ones((480, 640, 3), dtype=np.uint8))
                ret, jpeg = cv2.imencode('.jpg', blank)
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(frame_interval)
                continue
                
            camera = cameras[name]
            if raw:
                frame = camera.raw_frame
            else:
                frame = camera.frame
                
            if frame is None:
                blank = (255 * np.ones((480, 640, 3), dtype=np.uint8))
                ret, jpeg = cv2.imencode('.jpg', blank)
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(frame_interval)
                continue
            
            # Only encode and send frame if enough time has passed
            time_since_last = current_time - last_frame_time
            if time_since_last >= frame_interval:
                # Frames are stored as RGB internally; cv2.imencode needs BGR.
                ret, jpeg = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                         [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    last_frame_time = current_time
                    
                # Calculate remaining time for this frame interval
                processing_time = time.time() - current_time
                remaining_time = max(0.001, frame_interval - processing_time)
                time.sleep(remaining_time)
            else:
                # Frame too early, small sleep
                time.sleep(0.001)
                
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/advanced/<name>', methods=['GET', 'POST'])
def advanced(name):
    # All advanced settings with defaults
    default_settings = {
        'recognition_threshold': 0.4,
        'min_face_size': 50,
        'movement_threshold': 3000,
        'motion_fps': 2,
        'no_face_timeout': 5.0,
        'consecutive_frames': 3,
        'fiaq_threshold': 0.4,
        'tracker_max_age': 3,
        'tracker_min_hits': 1,
        'tracker_iou_threshold': 0.2,
        'detector_conf_thresh': 0.6
    }
    settings = settings_store.get(name, {})
    # Fill missing with defaults
    for k, v in default_settings.items():
        if k not in settings:
            settings[k] = v
    if request.method == 'POST':
        for k in default_settings:
            val = request.form.get(k, None)
            if val is not None:
                try:
                    if '.' in str(default_settings[k]):
                        settings[k] = float(val)
                    else:
                        settings[k] = int(val)
                except Exception:
                    settings[k] = val
        settings_store[name] = settings
        save_config()
        restart_camera(name, settings)
        return redirect(url_for('advanced', name=name))
    return render_template('advanced.html', name=name, settings=settings)

@app.route('/settings/<name>', methods=['GET', 'POST'])
def camera_settings(name):
    # Default settings
    default = {
        'width': 640,
        'height': 480,
        'crop': False,
        'x': 50,
        'y': 50,
        'w': 200,
        'h': 150,
        'fps': 5,
        'rotate': 0,
        'detection_mode': 'line'  # Default to line detection mode
    }
    settings = settings_store.get(name, default.copy())
    if request.method == 'POST':
        settings['width'] = int(request.form.get('width', 640))
        settings['height'] = int(request.form.get('height', 480))
        settings['crop'] = request.form.get('crop') == '1' or request.form.get('crop') == 'on'
        settings['x'] = int(request.form.get('x', 50))
        settings['y'] = int(request.form.get('y', 50))
        settings['w'] = int(request.form.get('w', 200))
        settings['h'] = int(request.form.get('h', 150))
        settings['fps'] = int(request.form.get('fps', 5))
        settings['rotate'] = int(request.form.get('rotate', 0))
        # Save detection mode and related settings
        detection_mode = request.form.get('detection_mode', 'line')
        settings['detection_mode'] = detection_mode
        
        if detection_mode == 'line':
            # Save entry and exit line config
            entry_orientation = request.form.get('entry_line_orientation', 'horizontal')
            if request.form.get('entry_line_enabled') is not None:
                entry_pos = int(request.form.get('entry_line_pos', 100))
                entry_dir = request.form.get('entry_line_dir', 'top')
                if entry_orientation == 'horizontal':
                    settings['entry_line'] = {'orientation': 'horizontal', 'y': entry_pos, 'direction': entry_dir}
                else:
                    settings['entry_line'] = {'orientation': 'vertical', 'x': entry_pos, 'direction': entry_dir}
            else:
                if 'entry_line' in settings:
                    del settings['entry_line']
            if request.form.get('exit_line_enabled') is not None:
                exit_orientation = request.form.get('exit_line_orientation', 'horizontal')
                exit_pos = int(request.form.get('exit_line_pos', 200))
                exit_dir = request.form.get('exit_line_dir', 'bottom')
                if exit_orientation == 'horizontal':
                    settings['exit_line'] = {'orientation': 'horizontal', 'y': exit_pos, 'direction': exit_dir}
                else:
                    settings['exit_line'] = {'orientation': 'vertical', 'x': exit_pos, 'direction': exit_dir}
            else:
                if 'exit_line' in settings:
                    del settings['exit_line']
            # Clear camera type if line mode is selected
            if 'camera_type' in settings:
                del settings['camera_type']
        elif detection_mode == 'camera':
            # Save camera type
            camera_type = request.form.get('camera_type', 'entry')
            settings['camera_type'] = camera_type
            # Clear line settings if camera mode is selected
            if 'entry_line' in settings:
                del settings['entry_line']
            if 'exit_line' in settings:
                del settings['exit_line']
        settings_store[name] = settings
        save_config()
        restart_camera(name, settings)
        return redirect(url_for('camera_settings', name=name))
    return render_template('settings.html', name=name, settings=settings)

@app.route('/wall')
def wall():
    return render_template('wall.html', cameras=cameras.keys())

@app.route('/delete/<name>', methods=['POST'])
def delete_camera(name):
    # Stop and remove camera
    if name in cameras:
        cameras[name].stop()
        del cameras[name]
    # Remove from settings and crop areas
    if name in settings_store:
        del settings_store[name]
    save_config()
    return redirect(url_for('index'))

@app.route('/face_capture/<user_id>')
def face_capture_page(user_id):
    return render_template('face_capture.html', user_id=user_id)

@app.route('/face_capture/<user_id>', methods=['POST'])
def save_face_dataset(user_id):
    cur = entry_exit_conn.cursor()
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        if not data or 'images' not in data:
            return jsonify({'success': False, 'error': 'No images provided'}), 400
        
        images = data['images']
        
        if len(images) != 10:
            return jsonify({'success': False, 'error': 'Expected 10 images'}), 400

        # Create user directory in dataset folder
        user_dir = os.path.join('dataset', user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Save each image
        for i, image_data in enumerate(images):
            if not image_data:
                continue
                
            # Remove the data:image/jpeg;base64, prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64 image
            try:
                image_bytes = base64.b64decode(image_data)
                
                # Save to file
                filename = f'face_sample_{i+1}.png'
                filepath = os.path.join(user_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(image_bytes)
                    
            except Exception as e:
                logging.error(f"Error saving image {i+1} for user {user_id}: {str(e)}")
                return jsonify({'success': False, 'error': f'Error saving image {i+1}'}), 500
        
        logging.info(f"Successfully saved face dataset for user: {user_id}")
        cur.execute("INSERT OR IGNORE INTO users (employee_id) VALUES (?)", (user_id,))
        entry_exit_conn.commit()
        threading.Thread(target=retrain_and_save_embeddings, daemon=True).start()
        return jsonify({'success': True, 'message': f'Face dataset saved for user {user_id}'})
    except Exception as e:
        logging.error(f"Error in save_face_dataset: {str(e)}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500


@app.route('/entry_exit_log')
def entry_exit_log():
    page = int(request.args.get('page', 1))
    per_page = 20
    date_str = request.args.get('date', '')
    offset = (page - 1) * per_page
    query = "SELECT entry_exit.id, users.employee_id, entry_exit.direction, entry_exit.timestamp FROM entry_exit LEFT JOIN users ON entry_exit.user = users.employee_id"
    params = []
    where = []
    if date_str:
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            start_ts = datetime.combine(dt, datetime.min.time()).timestamp()
            end_ts = datetime.combine(dt, datetime.max.time()).timestamp()
            where.append("timestamp BETWEEN ? AND ?")
            params += [start_ts, end_ts]
        except Exception:
            pass
    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params += [per_page + 1, offset]
    cur = entry_exit_conn.cursor()
    rows = cur.execute(query, params).fetchall()
    has_next = len(rows) > per_page
    rows = rows[:per_page]
    # Convert to dicts and format timestamp
    result = []
    for row in rows:
        name = row[1] if row[1] is not None else 'PENDING'
        result.append({
            'id': row[0],
            'name': name,
            'direction': row[2],
            'timestamp': datetime.fromtimestamp(row[3]).strftime('%Y-%m-%d %H:%M:%S') if row[3] else '',
        })
    return render_template('entry_exit_log.html', rows=result, page=page, has_next=has_next, date=date_str)

@app.route('/entry_exit_image/<int:id>')
def entry_exit_image(id):
    import glob as _glob
    matches = sorted(_glob.glob(
        os.path.join(entry_exit_persistence.save_dir, f'face_{id}_*.jpg')
    ))
    if not matches:
        return '', 404
    image_path = matches[0]
    return send_from_directory(os.path.dirname(image_path), os.path.basename(image_path))

@app.route('/face_datasets')
def face_datasets():
    dataset_path = 'dataset'
    users = []
    if os.path.exists(dataset_path):
        for user_name in sorted(os.listdir(dataset_path)):
            user_dir = os.path.join(dataset_path, user_name)
            if os.path.isdir(user_dir):
                images = sorted([f for f in os.listdir(user_dir) if f.endswith('.png')])
                users.append({
                    'name': user_name,
                    'images': images,
                    'image_count': len(images)
                })
    return render_template('face_datasets.html', users=users)

@app.route('/dataset/<user_id>/<filename>')
def dataset_image(user_id, filename):
    image_path = os.path.join('dataset', user_id, filename)
    if not os.path.exists(image_path):
        return '', 404
    return send_from_directory(os.path.join('dataset', user_id), filename)

if __name__ == '__main__':
    load_config()
    restore_cameras()
    app.run(host='0.0.0.0', port=8080, ssl_context=('cert.pem', 'key.pem'))
