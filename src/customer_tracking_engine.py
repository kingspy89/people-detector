#!/usr/bin/env python3
"""
IntelliSight - Customer Tracking Engine
Beautiful UI with individual customer tracking
"""

import cv2
import time
import pandas as pd
import numpy as np
import argparse
import os
import json
import sqlite3
import uuid
import math
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import threading

# Fix PyTorch compatibility# Fix PyTorch compatibility if supported
import torch
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  Install: pip install ultralytics")

@dataclass
class CustomerJourney:
    customer_id: str
    session_id: str
    entry_time: float
    exit_time: Optional[float] = None
    total_duration: Optional[float] = None
    zones_visited: List[str] = None
    path_coordinates: List[Tuple] = None
    dwell_times: Dict[str, float] = None
    behavior_pattern: str = "unknown"
    conversion_status: str = "in_progress"

    def __post_init__(self):
        if self.zones_visited is None:
            self.zones_visited = []
        if self.path_coordinates is None:
            self.path_coordinates = []
        if self.dwell_times is None:
            self.dwell_times = {}

class CustomerTracker:
    def __init__(self, model_name='yolov8m.pt', confidence=0.35, downscale: float = 0.5, skip_frames: int = 1, use_gpu: bool = False):
        self.model = None
        self.confidence = confidence
        self.tracking_id = 0
        self.active_customers = {}
        self.completed_journeys = []
        # Performance tuning
        # downscale: scale to resize frame for detection (0.5 => half size)
        # skip_frames: run detection every N frames (1 = every frame)
        # use_gpu: try to move model to CUDA if available
        self.downscale = downscale if downscale and downscale > 0 and downscale <= 1.0 else 1.0
        self.skip_frames = max(1, int(skip_frames))
        self.use_gpu = bool(use_gpu)

        # Zone definitions with colors
        self.zones = {
            'entrance': {'x1': 0, 'y1': 0, 'x2': 200, 'y2': 480, 
                        'color': (255, 255, 0), 'name': '🚪 Entrance'},
            'browsing': {'x1': 200, 'y1': 0, 'x2': 440, 'y2': 300,
                        'color': (255, 0, 255), 'name': '🛍️ Browsing'},
            'counter': {'x1': 440, 'y1': 300, 'x2': 640, 'y2': 480,
                       'color': (0, 255, 0), 'name': '💳 Counter'},
            'seating': {'x1': 200, 'y1': 300, 'x2': 440, 'y2': 480,
                       'color': (0, 255, 255), 'name': '🍽️ Seating'}
        }

        self.start_time = time.time()
        self.frame_count = 0
        self.total_customers = 0
        self.peak_occupancy = 0

        self.setup_system()
        self.load_model(model_name)

    def setup_system(self):
        os.makedirs('data', exist_ok=True)
        os.makedirs('exports', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        self.setup_database()

    def setup_database(self):
        self.db_path = "data/analytics.db"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS journeys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT, entry_time REAL, exit_time REAL,
                duration REAL, zones TEXT, behavior TEXT, conversion TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def load_model(self, model_name):
        if YOLO_AVAILABLE:
            print(f"🤖 Loading {model_name}...")
            self.model = YOLO(model_name)
            # move to GPU if requested and available
            try:
                if self.use_gpu and torch.cuda.is_available():
                    self.model.to('cuda')
                    print("✅ AI Model Ready on CUDA!")
                else:
                    print("✅ AI Model Ready (CPU)")
            except Exception:
                print("⚠️ Could not move model to CUDA; running on CPU")
        else:
            print("❌ YOLO not available")

    def detect_people(self, frame):
        if not self.model:
            return []
        # Optionally downscale frame for faster inference
        orig_h, orig_w = frame.shape[:2]
        if self.downscale and self.downscale < 1.0:
            small_w = max(2, int(orig_w * self.downscale))
            small_h = max(2, int(orig_h * self.downscale))
            small = cv2.resize(frame, (small_w, small_h))
            scale_x = orig_w / small_w
            scale_y = orig_h / small_h
        else:
            small = frame
            scale_x = scale_y = 1.0

        results = self.model(small, verbose=False, conf=self.confidence, classes=[0])
        detections = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    # scale coords back to original frame size
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'center': [int((x1+x2)/2), int((y1+y2)/2)]
                    })
        return detections

    def get_zone(self, x, y):
        for zone_name, zone_data in self.zones.items():
            if (zone_data['x1'] <= x <= zone_data['x2'] and 
                zone_data['y1'] <= y <= zone_data['y2']):
                return zone_name
        return 'unknown'

    def distance(self, pos1, pos2):
        return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

    def track_customers(self, detections):
        current_time = time.time()
        unmatched = detections.copy()
        updated = set()

        for cid, data in self.active_customers.items():
            best_match, best_dist = None, float('inf')
            for i, det in enumerate(unmatched):
                dist = self.distance(data['last_pos'], det['center'])
                if dist < best_dist and dist < 100:
                    best_dist, best_match = dist, i

            if best_match is not None:
                det = unmatched.pop(best_match)
                self.update_customer(cid, det, current_time)
                updated.add(cid)

        for det in unmatched:
            self.create_customer(det, current_time)

        to_remove = []
        for cid, data in self.active_customers.items():
            if cid not in updated and current_time - data['last_update'] > 5.0:
                self.complete_journey(cid, current_time)
                to_remove.append(cid)

        for cid in to_remove:
            del self.active_customers[cid]

    def create_customer(self, detection, timestamp):
        cid = f"Customer_{self.tracking_id}"
        self.tracking_id += 1
        center = detection['center']
        zone = self.get_zone(center[0], center[1])

        journey = CustomerJourney(
            customer_id=cid,
            session_id=f"session_{uuid.uuid4().hex[:8]}",
            entry_time=timestamp
        )

        if zone != 'unknown':
            journey.zones_visited.append(zone)

        self.active_customers[cid] = {
            'journey': journey,
            'last_pos': center,
            'last_update': timestamp,
            'last_zone': zone,
            'zone_enter_time': timestamp,
            'positions': [center],
            'bbox': detection.get('bbox', [center[0]-20, center[1]-40, center[0]+20, center[1]+40])
        }
        self.total_customers += 1

    def update_customer(self, cid, detection, timestamp):
        data = self.active_customers[cid]
        journey = data['journey']
        center = detection['center']
        zone = self.get_zone(center[0], center[1])

        data['last_pos'] = center
        data['last_update'] = timestamp
        data['positions'].append(center)
        data['bbox'] = detection.get('bbox', data.get('bbox'))

        if zone != data['last_zone'] and zone != 'unknown':
            if data['last_zone'] != 'unknown':
                dwell = timestamp - data['zone_enter_time']
                if dwell >= 3.0:
                    journey.dwell_times[data['last_zone']] = dwell

            if zone not in journey.zones_visited:
                journey.zones_visited.append(zone)

            data['last_zone'] = zone
            data['zone_enter_time'] = timestamp

    def complete_journey(self, cid, exit_time):
        data = self.active_customers[cid]
        journey = data['journey']
        journey.exit_time = exit_time
        journey.total_duration = exit_time - journey.entry_time

        if data['last_zone'] != 'unknown':
            dwell = exit_time - data['zone_enter_time']
            if dwell >= 3.0:
                journey.dwell_times[data['last_zone']] = dwell

        journey.behavior_pattern = self.analyze_behavior(journey)
        journey.conversion_status = self.analyze_conversion(journey)

        self.completed_journeys.append(journey)
        self.save_journey(journey)

    def analyze_behavior(self, journey):
        duration = journey.total_duration or 0
        zones = journey.zones_visited

        if duration < 30:
            return "🚶 Quick Pass"
        elif 'counter' in zones and duration > 60:
            return "🛒 Purchaser"
        elif 'browsing' in zones and duration > 180:
            return "🔍 Browser"
        elif 'seating' in zones:
            return "🍽️ Diner"
        return "👀 Visitor"

    def analyze_conversion(self, journey):
        zones = journey.zones_visited
        dwell = journey.dwell_times

        if 'counter' in zones and dwell.get('counter', 0) > 30:
            return "💳 Purchased"
        elif 'counter' in zones:
            return "🤔 Considered"
        elif 'browsing' in zones:
            return "👀 Browsed"
        return "🚶 Passed"

    def save_journey(self, journey):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO journeys (customer_id, entry_time, exit_time, duration, zones, behavior, conversion)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (journey.customer_id, journey.entry_time, journey.exit_time,
                  journey.total_duration, json.dumps(journey.zones_visited),
                  journey.behavior_pattern, journey.conversion_status))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Save error: {e}")

    def draw_visualization(self, frame):
        # Minimal visualization: only boxes, trails, movement vector and small stats
        h, w = frame.shape[:2]

        # Small stats panel (top-left)
        elapsed = time.time() - self.start_time
        fps = self.frame_count/elapsed if elapsed > 0 else 0
        occupancy = len(self.active_customers)

        panel_w, panel_h = 260, 70
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (0, 200, 100), 1)
        cv2.putText(frame, f"Live: {occupancy}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 100), 2)
        cv2.putText(frame, f"Total: {self.total_customers}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (140, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Draw each customer as a bbox with trail and movement arrow
        for cid, data in self.active_customers.items():
            bbox = data.get('bbox')
            positions = data.get('positions', [])
            if bbox:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 127), 2)

            # Trail
            if len(positions) > 1:
                for i in range(1, min(len(positions), 20)):
                    p0 = tuple(positions[-i-1])
                    p1 = tuple(positions[-i])
                    alpha = 1.0 - (i/20.0)
                    color = (int(0 * alpha + 0 * (1-alpha)), int(255 * alpha), int(127 * alpha))
                    cv2.line(frame, p0, p1, color, 2)

            # Movement vector (arrow) from previous to last
            if len(positions) > 1:
                p_prev = tuple(positions[-2])
                p_curr = tuple(positions[-1])
                cv2.arrowedLine(frame, p_prev, p_curr, (255, 200, 0), 2, tipLength=0.3)

            # Label with ID and time-on-screen
            duration = time.time() - data['journey'].entry_time
            label = f"{cid} {int(duration)}s"
            text_pos = (positions[-1][0] - 5, positions[-1][1] - 10) if positions else (x1, y1 - 10)
            cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Update peak occupancy
        if occupancy > self.peak_occupancy:
            self.peak_occupancy = occupancy

        return frame

    def save_report(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.completed_journeys:
            data = []
            for j in self.completed_journeys:
                data.append({
                    'Customer': j.customer_id,
                    'Duration_Min': (j.total_duration or 0)/60,
                    'Zones': ', '.join(j.zones_visited),
                    'Behavior': j.behavior_pattern,
                    'Conversion': j.conversion_status
                })

            df = pd.DataFrame(data)
            excel_file = f"exports/analytics_{timestamp}.xlsx"
            df.to_excel(excel_file, index=False)
            print(f"📊 Report: {excel_file}")

        print("\n" + "="*60)
        print("🎯 SESSION SUMMARY")
        print("="*60)
        print(f"Duration: {(time.time()-self.start_time)/60:.1f} min")
        print(f"Total Customers: {self.total_customers}")
        print(f"Peak Occupancy: {self.peak_occupancy}")
        print(f"Completed: {len(self.completed_journeys)}")
        print("="*60)

    def process_video(self, source):
        # Threaded frame reader to reduce capture latency
        class FrameGrabber:
            def __init__(self, src):
                self.cap = cv2.VideoCapture(src)
                self.ret = False
                self.frame = None
                self.stopped = False
                self.lock = threading.Lock()
                self.thread = threading.Thread(target=self.update, daemon=True)
                # try to read FPS (0 or NaN if not available for cameras)
                try:
                    fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
                except Exception:
                    fps = 0.0
                # some cameras return 0 or 30; treat <=1 as unknown
                self.fps = fps if fps and fps > 1.0 else 0.0

            def start(self):
                if not self.cap.isOpened():
                    return False
                self.thread.start()
                return True

            def update(self):
                while not self.stopped:
                    ret, frame = self.cap.read()
                    with self.lock:
                        self.ret = ret
                        self.frame = frame
                    if not ret:
                        self.stopped = True

            def read(self):
                with self.lock:
                    return self.ret, self.frame

            def release(self):
                self.stopped = True
                try:
                    self.thread.join(timeout=1.0)
                except Exception:
                    pass
                self.cap.release()

        # Decide strategy based on source type: file vs camera
        is_file = isinstance(source, str) and os.path.exists(source)

        print(f"🎬 Processing: {source}")
        print("🎮 Controls: 'q'=Quit, 's'=Screenshot, 'r'=Reset")

        try:
            display_name = 'IntelliSight Analytics'
            cv2.namedWindow(display_name, cv2.WINDOW_NORMAL)
            frame_idx = 0

            if is_file:
                # Use blocking capture for files so we can respect FPS/timestamps
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    print(f"❌ Cannot open: {source}")
                    return

                try:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

                    # If FPS not available, try to estimate using frame timestamps
                    def estimate_frame_interval(capture, samples=3):
                        times = []
                        # read a few frames to sample POS_MSEC
                        for i in range(samples):
                            ret, _ = capture.read()
                            if not ret:
                                break
                            t = capture.get(cv2.CAP_PROP_POS_MSEC)
                            if t and t > 0:
                                times.append(t)
                        # rewind to start
                        try:
                            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        except Exception:
                            pass
                        if len(times) >= 2:
                            # compute average delta in seconds
                            deltas = [(times[i+1] - times[i]) / 1000.0 for i in range(len(times)-1)]
                            avg = sum(deltas) / len(deltas)
                            return avg if avg > 0 else None
                        return None

                    frame_interval = None
                    if fps and fps > 1.0:
                        frame_interval = 1.0 / fps
                    else:
                        est = estimate_frame_interval(cap, samples=4)
                        if est:
                            frame_interval = est

                    # Use frame timestamps to align playback wall-clock to video timeline
                    play_start_wall = None

                    while True:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            break

                        frame_idx += 1
                        self.frame_count += 1

                        # Get current frame timestamp in milliseconds
                        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
                        if play_start_wall is None:
                            # Align the wall-clock so that play_start_wall + pos_msec/1000 == now
                            play_start_wall = time.time() - (pos_msec / 1000.0 if pos_msec else 0.0)

                        start_t = time.time()

                        if (frame_idx % self.skip_frames) == 0:
                            detections = self.detect_people(frame)
                            self.track_customers(detections)

                        vis = self.draw_visualization(frame)
                        cv2.imshow(display_name, vis)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('s'):
                            cv2.imwrite(f"output/screenshot_{time.time():.0f}.jpg", vis)
                            print("📸 Screenshot saved")
                        elif key == ord('r'):
                            self.active_customers.clear()
                            self.completed_journeys.clear()
                            print("🔄 Reset")

                        # Compute expected wall-clock time for this frame and wait if needed
                        if play_start_wall is not None and pos_msec:
                            expected_wall = play_start_wall + (pos_msec / 1000.0)
                            remaining = expected_wall - time.time()
                            if remaining > 0:
                                time.sleep(remaining)

                finally:
                    cap.release()

            else:
                # Camera path: use threaded grabber for low latency
                grabber = FrameGrabber(source)
                if not grabber.start():
                    print(f"❌ Cannot open: {source}")
                    return

                try:
                    # For cameras we don't have reliable FPS; draw as frames arrive
                    while True:
                        ret, frame = grabber.read()
                        if not ret or frame is None:
                            break

                        frame_idx += 1
                        self.frame_count += 1

                        # Run detection on skip schedule
                        if (frame_idx % self.skip_frames) == 0:
                            detections = self.detect_people(frame)
                            self.track_customers(detections)

                        vis = self.draw_visualization(frame)
                        cv2.imshow(display_name, vis)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('s'):
                            cv2.imwrite(f"output/screenshot_{time.time():.0f}.jpg", vis)
                            print("📸 Screenshot saved")
                        elif key == ord('r'):
                            self.active_customers.clear()
                            self.completed_journeys.clear()
                            print("🔄 Reset")

                finally:
                    grabber.release()

        finally:
            cv2.destroyAllWindows()

            current_time = time.time()
            for cid in list(self.active_customers.keys()):
                self.complete_journey(cid, current_time)

            self.save_report()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', default=0)
    parser.add_argument('--model', '-m', default='yolov8m.pt')
    parser.add_argument('--confidence', '-c', type=float, default=0.35)
    parser.add_argument('--downscale', '-d', type=float, default=0.5,
                        help='Resize factor for detection (0.5 = half size). Use 1.0 to disable')
    parser.add_argument('--skip', type=int, default=1,
                        help='Process detection every N frames (1 = every frame)')
    parser.add_argument('--gpu', action='store_true', help='Attempt to use GPU for model')
    args = parser.parse_args()

    try:
        source = int(args.source) if str(args.source).isdigit() else args.source
    except:
        source = args.source

    tracker = CustomerTracker(args.model, args.confidence, downscale=args.downscale, skip_frames=args.skip, use_gpu=args.gpu)
    tracker.process_video(source)

if __name__ == "__main__":
    main()
