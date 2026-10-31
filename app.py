"""Native iPhone LiDAR receiver. Run app.py --demo or app.py --transport usb."""
import argparse
from collections import deque
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import ipaddress
import json
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urlparse
import webbrowser

import numpy as np
from PIL import Image, ImageDraw

from model import Intrinsics, TargetTracker, TrackLost
from protocol import NativeDecoder, NativeFrame, ProtocolError

ROOT = Path(__file__).resolve().parent
MAX_GAP = .6
MAX_TRANSLATION = .02
MAX_ROTATION = np.deg2rad(2.)


def pose_change(reference, current):
    """Camera-to-world poses; translation in meters and rotation in radians."""
    translation = float(np.linalg.norm(current[:3, 3]-reference[:3, 3]))
    relative = reference[:3, :3].T @ current[:3, :3]
    rotation = float(np.arccos(np.clip((np.trace(relative)-1)/2, -1., 1.)))
    return translation, rotation


class State:
    def __init__(self, demo=False):
        self.lock = threading.RLock()
        self.demo = demo
        self.generation = 0
        self.scenario = "approach"
        self.connection = "Waiting for DebrisCapture"
        self._reset_stream()

    def _reset_stream(self):
        self.frame = None
        self.raw_depth = None
        self.camera = None
        self.camera_error = None
        self.depth_count = self.camera_count = 0
        self.calibration = None
        self.tracker = None
        self.reference_pose = None
        self.packet = None
        self.stream_session = None
        self.last_capture = self.last_sequence = self.last_processed = self.baseline_offset = None
        self.receipts = deque(maxlen=300)
        self.sensor = {"tracking_state": "waiting", "tracking_reason": "",
                       "gyro_rads": None, "gravity_g": None, "user_accel_g": None,
                       "motion_age_s": None, "confidence_available": False,
                       "pose_gate": "Waiting for camera pose", "intrinsics_source": "Waiting for ARKit",
                       "usable_depth_pixels": 0}
        self.result = {"state": "UNKNOWN", "message": "Start capture in DebrisCapture on the iPhone"}

    def connected(self, peer):
        with self.lock:
            self.generation += 1
            self._reset_stream()
            self.connection = peer
            self.result = {"state": "UNKNOWN", "message": "Connected; waiting for native LiDAR frames"}

    def disconnected(self, reason):
        with self.lock:
            self._reset_stream()
            self.connection = reason
            self.clear("Connection lost; reconnect, then select the target again")

    def clear(self, reason="Select one small isolated target in the depth image"):
        with self.lock:
            self.tracker = None
            self.reference_pose = None
            self.last_processed = None
            self.result = {"state": "UNKNOWN", "message": reason}

    def expire(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.frame is not None and now-self.frame[2] > MAX_GAP:
                self.sensor["pose_gate"] = "Depth stream is stale"
                self.clear("Depth stream is stale; resume streaming and select the target again")

    def _camera(self, packet, now):
        if not packet.image_data:
            return
        try:
            with Image.open(io.BytesIO(packet.image_data)) as image:
                if image.format != "JPEG" or image.size != (packet.camera_width, packet.camera_height):
                    raise ValueError("JPEG dimensions or format disagree with the native stream")
                image.load()
            self.camera = (packet.image_data, [packet.camera_width, packet.camera_height], now)
            self.camera_count += 1
            self.camera_error = None
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            self.camera = None
            self.camera_error = f"Camera preview unavailable: {exc}"

    def ingest(self, packet, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            new_session = self.stream_session != packet.session_id
            if new_session:
                # A capture restart changes ARKit world origin and capture sequence.
                if self.stream_session is not None:
                    self.generation += 1
                    self.clear("iPhone capture restarted; select the target again")
                self.stream_session = packet.session_id
                self.last_capture = self.last_sequence = self.baseline_offset = None
                self.calibration = None
            if self.last_sequence is not None and packet.sequence <= self.last_sequence:
                self.clear("Frame sequence repeated/backwards; restart capture and select again")
                return
            if self.last_capture is not None and packet.timestamp_s <= self.last_capture:
                self.clear("Capture timestamps repeated/backwards; restart capture and select again")
                return
            capture_gap = (self.last_capture is not None and packet.timestamp_s-self.last_capture > MAX_GAP)
            arrival_gap = self.frame is not None and now-self.frame[2] > MAX_GAP
            offset = now-packet.timestamp_s
            self.baseline_offset = offset if self.baseline_offset is None else min(self.baseline_offset, offset)
            self.last_capture, self.last_sequence = packet.timestamp_s, packet.sequence
            if offset-self.baseline_offset > .5:
                self.sensor["pose_gate"] = "Stream is falling behind"
                self.clear("Stream is falling behind; restart capture and select again")
                return
            if capture_gap or arrival_gap:
                self.clear("Stream gap; select the target again")
            depth = packet.depth_m.copy()
            confidence_available = packet.confidence is not None
            if confidence_available:
                depth[packet.confidence < 1] = np.nan
            else:
                # Raw preview remains visible; prediction requires confidence data.
                depth[:] = np.nan
            h, w = depth.shape
            calibration = Intrinsics(w, h, **packet.intrinsics, method="ARKit intrinsics scaled to depth map")
            if self.calibration is not None:
                old = self.calibration
                geometry_changed = ((old.width, old.height) != (w, h)
                                    or abs(old.fx/calibration.fx-1) > .02
                                    or abs(old.fy/calibration.fy-1) > .02
                                    or abs(old.cx-calibration.cx) > 1
                                    or abs(old.cy-calibration.cy) > 1)
                if geometry_changed:
                    self.clear("Camera calibration changed substantially; select the target again")
            self.calibration = calibration
            self.raw_depth = packet.depth_m
            self.frame = (depth, packet.timestamp_s, now, "float32_le_m")
            self.packet = packet
            self.depth_count += 1
            self.receipts.append(now)
            motion = packet.motion or {}
            motion_age = packet.timestamp_s-motion["timestamp_s"] if motion else None
            self.sensor.update({"tracking_state": packet.tracking_state,
                                "tracking_reason": packet.tracking_reason,
                                "gyro_rads": motion.get("rotation_rate_rads"),
                                "gravity_g": motion.get("gravity_g"),
                                "user_accel_g": motion.get("user_accel_g"),
                                "motion_age_s": motion_age,
                                "confidence_available": confidence_available,
                                "intrinsics_source": calibration.method,
                                "usable_depth_pixels": int(np.count_nonzero(np.isfinite(depth) & (depth >= .25) & (depth <= 4)))})
            self._camera(packet, now)
            if not confidence_available:
                self.sensor["pose_gate"] = "Confidence map unavailable; preview only"
                self.clear("LiDAR confidence is unavailable; prediction requires a confidence map")
                return
            if packet.tracking_state != "normal":
                self.sensor["pose_gate"] = "ARKit tracking is not normal"
                self.clear("ARKit tracking is limited/unavailable; hold the phone fixed, then select again")
                return
            self.sensor["pose_gate"] = "Fixed-phone check ready; select a target"
            if self.tracker is None:
                if self.result["message"] in ("Connected; waiting for native LiDAR frames", "Start capture in DebrisCapture on the iPhone"):
                    self.result = {"state": "READY", "message": "ARKit calibration ready. Keep the phone fixed and select one small target"}
                return
            translation, rotation = pose_change(self.reference_pose, packet.camera_transform)
            if translation > MAX_TRANSLATION or rotation > MAX_ROTATION:
                self.sensor["pose_gate"] = f"Phone moved {translation*100:.1f} cm / {np.rad2deg(rotation):.1f} degrees"
                self.clear("Phone moved since selection; fix its position and select the target again")
                return
            self.sensor["pose_gate"] = f"Phone fixed: {translation*100:.1f} cm / {np.rad2deg(rotation):.1f} degrees since selection"
            if self.last_processed is not None and packet.timestamp_s-self.last_processed < .18:
                return
            self.last_processed = packet.timestamp_s
            try:
                self.result = self.tracker.update(depth, calibration, packet.timestamp_s)
            except (TrackLost, ValueError) as exc:
                self.clear(str(exc))

    def select(self, pixel, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            self.expire(now)
            if self.frame is None or now-self.frame[2] > MAX_GAP:
                raise ValueError("Need a fresh depth stream")
            if self.packet.tracking_state != "normal":
                raise ValueError("Wait for normal ARKit tracking before selection")
            if self.packet.confidence is None:
                raise ValueError("A LiDAR confidence map is required for prediction")
            if not isinstance(pixel, list) or len(pixel) != 2 or any(type(v) is not int for v in pixel):
                raise ValueError("Select an integer depth pixel")
            tracker = TargetTracker()
            tracker.select(self.frame[0], self.calibration, tuple(pixel), self.frame[1])
            self.tracker = tracker
            self.reference_pose = self.packet.camera_transform.copy()
            self.last_processed = self.frame[1]
            self.result = {"state": "WAIT", "message": "Selected. Keep the phone fixed and move the target slowly"}
            self.sensor["pose_gate"] = "Phone pose saved at target selection"

    def snapshot(self):
        self.expire()
        with self.lock:
            now = time.monotonic()
            recent = [t for t in self.receipts if now-t < 2]
            fps = (len(recent)-1)/(recent[-1]-recent[0]) if len(recent)>1 else 0.
            dimensions = [self.frame[0].shape[1], self.frame[0].shape[0]] if self.frame else [0, 0]
            camera_age = max(0., now-self.camera[2]) if self.camera else None
            camera_status = "live" if camera_age is not None and camera_age <= .75 else "stale" if self.camera else "error" if self.camera_error else "waiting"
            sensor = dict(self.sensor)
            if sensor["motion_age_s"] is not None and self.frame is not None:
                # Add time since receipt so a frozen stream cannot show fresh gyro.
                sensor["motion_age_s"] = max(0., sensor["motion_age_s"]) + max(0., now-self.frame[2])
            return {**self.result, "demo": self.demo, "connection": self.connection,
                    "dimensions": dimensions,
                    "arrival_age": max(0., now-self.frame[2]) if self.frame else None,
                    "fps": fps, "calibration": asdict(self.calibration) if self.calibration else None,
                    "box": self.tracker.box if self.tracker else None,
                    "scenario": self.scenario, "depth_frames": self.depth_count, "session": self.generation,
                    "frame_sequence": self.last_sequence, "capture_session": self.stream_session,
                    "usable_depth_pixels": self.sensor["usable_depth_pixels"], "sensor": sensor,
                    "camera": {"status": camera_status, "frames": self.camera_count,
                               "dimensions": self.camera[1] if self.camera else [0, 0],
                               "arrival_age": camera_age, "error": self.camera_error}}

    def camera_image(self):
        with self.lock:
            if self.camera:
                return self.camera[0], "image/jpeg"
        image = Image.new("RGB", (640, 360), "#182234")
        ImageDraw.Draw(image).text((24, 170), "Start capture in DebrisCapture on the iPhone", fill="white")
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue(), "image/png"

    def png(self):
        with self.lock:
            if self.frame is None:
                image = Image.new("RGB", (512, 384), "#182234")
                ImageDraw.Draw(image).text((24, 180), "Waiting for native LiDAR frames", fill="white")
            else:
                # Show the raw depth when confidence is absent, but never track it.
                depth = self.frame[0] if self.packet.confidence is not None else self.raw_depth
                valid = np.isfinite(depth) & (depth > 0)
                p = np.nan_to_num(np.clip(depth/4, 0, 1), nan=0.)
                rgb = np.stack((255*(1-p), 200*(1-np.abs(2*p-1)), 255*p), axis=-1).astype(np.uint8)
                rgb[~valid] = [12, 17, 28]
                image = Image.fromarray(rgb)
                draw = ImageDraw.Draw(image)
                if self.tracker and self.tracker.box:
                    draw.rectangle(tuple(self.tracker.box), outline="#75f0ba", width=2)
                if time.monotonic()-self.frame[2] > MAX_GAP:
                    draw.rectangle((0, 0, image.width, 22), fill="#8c2936")
                    draw.text((6, 5), "STALE - select again after recovery", fill="white")
            output = io.BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()


def receive_packets(client, state, stop):
    client.settimeout(.25)
    decoder = NativeDecoder()
    last_bytes = time.monotonic()
    while not stop.is_set():
        try:
            data = client.recv(65536)
        except socket.timeout:
            state.expire()
            if time.monotonic()-last_bytes > 3:
                decoder.finish()
                return
            continue
        if not data:
            decoder.finish()
            return
        last_bytes = time.monotonic()
        for packet in decoder.feed(data):
            state.ingest(packet)
        state.expire()


class NativeClient:
    """Mac initiates both USB-tunnel and direct Wi-Fi connections."""
    def __init__(self, state, host="127.0.0.1", port=8765, transport="usb"):
        self.state, self.host, self.port, self.transport = state, host, port, transport
        self.stop, self.ready = threading.Event(), threading.Event()
        self.error = None
        self.client = None

    def run(self):
        self.ready.set()
        while not self.stop.is_set():
            try:
                with socket.create_connection((self.host, self.port), timeout=.8) as client:
                    self.client = client
                    self.state.connected(f"{self.transport.upper()} phone stream {self.host}:{self.port}")
                    receive_packets(client, self.state, self.stop)
                    self.state.disconnected("Phone stream ended; retrying connection")
            except (OSError, ProtocolError) as exc:
                self.state.disconnected(f"Waiting for {self.transport} {self.host}:{self.port}: {exc}")
            finally:
                self.client = None
            self.stop.wait(.5)

    def close(self):
        self.stop.set()
        if self.client is not None:
            try:
                self.client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def demo_frame(elapsed, scenario="approach"):
    cal = Intrinsics(256, 192, 180., 180., 127.5, 95.5, "synthetic ground truth")
    depth = np.full((192, 256), 4., dtype=np.float32)
    z = max(.24, 2.1-.30*elapsed)
    x = .025 if scenario == "approach" else .55
    cx, cy = int(round(cal.cx+cal.fx*x/z)), int(round(cal.cy))
    half = max(3, int(round(cal.fx*.06/z)))
    depth[max(0, cy-half):min(192, cy+half+1), max(0, cx-half):min(256, cx+half+1)] = z
    return cal, depth, [min(255, max(0, cx)), cy]


def run_demo(state, stop):
    import uuid
    previous_scenario, start, sequence, session = None, 0, 0, str(uuid.uuid4())
    while not stop.is_set():
        now = time.monotonic()
        with state.lock:
            scenario = state.scenario
        restart = scenario != previous_scenario or now-start > (6.4 if scenario == "approach" else 3.5)
        if restart:
            start, previous_scenario, sequence, session = now, scenario, 0, str(uuid.uuid4())
            state.connected("Synthetic iPhone - no device required")
        cal, depth, pixel = demo_frame(now-start, scenario)
        scene = Image.new("RGB", (512, 384), "#263a4f")
        draw = ImageDraw.Draw(scene)
        for y in range(0, 384, 48):
            draw.line((0, y, 512, y), fill="#35516a")
        ys, xs = np.where(depth < 3.9)
        if len(xs):
            draw.rectangle((int(xs.min())*2, int(ys.min())*2, int(xs.max())*2, int(ys.max())*2), fill="#eead70")
        draw.text((18, 18), "SIMULATED CAMERA - not physical footage", fill="white")
        image_bytes = io.BytesIO()
        scene.save(image_bytes, format="JPEG")
        packet = NativeFrame(session_id=session, sequence=sequence, timestamp_s=now,
                             depth_m=depth, confidence=np.full(depth.shape, 2, dtype=np.uint8),
                             image_data=image_bytes.getvalue(), camera_width=512, camera_height=384,
                             intrinsics={key: getattr(cal, key) for key in ("fx", "fy", "cx", "cy")},
                             camera_transform=np.eye(4), tracking_state="normal", tracking_reason="",
                             motion=None)
        state.ingest(packet, now)
        if restart:
            state.select(pixel, now)
        sequence += 1
        stop.wait(.1)


def make_handler(state, phone_port=8765, transport="usb", phone_host="127.0.0.1"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, content, mime):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send(200, (ROOT/"index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/state":
                value = {**state.snapshot(), "phone_port": 8765 if transport == "usb" else phone_port, "usb_port": phone_port,
                         "phone_host": phone_host, "transport": "demo" if state.demo else transport}
                self.send(200, json.dumps(value, allow_nan=False).encode(), "application/json")
            elif path == "/depth.png":
                self.send(200, state.png(), "image/png")
            elif path == "/camera":
                data, mime = state.camera_image()
                self.send(200, data, mime)
            else:
                self.send(404, b"Not found", "text/plain")

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                self.send(403, b"Local viewer only", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request size")
                value = json.loads(self.rfile.read(length))
                if self.path == "/select":
                    state.select(value["pixel"])
                elif self.path == "/reset":
                    state.clear()
                elif self.path == "/scenario" and state.demo:
                    if value["scenario"] not in ("approach", "miss"):
                        raise ValueError("Unknown scenario")
                    with state.lock:
                        state.scenario = value["scenario"]
                else:
                    self.send(404, b"Not found", "text/plain")
                    return
                self.send(200, b'{"ok":true}', "application/json")
            except (ValueError, KeyError, TypeError, OverflowError, OSError) as exc:
                self.send(400, json.dumps({"error": str(exc)}).encode(), "application/json")
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Synthetic camera/depth; no phone")
    parser.add_argument("--transport", choices=("usb", "wifi"), default="usb")
    parser.add_argument("--host", default="127.0.0.1", help="Phone IPv4 for Wi-Fi; localhost for USB")
    parser.add_argument("--port", "--usb-port", dest="port", type=int, default=8765)
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    if not all(1 <= n <= 65535 for n in (args.port, args.web_port)):
        parser.error("Ports must be between 1 and 65535")
    try:
        address = ipaddress.ip_address(args.host)
    except ValueError:
        parser.error("Use a numeric phone IP address (USB uses 127.0.0.1)")
    if address.version != 4:
        parser.error("Use an IPv4 address")
    if args.transport == "usb" and not address.is_loopback:
        parser.error("USB connects only to the local iproxy tunnel at 127.0.0.1")
    state = State(demo=args.demo)
    stop = threading.Event()
    receiver = None if args.demo else NativeClient(state, args.host, args.port, args.transport)
    worker = threading.Thread(target=run_demo, args=(state, stop), daemon=True) if args.demo else threading.Thread(target=receiver.run, daemon=True)
    worker.start()
    print("SIMULATED DATA" if args.demo else f"DebrisCapture {args.transport.upper()}: connecting to {args.host}:{args.port}")
    print(f"Viewer: http://127.0.0.1:{args.web_port}")
    try:
        with ThreadingHTTPServer(("127.0.0.1", args.web_port), make_handler(state, args.port, args.transport, args.host)) as http:
            if args.open_browser:
                timer = threading.Timer(.4, webbrowser.open, args=(f"http://127.0.0.1:{args.web_port}",))
                timer.daemon = True
                timer.start()
            http.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        parser.error(f"Viewer could not start: {exc}")
    finally:
        stop.set()
        if receiver:
            receiver.close()
        worker.join(2)


if __name__ == "__main__":
    main()
