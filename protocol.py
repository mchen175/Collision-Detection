"""Bounded Debris Native v1 decoder. See ../docs/PROTOCOL.md.

TCP is a byte stream: packets may be split anywhere or arrive together. No
third-party sensor-stream framing is accepted by this decoder.
"""
from dataclasses import dataclass
import json
import math
import struct
import uuid

import numpy as np


MAGIC = b"DBR1"
HEADER = struct.Struct("!4sII")
MAX_METADATA = 65536
MAX_PAYLOAD = 8 * 1024 * 1024


class ProtocolError(ValueError):
    """The peer sent a frame that cannot be safely interpreted."""


@dataclass(frozen=True)
class NativeFrame:
    session_id: str
    sequence: int
    timestamp_s: float
    depth_m: np.ndarray
    confidence: np.ndarray | None
    image_data: bytes
    camera_width: int
    camera_height: int
    intrinsics: dict
    camera_transform: np.ndarray
    tracking_state: str
    tracking_reason: str
    motion: dict | None = None


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ProtocolError(f"Invalid {name}")
    return value


def _number(value, name, low=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"Invalid {name}")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ProtocolError(f"Invalid {name}") from exc
    if not math.isfinite(number) or (low is not None and number < low):
        raise ProtocolError(f"Invalid {name}")
    return number


def _vector(value, name, count):
    if not isinstance(value, list) or len(value) != count:
        raise ProtocolError(f"Invalid {name}")
    return [_number(v, name) for v in value]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"Duplicate metadata key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ProtocolError(f"Non-finite JSON constant: {value}")


def decode_frame(metadata, payload):
    """Validate one complete packet and return independent NumPy buffers."""
    try:
        meta = json.loads(metadata.decode("utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ProtocolError(f"Invalid frame metadata: {exc}") from exc
    if not isinstance(meta, dict):
        raise ProtocolError("Frame metadata must be an object")
    required = ("schema_version", "type", "session_id", "sequence", "timestamp_s",
                "depth_width", "depth_height", "depth_encoding", "depth_bytes",
                "confidence_bytes", "jpeg_bytes", "camera_width", "camera_height",
                "intrinsics", "camera_transform", "tracking_state", "tracking_reason")
    if any(key not in meta for key in required):
        raise ProtocolError("Frame metadata is missing required fields")
    if type(meta["schema_version"]) is not int or meta["schema_version"] != 1:
        raise ProtocolError("Unsupported Debris Native schema version")
    if meta["type"] != "frame":
        raise ProtocolError("Unsupported packet type")
    session_id = meta["session_id"]
    if not isinstance(session_id, str) or len(session_id) > 64:
        raise ProtocolError("Invalid session_id")
    try:
        uuid.UUID(session_id)
    except (ValueError, AttributeError) as exc:
        raise ProtocolError("Invalid session_id UUID") from exc
    sequence = _integer(meta["sequence"], "sequence", 0, 2**63 - 1)
    timestamp = _number(meta["timestamp_s"], "timestamp_s", 0)
    width = _integer(meta["depth_width"], "depth_width", 1, 2048)
    height = _integer(meta["depth_height"], "depth_height", 1, 2048)
    if meta["depth_encoding"] != "float32_le_m":
        raise ProtocolError("Unsupported depth encoding")
    pixels = width * height
    depth_bytes = _integer(meta["depth_bytes"], "depth_bytes", 1, MAX_PAYLOAD)
    confidence_bytes = _integer(meta["confidence_bytes"], "confidence_bytes", 0, MAX_PAYLOAD)
    jpeg_bytes = _integer(meta["jpeg_bytes"], "jpeg_bytes", 0, MAX_PAYLOAD)
    if (depth_bytes != 4 * pixels or confidence_bytes not in (0, pixels)
            or depth_bytes + confidence_bytes + jpeg_bytes != len(payload)):
        raise ProtocolError("Frame byte counts do not match dimensions or payload")
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError("Frame payload is too large")
    camera_width = _integer(meta["camera_width"], "camera_width", 0, 4096)
    camera_height = _integer(meta["camera_height"], "camera_height", 0, 4096)
    if jpeg_bytes and (camera_width == 0 or camera_height == 0):
        raise ProtocolError("JPEG dimensions must be positive")
    intrinsics = meta["intrinsics"]
    if not isinstance(intrinsics, dict) or any(k not in intrinsics for k in ("fx", "fy", "cx", "cy")):
        raise ProtocolError("Invalid depth intrinsics")
    intrinsics = {k: _number(intrinsics[k], f"intrinsics.{k}") for k in ("fx", "fy", "cx", "cy")}
    if (intrinsics["fx"] <= 0 or intrinsics["fy"] <= 0
            or not 0 <= intrinsics["cx"] < width or not 0 <= intrinsics["cy"] < height):
        raise ProtocolError("Depth intrinsics fall outside the depth geometry")
    transform = np.array(_vector(meta["camera_transform"], "camera_transform", 16), dtype=float).reshape((4, 4), order="F")
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-4)
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-3)
            or abs(float(np.linalg.det(rotation)) - 1) > 1e-3):
        raise ProtocolError("Camera transform must be a rigid column-major camera-to-world matrix")
    tracking_state = meta["tracking_state"]
    if tracking_state not in ("normal", "limited", "notAvailable"):
        raise ProtocolError("Invalid tracking_state")
    reason = meta["tracking_reason"]
    if not isinstance(reason, str) or len(reason) > 1024:
        raise ProtocolError("Invalid tracking_reason")
    motion = meta.get("motion")
    if motion is not None:
        if not isinstance(motion, dict) or any(k not in motion for k in ("timestamp_s", "rotation_rate_rads", "gravity_g", "user_accel_g")):
            raise ProtocolError("Invalid motion metadata")
        motion = {"timestamp_s": _number(motion["timestamp_s"], "motion.timestamp_s", 0),
                  **{k: _vector(motion[k], f"motion.{k}", 3) for k in ("rotation_rate_rads", "gravity_g", "user_accel_g")}}
    depth = np.frombuffer(payload, dtype="<f4", count=pixels).reshape((height, width)).copy()
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    confidence = None
    if confidence_bytes:
        confidence = np.frombuffer(payload, dtype=np.uint8, count=pixels, offset=depth_bytes).reshape((height, width)).copy()
        if np.any(confidence > 2):
            raise ProtocolError("Confidence map contains an unknown confidence level")
    image = bytes(payload[depth_bytes + confidence_bytes:])
    return NativeFrame(session_id, sequence, timestamp, depth, confidence, image,
                       camera_width, camera_height, intrinsics, transform,
                       tracking_state, reason, motion)


class NativeDecoder:
    """Incremental length-framed decoder with at most one packet buffered."""
    def __init__(self):
        self.buffer = bytearray()
        self.expected = HEADER.size
        self.lengths = None

    def feed(self, data):
        view = memoryview(data)
        offset = 0
        output = []
        while offset < len(view):
            take = min(self.expected - len(self.buffer), len(view) - offset)
            self.buffer.extend(view[offset:offset + take])
            offset += take
            if len(self.buffer) != self.expected:
                continue
            if self.lengths is None:
                magic, metadata_length, payload_length = HEADER.unpack(self.buffer)
                if magic != MAGIC:
                    raise ProtocolError("Unknown stream magic; expected Debris Native DBR1")
                if not 1 <= metadata_length <= MAX_METADATA or not 1 <= payload_length <= MAX_PAYLOAD:
                    raise ProtocolError("Invalid or oversized frame lengths")
                self.lengths = (metadata_length, payload_length)
                self.expected = HEADER.size + metadata_length + payload_length
            else:
                metadata_length, _ = self.lengths
                split = HEADER.size + metadata_length
                output.append(decode_frame(bytes(self.buffer[HEADER.size:split]), bytes(self.buffer[split:])))
                self.buffer.clear()
                self.expected = HEADER.size
                self.lengths = None
        return output

    def finish(self):
        """Reject an EOF halfway through a packet; clean EOF is allowed."""
        if self.buffer:
            raise ProtocolError("Stream ended during an incomplete frame")
