"""Single slow target; fixed camera; finite-horizon educational prediction."""
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    method: str = "supplied calibration"

    def __post_init__(self):
        values = (self.fx, self.fy, self.cx, self.cy)
        if (not isinstance(self.width, int) or not isinstance(self.height, int)
                or not 1 <= self.width <= 2048 or not 1 <= self.height <= 2048
                or not np.isfinite(values).all() or self.fx <= 0 or self.fy <= 0
                or not 0 <= self.cx < self.width or not 0 <= self.cy < self.height):
            raise ValueError("Invalid calibration")

    def points(self, depth):
        if depth.shape != (self.height, self.width):
            raise ValueError("Depth resolution changed; calibrate again")
        y, x = np.indices(depth.shape)
        return np.stack(((x - self.cx) * depth / self.fx,
                         -(y - self.cy) * depth / self.fy, depth), axis=-1)


def board_calibration(depth, box, board_width, board_height):
    """Approximation: centered principal point, flat fronto-parallel rectangle."""
    h, w = depth.shape
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 < w and 0 <= y0 < y1 < h and x1-x0 >= 12 and y1-y0 >= 12):
        raise ValueError("Select both opposite board corners, at least 12 pixels apart")
    if not (np.isfinite([board_width, board_height]).all()
            and 0.05 <= board_width <= 2 and 0.05 <= board_height <= 2):
        raise ValueError("Enter the measured board width and height in meters")
    dx, dy = max(1, (x1-x0)//5), max(1, (y1-y0)//5)
    region = depth[y0+dy:y1-dy, x0+dx:x1-dx]
    good = region[np.isfinite(region) & (region >= .25) & (region <= 4)]
    if good.size < 20 or good.size < region.size * .8:
        raise ValueError("Not enough valid board depth; try a larger/closer board")
    distance = float(np.median(good))
    if np.median(np.abs(good - distance)) > .025:
        raise ValueError("Board depth is not flat/stable enough")
    return Intrinsics(w, h, (x1-x0)*distance/board_width,
                      (y1-y0)*distance/board_height, (w-1)/2, (h-1)/2,
                      "approximate flat-board calibration")


@dataclass(frozen=True)
class MotionConfig:
    horizon: float = 3.0
    radius: float = .30
    acceleration: float = .40
    burn: float = 1.0
    latency: float = .20
    step: float = .025
    position_allowance: float = .05


AXES = {"HOLD": np.array([0., 0., 0.]),
        "+X": np.array([1., 0., 0.]), "-X": np.array([-1., 0., 0.]),
        "+Z": np.array([0., 0., 1.]), "-Z": np.array([0., 0., -1.])}


def displacement(t, config=MotionConfig()):
    elapsed = np.maximum(np.asarray(t) - config.latency, 0)
    return np.where(elapsed < config.burn, .5*config.acceleration*elapsed**2,
                    config.acceleration*config.burn*(elapsed-.5*config.burn))


def prediction(r, v, config=MotionConfig()):
    r, v = np.asarray(r, float), np.asarray(v, float)
    if r.shape != (3,) or v.shape != (3,) or not np.isfinite([r, v]).all():
        raise ValueError("Position and velocity must be finite 3D vectors")
    vv = float(v @ v)
    tcpa = 0. if vv < 1e-10 else float(np.clip(-(r @ v)/vv, 0, config.horizon))
    dcpa = float(np.linalg.norm(r + v*tcpa))
    times = np.linspace(0, config.horizon, int(np.ceil(config.horizon/config.step))+1)
    actual_step = float(times[1]-times[0])
    scores = {}
    for name, axis in AXES.items():
        path = r + times[:, None]*v - displacement(times, config)[:, None]*axis
        max_own_speed = 0 if name == "HOLD" else config.acceleration*config.burn
        bound = (float(np.linalg.norm(v))+max_own_speed)*actual_step/2
        scores[name] = max(0., float(np.linalg.norm(path, axis=1).min())
                           - bound - config.position_allowance)
    threat = scores["HOLD"] <= config.radius
    action = "HOLD"
    if threat:
        action = max(scores, key=scores.get)
        if scores[action] <= config.radius or np.linalg.norm(r) <= config.radius:
            action = "NONE"
    return {"state": "THREAT" if threat else "CLEAR",
            "tcpa": tcpa, "dcpa": dcpa, "action": action,
            "score": scores.get(action), "position": r.tolist(), "velocity": v.tolist(),
            "scores": scores, "config": asdict(config)}


class TrackLost(ValueError):
    pass


class TargetTracker:
    """Depth connected-component tracking, after a user selects one isolated object.

    This cannot prove identity. The operator must monitor the selected component.
    The current point represents a visible surface, not the object's center of mass.
    """
    def __init__(self):
        self.history = deque(maxlen=8)
        self.last_position = None
        self.last_size = None
        self.box = None
        self.velocity = np.zeros(3)

    def _component(self, points, anchor, seed):
        good = np.isfinite(points).all(axis=2)
        good &= (points[:, :, 2] >= .25) & (points[:, :, 2] <= 4)
        good &= np.linalg.norm(points-anchor, axis=2) <= .28
        good &= np.abs(points[:, :, 2]-anchor[2]) <= .12
        h, w = good.shape
        x, y = seed
        if not (0 <= x < w and 0 <= y < h and good[y, x]):
            raise TrackLost("Selected point has no usable depth")
        mask = np.zeros_like(good)
        stack = [(x, y)]
        mask[y, x] = True
        coords = []
        while stack:
            x, y = stack.pop()
            coords.append((x, y))
            if len(coords) > w*h*.20:
                raise TrackLost("Selection merged with a large surface/background")
            for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if 0 <= nx < w and 0 <= ny < h and good[ny,nx] and not mask[ny,nx]:
                    mask[ny,nx] = True
                    stack.append((nx,ny))
        if len(coords) < 8:
            raise TrackLost("Too few target depth pixels")
        xy = np.array(coords)
        values = points[mask]
        if np.max(np.ptp(values, axis=0)) > .45:
            raise TrackLost("Selection is too large for this small-target demo")
        position = np.median(values, axis=0)
        if np.median(np.abs(values[:,2]-position[2])) > .045:
            raise TrackLost("Target depth spread is too large")
        box = [int(xy[:,0].min()), int(xy[:,1].min()),
               int(xy[:,0].max()), int(xy[:,1].max())]
        return position, len(coords), box

    def select(self, depth, calibration, pixel, timestamp):
        points = calibration.points(depth)
        x, y = pixel
        h, w = depth.shape
        if not (0 <= x < w and 0 <= y < h):
            raise TrackLost("Select a point inside the depth image")
        anchor = points[y,x]
        if not np.isfinite(anchor).all():
            raise TrackLost("Click the interior of a target with valid depth")
        position, size, box = self._component(points, anchor, (x,y))
        self.history.clear()
        self.history.append((timestamp, position))
        self.last_position, self.last_size, self.box = position, size, box
        self.velocity = np.zeros(3)

    def update(self, depth, calibration, timestamp):
        if self.last_position is None:
            raise TrackLost("Select a target first")
        dt = timestamp-self.history[-1][0]
        if dt <= 0 or dt > .6:
            raise TrackLost("Missing, repeated or out-of-order capture time; select again")
        points = calibration.points(depth)
        expected = self.last_position + self.velocity*dt
        distances = np.linalg.norm(points-expected, axis=2)
        distances[~np.isfinite(distances)] = np.inf
        index = np.unravel_index(int(np.argmin(distances)), distances.shape)
        if distances[index] > .25:
            raise TrackLost("Selected target disappeared or moved too quickly")
        y, x = index
        position, size, box = self._component(points, points[y,x], (x,y))
        if not .4 <= size/self.last_size <= 2.5:
            raise TrackLost("Target shape/size changed abruptly; select again")
        if np.linalg.norm(position-self.last_position) > 1.5*dt+.025:
            raise TrackLost("Target position jumped; select again")
        self.history.append((timestamp, position))
        self.last_position, self.last_size, self.box = position, size, box
        if len(self.history) < 5 or timestamp-self.history[0][0] < .6:
            return {"state": "WAIT", "message": "Collecting at least 0.6 s of target motion"}
        times = np.array([t for t, _ in self.history])
        times -= times.mean()
        positions = np.array([p for _, p in self.history])
        self.velocity = (times[:,None]*positions).sum(axis=0)/(times @ times)
        if np.linalg.norm(self.velocity) > 1.5:
            raise TrackLost("Target exceeds this slow-motion demo's speed limit")
        result = prediction(position, self.velocity)
        result["message"] = "One selected target; fixed-camera model"
        return result
