import json
from bisect import bisect_right
from pathlib import Path


class ServoPointingCalibration:
    """Piecewise-linear full-frame pixel to absolute servo-angle mapping."""

    def __init__(self, path, frame_size):
        self.path = Path(path)
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.pan_points = []
        self.tilt_points = []
        self.loaded = False
        self.error = None
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            stored_size = tuple(int(value) for value in data.get("frame_size", ()))
            if stored_size and stored_size != self.frame_size:
                raise ValueError(
                    f"frame_size mismatch: calibration={stored_size}, runtime={self.frame_size}"
                )
            self.pan_points = self._validate_axis(data.get("pan"), "pan")
            self.tilt_points = self._validate_axis(data.get("tilt"), "tilt")
            self.loaded = True
        except Exception as exc:
            self.error = str(exc)
            self.pan_points = []
            self.tilt_points = []

    @staticmethod
    def _validate_axis(raw_points, axis_name):
        if not isinstance(raw_points, list) or len(raw_points) < 2:
            raise ValueError(f"{axis_name} requires at least two calibration points")
        points = sorted(
            (float(item["pixel"]), float(item["servo_deg"]))
            for item in raw_points
        )
        pixels = [pixel for pixel, _ in points]
        if len(set(pixels)) != len(pixels):
            raise ValueError(f"{axis_name} contains duplicate pixel positions")
        return points

    @staticmethod
    def _interpolate(points, pixel):
        pixel = float(pixel)
        if pixel <= points[0][0]:
            return points[0][1]
        if pixel >= points[-1][0]:
            return points[-1][1]

        right = bisect_right([item[0] for item in points], pixel)
        x0, y0 = points[right - 1]
        x1, y1 = points[right]
        ratio = (pixel - x0) / max(x1 - x0, 1e-9)
        return y0 + (y1 - y0) * ratio

    def map_pixel(self, target_px):
        if not self.loaded:
            return None
        x, y = target_px
        return (
            self._interpolate(self.pan_points, x),
            self._interpolate(self.tilt_points, y),
        )
