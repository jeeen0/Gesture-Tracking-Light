import json
from bisect import bisect_right
from pathlib import Path


class ServoPointingCalibration:
    """Pixel LUT for 2D residual, axis residual, or legacy absolute mapping."""

    def __init__(self, path, frame_size):
        self.path = Path(path)
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.pan_points = []
        self.tilt_points = []
        self.grid_x = []
        self.grid_y = []
        self.grid_values = {}
        self.mode = None
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
            requested_mode = str(data.get("mode", "")).strip().lower()
            if not requested_mode:
                # Files created by the first calibration implementation stored
                # final absolute angles without an explicit mode.
                requested_mode = "absolute"
            if requested_mode not in {"residual_2d", "residual", "absolute"}:
                raise ValueError(f"unsupported calibration mode: {requested_mode}")

            if requested_mode == "residual_2d":
                self.grid_x, self.grid_y, self.grid_values = self._validate_grid(
                    data.get("points")
                )
            else:
                value_key = (
                    "correction_deg"
                    if requested_mode == "residual"
                    else "servo_deg"
                )
                self.pan_points = self._validate_axis(
                    data.get("pan"),
                    "pan",
                    value_key,
                )
                self.tilt_points = self._validate_axis(
                    data.get("tilt"),
                    "tilt",
                    value_key,
                )
            self.mode = requested_mode
            self.loaded = True
        except Exception as exc:
            self.error = str(exc)
            self.pan_points = []
            self.tilt_points = []
            self.grid_x = []
            self.grid_y = []
            self.grid_values = {}
            self.mode = None

    @staticmethod
    def _validate_axis(raw_points, axis_name, value_key):
        if not isinstance(raw_points, list) or len(raw_points) < 2:
            raise ValueError(f"{axis_name} requires at least two calibration points")
        points = sorted(
            (float(item["pixel"]), float(item[value_key]))
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

    @staticmethod
    def _validate_grid(raw_points):
        if not isinstance(raw_points, list) or len(raw_points) < 4:
            raise ValueError("residual_2d requires a rectangular point grid")

        values = {}
        for item in raw_points:
            x = float(item["x"])
            y = float(item["y"])
            key = (x, y)
            if key in values:
                raise ValueError(f"duplicate residual_2d point: x={x}, y={y}")
            values[key] = (
                float(item["pan_correction_deg"]),
                float(item["tilt_correction_deg"]),
            )

        grid_x = sorted({x for x, _ in values})
        grid_y = sorted({y for _, y in values})
        if len(grid_x) < 2 or len(grid_y) < 2:
            raise ValueError("residual_2d requires at least a 2x2 grid")
        missing = [
            (x, y)
            for y in grid_y
            for x in grid_x
            if (x, y) not in values
        ]
        if missing:
            raise ValueError(f"residual_2d grid is incomplete: missing={missing}")
        return grid_x, grid_y, values

    @staticmethod
    def _grid_bounds(values, value):
        value = float(value)
        if value <= values[0]:
            return values[0], values[0], 0.0
        if value >= values[-1]:
            return values[-1], values[-1], 0.0
        right = bisect_right(values, value)
        low = values[right - 1]
        high = values[right]
        ratio = (value - low) / max(high - low, 1e-9)
        return low, high, ratio

    def _interpolate_grid(self, target_px):
        x, y = (float(target_px[0]), float(target_px[1]))
        x0, x1, tx = self._grid_bounds(self.grid_x, x)
        y0, y1, ty = self._grid_bounds(self.grid_y, y)

        top_left = self.grid_values[(x0, y0)]
        top_right = self.grid_values[(x1, y0)]
        bottom_left = self.grid_values[(x0, y1)]
        bottom_right = self.grid_values[(x1, y1)]

        result = []
        for value_index in (0, 1):
            top = top_left[value_index] + (
                top_right[value_index] - top_left[value_index]
            ) * tx
            bottom = bottom_left[value_index] + (
                bottom_right[value_index] - bottom_left[value_index]
            ) * tx
            result.append(top + (bottom - top) * ty)
        return tuple(result)

    def map_pixel(self, target_px):
        if not self.loaded:
            return None
        if self.mode == "residual_2d":
            return self._interpolate_grid(target_px)
        x, y = target_px
        return (
            self._interpolate(self.pan_points, x),
            self._interpolate(self.tilt_points, y),
        )
