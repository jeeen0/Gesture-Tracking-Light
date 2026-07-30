import importlib
import sys
import threading
import time
import types
import unittest

import numpy as np


if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.SimpleNamespace()

pointing = importlib.import_module("src.vision.pointing_target")


class _FakeDepthEstimator:
    @staticmethod
    def midas_to_meters(value):
        return 1.0 / max(float(value), 0.001)


class PointingGeometryTests(unittest.TestCase):
    def make_estimator(self):
        estimator = pointing.PointingTargetEstimator.__new__(
            pointing.PointingTargetEstimator
        )
        estimator.frame_w = 100
        estimator.frame_h = 60
        estimator.cam_fx = 70.0
        estimator.cam_fy = 70.0
        estimator.cx = 50.0
        estimator.cy = 30.0
        estimator.depth_est = _FakeDepthEstimator()
        return estimator

    def test_surface_patch_rejects_single_pixel_outlier(self):
        estimator = self.make_estimator()
        depth_map = np.full((60, 100), 0.5, dtype=np.float32)
        depth_map[30, 50] = 1.0

        value = estimator._surface_midas_at(depth_map, 50, 30)

        self.assertAlmostEqual(value, 0.5, places=5)

    def test_surface_patch_rejects_inconsistent_depth_edge(self):
        estimator = self.make_estimator()
        depth_map = np.full((60, 100), 0.5, dtype=np.float32)
        for y in range(27, 34):
            for x in range(47, 54):
                depth_map[y, x] = 0.1 if (x + y) % 2 else 1.0

        value = estimator._surface_midas_at(depth_map, 50, 30)

        self.assertIsNone(value)

    def test_2d_march_requires_consecutive_surface_hits(self):
        estimator = self.make_estimator()
        depth_map = np.full((60, 100), 0.5, dtype=np.float32)

        target = estimator._march(depth_map, (10, 30), (20, 30))

        self.assertEqual(target, (59, 30))

    def test_tracker_requires_minimum_sample_count(self):
        old_seconds = pointing.STABLE_SECONDS
        old_min_samples = pointing.STABLE_MIN_SAMPLES
        pointing.STABLE_SECONDS = 0.1
        pointing.STABLE_MIN_SAMPLES = 3
        try:
            tracker = pointing.MotorAngleTracker(100.0, 100.0, 50.0, 30.0)
            first = tracker.update((50, 30), True)
            tracker.stable_start = time.time() - 1.0
            second = tracker.update((51, 30), True)
            third = tracker.update((49, 30), True)
        finally:
            pointing.STABLE_SECONDS = old_seconds
            pointing.STABLE_MIN_SAMPLES = old_min_samples

        self.assertIsNone(first["confirmed"])
        self.assertIsNone(second["confirmed"])
        self.assertEqual(third["confirmed"], (50, 30))

    def test_world_3d_ray_hits_background_surface(self):
        estimator = self.make_estimator()
        estimator.frame_w = 200
        estimator.frame_h = 100
        estimator.cam_fx = 100.0
        estimator.cam_fy = 100.0
        estimator.cx = 100.0
        estimator.cy = 50.0

        depth_map = np.full((100, 200), 0.5, dtype=np.float32)
        depth_map[47:54, 97:104] = 1.0
        points = {"index_mcp": (100, 50)}
        world_points = {
            "index_mcp": (0.00, 0.0, 0.00),
            "index_pip": (0.02, 0.0, 0.02),
            "index_dip": (0.04, 0.0, 0.04),
            "index_tip": (0.06, 0.0, 0.06),
        }

        target = estimator._march_world_3d(depth_map, points, world_points)

        self.assertIsNotNone(target)
        self.assertGreater(target[0], 120)
        self.assertEqual(target[1], 50)

    def test_async_result_is_added_to_tracker_only_once(self):
        estimator = self.make_estimator()
        estimator.ray_mode = "world_3d"
        estimator.depth_error = None
        estimator.async_depth = True
        estimator.tracker = pointing.MotorAngleTracker(
            100.0,
            100.0,
            50.0,
            30.0,
        )
        estimator._generation = 0
        estimator._consumed_sequence = 0
        estimator._async_condition = threading.Condition()
        estimator._latest_depth_captured_at = None
        estimator._latest_depth_error = None
        estimator._last_result = estimator._empty_result()
        captured_at = time.monotonic()
        estimator._async_result = {
            "sequence": 1,
            "generation": 0,
            "captured_at": captured_at,
            "completed_at": captured_at + 0.1,
            "depth_map": np.full((60, 100), 0.5, dtype=np.float32),
            "target_payload": {
                "raw_target": (60, 30),
                "ray_start_px": (40, 30),
                "ray_tip_px": (50, 30),
                "calibrated": True,
                "depth_available": True,
                "used_depth_hit": True,
                "hit_method": "depth_march_world_3d",
                "depth_error": None,
                "depth_map": np.full((60, 100), 0.5, dtype=np.float32),
                "ray_mode": "world_3d",
            },
            "depth_error": None,
            "ray_start": (40, 30),
            "ray_tip": (50, 30),
        }

        first = estimator._consume_async_result((40, 30), (50, 30))
        second = estimator._consume_async_result((41, 30), (51, 30))

        self.assertEqual(first["sample_count"], 1)
        self.assertEqual(second["sample_count"], 1)
        self.assertTrue(second["async_pending"])


if __name__ == "__main__":
    unittest.main()
