import math
import os
import threading
import time
from collections import deque

import cv2
import numpy as np


HAND_WRIST_TO_MIDDLE_MCP_M = 0.10
RAY_STEP_PX = 4
RAY_MAX_STEPS = 300
DEPTH_HIT_THRESHOLD = 0.12
DEPTH_UPDATE_INTERVAL = 4
STABLE_SECONDS = 3.0
STABLE_STD_PX = 55.0
JITTER_RESET_PX = 140.0
BUFFER_MAXLEN = 50
EMA_ALPHA = 0.25
DEPTH_ASYNC = os.environ.get("PI_DEPTH_ASYNC", "1") != "0"
DEPTH_ASYNC_FPS = max(0.5, float(os.environ.get("PI_DEPTH_ASYNC_FPS", "4.0")))
DEPTH_RESULT_MAX_AGE_SECONDS = max(
    0.05,
    float(os.environ.get("PI_DEPTH_RESULT_MAX_AGE", "0.75")),
)
TORCH_NUM_THREADS = max(1, int(os.environ.get("PI_TORCH_NUM_THREADS", "2")))


class DepthEstimator:
    def __init__(self):
        import torch

        self.torch = torch
        try:
            torch.set_num_threads(TORCH_NUM_THREADS)
        except RuntimeError as exc:
            print(f"[Pointing] could not set torch threads: {exc}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[Pointing] MiDaS loading... first run can take a while")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.model.to(self.device)
        self.model.eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform
        self.depth_scale = 1.0
        self.is_calibrated = False
        print(f"[Pointing] MiDaS loaded ({self.device})")

    def estimate(self, frame_bgr):
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).to(self.device)

        with self.torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = self.torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(frame_bgr.shape[0], frame_bgr.shape[1]),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 1e-6:
            depth_map = (depth_map - d_min) / (d_max - d_min)
        return depth_map.astype(np.float32)

    def calibrate(self, depth_map, wrist_px, middle_mcp_px, cam_fx):
        wx, wy = wrist_px
        mx, my = middle_mcp_px
        px_dist = math.hypot(mx - wx, my - wy)
        if px_dist < 20:
            return False

        estimated_depth_m = HAND_WRIST_TO_MIDDLE_MCP_M * cam_fx / px_dist
        h, w = depth_map.shape
        wx_c = int(np.clip(wx, 5, w - 5))
        wy_c = int(np.clip(wy, 5, h - 5))
        midas_val = float(np.mean(depth_map[wy_c - 5:wy_c + 5, wx_c - 5:wx_c + 5]))
        if midas_val < 0.01:
            return False

        self.depth_scale = estimated_depth_m * midas_val
        self.is_calibrated = True
        print(
            f"[Pointing] calibrated scale={self.depth_scale:.4f} "
            f"hand_depth={estimated_depth_m:.2f}m midas={midas_val:.3f}"
        )
        return True

    def midas_to_meters(self, midas_val):
        if midas_val < 0.001:
            return 99.0
        return self.depth_scale / midas_val


class MotorAngleTracker: #0520_v2m
    def __init__(self, cam_fx, cam_fy, cx, cy):
        self.fx = cam_fx
        self.fy = cam_fy
        self.cx = cx
        self.cy = cy
        self.buf_x = deque(maxlen=BUFFER_MAXLEN)
        self.buf_y = deque(maxlen=BUFFER_MAXLEN)
        self.stable_start = None
        self.confirmed_target = None
        self.confirmed_angles = None
        self.ema_x = None
        self.ema_y = None
        self.grace_until = 0.0
        self.last_pointing_time = 0.0
        self.pointing_grace_seconds = float(os.environ.get("PI_POINTING_GRACE_SECONDS", "0.35"))
        self.entry_grace_seconds = float(
            os.environ.get("PI_POINT_ENTRY_GRACE_SECONDS", "1.0")
        )

    def update(self, raw, is_pointing):
        result = {
            "display_target": None,
            "confirmed": self.confirmed_target,
            "pan_deg": self.confirmed_angles[0] if self.confirmed_angles else None,
            "tilt_deg": self.confirmed_angles[1] if self.confirmed_angles else None,
            "stable_ratio": 0.0,
            "std_px": 0.0,
            "stable_rejected": False,
        }

        if self.confirmed_target is not None:
            result["display_target"] = self.confirmed_target
            result["stable_ratio"] = 1.0
            return result

        # 포인트모드 진입 직후 유예시간 — 피스→포인트 전환 중 잘못된 데이터 방지
        if time.time() < self.grace_until: #0520_v2m
            return result

        now = time.time()

        if is_pointing and raw is not None:
            self.last_pointing_time = now
        else:
            if now - self.last_pointing_time <= self.pointing_grace_seconds:
                result["display_target"] = (
                    (int(self.ema_x), int(self.ema_y))
                    if self.ema_x is not None and self.ema_y is not None
                    else None
                )
                if self.stable_start is not None:
                    elapsed = now - self.stable_start
                    result["stable_ratio"] = min(elapsed / STABLE_SECONDS, 1.0)
                return result

            self._reset_buffer()
            return result

        rx, ry = raw
        self.buf_x.append(rx)
        self.buf_y.append(ry)

        if self.ema_x is None:
            self.ema_x, self.ema_y = float(rx), float(ry)
        else:
            self.ema_x = EMA_ALPHA * rx + (1 - EMA_ALPHA) * self.ema_x
            self.ema_y = EMA_ALPHA * ry + (1 - EMA_ALPHA) * self.ema_y
        result["display_target"] = (int(self.ema_x), int(self.ema_y))

        if self.stable_start is None:
            self.stable_start = time.time()

        elapsed = time.time() - self.stable_start
        result["stable_ratio"] = min(elapsed / STABLE_SECONDS, 1.0)

        if len(self.buf_x) >= 5:
            result["std_px"] = float(np.hypot(np.std(self.buf_x), np.std(self.buf_y)))
            if result["std_px"] > JITTER_RESET_PX:
                print(f"[Pointing] too much jitter, reset buffer std={result['std_px']:.1f}px")
                self._reset_buffer()
                result["stable_ratio"] = 0.0
                result["stable_rejected"] = True
                return result

        if elapsed >= STABLE_SECONDS:
            if result["std_px"] > STABLE_STD_PX: #0520_v2m
                print(f"[Pointing] not stable enough, retry std={result['std_px']:.1f}px")
                self._reset_buffer()          # ← 버퍼도 비워서 과거 흔들림 데이터 제거
                result["stable_ratio"] = 0.0
                result["stable_rejected"] = True
                return result

            tx = int(np.mean(self.buf_x))
            ty = int(np.mean(self.buf_y))
            pan, tilt = self._to_angles(tx, ty)
            self.confirmed_target = (tx, ty)
            self.confirmed_angles = (pan, tilt)
            result.update({
                "confirmed": self.confirmed_target,
                "pan_deg": pan,
                "tilt_deg": tilt,
                "stable_ratio": 1.0,
            })
            print(
                f"[Pointing] target confirmed pixel=({tx},{ty}) "
                f"pan={pan:+.2f} tilt={tilt:+.2f} std={result['std_px']:.1f}px"
            )
            self._reset_buffer()

        return result

    def _to_angles(self, tx, ty):
        pan_deg = math.degrees(math.atan((tx - self.cx) / self.fx))
        tilt_deg = math.degrees(math.atan((ty - self.cy) / self.fy))
        return pan_deg, tilt_deg

    def _reset_buffer(self):
        self.buf_x.clear()
        self.buf_y.clear()
        self.stable_start = None
        self.ema_x = None
        self.ema_y = None

    def reset(self, clear_confirmed=False): #0520_v2m
        self._reset_buffer()
        if clear_confirmed:
            self.confirmed_target = None
            self.confirmed_angles = None
            self.grace_until = time.time() + self.entry_grace_seconds


class PointingTargetEstimator:
    WRIST = 0
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9

    def __init__(
        self,
        frame_w,
        frame_h,
        cam_fx=None,
        cam_fy=None,
        cx=None,
        cy=None,
        ray_mode=None,
    ):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.cam_fx = float(cam_fx) if cam_fx is not None else frame_w * 0.7
        self.cam_fy = float(cam_fy) if cam_fy is not None else self.cam_fx
        self.cx = float(cx) if cx is not None else frame_w / 2.0
        self.cy = float(cy) if cy is not None else frame_h / 2.0
        self.ray_mode = (
            ray_mode or os.environ.get("PI_POINT_RAY_MODE", "mcp_tip")
        ).strip().lower()
        if self.ray_mode not in {"mcp_tip", "finger_axis"}:
            print(f"[Pointing] unknown ray mode {self.ray_mode!r}; using mcp_tip")
            self.ray_mode = "mcp_tip"

        self.depth_est = None
        self.depth_error = None
        try:
            self.depth_est = DepthEstimator()
        except ModuleNotFoundError as e:
            self.depth_error = str(e)
            print(f"[Pointing] MiDaS disabled, using 2D ray fallback: {e}")
        except Exception as e:
            self.depth_error = str(e)
            print(f"[Pointing] MiDaS disabled, using 2D ray fallback: {e}")

        self.tracker = MotorAngleTracker(self.cam_fx, self.cam_fy, self.cx, self.cy)
        self.depth_map = None
        self.frame_count = 0
        self.async_depth = bool(self.depth_est is not None and DEPTH_ASYNC)
        self._last_result = self._empty_result()
        self._generation = 0
        self._job_sequence = 0
        self._consumed_sequence = 0
        self._pending_job = None
        self._async_result = None
        self._latest_depth_captured_at = None
        self._latest_depth_error = None
        self._async_stop = False
        self._async_condition = threading.Condition()
        self._async_thread = None
        if self.async_depth:
            self._async_thread = threading.Thread(
                target=self._depth_worker,
                daemon=True,
            )
            self._async_thread.start()
            print(
                f"[Pointing] async MiDaS enabled "
                f"fps={DEPTH_ASYNC_FPS:g} max_age={DEPTH_RESULT_MAX_AGE_SECONDS:g}s"
            )

    def _empty_result(self):
        return {
            "display_target": None,
            "confirmed": None,
            "pan_deg": None,
            "tilt_deg": None,
            "stable_ratio": 0.0,
            "std_px": 0.0,
            "stable_rejected": False,
            "raw_target": None,
            "ray_start_px": None,
            "ray_tip_px": None,
            "calibrated": False,
            "depth_available": self.depth_est is not None,
            "used_depth_hit": False,
            "hit_method": "waiting_for_depth" if self.depth_est is not None else "2d_fallback_no_depth",
            "depth_error": self.depth_error,
            "depth_map": None,
            "async_pending": self.async_depth,
            "ray_mode": self.ray_mode,
        }

    def update(self, frame_bgr, hand_landmarks):
        self.frame_count += 1
        points = self._key_points(hand_landmarks.landmark)
        ray_start, ray_tip = self._ray_segment(points)

        if self.depth_est is None:
            raw_target = self._project_to_screen_edge(ray_start, ray_tip)
            result = self.tracker.update(raw_target, True)
            result.update(
                {
                    "raw_target": raw_target,
                    "ray_start_px": ray_start,
                    "ray_tip_px": ray_tip,
                    "calibrated": False,
                    "depth_available": False,
                    "used_depth_hit": False,
                    "hit_method": "2d_fallback_no_depth",
                    "depth_error": self.depth_error,
                    "depth_map": None,
                    "async_pending": False,
                    "ray_mode": self.ray_mode,
                }
            )
            self._last_result = result
            return result

        if self.async_depth:
            self._submit_depth_job(frame_bgr)
            return self._consume_async_result(points, ray_start, ray_tip)

        if self.depth_map is None or self.frame_count % DEPTH_UPDATE_INTERVAL == 0:
            self.depth_map = self.depth_est.estimate(frame_bgr)
        payload = self._target_from_depth(
            self.depth_map,
            points,
            ray_start,
            ray_tip,
        )
        result = self.tracker.update(payload["raw_target"], True)
        result.update(payload)
        result["async_pending"] = False
        self._last_result = result
        return result

    def _submit_depth_job(self, frame_bgr):
        with self._async_condition:
            self._job_sequence += 1
            self._pending_job = {
                "sequence": self._job_sequence,
                "generation": self._generation,
                "captured_at": time.monotonic(),
                "frame": frame_bgr.copy(),
            }
            self._async_condition.notify()

    def _depth_worker(self):
        last_start = 0.0
        interval = 1.0 / DEPTH_ASYNC_FPS
        while True:
            with self._async_condition:
                while self._pending_job is None and not self._async_stop:
                    self._async_condition.wait()
                if self._async_stop:
                    return
                job = self._pending_job
                self._pending_job = None

                while True:
                    remaining = interval - (time.monotonic() - last_start)
                    if remaining <= 0 or self._async_stop:
                        break
                    self._async_condition.wait(timeout=remaining)
                    if self._pending_job is not None:
                        job = self._pending_job
                        self._pending_job = None
                if self._async_stop:
                    return

            last_start = time.monotonic()
            try:
                depth_map = self.depth_est.estimate(job["frame"])
                payload = {
                    "depth_map": depth_map,
                    "depth_error": None,
                    "completed_at": time.monotonic(),
                }
            except Exception as exc:
                payload = {
                    "depth_map": None,
                    "depth_error": str(exc),
                    "completed_at": time.monotonic(),
                }

            payload.update(
                {
                    "sequence": job["sequence"],
                    "generation": job["generation"],
                    "captured_at": job["captured_at"],
                }
            )
            with self._async_condition:
                self._async_result = payload

    def _consume_async_result(self, points, ray_start, ray_tip):
        with self._async_condition:
            payload = dict(self._async_result) if self._async_result else None

        if (
            payload is not None
            and payload["generation"] == self._generation
            and payload["sequence"] > self._consumed_sequence
        ):
            self._consumed_sequence = payload["sequence"]
            if payload.get("depth_map") is not None:
                self.depth_map = payload["depth_map"]
                self._latest_depth_captured_at = payload["captured_at"]
                self._latest_depth_error = None
            else:
                self._latest_depth_error = payload.get("depth_error")

        now = time.monotonic()
        depth_age = (
            max(0.0, now - self._latest_depth_captured_at)
            if self._latest_depth_captured_at is not None
            else None
        )
        depth_is_fresh = (
            self.depth_map is not None
            and depth_age is not None
            and depth_age <= DEPTH_RESULT_MAX_AGE_SECONDS
        )

        # MiDaS remains rate-limited in the worker, but the lightweight ray
        # intersection and EMA use the current hand landmarks every frame.
        if depth_is_fresh:
            target_payload = self._target_from_depth(
                self.depth_map,
                points,
                ray_start,
                ray_tip,
            )
            result = self.tracker.update(target_payload["raw_target"], True)
            result.update(target_payload)
            result["async_pending"] = False
            result["depth_age_seconds"] = depth_age
            self._last_result = result
            return result

        # If depth inference failed before producing any usable map, retain the
        # existing 2D fallback while continuing to request fresh depth frames.
        if self._latest_depth_error:
            raw_target = self._project_to_screen_edge(ray_start, ray_tip)
            result = self.tracker.update(raw_target, True)
            result.update(
                {
                    "raw_target": raw_target,
                    "ray_start_px": ray_start,
                    "ray_tip_px": ray_tip,
                    "calibrated": False,
                    "depth_available": False,
                    "used_depth_hit": False,
                    "hit_method": "2d_fallback_depth_error",
                    "depth_error": self._latest_depth_error,
                    "depth_map": None,
                    "async_pending": False,
                    "ray_mode": self.ray_mode,
                }
            )
            self._last_result = result
            return result

        # Before the first fresh depth map (or while a previous map is stale),
        # keep the last target but draw the ray from the current hand pose.
        result = dict(self._last_result)
        result["ray_start_px"] = ray_start
        result["ray_tip_px"] = ray_tip
        result["async_pending"] = True
        result["ray_mode"] = self.ray_mode
        if depth_age is not None:
            result["depth_age_seconds"] = depth_age
        return result

    def _target_from_depth(self, depth_map, points, ray_start, ray_tip):
        if not self.depth_est.is_calibrated:
            self.depth_est.calibrate(
                depth_map,
                points["wrist"],
                points["middle_mcp"],
                self.cam_fx,
            )

        raw_target = self._march(depth_map, ray_start, ray_tip)
        used_depth_hit = raw_target is not None
        if raw_target is None:
            raw_target = self._project_to_screen_edge(ray_start, ray_tip)

        return {
            "raw_target": raw_target,
            "ray_start_px": ray_start,
            "ray_tip_px": ray_tip,
            "calibrated": self.depth_est.is_calibrated,
            "depth_available": True,
            "used_depth_hit": used_depth_hit,
            "hit_method": "depth_march" if used_depth_hit else "2d_fallback_after_depth",
            "depth_error": None,
            "depth_map": depth_map,
            "ray_mode": self.ray_mode,
        }

    def reset(self, clear_confirmed=False):
        self.tracker.reset(clear_confirmed=clear_confirmed)
        self._last_result = self._empty_result()
        self.depth_map = None
        self._latest_depth_captured_at = None
        self._latest_depth_error = None
        with self._async_condition:
            self._generation += 1
            self._pending_job = None
            self._async_result = None
            self._consumed_sequence = self._job_sequence

    def close(self):
        if not self.async_depth:
            return
        with self._async_condition:
            self._async_stop = True
            self._async_condition.notify_all()
        if self._async_thread is not None:
            self._async_thread.join(timeout=2.0)

    def _key_points(self, landmarks):
        def to_px(lm):
            return (
                int(np.clip(lm.x, 0.0, 1.0) * self.frame_w),
                int(np.clip(lm.y, 0.0, 1.0) * self.frame_h),
            )

        return {
            "wrist": to_px(landmarks[self.WRIST]),
            "index_mcp": to_px(landmarks[self.INDEX_MCP]),
            "index_pip": to_px(landmarks[self.INDEX_PIP]),
            "index_dip": to_px(landmarks[self.INDEX_DIP]),
            "index_tip": to_px(landmarks[self.INDEX_TIP]),
            "middle_mcp": to_px(landmarks[self.MIDDLE_MCP]),
        }

    def _ray_segment(self, points):
        if self.ray_mode == "mcp_tip":
            return points["index_mcp"], points["index_tip"]

        joints = np.asarray(
            [
                points["index_pip"],
                points["index_dip"],
                points["index_tip"],
            ],
            dtype=np.float32,
        )
        centered = joints - joints.mean(axis=0)
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        direction = axes[0]
        forward = joints[-1] - joints[0]
        if float(np.dot(direction, forward)) < 0:
            direction = -direction
        segment_length = max(float(np.linalg.norm(forward)), 1.0)
        start = joints[0]
        tip = start + direction * segment_length
        return tuple(start.tolist()), tuple(tip.tolist())

    def _project_to_screen_edge(self, start_px, tip_px):
        mx, my = start_px
        tx, ty = tip_px
        dx = tx - mx
        dy = ty - my
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return None

        ux = dx / dist
        uy = dy / dist
        start = dist + 10
        last = None
        max_steps = int(max(self.frame_w, self.frame_h) / RAY_STEP_PX) + 50
        for step in range(max_steps):
            t = start + step * RAY_STEP_PX
            cx = int(mx + ux * t)
            cy = int(my + uy * t)
            if cx < 0 or cx >= self.frame_w or cy < 0 or cy >= self.frame_h:
                return last
            last = (cx, cy)
        return last

    def _march(self, depth_map, start_px, tip_px):
        mx, my = start_px
        tx, ty = tip_px
        dx = tx - mx
        dy = ty - my
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return None

        ux = dx / dist
        uy = dy / dist
        mx_c = int(np.clip(mx, 0, self.frame_w - 1))
        my_c = int(np.clip(my, 0, self.frame_h - 1))
        start_midas = float(depth_map[my_c, mx_c])
        start_depth_m = self.depth_est.midas_to_meters(start_midas)
        start_offset = dist + 35

        for step in range(RAY_MAX_STEPS):
            t = start_offset + step * RAY_STEP_PX
            cx = int(mx + ux * t)
            cy = int(my + uy * t)
            if cx < 0 or cx >= self.frame_w or cy < 0 or cy >= self.frame_h:
                return None

            surface_midas = float(depth_map[cy, cx])
            surface_depth_m = self.depth_est.midas_to_meters(surface_midas)
            ray_depth_m = start_depth_m * (t / dist)
            if ray_depth_m >= surface_depth_m * (1.0 - DEPTH_HIT_THRESHOLD):
                return (cx, cy)
        return None
