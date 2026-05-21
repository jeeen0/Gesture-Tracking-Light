import math
import os
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


class DepthEstimator:
    def __init__(self):
        import torch

        self.torch = torch
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
            self.grace_until = time.time() + 1.0   # ← 추가: 1초간 데이터 무시


class PointingTargetEstimator:
    WRIST = 0
    INDEX_MCP = 5
    INDEX_TIP = 8
    MIDDLE_MCP = 9

    def __init__(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.cam_fx = frame_w * 0.7
        self.cam_fy = self.cam_fx
        self.cx = frame_w / 2.0
        self.cy = frame_h / 2.0
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

    def update(self, frame_bgr, hand_landmarks):
        self.frame_count += 1
        landmarks = hand_landmarks.landmark
        points = self._key_points(landmarks)

        if self.depth_est is None:
            raw_target = self._project_to_screen_edge(points["index_mcp"], points["index_tip"])
            result = self.tracker.update(raw_target, True)
            result["raw_target"] = raw_target
            result["calibrated"] = False
            result["depth_available"] = False
            result["used_depth_hit"] = False
            result["hit_method"] = "2d_fallback_no_depth"
            result["depth_error"] = self.depth_error
            return result

        if self.depth_map is None or self.frame_count % DEPTH_UPDATE_INTERVAL == 0:
            self.depth_map = self.depth_est.estimate(frame_bgr)

        if not self.depth_est.is_calibrated:
            self.depth_est.calibrate(
                self.depth_map,
                points["wrist"],
                points["middle_mcp"],
                self.cam_fx,
            )

        raw_target = self._march(points["index_mcp"], points["index_tip"])
        used_depth_hit = raw_target is not None

        # ray가 표면을 못 찾았을 때 2D fallback으로 임시 타겟 제공
        if raw_target is None:
            raw_target = self._project_to_screen_edge(points["index_mcp"], points["index_tip"])

        result = self.tracker.update(raw_target, True)
        result["raw_target"] = raw_target
        result["calibrated"] = self.depth_est.is_calibrated
        result["depth_available"] = True
        result["used_depth_hit"] = used_depth_hit
        result["hit_method"] = "depth_march" if used_depth_hit else "2d_fallback_after_depth"
        result["depth_error"] = None
        return result

    def reset(self, clear_confirmed=False):
        self.tracker.reset(clear_confirmed=clear_confirmed)

    def _key_points(self, landmarks):
        def to_px(lm):
            return (
                int(np.clip(lm.x, 0.0, 1.0) * self.frame_w),
                int(np.clip(lm.y, 0.0, 1.0) * self.frame_h),
            )

        return {
            "wrist": to_px(landmarks[self.WRIST]),
            "index_mcp": to_px(landmarks[self.INDEX_MCP]),
            "index_tip": to_px(landmarks[self.INDEX_TIP]),
            "middle_mcp": to_px(landmarks[self.MIDDLE_MCP]),
        }

    def _project_to_screen_edge(self, mcp_px, tip_px):
        mx, my = mcp_px
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

    def _march(self, mcp_px, tip_px):
        if self.depth_map is None:
            return None

        mx, my = mcp_px
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
        start_midas = float(self.depth_map[my_c, mx_c])
        start_depth_m = self.depth_est.midas_to_meters(start_midas)
        start_offset = dist + 10

        for step in range(RAY_MAX_STEPS):
            t = start_offset + step * RAY_STEP_PX
            cx = int(mx + ux * t)
            cy = int(my + uy * t)
            if cx < 0 or cx >= self.frame_w or cy < 0 or cy >= self.frame_h:
                return None

            surface_midas = float(self.depth_map[cy, cx])
            surface_depth_m = self.depth_est.midas_to_meters(surface_midas)
            ray_depth_m = start_depth_m * (t / dist)
            if ray_depth_m >= surface_depth_m * (1.0 - DEPTH_HIT_THRESHOLD):
                return (cx, cy)

        return None
