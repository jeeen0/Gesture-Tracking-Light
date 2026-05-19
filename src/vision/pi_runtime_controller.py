import json
import os
import threading
import time

from gestures import GestureRecognizer
from pi_runtime_config import (
    ACTIVE_TIMEOUT_SECONDS,
    BRIGHTNESS_UPDATE_INTERVAL,
    BRIGHTNESS_STEP,
    COMMAND_HOLD_SECONDS,
    ENABLE_POINTING,
    ENABLE_YOLO,
    FRAME_H,
    FRAME_W,
    KEEP_AWAKE,
    MAX_BRIGHTNESS,
    MIN_BRIGHTNESS,
    PRELOAD_POINTING,
    PRELOAD_YOLO,
    STATE_FILE,
    YOLO_MODEL_PATH,
)
from pi_runtime_events import emit

try:
    from pointing_target import PointingTargetEstimator
except Exception as e:
    PointingTargetEstimator = None
    print(f"[PI] PointingTargetEstimator unavailable: {e}")


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


class PiSmartLightController:
    def __init__(self):
        self.recognizer = GestureRecognizer()
        self.standby = True
        self.active_until = 0.0
        self.point_mode = False

        self.power = False
        self.brightness = 0
        self.saved_brightness = 50
        self.mode = "Spot"
        self.pan_deg = 0.0
        self.tilt_deg = 0.0
        self.preview_pan_deg = 0.0
        self.preview_tilt_deg = 0.0
        self.last_delta_pan_deg = 0.0
        self.last_delta_tilt_deg = 0.0
        self.point_target = None
        self.point_display_target = None
        self.point_stable_ratio = 0.0
        self.point_std_px = 0.0

        self.point_estimator = None
        self.point_status = "idle"
        self.point_error = None
        self.point_preload_started = False
        self.yolo_model = None
        self.yolo_status = "idle"
        self.yolo_error = None
        self.yolo_preload_started = False
        self.last_brightness_gesture = None
        self.last_brightness_time = 0.0
        self.latched_gesture = None
        self.hold_gesture = None
        self.hold_start_time = 0.0
        self.load_state()

    def start_preloads(self):
        self.start_point_preload()
        self.start_yolo_preload()

    def load_state(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.saved_brightness = clamp(int(data.get("brightness", self.saved_brightness)), 0, MAX_BRIGHTNESS)
            self.power = False
            self.brightness = 0
            self.mode = data.get("mode", self.mode)
            self.pan_deg = float(data.get("pan_deg", self.pan_deg))
            self.tilt_deg = float(data.get("tilt_deg", self.tilt_deg))
            self.preview_pan_deg = self.pan_deg
            self.preview_tilt_deg = self.tilt_deg
            self.point_target = data.get("point_target", self.point_target)
            emit("state_loaded", file=STATE_FILE, saved_brightness=self.saved_brightness, mode=self.mode)
        except Exception as e:
            emit("state_load_error", file=STATE_FILE, error=str(e))

    def save_state(self):
        data = {
            "power": self.power,
            "brightness": self.saved_brightness,
            "mode": self.mode,
            "pan_deg": self.pan_deg,
            "tilt_deg": self.tilt_deg,
            "point_target": self.point_target,
        }
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            emit("state_save_error", file=STATE_FILE, error=str(e))

    def start_point_preload(self):
        if not ENABLE_POINTING or not PRELOAD_POINTING or self.point_preload_started:
            return
        self.point_preload_started = True
        threading.Thread(target=self._preload_pointing, daemon=True).start()

    def _preload_pointing(self):
        if PointingTargetEstimator is None:
            self.point_status = "module_unavailable"
            return
        try:
            self.point_status = "preloading"
            self.point_estimator = PointingTargetEstimator(FRAME_W, FRAME_H)
            self.point_status = "ready"
            emit("pointing_ready")
        except Exception as e:
            self.point_estimator = None
            self.point_error = str(e)
            self.point_status = f"preload_error:{e}"
            emit("pointing_error", error=str(e))

    def start_yolo_preload(self):
        if not ENABLE_YOLO or not PRELOAD_YOLO or self.yolo_preload_started:
            return
        self.yolo_preload_started = True
        threading.Thread(target=self._preload_yolo, daemon=True).start()

    def _preload_yolo(self):
        try:
            self.yolo_status = "preloading"
            from ultralytics import YOLO

            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            self.yolo_status = "ready"
            emit("yolo_ready", model=YOLO_MODEL_PATH)
        except Exception as e:
            self.yolo_model = None
            self.yolo_error = str(e)
            self.yolo_status = f"preload_error:{e}"
            emit("yolo_error", model=YOLO_MODEL_PATH, error=str(e))

    def wake(self):
        self.standby = False
        self.active_until = time.time() + ACTIVE_TIMEOUT_SECONDS
        self.point_mode = False
        self.point_display_target = None
        self.point_stable_ratio = 0.0
        self.point_std_px = 0.0
        if self.point_status not in ("preloading", "ready"):
            self.point_status = "idle"
        self.power = True
        self.brightness = self.saved_brightness
        self.save_state()
        emit("wake", brightness=self.brightness)

    def sleep(self):
        self.standby = True
        self.active_until = 0.0
        self.point_mode = False
        self.power = False
        self.brightness = 0
        self.save_state()
        emit("sleep")

    def update_activity_timeout(self, hand_detected):
        if self.standby:
            return
        if hand_detected or KEEP_AWAKE:
            self.active_until = time.time() + ACTIVE_TIMEOUT_SECONDS
            return
        if time.time() >= self.active_until:
            self.sleep()

    def enter_point_mode(self):
        self.point_mode = True
        self.mode = "Point"
        self.point_target = None
        self.point_display_target = None
        self.point_stable_ratio = 0.0
        self.point_std_px = 0.0
        if self.point_estimator:
            self.point_estimator.reset(clear_confirmed=True)
        self.save_state()
        emit("point_mode", enabled=True)

    def _reset_hold(self):
        self.hold_gesture = None
        self.hold_start_time = 0.0

    def _hold_ready(self, gesture):
        now = time.time()
        if self.hold_gesture != gesture:
            self.hold_gesture = gesture
            self.hold_start_time = now
            return False
        return now - self.hold_start_time >= COMMAND_HOLD_SECONDS

    def apply_brightness_gesture(self, gesture):
        now = time.time()
        if now - self.last_brightness_time < BRIGHTNESS_UPDATE_INTERVAL:
            return
        old = self.brightness
        if gesture == "THUMBS_UP":
            self.brightness = clamp(old + BRIGHTNESS_STEP, MIN_BRIGHTNESS, MAX_BRIGHTNESS)
        elif gesture == "THUMBS_DOWN":
            self.brightness = clamp(old - BRIGHTNESS_STEP, MIN_BRIGHTNESS, MAX_BRIGHTNESS)
        else:
            return
        self.power = self.brightness > 0
        self.last_brightness_time = now
        if self.brightness != old:
            self.saved_brightness = self.brightness
            self.last_brightness_gesture = gesture
            self.save_state()
            emit("brightness", gesture=gesture, brightness=self.brightness, delta=self.brightness - old)

    def update_point_target(self, frame, hand_landmarks):
        if not ENABLE_POINTING:
            self.point_status = "disabled"
            return
        if self.point_estimator is None:
            if self.point_preload_started:
                self.point_status = self.point_status or "preloading"
                return
            self.point_status = "loading"
            self.point_estimator = PointingTargetEstimator(FRAME_W, FRAME_H)

        target = self.point_estimator.update(frame, hand_landmarks)
        self.point_status = "tracking_2d_fallback" if target.get("depth_available") is False else "tracking"
        self.point_display_target = target.get("display_target")
        self.point_stable_ratio = float(target.get("stable_ratio", 0.0))
        self.point_std_px = float(target.get("std_px", 0.0))
        if target.get("pan_deg") is not None:
            self.preview_pan_deg = float(target["pan_deg"])
            self.preview_tilt_deg = float(target["tilt_deg"])
        if target.get("confirmed") and target["confirmed"] != self.point_target:
            previous_pan = self.pan_deg
            previous_tilt = self.tilt_deg
            if target.get("pan_deg") is not None:
                self.pan_deg = float(target["pan_deg"])
                self.tilt_deg = float(target["tilt_deg"])
            self.last_delta_pan_deg = self.pan_deg - previous_pan
            self.last_delta_tilt_deg = self.tilt_deg - previous_tilt
            self.point_target = target["confirmed"]
            self.point_mode = False
            self.point_status = "locked"
            self.save_state()
            emit(
                "point_target_locked",
                target_px=self.point_target,
                pan_deg=round(self.pan_deg, 2),
                tilt_deg=round(self.tilt_deg, 2),
                previous_pan_deg=round(previous_pan, 2),
                previous_tilt_deg=round(previous_tilt, 2),
                delta_pan_deg=round(self.last_delta_pan_deg, 2),
                delta_tilt_deg=round(self.last_delta_tilt_deg, 2),
                std_px=round(target.get("std_px", 0.0), 1),
            )
            emit("point_mode", enabled=False, reason="target_locked")

    def apply_gesture(self, gesture, results, frame, wave_active):
        if gesture is None:
            self.latched_gesture = None
            self._reset_hold()
            return None

        if self.standby:
            if gesture == "WAVE" and wave_active:
                self.wake()
                self.latched_gesture = "WAVE"
                self._reset_hold()
                return "WAVE"
            return None

        if gesture == "FIST":
            if not self._hold_ready("FIST"):
                return None
            self.sleep()
            self._reset_hold()
        elif gesture == "POINT_MODE":
            if self.latched_gesture == "POINT_MODE":
                return gesture
            if not self._hold_ready("POINT_MODE"):
                return None
            self.enter_point_mode()
            self._reset_hold()
        elif gesture == "POINT":
            self._reset_hold()
            if not self.point_mode:
                emit("point_blocked", reason="point_mode_required")
                return None
            self.update_point_target(frame, results.multi_hand_landmarks[0])
        elif gesture in ("THUMBS_UP", "THUMBS_DOWN"):
            self._reset_hold()
            self.apply_brightness_gesture(gesture)
        elif gesture == "MODE_SWITCH":
            if self.latched_gesture == "MODE_SWITCH":
                return gesture
            if not self._hold_ready("MODE_SWITCH"):
                return None
            self.mode = "Mood" if self.mode == "Spot" else "Spot"
            self.save_state()
            emit("mode", mode=self.mode)
            self._reset_hold()
        else:
            self._reset_hold()
        self.latched_gesture = gesture
        return gesture

    def state_payload(self, fps, gesture, hand_detected, bbox_px, wave_active):
        active_remaining = max(0.0, self.active_until - time.time()) if not self.standby else 0.0
        return {
            "standby": self.standby,
            "power": self.power,
            "brightness": self.brightness,
            "mode": self.mode,
            "point_mode": self.point_mode,
            "point_status": self.point_status,
            "yolo_status": self.yolo_status,
            "keep_awake": KEEP_AWAKE,
            "active_timeout_seconds": ACTIVE_TIMEOUT_SECONDS,
            "active_remaining_seconds": round(active_remaining, 1),
            "point_target": self.point_target,
            "point_display_target": self.point_display_target,
            "point_stable_ratio": round(self.point_stable_ratio, 2),
            "point_std_px": round(self.point_std_px, 1),
            "pan_deg": round(self.pan_deg, 2),
            "tilt_deg": round(self.tilt_deg, 2),
            "preview_pan_deg": round(self.preview_pan_deg, 2),
            "preview_tilt_deg": round(self.preview_tilt_deg, 2),
            "last_delta_pan_deg": round(self.last_delta_pan_deg, 2),
            "last_delta_tilt_deg": round(self.last_delta_tilt_deg, 2),
            "last_brightness_gesture": self.last_brightness_gesture,
            "gesture": gesture,
            "hand_detected": hand_detected,
            "hand_bbox_width_px": bbox_px,
            "wave_active": wave_active,
            "fps": round(fps, 1),
        }

