import json
import os
import tempfile
import threading
import time
from collections import deque

from gestures import GestureRecognizer
from src.config import PAN_MAX_DEG, PAN_MIN_DEG, TILT_MAX_DEG, TILT_MIN_DEG
from src.core.servo_controller import ServoController  # 0521_v2m
from pi_runtime_config import (
    ACTIVE_TIMEOUT_SECONDS,
    BRIGHTNESS_UPDATE_INTERVAL,
    BRIGHTNESS_STEP,
    COMMAND_HOLD_SECONDS,
    ENABLE_FISHEYE_UNDISTORT,
    ENABLE_POINTING,
    ENABLE_YOLO,
    FISHEYE_CALIB_PATH,
    POINT_PAN_GAIN,
    POINT_PAN_OFFSET_DEG,
    POINT_ARM_MIN_HITS,
    POINT_ARM_WINDOW,
    POINT_ENTRY_GRACE_SECONDS,
    POINT_RAY_MODE,
    POINT_TILT_GAIN,
    POINT_TILT_OFFSET_DEG,
    SERVO_PAN_CENTER,
    SERVO_PAN_SIGN,
    SERVO_TILT_CENTER,
    SERVO_TILT_SIGN,
    SERVO_POINTING_CALIB_PATH,
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
from servo_pointing_calibration import ServoPointingCalibration

try:
    from fisheye_undistort import FisheyeUndistorter
except Exception as e:
    FisheyeUndistorter = None
    print(f"[PI] FisheyeUndistorter unavailable: {e}")

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

        # ── 서보 (모터) 제어 ── 0521_v2m
        self.servo = None
        try:
            self.servo = ServoController(
                initial_pan=SERVO_PAN_CENTER,
                initial_tilt=SERVO_TILT_CENTER,
            )
            emit("servo_ready")
        except Exception as e:
            emit("servo_error", error=str(e))
            print(f"[PI] ServoController init failed (DRY mode): {e}", flush=True)

        # 서보 절대 PWM 각도 (0~180°). 시작 시 CENTER 위치 = 정면.
        self.servo_pan_deg = SERVO_PAN_CENTER
        self.servo_tilt_deg = SERVO_TILT_CENTER

        # ── Fisheye 카메라 보정 (있으면 frame 보정 + 실제 intrinsics 사용) ──
        self.undistorter = None
        if ENABLE_FISHEYE_UNDISTORT and FisheyeUndistorter is not None:
            try:
                self.undistorter = FisheyeUndistorter(
                    FISHEYE_CALIB_PATH, (FRAME_W, FRAME_H)
                )
                emit(
                    "undistort_ready",
                    file=FISHEYE_CALIB_PATH,
                    fx=round(self.undistorter.fx, 2),
                    fy=round(self.undistorter.fy, 2),
                    cx=round(self.undistorter.cx, 2),
                    cy=round(self.undistorter.cy, 2),
                    rms=round(self.undistorter.rms, 3),
                )
            except Exception as e:
                emit("undistort_unavailable", file=FISHEYE_CALIB_PATH, error=str(e))
                print(f"[PI] Fisheye undistort 비활성: {e}", flush=True)

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
        self.mode_switch_armed = True
        self.mode_switch_release_start = None
        self._last_save_time = 0.0
        self._pending_save = False
        self.point_ray_start_px = None
        self.point_ray_tip_px = None
        self.point_ray_hit_px = None
        self.point_depth_map = None
        self.point_mode_entered_at = 0.0
        self.point_tracking_armed = False
        self.point_arm_history = deque(maxlen=POINT_ARM_WINDOW)
        self.servo_pointing_calibration = ServoPointingCalibration(
            SERVO_POINTING_CALIB_PATH,
            (FRAME_W, FRAME_H),
        )
        if self.servo_pointing_calibration.loaded:
            emit(
                "servo_pointing_calibration_ready",
                file=SERVO_POINTING_CALIB_PATH,
                pan_points=len(self.servo_pointing_calibration.pan_points),
                tilt_points=len(self.servo_pointing_calibration.tilt_points),
            )
        elif self.servo_pointing_calibration.error:
            emit(
                "servo_pointing_calibration_error",
                file=SERVO_POINTING_CALIB_PATH,
                error=self.servo_pointing_calibration.error,
            )
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
            # 서보 절대 각도 복원 (저장된 값이 없으면 90° 중앙 유지) 0521_v2m
            self.servo_pan_deg = float(data.get("servo_pan_deg", self.servo_pan_deg))
            self.servo_tilt_deg = float(data.get("servo_tilt_deg", self.servo_tilt_deg))
            if self.servo is not None:
                self.servo.move_to(self.servo_pan_deg, self.servo_tilt_deg)
            emit("state_loaded", file=STATE_FILE, saved_brightness=self.saved_brightness, mode=self.mode)
        except Exception as e:
            emit("state_load_error", file=STATE_FILE, error=str(e))

    _SAVE_DEBOUNCE_S = 0.4

    def save_state(self, debounce: bool = False):
        now = time.time()
        if debounce:
            self._pending_save = True
            if now - self._last_save_time < self._SAVE_DEBOUNCE_S:
                return
        self._pending_save = False
        self._last_save_time = now
        data = {
            "power": self.power,
            "brightness": self.saved_brightness,
            "mode": self.mode,
            "pan_deg": self.pan_deg,
            "tilt_deg": self.tilt_deg,
            "point_target": self.point_target,
            "servo_pan_deg": self.servo_pan_deg,  # 0521_v2m
            "servo_tilt_deg": self.servo_tilt_deg,
        }
        state_dir = os.path.dirname(STATE_FILE) or "."
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=state_dir,
                prefix=".pi_state.",
                suffix=".tmp",
                delete=False,
                encoding="utf-8",
            ) as f:
                tmp_path = f.name
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, STATE_FILE)
        except Exception as e:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            emit("state_save_error", file=STATE_FILE, error=str(e))

    def flush_pending_save(self):
        """debounce로 미뤄진 save_state를 강제 실행. 종료 직전에 호출."""
        if self._pending_save:
            self.save_state()

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
            self.point_estimator = PointingTargetEstimator(
                FRAME_W,
                FRAME_H,
                ray_mode=POINT_RAY_MODE,
                **self._pointing_intrinsics(),
            )
            self.point_status = "ready"
            emit("pointing_ready")
        except Exception as e:
            self.point_estimator = None
            self.point_error = str(e)
            self.point_status = f"preload_error:{e}"
            emit("pointing_error", error=str(e))

    def _pointing_intrinsics(self):
        """undistorter가 있으면 실제 intrinsics, 없으면 빈 dict (PointingTargetEstimator가 기본 추정값 사용)."""
        if self.undistorter is None:
            return {}
        return {
            "cam_fx": self.undistorter.fx,
            "cam_fy": self.undistorter.fy,
            "cx": self.undistorter.cx,
            "cy": self.undistorter.cy,
        }

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
        self.point_tracking_armed = False
        self.point_arm_history.clear()
        self.point_display_target = None
        self.point_stable_ratio = 0.0
        self.point_std_px = 0.0
        if self.point_status not in ("preloading", "ready"):
            self.point_status = "idle"
        self.power = True
        self.brightness = self.saved_brightness
        # sleep() 시 PWM이 꺼져 있을 수 있으니 마지막 위치로 재기동
        if self.servo is not None:
            self.servo.resume()
            self.servo.move_to(self.servo_pan_deg, self.servo_tilt_deg)
        self.save_state()
        emit("wake", brightness=self.brightness)

    def sleep(self):
        self.standby = True
        self.active_until = 0.0
        self.point_mode = False
        self.point_tracking_armed = False
        self.point_arm_history.clear()
        self.power = False
        self.brightness = 0
        self.save_state()
        # 서보 PWM 해제 (전류 절약, 서보 위치는 state에 저장됨) 0521_v2m
        if self.servo is not None:
            self.servo.pause()
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
        # mode(조명 모드: Spot/Mood)는 건드리지 않음 — point_mode는 별도 플래그
        self.point_target = None
        self.point_display_target = None
        self.point_stable_ratio = 0.0
        self.point_std_px = 0.0
        self.point_mode_entered_at = time.time()
        self.point_tracking_armed = False
        self.point_arm_history.clear()
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
            self.save_state(debounce=True)
            emit("brightness", gesture=gesture, brightness=self.brightness, delta=self.brightness - old)

    def update_point_target(self, frame, hand_landmarks):
        if not ENABLE_POINTING:
            self.point_status = "disabled"
            return
        if self.point_estimator is None:
            if self.point_preload_started and self.point_status == "preloading":
                # 프리로딩이 아직 진행 중이면 잠시 대기하되,
                # 10초 이상 걸리면 동기 생성으로 전환 (2D fallback이라도 동작)
                if not hasattr(self, '_preload_wait_start'):
                    self._preload_wait_start = time.time()
                if time.time() - self._preload_wait_start < 10.0:
                    return
                print("[PI] preload timeout, creating estimator synchronously", flush=True)
            self.point_status = "loading"
            if PointingTargetEstimator is None:
                self.point_status = "module_unavailable"
                emit("pointing_error", error="PointingTargetEstimator module not available")
                return
            self.point_estimator = PointingTargetEstimator(
                FRAME_W,
                FRAME_H,
                ray_mode=POINT_RAY_MODE,
                **self._pointing_intrinsics(),
            )
            self.point_status = "ready"
            emit("pointing_ready", method="sync_fallback")

        target = self.point_estimator.update(frame, hand_landmarks)
        if target.get("async_pending"):
            self.point_status = "tracking_depth_waiting"
        elif target.get("depth_available") is False:
            self.point_status = "tracking_2d_fallback"
        elif target.get("used_depth_hit"):
            self.point_status = "tracking_depth"
        else:
            self.point_status = "tracking_depth_fallback"
        self.point_display_target = target.get("display_target")
        self.point_stable_ratio = float(target.get("stable_ratio", 0.0))
        self.point_std_px = float(target.get("std_px", 0.0))
        self.point_ray_start_px = target.get("ray_start_px")
        self.point_ray_tip_px = target.get("ray_tip_px")
        self.point_ray_hit_px = target.get("raw_target")
        self.point_depth_map = target.get("depth_map")
        if target.get("pan_deg") is not None:
            self.preview_pan_deg = float(target["pan_deg"])
            self.preview_tilt_deg = float(target["tilt_deg"])
        if target.get("confirmed") and target["confirmed"] != self.point_target:
            previous_pan = self.pan_deg
            previous_tilt = self.tilt_deg
            if target.get("pan_deg") is not None:
                self.pan_deg = float(target["pan_deg"])
                self.tilt_deg = float(target["tilt_deg"])
                
            previous_servo_pan = self.servo_pan_deg
            previous_servo_tilt = self.servo_tilt_deg

            # Prefer the measured full-frame pixel LUT when present. Otherwise
            # keep the existing center/gain/offset conversion.
            lut_angles = self.servo_pointing_calibration.map_pixel(
                target["confirmed"]
            )
            if lut_angles is not None:
                new_servo_pan, new_servo_tilt = lut_angles
                servo_mapping = "pixel_lut"
            else:
                new_servo_pan = (
                    SERVO_PAN_CENTER
                    + self.pan_deg * SERVO_PAN_SIGN * POINT_PAN_GAIN
                    + POINT_PAN_OFFSET_DEG
                )
                new_servo_tilt = (
                    SERVO_TILT_CENTER
                    + self.tilt_deg * SERVO_TILT_SIGN * POINT_TILT_GAIN
                    + POINT_TILT_OFFSET_DEG
                )
                servo_mapping = "angle_gain_offset"
            new_servo_pan = clamp(new_servo_pan, PAN_MIN_DEG, PAN_MAX_DEG)
            new_servo_tilt = clamp(new_servo_tilt, TILT_MIN_DEG, TILT_MAX_DEG)
            self.last_delta_pan_deg = new_servo_pan - previous_servo_pan
            self.last_delta_tilt_deg = new_servo_tilt - previous_servo_tilt
            self.point_target = target["confirmed"]
            self.point_mode = False
            self.point_status = "locked"

            self.servo_pan_deg = new_servo_pan
            self.servo_tilt_deg = new_servo_tilt
            if self.servo is not None:
                self.servo.move_to(new_servo_pan, new_servo_tilt)
                emit(
                    "servo_moved",
                    servo_pan=round(new_servo_pan, 2),
                    servo_tilt=round(new_servo_tilt, 2),
                    delta_pan=round(self.last_delta_pan_deg, 2),
                    delta_tilt=round(self.last_delta_tilt_deg, 2),
                    pan_center=round(SERVO_PAN_CENTER, 2),
                    tilt_center=round(SERVO_TILT_CENTER, 2),
                    point_pan_gain=round(POINT_PAN_GAIN, 3),
                    point_tilt_gain=round(POINT_TILT_GAIN, 3),
                    point_pan_offset_deg=round(POINT_PAN_OFFSET_DEG, 2),
                    point_tilt_offset_deg=round(POINT_TILT_OFFSET_DEG, 2),
                    servo_mapping=servo_mapping,
                )

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
                hit_method=target.get("hit_method"),
                used_depth_hit=target.get("used_depth_hit"),
                servo_mapping=servo_mapping,
            )
            emit("point_mode", enabled=False, reason="target_locked")

    def _update_point_arm(self, gesture, sample_updated):
        if (
            not self.point_mode
            or self.point_tracking_armed
            or not sample_updated
            or time.time() - self.point_mode_entered_at < POINT_ENTRY_GRACE_SECONDS
        ):
            return
        self.point_arm_history.append(gesture == "POINT")
        if (
            len(self.point_arm_history) >= POINT_ARM_WINDOW
            and sum(self.point_arm_history) >= POINT_ARM_MIN_HITS
        ):
            self.point_tracking_armed = True
            if self.point_estimator:
                self.point_estimator.reset(clear_confirmed=False)
            emit(
                "point_tracking_armed",
                hits=sum(self.point_arm_history),
                window=len(self.point_arm_history),
            )

    def apply_gesture(self, gesture, results, frame, wave_active, sample_updated=True):
        self._update_point_arm(gesture, sample_updated)
        if gesture is None:
            if not self.mode_switch_armed:
                now = time.time()
                if self.mode_switch_release_start is None:
                    self.mode_switch_release_start = now
                elif now - self.mode_switch_release_start >= 1.0:
                    self.mode_switch_armed = True
                    self.mode_switch_release_start = None
            # point_mode 활성화 중에는 제스처가 잠깐 끊겨도 point_mode 유지
            if not self.point_mode:
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


        if gesture == "FIST": ##0519_v4 수정부분
            hold_time = COMMAND_HOLD_SECONDS * (4 if self.point_mode else 1)
            if not self._hold_ready_with("FIST", hold_time):
                return None
            self.sleep()
            self._reset_hold()
        elif gesture == "POINT_MODE":
            if self.point_mode:
                return gesture
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
            if not self.point_tracking_armed:
                return gesture
            self.update_point_target(frame, results.multi_hand_landmarks[0])
        elif gesture in ("THUMBS_UP", "THUMBS_DOWN"):    #0520_v2m   
            self._reset_hold()
            if not self.point_mode:              # ← 추가: 포인트모드 중엔 밝기 변경 무시
                self.apply_brightness_gesture(gesture)
        elif gesture == "MODE_SWITCH":
            self.mode_switch_release_start = None
            if not self.mode_switch_armed:
                return None
            if self.latched_gesture == "MODE_SWITCH":
                return gesture
            # cooldown: 마지막 전환 후 2초 이내면 무시 (제스처 인식 변동으로 인한 중복 토글 방지)
            if time.time() - getattr(self, "last_mode_switch_time", 0.0) < 2.0:
                return None
            if not self._hold_ready("MODE_SWITCH"):
                return None
            self.mode = "Mood" if self.mode == "Spot" else "Spot"
            self.last_mode_switch_time = time.time()
            self.mode_switch_armed = False
            self.save_state()
            emit("mode", mode=self.mode)
            self._reset_hold()
        else:
            self.mode_switch_armed = True
            self.mode_switch_release_start = None
            self._reset_hold()
        self.latched_gesture = gesture
        return gesture

    def _hold_ready_with(self, gesture, seconds): ##0519_v4 추가부분
        now = time.time()
        if self.hold_gesture != gesture:
            self.hold_gesture = gesture
            self.hold_start_time = now
            return False
        return now - self.hold_start_time >= seconds

    def state_payload(self, fps, gesture, hand_detected, bbox_px, wave_active):
        active_remaining = max(0.0, self.active_until - time.time()) if not self.standby else 0.0
        return {
            "standby": self.standby,
            "power": self.power,
            "brightness": self.brightness,
            "mode": self.mode,
            "point_mode": self.point_mode,
            "point_tracking_armed": self.point_tracking_armed,
            "point_arm_hits": sum(self.point_arm_history),
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
            "servo_pan_deg": round(self.servo_pan_deg, 2),  # 0521_v2m (디버깅용)
            "servo_tilt_deg": round(self.servo_tilt_deg, 2),
            "gesture": gesture,
            "hand_detected": hand_detected,
            "hand_bbox_width_px": bbox_px,
            "wave_active": wave_active,
            "fps": round(fps, 1),
        }

