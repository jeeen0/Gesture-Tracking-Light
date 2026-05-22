"""
서보 컨트롤러 - PCA9685 I2C PWM 드라이버를 통해 Pan/Tilt 짐벌을 제어
회로: Raspberry Pi → I2C → PCA9685 → MG946R/MG90S
"""
import logging
import math
import threading
import time

try:
    from adafruit_servokit import ServoKit
except ImportError:
    ServoKit = None  # 실제 라즈베리파이가 아닐 때 (테스트용)

from src.config import (
    PCA9685_ADDRESS, PCA9685_FREQ,
    SERVO_PAN_CH, SERVO_TILT_CH,
    PAN_MIN_DEG, PAN_MAX_DEG, TILT_MIN_DEG, TILT_MAX_DEG,
    SERVO_MAX_STEP_DEG, SERVO_STEP_DELAY_S,
)

log = logging.getLogger(__name__)


class ServoController:
    """Pan-Tilt 짐벌 제어 클래스.

    move_to()는 즉시 반환하고 워커 스레드가 실제 이동을 처리.
    새 명령이 도착하면 진행 중인 이동을 중단하고 현재 위치에서 새 타깃으로 재계획.
    """

    def __init__(self, initial_pan: float = 90.0, initial_tilt: float = 90.0):
        """
        initial_pan, initial_tilt: 시작 시 즉시 송신할 PWM 절대 각도.
        호출자가 전달하는 angle은 항상 PWM 절대 각도로 해석된다 (OFFSET 없음).
        포인팅 각도 → PWM 절대각 변환은 호출자가 담당.
        """
        if ServoKit is None:
            log.warning("adafruit_servokit not available - running in DRY mode")
            self.kit = None
        else:
            self.kit = ServoKit(channels=16, address=PCA9685_ADDRESS, frequency=PCA9685_FREQ)
            log.info(f"PCA9685 initialized at 0x{PCA9685_ADDRESS:02X}, {PCA9685_FREQ}Hz")

        self._lock = threading.Lock()
        self._current_pan = initial_pan
        self._current_tilt = initial_tilt
        self._target_pan = initial_pan
        self._target_tilt = initial_tilt
        self._new_target = threading.Event()
        self._stop = False

        self._set_raw(SERVO_PAN_CH, initial_pan)
        self._set_raw(SERVO_TILT_CH, initial_tilt)

        self._worker = threading.Thread(target=self._move_worker, daemon=True)
        self._worker.start()

    def _clamp_pan(self, deg: float) -> float:
        return max(PAN_MIN_DEG, min(PAN_MAX_DEG, deg))

    def _clamp_tilt(self, deg: float) -> float:
        return max(TILT_MIN_DEG, min(TILT_MAX_DEG, deg))

    def _set_raw(self, channel: int, angle: float):
        """PCA9685 채널에 직접 각도 명령 전송 (PWM 절대 각도). 0~180 안전 clamp."""
        actual = max(0.0, min(180.0, angle))
        if self.kit is None:
            log.debug(f"[DRY] CH{channel} → {actual:.1f}°")
            return
        try:
            self.kit.servo[channel].angle = actual
        except Exception as e:
            log.error(f"Failed to set CH{channel} to {actual}°: {e}")

    def move_to(self, pan_deg: float, tilt_deg: float, smooth: bool = True):
        pan_target = self._clamp_pan(pan_deg)
        tilt_target = self._clamp_tilt(tilt_deg)

        if not smooth:
            with self._lock:
                self._current_pan = pan_target
                self._current_tilt = tilt_target
                self._target_pan = pan_target
                self._target_tilt = tilt_target
            self._set_raw(SERVO_PAN_CH, pan_target)
            self._set_raw(SERVO_TILT_CH, tilt_target)
            return

        with self._lock:
            self._target_pan = pan_target
            self._target_tilt = tilt_target
        self._new_target.set()

    def _move_worker(self):
        while not self._stop:
            self._new_target.wait(timeout=0.1)
            if self._stop:
                break
            self._new_target.clear()

            while not self._stop:
                with self._lock:
                    pan_target = self._target_pan
                    tilt_target = self._target_tilt
                    pan_start = self._current_pan
                    tilt_start = self._current_tilt

                pan_delta = pan_target - pan_start
                tilt_delta = tilt_target - tilt_start
                max_delta = max(abs(pan_delta), abs(tilt_delta))

                if max_delta < 0.1:
                    break

                steps = max(int(max_delta / SERVO_MAX_STEP_DEG), 1)
                interrupted = False

                for i in range(1, steps + 1):
                    if self._stop or self._new_target.is_set():
                        interrupted = True
                        break
                    t = i / steps
                    eased = 0.5 - 0.5 * math.cos(math.pi * t)
                    pan_now = pan_start + pan_delta * eased
                    tilt_now = tilt_start + tilt_delta * eased
                    self._set_raw(SERVO_PAN_CH, pan_now)
                    self._set_raw(SERVO_TILT_CH, tilt_now)
                    with self._lock:
                        self._current_pan = pan_now
                        self._current_tilt = tilt_now
                    time.sleep(SERVO_STEP_DELAY_S)

                if not interrupted:
                    with self._lock:
                        self._current_pan = pan_target
                        self._current_tilt = tilt_target
                    break

                # 새 명령이 와서 중단 → 현재 위치에서 재계획
                if self._new_target.is_set():
                    self._new_target.clear()
                    continue
                break

    def get_position(self) -> tuple[float, float]:
        """현재 (pan, tilt) 각도 반환."""
        with self._lock:
            return self._current_pan, self._current_tilt

    def home(self):
        """중앙 위치로 복귀 (서보 보호용)."""
        log.info("Homing to center")
        self.move_to(90, 90, smooth=True)

    def shutdown(self):
        """워커 스레드 종료 후 서보 PWM 끄기."""
        self._stop = True
        self._new_target.set()
        if self.kit is None:
            return
        try:
            self.kit.servo[SERVO_PAN_CH].angle = None
            self.kit.servo[SERVO_TILT_CH].angle = None
            log.info("Servo PWM disabled")
        except Exception as e:
            log.error(f"Shutdown error: {e}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
