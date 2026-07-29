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

    def __init__(
        self,
        initial_pan: float = 90.0,
        initial_tilt: float = 90.0,
        max_step_deg: float = SERVO_MAX_STEP_DEG,
        step_delay_s: float = SERVO_STEP_DELAY_S,
    ):
        """
        initial_pan, initial_tilt: 시작 시 즉시 송신할 PWM 절대 각도.
        호출자가 전달하는 angle은 항상 PWM 절대 각도로 해석된다 (OFFSET 없음).
        포인팅 각도 → PWM 절대각 변환은 호출자가 담당.
        """
        initial_pan = self._clamp_pan(float(initial_pan))
        initial_tilt = self._clamp_tilt(float(initial_tilt))
        if ServoKit is None:
            log.warning("adafruit_servokit not available - running in DRY mode")
            self.kit = None
        else:
            self.kit = ServoKit(channels=16, address=PCA9685_ADDRESS, frequency=PCA9685_FREQ)
            log.info(f"PCA9685 initialized at 0x{PCA9685_ADDRESS:02X}, {PCA9685_FREQ}Hz")

        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._current_pan = initial_pan
        self._current_tilt = initial_tilt
        self._target_pan = initial_pan
        self._target_tilt = initial_tilt
        self._home_pan = initial_pan
        self._home_tilt = initial_tilt
        self._max_step_deg = max(0.01, float(max_step_deg))
        self._step_delay_s = max(0.0, float(step_delay_s))
        self._new_target = threading.Event()
        self._paused = threading.Event()
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
        log.debug(f"CH{channel} cmd={angle:.1f} → PWM={actual:.1f}")
        if self.kit is None:
            return
        with self._io_lock:
            try:
                self.kit.servo[channel].angle = actual
            except Exception as e:
                log.error(f"Failed to set CH{channel} to {actual}°: {e}")

    def move_to(self, pan_deg: float, tilt_deg: float, smooth: bool = True):
        if self._stop:
            return
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
            if self._paused.is_set():
                continue

            while not self._stop:
                if self._paused.is_set():
                    break
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

                with self._lock:
                    max_step_deg = self._max_step_deg
                    step_delay_s = self._step_delay_s
                steps = max(math.ceil(max_delta / max_step_deg), 1)
                interrupted = False

                for i in range(1, steps + 1):
                    if self._stop or self._paused.is_set() or self._new_target.is_set():
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
                    time.sleep(step_delay_s)

                if not interrupted:
                    with self._lock:
                        self._current_pan = pan_target
                        self._current_tilt = tilt_target
                    break

                if self._paused.is_set():
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

    def set_motion_profile(self, max_step_deg: float, step_delay_s: float):
        """다음 보간 스텝부터 이동 속도/부드러움 설정을 적용한다."""
        if max_step_deg <= 0:
            raise ValueError("max_step_deg must be greater than 0")
        if step_delay_s < 0:
            raise ValueError("step_delay_s must be 0 or greater")
        with self._lock:
            self._max_step_deg = float(max_step_deg)
            self._step_delay_s = float(step_delay_s)

    def get_motion_profile(self) -> tuple[float, float]:
        """현재 (스텝당 최대 각도, 스텝 지연 초) 설정을 반환한다."""
        with self._lock:
            return self._max_step_deg, self._step_delay_s

    def wait_until_done(self, timeout: float = 5.0, tol: float = 0.5) -> bool:
        """현재 진행 중인 이동이 끝날 때까지 대기.
        servo_test 등 동기적 시퀀스 진행에 사용. 비동기 사용에는 호출 불필요."""
        start = time.time()
        while time.time() - start < timeout:
            if self._stop or self._paused.is_set():
                return False
            with self._lock:
                at_target = (
                    abs(self._current_pan - self._target_pan) < tol
                    and abs(self._current_tilt - self._target_tilt) < tol
                )
            if at_target:
                return True
            time.sleep(0.02)
        return False

    def home(self):
        """초기화할 때 지정한 Pan/Tilt 중심 위치로 복귀."""
        log.info("Homing to center")
        self.move_to(self._home_pan, self._home_tilt, smooth=True)

    def pause(self):
        """Keep the worker alive while releasing servo PWM during sleep."""
        if self._stop or self._paused.is_set():
            return
        self._paused.set()
        self._new_target.set()
        if self.kit is None:
            return
        with self._io_lock:
            try:
                self.kit.servo[SERVO_PAN_CH].angle = None
                self.kit.servo[SERVO_TILT_CH].angle = None
                log.info("Servo PWM paused")
            except Exception as e:
                log.error(f"Pause error: {e}")

    def resume(self):
        """Re-enable PWM at the last known position and resume queued motion."""
        if self._stop or not self._paused.is_set():
            return
        with self._lock:
            pan_now = self._current_pan
            tilt_now = self._current_tilt
        self._paused.clear()
        self._set_raw(SERVO_PAN_CH, pan_now)
        self._set_raw(SERVO_TILT_CH, tilt_now)
        self._new_target.set()
        log.info("Servo PWM resumed")

    def shutdown(self):
        """워커 스레드 종료 후 서보 PWM 끄기."""
        if self._stop:
            return
        self._stop = True
        self._new_target.set()
        if (
            self._worker.is_alive()
            and threading.current_thread() is not self._worker
        ):
            self._worker.join(timeout=1.0)
        if self.kit is None:
            return
        with self._io_lock:
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
