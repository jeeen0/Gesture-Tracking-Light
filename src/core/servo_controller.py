"""
서보 컨트롤러 - PCA9685 I2C PWM 드라이버를 통해 Pan/Tilt 짐벌을 제어
회로: Raspberry Pi → I2C → PCA9685 → MG946R/MG90S
"""
import logging
import time
from typing import Optional
import math

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
    
    PCA9685를 통해 두 서보의 각도를 부드럽게 제어.
    급격한 이동 시 발생하는 지터를 방지하기 위해 단계적 이동(step) 사용.
    """
    
    def __init__(self):
        if ServoKit is None:
            log.warning("adafruit_servokit not available - running in DRY mode")
            self.kit = None
        else:
            self.kit = ServoKit(channels=16, address=PCA9685_ADDRESS, frequency=PCA9685_FREQ)
            log.info(f"PCA9685 initialized at 0x{PCA9685_ADDRESS:02X}, {PCA9685_FREQ}Hz")
        
        # 현재 각도 추적 (부드러운 이동용)
        self._current_pan = 90.0
        self._current_tilt = 90.0
        
        # 초기 위치로 이동 (중앙)
        self.move_to(90, 90, smooth=False)
    
    def _clamp_pan(self, deg: float) -> float:
        return max(PAN_MIN_DEG, min(PAN_MAX_DEG, deg))
    
    def _clamp_tilt(self, deg: float) -> float:
        return max(TILT_MIN_DEG, min(TILT_MAX_DEG, deg))
    
    def _set_raw(self, channel: int, angle: float):
        """PCA9685 채널에 직접 각도 명령 전송."""
        if self.kit is None:
            log.debug(f"[DRY] CH{channel} → {angle:.1f}°")
            return
        try:
            self.kit.servo[channel].angle = angle
        except Exception as e:
            log.error(f"Failed to set CH{channel} to {angle}°: {e}")
    
    def move_to(self, pan_deg: float, tilt_deg: float, smooth: bool = True):
        pan_target = self._clamp_pan(pan_deg)
        tilt_target = self._clamp_tilt(tilt_deg)
        
        if not smooth:
            self._set_raw(SERVO_PAN_CH, pan_target)
            self._set_raw(SERVO_TILT_CH, tilt_target)
            self._current_pan = pan_target
            self._current_tilt = tilt_target
            return
        
        pan_delta = pan_target - self._current_pan
        tilt_delta = tilt_target - self._current_tilt
        max_delta = max(abs(pan_delta), abs(tilt_delta))
        
        if max_delta < 0.1:  # 거의 안 움직이면 skip
            return
        
        steps = max(int(max_delta / SERVO_MAX_STEP_DEG), 1)
        pan_start = self._current_pan
        tilt_start = self._current_tilt
        
        for i in range(1, steps + 1):
            t = i / steps
            # ease-in-out: 0.5 - 0.5*cos(pi*t)
            eased = 0.5 - 0.5 * math.cos(math.pi * t)
            pan_now = pan_start + pan_delta * eased
            tilt_now = tilt_start + tilt_delta * eased
            self._set_raw(SERVO_PAN_CH, pan_now)
            self._set_raw(SERVO_TILT_CH, tilt_now)
            time.sleep(SERVO_STEP_DELAY_S)
        
        self._current_pan = pan_target
        self._current_tilt = tilt_target
    
    def get_position(self) -> tuple[float, float]:
        """현재 (pan, tilt) 각도 반환."""
        return self._current_pan, self._current_tilt
    
    def home(self):
        """중앙 위치로 복귀 (서보 보호용)."""
        log.info("Homing to center")
        self.move_to(90, 90, smooth=True)
    
    def shutdown(self):
        """서보 PWM 끄기 (idle 상태 진입 시 사용)."""
        if self.kit is None:
            return
        try:
            # PCA9685 채널에 None 할당하면 PWM 출력 정지
            self.kit.servo[SERVO_PAN_CH].angle = None
            self.kit.servo[SERVO_TILT_CH].angle = None
            log.info("Servo PWM disabled")
        except Exception as e:
            log.error(f"Shutdown error: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown()
