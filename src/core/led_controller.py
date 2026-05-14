"""
LED 컨트롤러 - 10W 스팟 + 3W 무드 LED 디밍 제어

회로:
- 10W 라인: Pi GPIO 12 → MOSFET 모듈 SIG → HAM2121 → 10W COB LED
- 3W 라인:  Pi GPIO 13 → HAM3005 DIM → 3W LED × 8 (병렬)

PWM 1kHz 고정. duty 0~100% 입력.
"""
import logging
from typing import Optional

try:
    import lgpio  # Raspberry Pi 5 권장 GPIO 라이브러리
except ImportError:
    lgpio = None

from src.config import (
    PIN_LED_SPOT_PWM, PIN_LED_MOOD_PWM, LED_PWM_FREQ,
)

log = logging.getLogger(__name__)


class LEDController:
    """10W 스팟 + 3W 무드 두 채널 LED 디밍 컨트롤러.
    
    내부적으로 lgpio의 하드웨어 PWM 사용.
    duty는 0~100 범위 입력, 내부에서 0~1 변환.
    """
    
    def __init__(self):
        self._spot_duty = 0.0
        self._mood_duty = 0.0
        
        if lgpio is None:
            log.warning("lgpio not available - running in DRY mode")
            self.h = None
            return
        
        self.h = lgpio.gpiochip_open(0)
        # 출력 모드 설정
        lgpio.gpio_claim_output(self.h, PIN_LED_SPOT_PWM, 0)
        lgpio.gpio_claim_output(self.h, PIN_LED_MOOD_PWM, 0)
        log.info(f"LED PWM ready (Spot=GPIO{PIN_LED_SPOT_PWM}, Mood=GPIO{PIN_LED_MOOD_PWM}, {LED_PWM_FREQ}Hz)")
    
    def _set_pwm(self, pin: int, duty_percent: float):
        """0~100 → 0~100 duty cycle PWM 설정."""
        duty = max(0.0, min(100.0, duty_percent))
        if self.h is None:
            log.debug(f"[DRY] PWM pin {pin} duty={duty:.1f}%")
            return
        try:
            # lgpio.tx_pwm(handle, gpio, freq, duty_cycle 0~100)
            lgpio.tx_pwm(self.h, pin, LED_PWM_FREQ, duty)
        except Exception as e:
            log.error(f"PWM error on pin {pin}: {e}")
    
    # ---------- Spot (10W) ----------
    
    def spot_set(self, duty_percent: float):
        """10W 스팟 조명 밝기 설정 (0~100%)."""
        self._spot_duty = max(0.0, min(100.0, duty_percent))
        self._set_pwm(PIN_LED_SPOT_PWM, self._spot_duty)
        log.debug(f"Spot LED → {self._spot_duty:.1f}%")
    
    def spot_on(self, duty_percent: float = 80.0):
        self.spot_set(duty_percent)
    
    def spot_off(self):
        self.spot_set(0)
    
    # ---------- Mood (3W × 8) ----------
    
    def mood_set(self, duty_percent: float):
        """3W 무드 LED 밝기 설정 (0~100%)."""
        self._mood_duty = max(0.0, min(100.0, duty_percent))
        self._set_pwm(PIN_LED_MOOD_PWM, self._mood_duty)
        log.debug(f"Mood LED → {self._mood_duty:.1f}%")
    
    def mood_on(self, duty_percent: float = 40.0):
        self.mood_set(duty_percent)
    
    def mood_off(self):
        self.mood_set(0)
    
    # ---------- 통합 제어 ----------
    
    def all_off(self):
        """모든 LED 끄기."""
        self.spot_off()
        self.mood_off()
        log.info("All LEDs off")
    
    def get_state(self) -> dict:
        return {
            "spot_duty": self._spot_duty,
            "mood_duty": self._mood_duty,
        }
    
    def shutdown(self):
        """GPIO 해제."""
        self.all_off()
        if self.h is not None:
            try:
                lgpio.gpiochip_close(self.h)
            except Exception as e:
                log.error(f"Close error: {e}")
        log.info("LED controller shutdown")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown()
