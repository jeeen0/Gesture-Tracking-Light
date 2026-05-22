"""
LED 컨트롤러 - 10W 스팟 + 3W 무드 LED 디밍 제어

회로:
- 10W 라인: Pi GPIO 12 → MOSFET 모듈 SIG → HAM2121 → 10W COB LED
- 3W 라인:  Pi GPIO 13 → HAM3005 DIM → 3W LED × 8 (병렬)

PWM 1kHz 고정. duty 0~100% 입력.
"""
import logging
import time
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
        # 시작 시 명시적으로 OFF 신호 송신 (floating 방지)
        self.spot_off()
        self.mood_off()
    
    GAMMA = 2.2  # LED dimming 인지 선형화 (사람 눈의 로그 응답 보정)

    def _to_pwm_duty(self, brightness_percent: float) -> float:
        """사용자 밝기 (0~100) → PWM duty (0~100), 감마 보정 적용."""
        b = max(0.0, min(100.0, brightness_percent)) / 100.0
        return (b ** self.GAMMA) * 100.0

    def _set_pwm(self, pin: int, duty_percent: float):
        """0~100 → 0~100 duty cycle PWM 설정."""
        duty = max(0.0, min(100.0, duty_percent))
        print(f"[LED-PWM] pin={pin} duty={duty:.1f}%", flush=True)
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
        """10W 스팟 조명 밝기 설정 (0~100%). 감마 보정으로 인지 선형화."""
        self._spot_duty = max(0.0, min(100.0, duty_percent))
        actual = self._to_pwm_duty(self._spot_duty)
        self._set_pwm(PIN_LED_SPOT_PWM, actual)
        log.debug(f"Spot LED → {self._spot_duty:.1f}% (PWM duty {actual:.1f}%)")
    
    def spot_on(self, duty_percent: float = 80.0):
        self.spot_set(duty_percent)
    
    def spot_off(self):
        self.spot_set(0)
    
    # ---------- Mood (3W × 8) ----------
    
    def mood_set(self, duty_percent: float):
        """3W 무드 LED 밝기 설정 (0~100%). 감마 보정 + HAM3005 active low 반전."""
        self._mood_duty = max(0.0, min(100.0, duty_percent))
        corrected = self._to_pwm_duty(self._mood_duty)
        actual = 100.0 - corrected
        self._set_pwm(PIN_LED_MOOD_PWM, actual)
        log.debug(f"Mood LED → {self._mood_duty:.1f}% (PWM duty {actual:.1f}%)")
    
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
        """GPIO 해제. PWM을 명시적으로 OFF 신호로 보낸 뒤 핀을 고정한다.
        Spot(MOSFET): active high → LOW = OFF.
        Mood(HAM3005): active low → HIGH = OFF."""
        self.all_off()
        if self.h is not None:
            try:
                # Spot: PWM duty 0 (LOW) → MOSFET OFF
                # Mood: PWM duty 100 (HIGH) → HAM3005 OFF
                lgpio.tx_pwm(self.h, PIN_LED_SPOT_PWM, LED_PWM_FREQ, 0)
                lgpio.tx_pwm(self.h, PIN_LED_MOOD_PWM, LED_PWM_FREQ, 100)
                time.sleep(0.02)
                # 핀 고정 (floating 방지)
                try:
                    lgpio.gpio_write(self.h, PIN_LED_SPOT_PWM, 0)  # Spot LOW = OFF
                    lgpio.gpio_write(self.h, PIN_LED_MOOD_PWM, 1)  # Mood HIGH = OFF
                except Exception:
                    pass
                # GPIO 해제
                try:
                    lgpio.gpio_free(self.h, PIN_LED_SPOT_PWM)
                    lgpio.gpio_free(self.h, PIN_LED_MOOD_PWM)
                except Exception:
                    pass
                lgpio.gpiochip_close(self.h)
            except Exception as e:
                log.error(f"Close error: {e}")
        log.info("LED controller shutdown")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown()
