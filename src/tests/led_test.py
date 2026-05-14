"""
LED 단독 테스트 - Phase 3 검증용

실행:
    sudo python3 -m src.tests.led_test

테스트:
1. Spot/Mood 각각 ON/OFF
2. 0~100% 디밍 스윕
3. 양쪽 동시 페이드
"""
import logging
import time
import sys

from src.config import LOG_FORMAT, LOG_LEVEL
from src.core.led_controller import LEDController

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
log = logging.getLogger("led_test")


def test_basic_onoff(led: LEDController):
    """1단계: ON/OFF 기본 확인."""
    log.info("=== Test 1: Basic ON/OFF ===")
    log.info("Spot ON (50%)")
    led.spot_set(50)
    time.sleep(2)
    led.spot_off()
    time.sleep(0.5)
    
    log.info("Mood ON (50%)")
    led.mood_set(50)
    time.sleep(2)
    led.mood_off()
    time.sleep(0.5)


def test_dimming_sweep(led: LEDController):
    """2단계: 0~100% 부드러운 스윕."""
    log.info("=== Test 2: Spot dimming sweep ===")
    for duty in range(0, 101, 5):
        led.spot_set(duty)
        time.sleep(0.05)
    for duty in range(100, -1, -5):
        led.spot_set(duty)
        time.sleep(0.05)
    
    log.info("=== Test 3: Mood dimming sweep ===")
    for duty in range(0, 101, 5):
        led.mood_set(duty)
        time.sleep(0.05)
    for duty in range(100, -1, -5):
        led.mood_set(duty)
        time.sleep(0.05)


def test_simultaneous(led: LEDController):
    """4단계: 양쪽 동시 페이드 (한쪽이 켜질 때 다른 쪽 꺼짐)."""
    log.info("=== Test 4: Cross-fade ===")
    for i in range(0, 101, 2):
        led.spot_set(i)
        led.mood_set(100 - i)
        time.sleep(0.03)
    for i in range(100, -1, -2):
        led.spot_set(i)
        led.mood_set(100 - i)
        time.sleep(0.03)
    led.all_off()


def main():
    log.info("LED test starting. Watch heat sink temperature!")
    log.info("⚠ Do NOT look directly at 10W LED. Check via reflection.")
    
    try:
        with LEDController() as led:
            test_basic_onoff(led)
            test_dimming_sweep(led)
            test_simultaneous(led)
            log.info("✓ All tests complete")
    except KeyboardInterrupt:
        log.warning("Interrupted")
        sys.exit(1)
    except Exception as e:
        log.error(f"Test failed: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
