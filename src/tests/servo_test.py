"""
서보 단독 테스트 - Phase 2 검증용
회로 결선 후 가장 먼저 실행해서 짐벌 동작 확인.

실행:
    sudo python3 -m src.tests.servo_test

테스트 시나리오:
1. 두 축이 0°, 45°, 90°, 135°, 180°로 차례로 이동
2. Pan만 좌우 스윕 (Tilt 고정)
3. Tilt만 위아래 스윕 (Pan 고정)
4. 대각선 이동
"""
import logging
import time
import sys

from src.config import LOG_FORMAT, LOG_LEVEL, TILT_MIN_DEG, TILT_MAX_DEG
from src.core.servo_controller import ServoController

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
log = logging.getLogger("servo_test")


def test_basic_angles(servo: ServoController):
    """1단계: 기본 각도 5개 위치 확인."""
    log.info("=== Test 1: Basic angles ===")
    angles = [0, 45, 90, 135, 180]
    for ang in angles:
        log.info(f"Moving both axes to {ang}°")
        # tilt은 기구 제한 안에서
        tilt_ang = max(TILT_MIN_DEG, min(TILT_MAX_DEG, ang))
        servo.move_to(ang, tilt_ang)
        time.sleep(1.0)


def test_pan_sweep(servo: ServoController):
    """2단계: Pan 축만 좌우 스윕."""
    log.info("=== Test 2: Pan sweep ===")
    servo.move_to(0, 90)
    time.sleep(0.5)
    for ang in range(0, 181, 10):
        servo.move_to(ang, 90, smooth=False)
        time.sleep(0.1)
    for ang in range(180, -1, -10):
        servo.move_to(ang, 90, smooth=False)
        time.sleep(0.1)


def test_tilt_sweep(servo: ServoController):
    """3단계: Tilt 축만 위아래 스윕."""
    log.info("=== Test 3: Tilt sweep ===")
    servo.move_to(90, TILT_MIN_DEG)
    time.sleep(0.5)
    for ang in range(TILT_MIN_DEG, TILT_MAX_DEG + 1, 10):
        servo.move_to(90, ang, smooth=False)
        time.sleep(0.1)
    for ang in range(TILT_MAX_DEG, TILT_MIN_DEG - 1, -10):
        servo.move_to(90, ang, smooth=False)
        time.sleep(0.1)


def test_diagonal(servo: ServoController):
    """4단계: 대각선 이동 (두 축 동시 구동)."""
    log.info("=== Test 4: Diagonal movement ===")
    waypoints = [
        (0, TILT_MIN_DEG),
        (180, TILT_MAX_DEG),
        (180, TILT_MIN_DEG),
        (0, TILT_MAX_DEG),
        (90, 90),  # 마지막 중앙
    ]
    for pan, tilt in waypoints:
        log.info(f"→ Pan={pan}°, Tilt={tilt}°")
        servo.move_to(pan, tilt)
        time.sleep(0.8)


def main():
    log.info("Servo test starting. Press Ctrl+C to abort.")
    try:
        with ServoController() as servo:
            test_basic_angles(servo)
            time.sleep(1.0)
            test_pan_sweep(servo)
            time.sleep(1.0)
            test_tilt_sweep(servo)
            time.sleep(1.0)
            test_diagonal(servo)
            
            log.info("✓ All tests complete. Homing.")
            servo.home()
            time.sleep(1.0)
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        log.error(f"Test failed: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
