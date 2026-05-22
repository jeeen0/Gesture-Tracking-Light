"""
서보 단독 테스트 - 정면(CENTER) 기준 상대 각도로 짐벌 동작 확인.

실행:
    sudo python3 -m src.tests.servo_test

테스트 시나리오 (CENTER 기준):
1. 정면에서 -45°, -22°, 0°, +22°, +45° 5개 위치 차례 이동
2. Pan 좌우 ±45° 스윕 (Tilt 정면 고정)
3. Tilt 상하 ±45° 스윕 (Pan 정면 고정)
4. 대각선 이동 (CENTER 기준 4방향)

CENTER 값은 pi_runtime_config.py 또는 환경변수 PI_SERVO_PAN_CENTER/PI_SERVO_TILT_CENTER로 설정.
"""
import logging
import time
import sys

from src.config import LOG_FORMAT, LOG_LEVEL, PAN_MIN_DEG, PAN_MAX_DEG, TILT_MIN_DEG, TILT_MAX_DEG
from src.core.servo_controller import ServoController
from src.vision.pi_runtime_config import SERVO_PAN_CENTER, SERVO_TILT_CENTER

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
log = logging.getLogger("servo_test")


def _clamp_pan(pwm: float) -> float:
    return max(PAN_MIN_DEG, min(PAN_MAX_DEG, pwm))


def _clamp_tilt(pwm: float) -> float:
    return max(TILT_MIN_DEG, min(TILT_MAX_DEG, pwm))


def test_basic_angles(servo: ServoController):
    """1단계: 정면 기준 5개 위치."""
    log.info("=== Test 1: Basic angles (CENTER 기준 ±45°) ===")
    offsets = [-45, -22, 0, 22, 45]
    for offset in offsets:
        pan_pwm = _clamp_pan(SERVO_PAN_CENTER + offset)
        tilt_pwm = _clamp_tilt(SERVO_TILT_CENTER + offset)
        log.info(f"offset={offset:+d}° → Pan PWM={pan_pwm:.1f}, Tilt PWM={tilt_pwm:.1f}")
        servo.move_to(pan_pwm, tilt_pwm)
        time.sleep(1.0)


def test_pan_sweep(servo: ServoController):
    """2단계: Pan 좌우 스윕 (Tilt 정면 고정)."""
    log.info("=== Test 2: Pan 좌우 스윕 (Tilt CENTER 고정) ===")
    pan_left = _clamp_pan(SERVO_PAN_CENTER - 45)
    pan_right = _clamp_pan(SERVO_PAN_CENTER + 45)
    tilt_center = _clamp_tilt(SERVO_TILT_CENTER)
    servo.move_to(pan_left, tilt_center)
    time.sleep(0.3)
    servo.move_to(pan_right, tilt_center)
    servo.move_to(pan_left, tilt_center)


def test_tilt_sweep(servo: ServoController):
    """3단계: Tilt 상하 스윕 (Pan 정면 고정)."""
    log.info("=== Test 3: Tilt 상하 스윕 (Pan CENTER 고정) ===")
    pan_center = _clamp_pan(SERVO_PAN_CENTER)
    tilt_up = _clamp_tilt(SERVO_TILT_CENTER - 45)
    tilt_down = _clamp_tilt(SERVO_TILT_CENTER + 45)
    servo.move_to(pan_center, tilt_up)
    time.sleep(0.3)
    servo.move_to(pan_center, tilt_down)
    servo.move_to(pan_center, tilt_up)


def test_diagonal(servo: ServoController):
    """4단계: 대각선 이동 (두 축 동시 구동)."""
    log.info("=== Test 4: Diagonal movement (CENTER 기준 4방향) ===")
    pan_c = SERVO_PAN_CENTER
    tilt_c = SERVO_TILT_CENTER
    waypoints = [
        (_clamp_pan(pan_c - 45), _clamp_tilt(tilt_c - 45)),  # ↖
        (_clamp_pan(pan_c + 45), _clamp_tilt(tilt_c + 45)),  # ↘
        (_clamp_pan(pan_c + 45), _clamp_tilt(tilt_c - 45)),  # ↗
        (_clamp_pan(pan_c - 45), _clamp_tilt(tilt_c + 45)),  # ↙
        (_clamp_pan(pan_c), _clamp_tilt(tilt_c)),            # 정면 복귀
    ]
    for pan, tilt in waypoints:
        log.info(f"→ Pan PWM={pan:.1f}, Tilt PWM={tilt:.1f}")
        servo.move_to(pan, tilt)
        time.sleep(0.8)


def main():
    log.info(f"Servo test starting. CENTER: Pan={SERVO_PAN_CENTER}°, Tilt={SERVO_TILT_CENTER}°")
    log.info("Press Ctrl+C to abort.")
    try:
        with ServoController(
            initial_pan=SERVO_PAN_CENTER,
            initial_tilt=SERVO_TILT_CENTER,
        ) as servo:
            time.sleep(0.5)  # 초기 정면 위치 안정화
            test_basic_angles(servo)
            time.sleep(1.0)
            test_pan_sweep(servo)
            time.sleep(1.0)
            test_tilt_sweep(servo)
            time.sleep(1.0)
            test_diagonal(servo)

            log.info("✓ All tests complete. Homing to CENTER.")
            servo.move_to(SERVO_PAN_CENTER, SERVO_TILT_CENTER)
            time.sleep(1.0)
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        log.error(f"Test failed: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
