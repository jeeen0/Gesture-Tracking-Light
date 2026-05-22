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


def _move_with_log(servo: ServoController, pan_pwm: float, tilt_pwm: float, label: str = "", hold: float = 0.3):
    """이전 위치 표시 + 이동 완료까지 대기 + 지정한 시간 동안 자세 유지."""
    prev_pan, prev_tilt = servo.get_position()
    prefix = f"  [{label}] " if label else "  "
    log.info(
        f"{prefix}Pan {prev_pan:5.1f} → {pan_pwm:5.1f}   "
        f"Tilt {prev_tilt:5.1f} → {tilt_pwm:5.1f}"
    )
    servo.move_to(pan_pwm, tilt_pwm)
    servo.wait_until_done(timeout=5.0)   # 이동 끝날 때까지 동기 대기
    time.sleep(hold)                     # 도착 후 잠시 멈춰서 눈으로 확인 가능


def test_basic_angles(servo: ServoController):
    """1단계: 정면 기준 큰 폭 이동.
    Pan은 ±90°, Tilt는 위 +45° (PWM 180) / 아래 -105° (PWM 30)."""
    log.info("")
    log.info("=" * 60)
    log.info(" Test 1: Basic angles (Pan ±90°, Tilt 위 +45 / 아래 -105)")
    log.info("=" * 60)
    # (pan_offset, tilt_offset, label)
    samples = [
        (-90, +45, "왼쪽 끝 + 위쪽"),
        (-45, +20, "왼쪽 + 약간 위"),
        (0, 0, "정면"),
        (45, -45, "오른쪽 + 아래"),
        (90, -90, "오른쪽 끝 + 아래쪽"),
    ]
    for pan_off, tilt_off, label in samples:
        pan_pwm = _clamp_pan(SERVO_PAN_CENTER + pan_off)
        tilt_pwm = _clamp_tilt(SERVO_TILT_CENTER + tilt_off)
        _move_with_log(servo, pan_pwm, tilt_pwm, label=label, hold=1.0)


def test_pan_sweep(servo: ServoController):
    """2단계: Pan 좌우 ±90° 스윕 (Tilt 정면 고정)."""
    log.info("")
    log.info("=" * 60)
    log.info(" Test 2: Pan 좌우 ±90° 스윕 (Tilt CENTER 고정)")
    log.info("=" * 60)
    pan_left = _clamp_pan(SERVO_PAN_CENTER - 90)
    pan_right = _clamp_pan(SERVO_PAN_CENTER + 90)
    tilt_center = _clamp_tilt(SERVO_TILT_CENTER)
    _move_with_log(servo, pan_left, tilt_center, label="왼쪽 끝", hold=0.3)
    _move_with_log(servo, pan_right, tilt_center, label="오른쪽 끝", hold=1.0)
    _move_with_log(servo, pan_left, tilt_center, label="왼쪽 끝", hold=1.0)


def test_tilt_sweep(servo: ServoController):
    """3단계: Tilt 위 +45° (PWM 180) / 아래 -105° (PWM 30) 스윕."""
    log.info("")
    log.info("=" * 60)
    log.info(" Test 3: Tilt 상하 스윕 (위 +45°, 아래 -105°)")
    log.info("=" * 60)
    pan_center = _clamp_pan(SERVO_PAN_CENTER)
    tilt_up = _clamp_tilt(SERVO_TILT_CENTER + 45)    # PWM 큰 값 = 위
    tilt_down = _clamp_tilt(SERVO_TILT_CENTER - 105) # PWM 작은 값 = 아래
    _move_with_log(servo, pan_center, tilt_up, label="위쪽 끝", hold=0.3)
    _move_with_log(servo, pan_center, tilt_down, label="아래쪽 끝", hold=1.0)
    _move_with_log(servo, pan_center, tilt_up, label="위쪽 끝", hold=1.0)


def test_diagonal(servo: ServoController):
    """4단계: 대각선 이동."""
    log.info("")
    log.info("=" * 60)
    log.info(" Test 4: Diagonal movement")
    log.info("=" * 60)
    pan_c = SERVO_PAN_CENTER
    tilt_c = SERVO_TILT_CENTER
    waypoints = [
        (_clamp_pan(pan_c - 90), _clamp_tilt(tilt_c + 45), "↖ 왼쪽 위"),
        (_clamp_pan(pan_c + 90), _clamp_tilt(tilt_c - 90), "↘ 오른쪽 아래"),
        (_clamp_pan(pan_c + 90), _clamp_tilt(tilt_c + 45), "↗ 오른쪽 위"),
        (_clamp_pan(pan_c - 90), _clamp_tilt(tilt_c - 90), "↙ 왼쪽 아래"),
        (_clamp_pan(pan_c), _clamp_tilt(tilt_c), "정면 복귀"),
    ]
    for pan, tilt, label in waypoints:
        _move_with_log(servo, pan, tilt, label=label, hold=0.8)


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
