"""키보드로 Pan/Tilt 서보 각도와 이동 속도를 실시간 조정한다.

라즈베리파이의 프로젝트 루트에서 실행:
    python -m src.tests.servo_keyboard_tuner

키를 누른 즉시 반응하므로 Enter를 누를 필요가 없다.
"""

import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from src.config import (
    PAN_MAX_DEG,
    PAN_MIN_DEG,
    SERVO_MAX_STEP_DEG,
    SERVO_STEP_DELAY_S,
    TILT_MAX_DEG,
    TILT_MIN_DEG,
)
from src.core.servo_controller import ServoController
from src.vision.pi_runtime_config import SERVO_PAN_CENTER, SERVO_TILT_CENTER


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "src" / "calibration" / "servo_keyboard_tuning.json"


@contextmanager
def raw_keyboard():
    """Windows와 Linux/SSH 터미널에서 한 글자씩 읽는 함수를 제공한다."""
    if os.name == "nt":
        import msvcrt

        yield lambda: msvcrt.getwch()
        return

    if not sys.stdin.isatty():
        raise RuntimeError("키보드 튜너는 대화형 터미널(TTY)에서 실행해야 합니다.")

    import termios
    import tty

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield lambda: sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def print_help():
    print(
        "\n"
        "  W / S       Tilt - / +\n"
        "  A / D       Pan  - / +\n"
        "  [ / ]       키 1회당 각도 감소 / 증가\n"
        "  - / =       보간 최대 스텝 감소 / 증가 (크면 빠르고 거칠 수 있음)\n"
        "  , / .       스텝 지연 감소 / 증가 (작으면 빠름)\n"
        "  H           설정한 CENTER로 복귀\n"
        "  Space       PWM 일시 해제 / 다시 활성화\n"
        "  P           현재 설정을 JSON으로 저장\n"
        "  ?           도움말\n"
        "  Q           종료(PWM 해제)\n",
        flush=True,
    )


def print_status(pan, tilt, angle_step, max_step, delay, paused=False):
    state = "PAUSED" if paused else "ACTIVE"
    print(
        f"[{state}] pan={pan:6.1f}°  tilt={tilt:6.1f}°  "
        f"key_step={angle_step:.2f}°  max_step={max_step:.2f}°  "
        f"delay={delay * 1000:.1f}ms",
        flush=True,
    )


def save_tuning(path, pan, tilt, angle_step, max_step, delay):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "servo_pan_center": round(pan, 3),
        "servo_tilt_center": round(tilt, 3),
        "keyboard_angle_step_deg": round(angle_step, 3),
        "servo_max_step_deg": round(max_step, 4),
        "servo_step_delay_s": round(delay, 5),
        "runtime_env": {
            "PI_SERVO_PAN_CENTER": round(pan, 3),
            "PI_SERVO_TILT_CENTER": round(tilt, 3),
        },
        "config_py": {
            "SERVO_MAX_STEP_DEG": round(max_step, 4),
            "SERVO_STEP_DELAY_S": round(delay, 5),
        },
        "note": (
            "This file is a tuning record and is not loaded automatically. "
            "Use runtime_env for src.main and copy config_py values to src/config.py."
        ),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[저장] {path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Pan/Tilt 서보 키보드 튜너")
    parser.add_argument("--pan", type=float, default=SERVO_PAN_CENTER, help="시작 Pan 각도")
    parser.add_argument("--tilt", type=float, default=SERVO_TILT_CENTER, help="시작 Tilt 각도")
    parser.add_argument("--step", type=float, default=1.0, help="키 1회당 조정 각도")
    parser.add_argument(
        "--max-step",
        type=float,
        default=SERVO_MAX_STEP_DEG,
        help="서보 보간 스텝 각도",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=SERVO_STEP_DELAY_S,
        help="서보 보간 스텝 지연(초)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="P 키 저장 경로")
    return parser.parse_args()


def main():
    args = parse_args()
    pan = clamp(args.pan, PAN_MIN_DEG, PAN_MAX_DEG)
    tilt = clamp(args.tilt, TILT_MIN_DEG, TILT_MAX_DEG)
    home_pan = pan
    home_tilt = tilt
    angle_step = clamp(args.step, 0.1, 20.0)
    max_step = clamp(args.max_step, 0.1, 10.0)
    delay = clamp(args.delay, 0.0, 0.1)
    paused = False

    print("서보 키보드 튜너 시작. 모터의 물리적 간섭 여부를 먼저 확인하세요.", flush=True)
    print_help()

    try:
        with ServoController(
            initial_pan=pan,
            initial_tilt=tilt,
            max_step_deg=max_step,
            step_delay_s=delay,
        ) as servo:
            print_status(pan, tilt, angle_step, max_step, delay)
            with raw_keyboard() as read_key:
                while True:
                    key = read_key().lower()
                    changed_position = False

                    if key == "q":
                        break
                    if key == " ":
                        if paused:
                            servo.resume()
                        else:
                            servo.pause()
                        paused = not paused
                    elif key == "?" or key == "/":
                        print_help()
                    elif key == "p":
                        if not paused and not servo.wait_until_done():
                            print("[저장 보류] 목표 위치에 도달하지 못했습니다.", flush=True)
                            continue
                        actual_pan, actual_tilt = servo.get_position()
                        save_tuning(
                            args.output.resolve(),
                            actual_pan,
                            actual_tilt,
                            angle_step,
                            max_step,
                            delay,
                        )
                    elif key == "h":
                        pan = home_pan
                        tilt = home_tilt
                        changed_position = True
                    elif key == "a":
                        pan = clamp(pan - angle_step, PAN_MIN_DEG, PAN_MAX_DEG)
                        changed_position = True
                    elif key == "d":
                        pan = clamp(pan + angle_step, PAN_MIN_DEG, PAN_MAX_DEG)
                        changed_position = True
                    elif key == "w":
                        tilt = clamp(tilt - angle_step, TILT_MIN_DEG, TILT_MAX_DEG)
                        changed_position = True
                    elif key == "s":
                        tilt = clamp(tilt + angle_step, TILT_MIN_DEG, TILT_MAX_DEG)
                        changed_position = True
                    elif key == "[":
                        angle_step = clamp(angle_step / 2.0, 0.1, 20.0)
                    elif key == "]":
                        angle_step = clamp(angle_step * 2.0, 0.1, 20.0)
                    elif key == "-":
                        max_step = clamp(max_step - 0.1, 0.1, 10.0)
                        servo.set_motion_profile(max_step, delay)
                    elif key in ("=", "+"):
                        max_step = clamp(max_step + 0.1, 0.1, 10.0)
                        servo.set_motion_profile(max_step, delay)
                    elif key == ",":
                        delay = clamp(delay - 0.001, 0.0, 0.1)
                        servo.set_motion_profile(max_step, delay)
                    elif key == ".":
                        delay = clamp(delay + 0.001, 0.0, 0.1)
                        servo.set_motion_profile(max_step, delay)
                    else:
                        continue

                    if changed_position:
                        if paused:
                            pan, tilt = servo.get_position()
                            print("[정지 중] 이동 키를 무시했습니다. Space로 먼저 활성화하세요.", flush=True)
                        else:
                            servo.move_to(pan, tilt, smooth=True)
                    print_status(pan, tilt, angle_step, max_step, delay, paused)
    except KeyboardInterrupt:
        print("\n[중단] Ctrl+C - PWM을 해제합니다.", flush=True)
    except Exception as exc:
        print(f"\n[오류] {exc}", file=sys.stderr, flush=True)
        return 1

    print("\n종료했습니다. 서보 PWM이 해제되었습니다.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
