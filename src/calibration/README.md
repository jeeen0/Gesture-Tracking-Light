# 포인팅 서보 조정 및 보정

권장 작업 순서는 다음과 같습니다.

1. 카메라 없이 키보드 튜너로 중심 각도와 이동 속도를 확인합니다.
2. 결정한 `CENTER`, `SIGN`, `GAIN`, `OFFSET`을 정상 실행 설정에 반영합니다.
3. ROI·깊이·포인팅 선 동작을 확인합니다.
4. 마지막으로 화면 지점별 잔차 LUT 보정을 실행합니다.

잔차 LUT는 기본 설정에서 남은 오차만 저장하므로, LUT를 만든 후
`CENTER`, `SIGN`, `GAIN`, `OFFSET`을 바꾸면 다시 보정해야 합니다.

## 1. 카메라 없이 서보 조정

라즈베리파이의 프로젝트 루트에서 실행합니다.

```bash
python -m src.tests.servo_keyboard_tuner
```

시작값을 직접 지정할 수도 있습니다.

```bash
python -m src.tests.servo_keyboard_tuner \
  --pan 90 \
  --tilt 135 \
  --step 1 \
  --max-step 0.5 \
  --delay 0.003
```

키:

- `A` / `D`: Pan 각도 감소 / 증가
- `W` / `S`: Tilt 각도 감소 / 증가
- `[` / `]`: 키 1회당 조정 각도 감소 / 증가
- `-` / `=`: 이동 보간 스텝 감소 / 증가
- `,` / `.`: 보간 스텝 지연 감소 / 증가
- `H`: 실행 시 설정된 중심 각도로 복귀
- `Space`: PWM 일시 해제 / 재활성화
- `P`: 현재 값 저장
- `?`: 도움말
- `Q`: 종료하고 PWM 해제

저장 파일은 다음 위치에 생성됩니다.

```text
src/calibration/servo_keyboard_tuning.json
```

이 파일은 측정 기록이며 런타임에서 자동으로 읽지 않습니다.

- `runtime_env`의 중심값은 `PI_SERVO_PAN_CENTER`,
  `PI_SERVO_TILT_CENTER`로 실행할 때 지정합니다.
- `config_py`의 이동값은 `src/config.py`의
  `SERVO_MAX_STEP_DEG`, `SERVO_STEP_DELAY_S`에 반영합니다.

예시:

```bash
PI_SERVO_PAN_CENTER=90 \
PI_SERVO_TILT_CENTER=135 \
PI_SHOW_PREVIEW=1 \
python -m src.main
```

## 2. 화면 지점별 잔차 LUT 보정

기본 모터 설정과 화면 포인팅 동작을 먼저 확정한 뒤 실행합니다.

```bash
python -m src.tests.pointing_servo_calibration
```

화면 중앙부터 시작해 3×3의 총 9개 지점을 보정합니다. 각 지점에서
`A/D`로 Pan, `W/S`로 Tilt를 모두 맞춘 뒤 `Enter`를 한 번 누르면
두 축의 보정값이 같은 지점에 함께 저장됩니다.

각 화면 지점에서 스크립트가 먼저 현재 기본 매핑 각도로 이동합니다.
조명 중심을 화면의 십자 표시에 맞춘 뒤 `Enter`를 누르면 기본 각도와
실제 각도의 차이만 저장됩니다.

키:

- `A` / `D`: Pan 감소 / 증가
- `W` / `S`: Tilt 감소 / 증가
- `[` / `]`: 조정 스텝 감소 / 증가
- `Enter`: 현재 지점의 잔차 보정값 기록
- `Q` 또는 `Esc`: 취소

완료된 파일:

```text
src/calibration/servo_pointing_calibration.json
```

새 파일 형식:

```json
{
  "frame_size": [640, 360],
  "mode": "residual_2d",
  "points": [
    {
      "x": 320,
      "y": 180,
      "pan_correction_deg": 1.5,
      "tilt_correction_deg": -2.0
    }
  ]
}
```

런타임 계산:

```text
최종 서보 각도
= CENTER/SIGN/GAIN/OFFSET 기본 각도
+ 화면 위치별 LUT correction_deg
```

기존 `servo_deg` 파일은 레거시 절대각 방식으로 계속 읽을 수 있지만,
새 누적 보정 방식을 사용하려면 보정 스크립트를 다시 실행해야 합니다.

정상 적용 로그:

```text
mode=residual_2d
servo_mapping=angle_gain_offset+residual_2d_lut
```

## 3. 포인팅 테스트 실행

```bash
PI_ROI_INPUT_SIZE=320 \
PI_LATEST_FRAME_CAPTURE=1 \
PI_DEPTH_ASYNC=1 \
PI_DEPTH_ASYNC_FPS=4 \
PI_DEPTH_RESULT_MAX_AGE=0.75 \
PI_TORCH_NUM_THREADS=2 \
PI_POINT_INFERENCE_FPS=20 \
PI_POINT_RAY_MODE=mcp_tip \
PI_POINT_ENTRY_GRACE_SECONDS=1.0 \
PI_POINT_ARM_WINDOW=5 \
PI_POINT_ARM_MIN_HITS=3 \
PI_SHOW_PREVIEW=1 \
python -m src.main
```

MiDaS 깊이 맵은 4fps로 갱신하지만, 최신 깊이 맵과 현재 손 좌표를
사용한 포인팅 선·교차점·EMA 계산은 MediaPipe 추론 프레임마다
갱신됩니다.

`finger_axis`는 화면 포인터가 `mcp_tip`보다 일관되게 정확할 때만
A/B 비교 후 사용합니다.
