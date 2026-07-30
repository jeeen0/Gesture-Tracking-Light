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

화면을 3×3으로 나눈 각 구역의 중앙, 총 9개 지점을 보정합니다.
첫 보정점은 전체 화면 중앙입니다. 각 지점에서
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

## 3. 포인팅 A/B 테스트

테스트 전 다음 조건을 고정합니다.

- 같은 카메라 위치와 조명 위치를 유지합니다.
- 같은 `servo_pointing_calibration.json`을 사용합니다.
- 벽의 중앙·좌·우·상·하에 목표점 5개를 표시합니다.
- 각 설정에서 목표점마다 3회씩 포인팅합니다.
- 화면 포인터 오차와 실제 조명 오차를 따로 기록합니다.

시작 로그에서 실제 적용값을 먼저 확인합니다.

```text
[Pointing] ray config hit_threshold=0.12 start_margin=35px depth_update_interval=4
```

### 테스트 A: 현재 기본 조건

```bash
PI_ROI_INPUT_SIZE=320 \
PI_LATEST_FRAME_CAPTURE=1 \
PI_DEPTH_ASYNC=1 \
PI_DEPTH_ASYNC_FPS=4 \
PI_DEPTH_RESULT_MAX_AGE=0.75 \
PI_DEPTH_UPDATE_INTERVAL=4 \
PI_DEPTH_HIT_THRESHOLD=0.12 \
PI_POINT_RAY_START_MARGIN_PX=35 \
PI_TORCH_NUM_THREADS=2 \
PI_POINT_INFERENCE_FPS=20 \
PI_POINT_RAY_MODE=mcp_tip \
PI_POINT_ENTRY_GRACE_SECONDS=1.0 \
PI_POINT_ARM_WINDOW=5 \
PI_POINT_ARM_MIN_HITS=3 \
PI_SHOW_PREVIEW=1 \
PI_SHOW_DEPTH=1 \
PI_DEBUG=1 \
python -m src.main
```

### 테스트 B: 초기 충돌 조건

비동기 처리 방식은 유지하고 충돌 허용치와 탐색 시작점만 초기 구현과
같게 만듭니다.

```bash
PI_DEPTH_ASYNC=1 \
PI_DEPTH_HIT_THRESHOLD=0.08 \
PI_POINT_RAY_START_MARGIN_PX=10 \
PI_SHOW_PREVIEW=1 \
PI_SHOW_DEPTH=1 \
PI_DEBUG=1 \
python -m src.main
```

### 테스트 C: 동일 프레임 깊이 비교

비동기 깊이맵과 현재 손 좌표의 시간 차이를 확인하는 진단용 설정입니다.
동기 깊이 추론과 매 추론 프레임 깊이 갱신을 사용하므로 매우 느릴 수
있지만 깊이맵과 손 좌표가 같은 프레임에서 계산됩니다.

```bash
PI_DEPTH_ASYNC=0 \
PI_DEPTH_UPDATE_INTERVAL=1 \
PI_DEPTH_HIT_THRESHOLD=0.08 \
PI_POINT_RAY_START_MARGIN_PX=10 \
PI_SHOW_PREVIEW=1 \
PI_SHOW_DEPTH=1 \
PI_DEBUG=1 \
python -m src.main
```

### 테스트 D: 충돌 허용치 세부 비교

테스트 B가 가장 좋지만 간헐적으로 표면을 놓친다면 다른 조건은 그대로
두고 허용치만 `0.05`, `0.08`, `0.12` 순서로 비교합니다.

```bash
PI_DEPTH_ASYNC=1 \
PI_DEPTH_HIT_THRESHOLD=0.05 \
PI_POINT_RAY_START_MARGIN_PX=10 \
PI_SHOW_PREVIEW=1 \
PI_SHOW_DEPTH=1 \
PI_DEBUG=1 \
python -m src.main
```

허용치가 클수록 표면에 도달하기 전에 충돌로 인정하기 쉬우며, 작을수록
판정은 엄격해지지만 표면을 놓치고 화면 끝 2D fallback으로 갈 가능성이
커집니다.

### 기록 항목

각 시도마다 다음 항목을 기록합니다.

| 항목 | 확인 방법 |
|---|---|
| 설정 | A, B, C 또는 D와 허용치 |
| 목표 위치 | 중앙, 좌, 우, 상, 하 |
| 화면 포인터 오차 | 목표 표시와 화면 원 사이 픽셀 거리 |
| 실제 조명 오차 | 목표 표시와 조명 중심 사이 거리(cm) |
| `hit_method` | `depth_march` 또는 `2d_fallback_after_depth` |
| `std_px` | 잠금 순간 로그의 흔들림 값 |
| 잠금 시간 | POINT 시작부터 LOCKED까지 걸린 시간 |
| FPS | 프리뷰 또는 상태 로그 |

### 결과 해석

- A보다 B가 정확하면 `0.12/35px` 변경이 주된 회귀 원인입니다.
- A와 B는 비슷하게 틀리고 C만 정확하면 비동기 깊이맵과 현재 손 좌표의
  시간 차이가 주원인입니다. C는 진단용으로만 쓰고, 최종 코드는 프레임
  번호가 같은 깊이맵과 랜드마크를 묶는 방식으로 수정해야 합니다.
- 화면 포인터는 정확하지만 실제 조명만 틀리면 깊이 로직이 아니라
  LUT·서보 중심·부호 문제입니다. 시작 로그에서
  `mode=residual_2d`와
  `servo_mapping=angle_gain_offset+residual_2d_lut`를 확인합니다.
- `depth_march`일 때만 틀리고 `2d_fallback_after_depth`일 때 맞으면
  깊이 충돌 판정이 오차를 만드는 것입니다.
- `2d_fallback_after_depth`가 자주 나오면 표면 충돌을 찾지 못한
  것입니다. 허용치를 조금 높이거나 손가락 광선 방향을 점검합니다.
- 허용치를 높일수록 포인터가 검지 가까이에 붙거나 너무 일찍 멈추면
  충돌 오탐이므로 허용치를 낮춥니다.
- 화면 포인터가 프레임마다 크게 튀고 `std_px`가 높으면 깊이보다
  손 랜드마크 또는 ROI 문제일 수 있습니다. 같은 조건에서
  `PI_POINT_USE_ROI=0`을 추가해 전체 화면 검출과 비교합니다.
- 화면 중앙은 맞지만 가장자리 오차가 같은 방향으로 커지면 어안 보정
  intrinsics 또는 화면 좌표 변환을 확인합니다.
- A, B, C 모두 화면 포인터가 같은 위치로 틀리면 검지의
  `mcp_tip` 광선 자체가 원인일 가능성이 큽니다. 이때만
  `PI_POINT_RAY_MODE=finger_axis`를 별도로 A/B 비교합니다.

MiDaS 깊이 맵은 4fps로 갱신하지만, 최신 깊이 맵과 현재 손 좌표를
사용한 포인팅 선·교차점·EMA 계산은 MediaPipe 추론 프레임마다
갱신됩니다.

`finger_axis`는 화면 포인터가 `mcp_tip`보다 일관되게 정확할 때만
A/B 비교 후 사용합니다.
