"""
Gaegle 캡스톤 · 공통 설정 파일
회로 결선도(circuit.html)와 일치해야 함. 핀 변경 시 여기만 수정.
"""

# ============ I2C ============
I2C_BUS = 1                  # Raspberry Pi 5 기본 I2C 버스
PCA9685_ADDRESS = 0x40       # PCA9685 기본 주소
PCA9685_FREQ = 333            # 50 → 333Hz. 주기 3ms로 단축

# ============ PCA9685 채널 ============
SERVO_PAN_CH = 0             # MG946R - 수평 회전 (하단)
SERVO_TILT_CH = 1            # MG90S  - 수직 회전 (상단)

# ============ 서보 캘리브레이션 ============
# PCA9685는 12-bit (0~4095) PWM
# 50Hz 기준 1ms = 약 205, 2ms = 약 410
SERVO_MIN_PULSE = 150        # 0도 위치 (약 0.73ms)
SERVO_MAX_PULSE = 600        # 180도 위치 (약 2.93ms)

# Pan/Tilt 안전 각도 제한 (기구적 한계)
PAN_MIN_DEG = 0
PAN_MAX_DEG = 180
TILT_MIN_DEG = 30            # 너무 위로 꺾이면 LED가 천장 향함
TILT_MAX_DEG = 150           # 너무 아래로 꺾이면 짐벌 부딪힘

# 부드러운 이동을 위한 속도 제한 (deg/step)
SERVO_MAX_STEP_DEG = 0.5       # 0.5° 단위로 쪼갬 (기존이 2~3°면 확 부드러워짐)
SERVO_STEP_DELAY_S = 0.003    # 15ms = 약 66Hz. PCA9685 50Hz면 0.02도 OK

# ============ GPIO 핀 (BCM 번호) ============
PIN_LED_SPOT_PWM = 12        # 10W LED MOSFET 게이트 → GPIO 12 (Pin 32)
PIN_LED_MOOD_PWM = 13        # 3W LED HAM3005 DIM → GPIO 13 (Pin 33)

# LED PWM 주파수
LED_PWM_FREQ = 1000          # 1kHz - 깜박임 안 보이고 LED 드라이버 호환 좋음

# ============ 카메라 ============
CAMERA_WIDTH_FULL = 640
CAMERA_HEIGHT_FULL = 480
CAMERA_FPS_FULL = 30

# 저전력 모드 (Standby에서 motion 감지용)
CAMERA_WIDTH_LOW = 160
CAMERA_HEIGHT_LOW = 120
CAMERA_FPS_LOW = 10

# 사용자 손 실제 길이 (mediapipe 0번 손목 ~ 5번 검지 MCP)
# 캘리브레이션 시 사용자 입력
HAND_REAL_LENGTH_M = 0.085   # 평균 8.5cm 기본값

# ============ 상태 머신 ============
class State:
    """시스템 상태 enum"""
    STANDBY = "STANDBY"          # PIR 대기, 카메라 저전력
    GESTURE_MODE = "GESTURE"     # MediaPipe 풀 가동
    LOCKED = "LOCKED"            # 마지막 위치 유지, 인식 OFF

# 상태 전환 시간
HAND_HOLD_SEC = 3.0              # 3초간 손 정지 → Locked
STANDBY_TIMEOUT_SEC = 60.0       # 1분간 입력 없으면 → Standby

# Wave 감지 (좌우 반복 움직임)
WAVE_FRAMES = 12                 # 약 0.4초 분량의 프레임으로 wave 판정
WAVE_MIN_AMPLITUDE = 80          # 픽셀 단위, 좌우 진폭

# ============ 제스처 임계값 ============
# Pinch (엄지+검지 거리 비율, 손 크기 기준)
PINCH_ENTRY_RATIO = 0.18         # 진입 조건: 거리/손크기 < 0.18
PINCH_EXIT_RATIO = 0.30          # 유지 종료: 거리/손크기 > 0.30
PINCH_BRIGHT_MIN_RATIO = 0.05    # 0% 밝기
PINCH_BRIGHT_MAX_RATIO = 0.50    # 100% 밝기

# ============ 로깅 ============
LOG_LEVEL = "INFO"               # DEBUG / INFO / WARNING / ERROR
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
