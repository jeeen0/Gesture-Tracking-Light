# ============ PCA9685 ============
PCA9685_ADDRESS = 0x40       # PCA9685 기본 주소
PCA9685_FREQ = 50             # SG90 아날로그 서보 호환 (50Hz 필수)

SERVO_PAN_CH = 0             # 좌우 회전 (CH0)
SERVO_TILT_CH = 1            # 상하 회전 (CH1)

# ============ 서보 가동 범위 (PWM 절대각) ============
# 정면 CENTER=135 기준: PWM 큰 값=위쪽, PWM 작은 값=아래쪽
PAN_MIN_DEG = 0
PAN_MAX_DEG = 180
TILT_MIN_DEG = 30            # 아래쪽 한계 (PWM 30 = 정면에서 아래 105°)
TILT_MAX_DEG = 180           # 위쪽 한계 확장 (PWM 180 = 정면에서 위 45°)

# 부드러운 이동을 위한 속도 제한 (deg/step)
SERVO_MAX_STEP_DEG = 0.5
SERVO_STEP_DELAY_S = 0.003

# ============ GPIO 핀 (BCM 번호) ============
PIN_LED_SPOT_PWM = 12        # 10W LED MOSFET 게이트 → GPIO 12 (Pin 32)
PIN_LED_MOOD_PWM = 13        # 3W LED HAM3005 DIM → GPIO 13 (Pin 33)

LED_PWM_FREQ = 5000          # 5kHz - 낮은 duty에서도 안정적 dimming

# ============ 로깅 ============
LOG_LEVEL = "INFO"               # DEBUG / INFO / WARNING / ERROR
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
