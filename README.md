# Vision-AI Based Gesture Tracking Lighting

> 손가락 포인팅으로 LED 조명의 방향을 추적하고, 손동작 제스처로 밝기·전원을 제어하는 비전 AI 기반 지향성 조명 로봇

**숭실대학교 AI소프트웨어학부 · 2026-1 캡스톤디자인 · Gaegle 팀**

---

## 🌐 프로젝트 사이트

📍 **https://jeeen0.github.io/Gesture-Tracking-Light/**

---

## 페이지 구성

| 페이지 | 용도 |
|---|---|
| 🏠 **홈** (`index.html`) | 메인 랜딩. 다른 페이지로 이동 |
| 🛠 **회로 빌드** (`build_plan.html`) | 단계별 회로 빌드 매뉴얼 (Phase 1~4) |
| ⚡ **회로 결선도** (`circuit.html`) | GPIO 핀까지 표기된 상세 결선도 |
| 💻 **코드 에디터** (`code_editor.html`) | Python 모듈 작성/편집 |
| 📦 **부품 카탈로그** (`parts_catalog.html`) | 36개 부품 사진 카탈로그 |

### ✅ 초기 데이터 불러오는 방법

#### 📦 부품 카탈로그 데이터 (사진 포함)

1. 사이트 상단 메뉴에서 **부품 카탈로그** 클릭
2. 우측 상단 **"복원"** 버튼 클릭
3. 레포의 **`backup/parts_catalog_2026-05-13.json`** 파일 선택
4. 36개 부품 + 사진이 자동으로 불러와짐

> 💾 백업 파일 다운로드: 레포에서 `backup/parts_catalog_2026-05-13.json` 파일 우클릭 → "다른 이름으로 링크 저장" 또는 GitHub에서 파일 클릭 → "Download raw file" 버튼

#### 💻 코드 에디터 초기 코드

1. 사이트 상단 메뉴에서 **코드 에디터** 클릭
2. 좌측 상단 **"초기 코드 불러오기"** 버튼 클릭
3. Phase별로 정리된 19개 Python 파일이 한 번에 들어옴

(코드는 페이지에 내장되어 있어서 별도 파일 다운로드 불필요)

---

## 💡 사이트 사용 팁

### 데이터는 각자 브라우저에 저장됨

- 카탈로그 사진을 추가하거나 코드를 수정해도 **본인 브라우저에만** 저장됨
- 팀원 모두가 같은 데이터를 보려면 → 누군가 **백업 JSON**을 만들어서 다른 사람도 **복원**해야 함

### 백업 / 공유 방법

**부품 카탈로그**:
1. 카탈로그 페이지에서 **"백업"** 버튼 → JSON 파일 다운로드
2. Github에 JSON 파일 Push
3. 다운받은 팀원은 **"복원"** 버튼으로 불러오기

**코드 에디터**:
1. 코드 에디터 페이지에서 **"백업"** 버튼 → JSON 파일
2. 또는 **"전체 다운로드"** 버튼 → 19개 `.py` 파일이 한 번에 다운로드됨
3. 라즈베리파이에서 쓰려면 `git pull`로 받는 게 더 편함

### 브라우저 데이터 삭제 시 주의

브라우저 캐시/사이트 데이터 삭제하면 카탈로그·코드가 날아가므로 작업한 거 있으면 미리 백업 받을 것.

---

## 🚀 라즈베리파이 실행 방법

### 1. 레포 클론 (한 번만)

```bash
git clone https://github.com/jeeen0/Gesture-Tracking-Light.git
cd Gesture-Tracking-Light
```

### 2. 환경 준비 (한 번만)

#### 2-1. 시스템 인터페이스 활성화 + 권한

```bash
# I2C / 카메라 활성화
sudo raspi-config   # → Interface Options → I2C: Enable, Camera: Enable

# sudo 없이 GPIO/I2C/카메라 쓰려면 그룹 등록 (권장)
sudo usermod -aG gpio,i2c,video,spi gaegle
sudo reboot         # 그룹 적용 위해 1회 재부팅
```

> 그룹 등록 안 하면 매번 `sudo` 필요.

#### 2-2. Python 가상환경 + 의존성

```bash
cd ~/Gesture-Tracking-Light

# venv 생성
python -m venv .gaegle2
source .gaegle2/bin/activate

# 라파 기본 의존성
pip install -r src/requirements.txt

# MediaPipe, OpenCV, MiDaS 깊이추정용 timm 은 별도 설치 (라파 5 / Python 3.11)
pip install mediapipe==0.10.9 opencv-contrib-python numpy
pip install timm   # MiDaS backbone (POINT 깊이추정에 사용)
```

---

### 3. 단계별 검증

```bash
cd ~/Gesture-Tracking-Light
source .gaegle2/bin/activate

# Phase 2: 모터 단독 테스트
python -m src.tests.servo_test

# Phase 3: LED 단독 테스트
python -m src.tests.led_test

# Phase 4: 전체 통합 실행
python -m src.main
```

> ⚠️ 반드시 **레포 루트**에서 `python -m src.main` 또는 `python src/main.py` 형태로 실행. 다른 경로/방식은 패키지 import가 깨짐.

> **그룹 등록 안 했으면** venv 절대경로 + sudo:
> ```bash
> sudo .gaegle2/bin/python -m src.main
> ```

---

### 4. 자주 쓰는 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `PI_CAMERA_BACKEND` | `rpicam-vid` | `opencv` / `picamera2` / `rpicam-vid` |
| `PI_CAM_INDEX` | `0` | OpenCV 백엔드일 때 카메라 인덱스 |
| `PI_FRAME_W` / `PI_FRAME_H` | 640 / 360 | 처리 해상도 |
| `PI_TARGET_FPS` | 20 | 카메라 목표 fps |
| `PI_MJPEG_QUALITY` | 90 | `rpicam-vid` MJPEG 압축 품질 (1~100) |
| `PI_SHOW_PREVIEW` | 1 | 프리뷰 창 표시 (성능 부담 있음) |
| `PI_DEBUG` | 0 | STATE JSON + 후보 로그 |
| `PI_ENABLE_FISHEYE` | 1 | fisheye 왜곡 보정 |
| `PI_SERVO_PAN_SIGN` | +1 | 카메라↔서보 pan 방향 일치 시 +1, 반대면 -1 |
| `PI_SERVO_TILT_SIGN` | -1 | tilt 부호. 짐벌 조립 방향에 따라 조정 |

디버그 실행 예:
```bash
PI_DEBUG=1 PI_SHOW_PREVIEW=1 python -m src.main
```

---

### 5. 최신 코드 받기

```bash
cd ~/Gesture-Tracking-Light
git pull
```

> Raspberry PI에서 코드 수정한 게 있어 `pull`이 충돌나면, `git stash` 또는 `git reset --hard origin/main` 으로 정리 후 다시.

---

## 📂 디렉토리 구조

```
Gesture-Tracking-Light/
├── README.md
├── .gitignore
├── robots.txt
│
├── 🌐 웹사이트 (GitHub Pages 자동 서빙)
│   ├── index.html                  # 메인 랜딩
│   ├── build_plan.html             # 회로 빌드 가이드
│   ├── circuit.html                # 회로 결선도
│   ├── code_editor.html            # 코드 에디터
│   └── parts_catalog.html          # 부품 카탈로그
│
├── 📁 backup/                      # 초기 데이터 백업 파일
│   ├── parts_catalog_2026-05-13.json   # 카탈로그 사진 포함
│   ├── parts_catalog_backup.json       # 카탈로그 메타데이터만
│   └── DESIGN.md                       # 디자인 가이드
│
└── 📁 src/                         # 라즈베리파이 실행 코드
    ├── __init__.py
    ├── config.py                   # 핀 번호, 서보/LED 캘리브 상수
    ├── main.py                     # 엔트리포인트 (raspberry_pi_runtime 호출)
    ├── requirements.txt
    │
    ├── 📁 core/                    # 핵심 하드웨어 제어
    │   ├── __init__.py
    │   ├── servo_controller.py     # PCA9685 + Pan/Tilt 서보
    │   └── led_controller.py       # 10W 스팟 + 3W 무드 LED PWM 디밍
    │
    ├── 📁 vision/                  # 비전 처리 (제스처/포인팅 런타임)
    │   ├── __init__.py
    │   ├── raspberry_pi_runtime.py # 메인 루프 (카메라 → MediaPipe → 제스처/포인팅)
    │   ├── pi_runtime_controller.py# 상태 컨트롤러 (서보/LED 통합 dispatch)
    │   ├── pi_runtime_config.py    # 환경변수 기반 런타임 설정
    │   ├── pi_runtime_events.py    # JSON 이벤트/STATE 출력
    │   ├── pi_runtime_motion.py    # 모션/wave 후보 감지 유틸
    │   ├── gestures.py             # 손 랜드마크 → 제스처 분류
    │   ├── pointing_target.py      # 손가락 ray + MiDaS depth로 가리키는 지점 추정
    │   └── fisheye_undistort.py    # 캘리브된 fisheye 왜곡 보정
    │
    ├── 📁 calibration/             # 카메라 캘리브레이션 데이터
    │   └── fisheye_undistort_map.npz   # K, dist, remap 맵 (rms ≈ 0.81)
    │
    ├── 📁 state/                   # 상태 머신 (구버전 main.py 잔존)
    │   ├── __init__.py
    │   └── state_machine.py        # Standby/Gesture/Locked 전이
    │
    └── 📁 tests/                   # 하드웨어 검증 스크립트
        ├── __init__.py
        ├── servo_test.py           # 모터 단독 검증
        └── led_test.py             # LED 단독 검증
```

---

<p align="center">
  <i>Made with 🤍 by Gaegle</i>
</p>
