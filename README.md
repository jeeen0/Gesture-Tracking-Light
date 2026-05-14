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
3. Phase별로 정리된 11개 Python 파일이 한 번에 들어옴

(코드는 페이지에 내장되어 있어서 별도 파일 다운로드 불필요)

---

## 💡 사이트 사용 팁

### 데이터는 각자 브라우저에 저장됨

- 카탈로그 사진을 추가하거나 코드를 수정해도 **본인 브라우저에만** 저장됨
- 팀원 모두가 같은 데이터를 보려면 → 누군가 **백업 JSON**을 만들어서 다른 사람도 **복원**해야 함

### 백업 / 공유 방법

**부품 카탈로그**:
1. 카탈로그 페이지에서 **"백업"** 버튼 → JSON 파일 다운로드
2. 팀원에게 카톡/메일로 공유
3. 받은 팀원은 **"복원"** 버튼으로 불러오기

**코드 에디터**:
1. 코드 에디터 페이지에서 **"백업"** 버튼 → JSON 파일
2. 또는 **"전체 다운로드"** 버튼 → 11개 `.py` 파일이 한 번에 다운로드됨
3. 라즈베리파이에서 쓰려면 `git pull`로 받는 게 더 편함

### 브라우저 데이터 삭제 시 주의

브라우저 캐시/사이트 데이터 삭제하면 카탈로그·코드가 **다 날아가요**. 작업한 거 있으면 미리 백업 받으세요.

---

## 🚀 라즈베리파이 실행 방법

### 1. 레포 클론

```bash
git clone https://github.com/jeeen0/Gesture-Tracking-Light.git
cd Gesture-Tracking-Light
```

### 2. 환경 준비 (한 번만)

```bash
# I2C 활성화
sudo raspi-config  # → Interface Options → I2C → Enable

# 의존성 설치
pip install -r src/requirements.txt
```

### 3. 단계별 검증

```bash
# 레포 루트에서 실행 (src/ 안으로 들어가지 마세요!)

# Phase 2: 모터 단독 테스트
sudo python3 -m src.tests.servo_test

# Phase 3: LED 단독 테스트
sudo python3 -m src.tests.led_test

# Phase 4: 전체 통합 실행
sudo python3 -m src.main
```

> ⚠️ `python -m` 형식으로 실행해야 패키지 import가 정상 동작함. `python3 src/main.py`처럼 직접 실행하면 import 에러남

### 4. 최신 코드 받기

```bash
cd <repo-name>
git pull
```

---

## 📂 디렉토리 구조

```
gaegle-capstone/
├── README.md                   # 이 파일
├── .gitignore
├── robots.txt
│
├── 🌐 웹사이트 (GitHub Pages 자동 서빙)
│   ├── index.html              # 메인 랜딩
│   ├── build_plan.html         # 회로 빌드 가이드
│   ├── circuit.html            # 회로 결선도
│   ├── code_editor.html        # 코드 에디터
│   └── parts_catalog.html      # 부품 카탈로그
│
├── 📁 backup/                  # 초기 데이터 백업 파일
│   ├── parts_catalog_2026-05-13.json   # 카탈로그 사진 포함
│   ├── parts_catalog_backup.json       # 카탈로그 메타데이터만
│   └── DESIGN.md                       # 디자인 가이드
│
└── 📁 src/                     # 라즈베리파이 실행 코드
    ├── __init__.py
    ├── config.py               # 핀 번호, 상수 (모두가 참조)
    ├── main.py                 # 엔트리포인트
    ├── requirements.txt
    │
    ├── 📁 core/                # 핵심 하드웨어 제어
    │   ├── servo_controller.py # PCA9685 + Pan/Tilt 서보
    │   ├── led_controller.py   # 10W/3W LED PWM 디밍
    │   └── pir_sensor.py       # PIR 인터럽트 핸들러
    │
    ├── 📁 vision/              # 비전 처리
    │   ├── vector_calc.py      # 손 랜드마크 → 각도 변환
    │   └── vision_pipeline.py  # MediaPipe + 제스처 분류
    │
    ├── 📁 state/               # 상태 머신
    │   └── state_machine.py    # Standby/Gesture/Locked 전이
    │
    └── 📁 tests/               # 하드웨어 검증 스크립트
        ├── servo_test.py       # Phase 2 모터 단독 검증
        └── led_test.py         # Phase 3 LED 단독 검증
```

---

<p align="center">
  <i>Made with 🤍 by Gaegle</i>
</p>
