"""
Gaegle 메인 엔트리포인트

실행:
    python -m src.main
    python src/main.py

내부적으로 src/vision/raspberry_pi_runtime.py 의 main()을 호출.
서보(PCA9685) + LED(lgpio PWM) 통합 제스처 인식 런타임이 그쪽에 있음.
"""
import sys
from pathlib import Path

# `python src/main.py`로 직접 실행해도 동작하도록 프로젝트 루트를 sys.path에 추가.
# raspberry_pi_runtime.py 및 내부 모듈들이 `from pi_runtime_config import ...`
# 식의 bare import를 쓰므로, src/vision/ 도 앞에 추가한다.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VISION_DIR = Path(__file__).resolve().parent / "vision"
for _p in (_PROJECT_ROOT, _VISION_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.vision.raspberry_pi_runtime import main  # noqa: E402


if __name__ == "__main__":
    main()
