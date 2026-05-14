"""
메인 엔트리포인트 - 전체 시스템 통합

실행:
    sudo python3 -m src.main

흐름:
1. 모든 컴포넌트 초기화 (servo, led, pir, vision, state)
2. PIR 인터럽트 등록
3. 메인 루프: 상태에 따라 다른 동작
   - STANDBY: PIR 대기, 카메라 저전력 motion check
   - GESTURE_MODE: 카메라 풀가동, 제스처 인식 → 제어
   - LOCKED: 마지막 위치 유지, Wave/Fist만 감지
"""
import logging
import signal
import sys
import time
from typing import Optional

try:
    import cv2
except ImportError:
    cv2 = None

from src.config import (
    LOG_FORMAT, LOG_LEVEL, State,
    CAMERA_WIDTH_FULL, CAMERA_HEIGHT_FULL, CAMERA_FPS_FULL,
)
from src.core.servo_controller import ServoController
from src.core.led_controller import LEDController
from src.core.pir_sensor import PIRSensor
from src.state.state_machine import StateMachine
from src.vision.vision_pipeline import VisionPipeline

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
log = logging.getLogger("main")


class GaegleApp:
    """전체 애플리케이션 라이프사이클."""
    
    def __init__(self):
        log.info("Initializing Gaegle system...")
        self.servo = ServoController()
        self.led = LEDController()
        self.vision = VisionPipeline()
        self.state = StateMachine(on_state_change=self._on_state_change)
        self.pir = PIRSensor(on_motion=self._on_pir_motion)
        
        # 카메라
        self.cap: Optional["cv2.VideoCapture"] = None
        self._init_camera()
        
        # 현재 모드: spot ↔ mood
        self._light_mode = "spot"
        
        self._running = True
        # Ctrl+C 핸들러
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
    
    def _init_camera(self):
        if cv2 is None:
            log.warning("OpenCV not available - camera disabled")
            return
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH_FULL)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT_FULL)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS_FULL)
        log.info(f"Camera: {CAMERA_WIDTH_FULL}x{CAMERA_HEIGHT_FULL} @ {CAMERA_FPS_FULL}fps")
    
    # ============ 이벤트 핸들러 ============
    
    def _on_signal(self, signum, frame):
        log.warning(f"Signal {signum} received")
        self._running = False
    
    def _on_pir_motion(self):
        """PIR 감지 콜백 (인터럽트). STANDBY일 때만 의미 있음."""
        self.state.on_pir_motion()
    
    def _on_state_change(self, old: str, new: str):
        """상태 전환 시 동작."""
        if new == State.STANDBY:
            self.led.all_off()
            self.servo.home()
        elif new == State.GESTURE_MODE:
            # 진입 시 약한 무드등으로 시작
            self.led.mood_on(30)
        elif new == State.LOCKED:
            # 위치/조명 그대로 유지, vision만 안 함
            log.info("Position locked")
    
    # ============ 액션 실행 ============
    
    def _apply_gesture(self, result):
        """제스처 → 실제 제어 명령."""
        if result.gesture == "WAVE":
            self.state.on_wave()
        elif result.gesture == "FIST":
            self.state.on_fist()
        elif result.gesture == "POINT" and self.state.state == State.GESTURE_MODE:
            if result.pan_deg is not None and result.tilt_deg is not None:
                self.servo.move_to(result.pan_deg, result.tilt_deg)
                # 포인트한 곳에 스팟 조명
                if self._light_mode == "spot":
                    self.led.spot_on(70)
        elif result.gesture == "PINCH" and self.state.state == State.GESTURE_MODE:
            if result.brightness is not None:
                if self._light_mode == "spot":
                    self.led.spot_set(result.brightness)
                else:
                    self.led.mood_set(result.brightness)
        elif result.gesture == "SHAKA" and self.state.state == State.GESTURE_MODE:
            # 모드 토글
            self._light_mode = "mood" if self._light_mode == "spot" else "spot"
            log.info(f"Light mode → {self._light_mode}")
            if self._light_mode == "spot":
                self.led.mood_off()
                self.led.spot_on(70)
            else:
                self.led.spot_off()
                self.led.mood_on(40)
        
        # 손 정지 → Lock 전환 체크
        if result.gesture in ("POINT", "PALM"):
            self.state.on_hand_present(result.confidence > 0.9)
        else:
            self.state.on_hand_present(False)
    
    # ============ 메인 루프 ============
    
    def run(self):
        log.info("System ready. Starting main loop.")
        try:
            while self._running:
                self.state.tick()
                
                # 카메라 프레임 처리 (STANDBY가 아닐 때만)
                if self.state.state != State.STANDBY and self.cap is not None:
                    ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.01)
                        continue
                    
                    # LOCKED 상태에서는 wake용 Wave/Fist만 감지
                    if self.state.state == State.LOCKED:
                        result = self.vision.process_frame(frame)
                        if result.gesture in ("WAVE", "FIST"):
                            self._apply_gesture(result)
                    else:
                        # GESTURE_MODE: 모든 제스처 처리
                        result = self.vision.process_frame(frame)
                        self._apply_gesture(result)
                else:
                    # STANDBY: 카메라 안 돌리고 PIR만 기다림
                    time.sleep(0.1)
        
        except Exception as e:
            log.error(f"Main loop error: {e}", exc_info=True)
        finally:
            self.shutdown()
    
    def shutdown(self):
        log.info("Shutting down...")
        if self.cap is not None:
            self.cap.release()
        self.led.shutdown()
        self.servo.shutdown()
        self.pir.shutdown()
        self.vision.shutdown()
        log.info("Bye!")


def main():
    app = GaegleApp()
    app.run()


if __name__ == "__main__":
    main()
