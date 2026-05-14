"""
PIR 센서 - 재실 감지용 모션 센서 (HC-SR501)
회로: PIR OUT → GPIO 23 → 인터럽트

OUT 핀이 LOW → HIGH 변할 때 사람 감지.
HIGH 유지 시간은 PIR 모듈의 TX 가변저항으로 조절 (기본 3초~5분).
"""
import logging
from typing import Callable, Optional

try:
    import lgpio
except ImportError:
    lgpio = None

from src.config import PIN_PIR

log = logging.getLogger(__name__)


class PIRSensor:
    """PIR 센서 인터럽트 핸들러.
    
    사용 예:
        def on_motion():
            print("Motion detected!")
        
        pir = PIRSensor(on_motion=on_motion)
        # 콜백이 별도 스레드에서 실행됨
    """
    
    def __init__(self, on_motion: Optional[Callable[[], None]] = None,
                 on_no_motion: Optional[Callable[[], None]] = None):
        self.on_motion = on_motion
        self.on_no_motion = on_no_motion
        self._cb = None
        
        if lgpio is None:
            log.warning("lgpio not available - PIR running in DRY mode")
            self.h = None
            return
        
        self.h = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_input(self.h, PIN_PIR)
        
        # 양 edge 모두 콜백 (HIGH → 감지, LOW → 종료)
        self._cb = lgpio.callback(self.h, PIN_PIR, lgpio.BOTH_EDGES, self._handle_edge)
        log.info(f"PIR sensor ready on GPIO{PIN_PIR}")
    
    def _handle_edge(self, chip, gpio, level, tick):
        """엣지 검출 콜백. level: 1=HIGH(감지), 0=LOW(종료)."""
        if level == 1:
            log.debug("PIR: motion detected")
            if self.on_motion:
                try:
                    self.on_motion()
                except Exception as e:
                    log.error(f"on_motion callback error: {e}")
        elif level == 0:
            log.debug("PIR: motion ended")
            if self.on_no_motion:
                try:
                    self.on_no_motion()
                except Exception as e:
                    log.error(f"on_no_motion callback error: {e}")
    
    def read(self) -> bool:
        """현재 PIR 상태 (True=감지 중)."""
        if self.h is None:
            return False
        return bool(lgpio.gpio_read(self.h, PIN_PIR))
    
    def shutdown(self):
        if self._cb is not None:
            try:
                self._cb.cancel()
            except Exception:
                pass
        if self.h is not None:
            try:
                lgpio.gpiochip_close(self.h)
            except Exception as e:
                log.error(f"Close error: {e}")
        log.info("PIR sensor shutdown")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown()
