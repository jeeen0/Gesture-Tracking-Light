"""
상태 머신 - Standby ↔ Gesture Mode ↔ Locked 전환 관리

상태 다이어그램 (중간보고서):
    [STANDBY]  --Wave-->         [GESTURE_MODE]
    [STANDBY]  <--Fist--         [GESTURE_MODE]
    [GESTURE_MODE] --3초 정지--> [LOCKED]
    [LOCKED]   --Wave-->         [GESTURE_MODE]

각 상태별 동작:
- STANDBY: 카메라 저전력, MediaPipe 미가동, 조명 OFF
- GESTURE_MODE: 카메라 풀 가동, 모든 제스처 인식
- LOCKED: 마지막 위치 유지, 제스처 인식 OFF, 조명 ON 유지
"""
import logging
import time
from typing import Callable, Optional

from src.config import State, HAND_HOLD_SEC, STANDBY_TIMEOUT_SEC

log = logging.getLogger(__name__)


class StateMachine:
    """시스템 상태 관리 + 자동 타임아웃 처리."""
    
    def __init__(self,
                 on_state_change: Optional[Callable[[str, str], None]] = None):
        """on_state_change(old_state, new_state) 콜백을 받음."""
        self._state = State.STANDBY
        self._on_state_change = on_state_change
        
        # 손 정지 감지를 위한 타임스탬프
        self._last_motion_time = time.time()
        self._hand_hold_start: Optional[float] = None
    
    @property
    def state(self) -> str:
        return self._state
    
    def _transition(self, new_state: str, reason: str = ""):
        """상태 전이 + 콜백 호출."""
        if new_state == self._state:
            return
        
        old = self._state
        self._state = new_state
        log.info(f"State: {old} → {new_state}  ({reason})")
        
        # 타이머 리셋
        self._last_motion_time = time.time()
        self._hand_hold_start = None
        
        if self._on_state_change:
            try:
                self._on_state_change(old, new_state)
            except Exception as e:
                log.error(f"State change callback error: {e}")
    
    # ============ 이벤트 핸들러 ============
    
    def on_wave(self):
        """Wave 제스처 감지 시 호출."""
        if self._state == State.STANDBY:
            self._transition(State.GESTURE_MODE, "Wave detected (wake)")
        elif self._state == State.LOCKED:
            self._transition(State.GESTURE_MODE, "Wave detected (unlock)")
        else:
            log.debug("Wave ignored (already in GESTURE_MODE)")
    
    def on_fist(self):
        """Fist 제스처 감지 시 호출 - 조명 OFF + Standby 복귀."""
        if self._state == State.GESTURE_MODE:
            self._transition(State.STANDBY, "Fist detected (shutdown)")
    
    def on_pir_motion(self):
        """PIR이 사람 감지. Standby 상태에서만 의미 있음."""
        if self._state == State.STANDBY:
            log.debug("PIR motion in STANDBY - camera wakeup ready")
            # 카메라 풀 가동 트리거는 vision pipeline에서 처리
        self._last_motion_time = time.time()
    
    def on_hand_present(self, is_holding_still: bool):
        """매 프레임 손 존재 여부 보고.
        
        is_holding_still: 손이 일정 위치에 정지해있는지.
                          True가 HAND_HOLD_SEC 이상 지속되면 → LOCKED 전환.
        """
        if self._state != State.GESTURE_MODE:
            self._hand_hold_start = None
            return
        
        now = time.time()
        self._last_motion_time = now
        
        if is_holding_still:
            if self._hand_hold_start is None:
                self._hand_hold_start = now
            elif now - self._hand_hold_start >= HAND_HOLD_SEC:
                self._transition(State.LOCKED, f"Hand held for {HAND_HOLD_SEC}s")
        else:
            self._hand_hold_start = None
    
    def tick(self):
        """매 메인 루프마다 호출. 타임아웃 체크.
        
        GESTURE_MODE에서 일정 시간 입력 없으면 자동 Standby.
        """
        if self._state == State.GESTURE_MODE:
            if time.time() - self._last_motion_time > STANDBY_TIMEOUT_SEC:
                self._transition(State.STANDBY, f"No input for {STANDBY_TIMEOUT_SEC}s")
    
    def force_state(self, new_state: str, reason: str = "manual"):
        """디버깅용 강제 상태 전환."""
        self._transition(new_state, reason)
