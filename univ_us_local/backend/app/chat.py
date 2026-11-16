"""F9 자연어 대화 — ../F9_Chat_agent 의 API(chat.api)를 이 서버에 붙인다.

LLM 호출 · 도구 실행 · 미리보기 · 확인 · 대화 기록은 전부 F9_Chat_agent 에 있다. 여기서는
  - 라우터(/api/chat*)를 include 하고,
  - 도구가 읽을 값을 함수로 넘겨준다(chat 패키지는 이 백엔드를 import 하지 않는다):
      합친 캘린더(CALENDAR_SOURCES + C1 내 일정) · C1 모듈(확인된 쓰기) · F2 졸업요건 · F3 이번 학기 과목 · F7 급한 순 · F5 시험 · F6 수집 시작,
  - /api/status 에 chat 칸(available · llmConfigured · model)을 싣는다.
각 읽기는 실패하면 빈 값이다 — 한 기능이 깨져도 대화는 '없음'으로 답한다.
F9 쪽 모듈은 표준 라이브러리 + fastapi 만 쓰므로 이 백엔드의 .venv 에 더 설치할 것이 없다.

F9 폴더가 없거나 불러오지 못해도 나머지 화면은 그대로 동작해야 한다 → available=False 로 알린다.
"""
from __future__ import annotations

import sys
from datetime import datetime
from typing import Any, Callable, Optional, Sequence

from . import config as C
from . import attendance, calendar_events, eclass_data, exams, graduation, priority

_error: Optional[str] = None

Source = Callable[[Optional[str], Optional[str]], list[dict]]


def _load():
    global _error
    if not (C.F9_AGENT_DIR / "chat" / "api.py").exists():
        _error = f"F9_Chat_agent 가 없습니다: {C.F9_AGENT_DIR}"
        return None
    if str(C.F9_AGENT_DIR) not in sys.path:
        sys.path.insert(0, str(C.F9_AGENT_DIR))
    try:
        from chat import api as f9_api                     # type: ignore[import-not-found]
        return f9_api
    except Exception as e:                      # noqa: BLE001 — 어떤 이유로든 F9 가 못 뜨면 나머지는 살린다
        _error = f"F9 모듈을 불러오지 못했습니다: {type(e).__name__}: {e}"
        return None


f9 = _load()


def _safe(fn: Callable[..., Any], default: Any) -> Callable[..., Any]:
    def run(*args: Any) -> Any:
        try:
            return fn(*args)
        except Exception:                       # noqa: BLE001 — 읽기 실패는 빈 값
            return default
    return run


def _attendance_courses() -> list[dict]:
    """F3 이번 학기 과목(집계 포함) — 출결 화면과 같은 계산. 경고 판정·과목 동기화(부수효과)는 하지 않는다(F3 today() 와 같은 길)."""
    f3 = attendance.f3
    if not f3:
        return []
    now = datetime.now()
    with f3.store.connect() as con:
        cal = f3.AC.load()
        sid = f3.service.current_semester(con, cal, now.date())
        _, courses = f3.service.build_semester(con, sid, now, cal)
    return courses


def _exams_overview() -> dict:
    """F5 시험 목록 {exams: 다가오는 것, past: 지난 것} — /exams 화면과 같은 계산(범위 계산은 뺀다)."""
    f5 = exams.f5
    if not f5:
        return {"exams": [], "past": []}
    with f5.store.connect() as con:
        return f5.service.overview(con, eclass_data.load_courses, None, with_scope=False)


def _deps(sources: Sequence[Source]):
    return f9.Deps(
        load_events=_safe(lambda start, end: calendar_events.collect(sources, start, end), []),
        c1=calendar_events.c1,
        graduation=_safe(graduation.status, {"available": False}),
        attendance=_safe(_attendance_courses, []),
        assignments=_safe(priority.ranked, []),
        exams=_safe(_exams_overview, {"exams": [], "past": []}),
        start_sync=eclass_data.start_sync,
        sync_state=_safe(eclass_data.sync_state, {}),
    )


def router(sources: Sequence[Source] = ()):
    """sources — main.py 의 CALENDAR_SOURCES (/api/events 와 같은 합치기 — list_events · 겹침 검사가 읽는다)."""
    if not f9:
        return None
    return f9.build_router(_deps(sources))


def status() -> dict:
    if not f9:
        return {"available": False, "llmConfigured": False, "error": _error}
    try:
        return f9.status_summary()
    except Exception as e:                      # noqa: BLE001
        return {"available": False, "llmConfigured": False, "error": f"{type(e).__name__}: {e}"}
