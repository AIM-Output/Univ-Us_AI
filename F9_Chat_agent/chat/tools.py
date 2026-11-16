"""도구 11개 — 목록(tools.json) 읽기 · 읽기 도구 실행 · 의존(Deps).

tools.json 은 Univ-Us_Train tools/f9_tools.json 을 그대로 복사한 것이다(모델이 그 목록으로 학습했다).
    `x-kind`(read|write|run|navigate)와 `_note` 는 이 저장소용 표시라 LLM 에 보내기 전에 뗀다.

읽기 도구는 **각 기능이 계산한 값을 그대로** 가져온다(F9-R21·R23). 이 패키지는 백엔드를 import 하지 않는다 —
백엔드 어댑터(univ_us_local/backend/app/chat.py)가 함수를 모아 Deps 로 넘겨준다. 못 읽으면 빈 값.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

from . import config as C

TOOLS_FILE = Path(__file__).resolve().parent / "tools.json"

EventsLoader = Callable[[str, str], list[dict]]


@dataclass
class Deps:
    """백엔드가 넘겨주는 함수들 — 없으면 None (그 도구는 빈 값으로 답한다).

    load_events(start, end)  합쳐진 캘린더 (/api/events 모양 — id · title · start · end · allDay · extendedProps.kind …)
    c1                       C1 calendar_core.api 모듈 (service.date_error · service.parse_user_id · store.* · UserEventIn). None 이면 쓰기 불가
    graduation()             F2 /api/status 칸 {available, verdict, remaining, earned, required, headline, hasData …}
    attendance()             F3 이번 학기 과목 목록 [{id, name, short, excluded, summary{remaining, level, levelLabel …}}]
    assignments()            F7 급한 순 목록 [{id, title, courseShort, kindLabel, due, group, groupLabel, stale …}]
    exams()                  F5 시험 목록 {exams: [다가오는 것], past: [지난 것]}
    start_sync()             F6 e클래스 수집 시작 {running, already_running?, error?}
    chat(messages, tools)    LLM 한 번 — 테스트가 바꿔 끼운다 (None 이면 llm.complete)
    now()                    지금 — 테스트가 바꿔 끼운다
    """
    load_events: Optional[EventsLoader] = None
    c1: Any = None
    graduation: Optional[Callable[[], dict]] = None
    attendance: Optional[Callable[[], Any]] = None
    assignments: Optional[Callable[[], list[dict]]] = None
    exams: Optional[Callable[[], dict]] = None
    start_sync: Optional[Callable[[], dict]] = None
    sync_state: Optional[Callable[[], dict]] = None
    chat: Optional[Callable[[list[dict], list[dict]], dict]] = None
    now: Optional[Callable[[], datetime]] = None


@lru_cache(maxsize=1)
def _raw() -> dict:
    return json.loads(TOOLS_FILE.read_text(encoding="utf-8"))


def load() -> tuple[list[dict], dict[str, str]]:
    """(LLM 에 보낼 tools 목록 — x-kind 뗀 것, {이름: x-kind})."""
    tools, kinds = [], {}
    for t in _raw()["tools"]:
        t = copy.deepcopy(t)
        kind = t.pop("x-kind", "read")
        t.pop("_note", None)
        kinds[t["function"]["name"]] = kind
        tools.append(t)
    return tools, kinds


def safe(fn: Optional[Callable[..., Any]], default: Any, *args: Any) -> Any:
    if fn is None:
        return default
    try:
        out = fn(*args)
    except Exception:                                   # noqa: BLE001 — 한 기능이 깨져도 대화는 '없음'으로 답한다
        return default
    return default if out is None else out


# ---------------------------------------------------------------- 읽기 도구 — 값만 가져온다 (문장은 service)

def events(deps: Deps, frm: str, to: str, kinds: Optional[list[str]]) -> list[dict]:
    """list_events — 기간 · 종류로 거르고, 종일 일정의 exclusive end 경계를 보정하고, 시각 순으로."""
    out = []
    for ev in safe(deps.load_events, [], frm, to):
        p = ev.get("extendedProps") or {}
        kind = p.get("kind") or "user"
        if kinds and kind not in kinds:
            continue
        if ev.get("allDay") and ev.get("end") and str(ev["end"])[:10] <= frm:
            continue                                    # 종일 end 는 exclusive — 전날 끝난 일정이 from 날짜에 걸리지 않게
        if kind == "deadline" and (p.get("done") or p.get("removed")):
            continue                                    # 끝낸 과제는 '남은 마감'이 아니다
        out.append(ev)
    out.sort(key=sort_key)
    return out


def sort_key(ev: dict) -> str:
    p = ev.get("extendedProps") or {}
    if p.get("kind") == "deadline" and p.get("due"):
        return str(p["due"])
    return str(ev.get("start") or "")


def event_brief(ev: dict) -> dict:
    """LLM 으로 되돌리는 일정 한 건 — id · title · start · end · kind 만 (F9-R42)."""
    return {"id": str(ev.get("id")), "title": ev.get("title"), "kind": (ev.get("extendedProps") or {}).get("kind") or "user",
            "start": ev.get("start"), "end": ev.get("end")}


def attendance_courses(deps: Deps) -> list[dict]:
    got = safe(deps.attendance, [])
    if isinstance(got, dict):
        got = got.get("courses") or []
    return [c for c in got if isinstance(c, dict) and not c.get("excluded")]


def assignments(deps: Deps) -> list[dict]:
    return [a for a in safe(deps.assignments, []) if isinstance(a, dict) and not a.get("stale")]


def exams(deps: Deps) -> dict:
    got = safe(deps.exams, {})
    return got if isinstance(got, dict) else {}


def graduation(deps: Deps) -> dict:
    got = safe(deps.graduation, {})
    return got if isinstance(got, dict) else {}


def start_sync(deps: Deps) -> Optional[dict]:
    if deps.start_sync is None:
        return None
    try:
        return deps.start_sync() or {}
    except Exception as e:                              # noqa: BLE001
        return {"running": False, "error": f"{type(e).__name__}: {e}"}


def route_of(route: Any) -> Optional[tuple[str, str]]:
    """허용 목록에 있는 화면이면 (경로, 이름). 없으면 None."""
    if not isinstance(route, str):
        return None
    r = route.strip()
    if len(r) > 1:
        r = r.rstrip("/")
    return (r, C.ROUTES[r]) if r in C.ROUTES else None
