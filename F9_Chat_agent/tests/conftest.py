"""테스트 공통 — 임시 data/ 를 쓴다. 내 대화 기록(F9_Chat_agent/data)도, 다른 기능의 DB 도 건드리지 않는다.

LLM 은 부르지 않는다 — 슬롯을 비워 두고(F9_LLM_* · LLM_MAIN_* = ""), 대화 루프는 Deps.chat 에 가짜 LLM(Script)을 끼워서 시험한다.
C1 캘린더도 가짜(FakeC1)다 — 백엔드 app 패키지는 import 하지 않고, build_router 를 맨 FastAPI 앱에 붙인다.
"""
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

_TMP = Path(tempfile.mkdtemp(prefix="f9test_"))
os.environ["F9_DATA_DIR"] = str(_TMP / "f9data")
os.environ["F9_LLM"] = ""
os.environ["F9_LLM_TIMEOUT"] = "5"
for k in ("F9_LLM_BASE_URL", "F9_LLM_API_KEY", "F9_LLM_MODEL", "LLM_MAIN_BASE_URL", "LLM_MAIN_API_KEY", "LLM_MAIN_MODEL"):
    os.environ[k] = ""                     # 빈 값도 '있는 환경변수'라 F9_Chat_agent/.env 가 있어도 테스트에 새지 않는다
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

NOW = datetime(2026, 11, 2, 10, 0)          # 월요일 10:00


@pytest.fixture(autouse=True)
def clean_db():
    from chat import config as C
    if C.DB_PATH.exists():
        C.DB_PATH.unlink()
    yield
    if C.DB_PATH.exists():
        C.DB_PATH.unlink()


# ---------------------------------------------------------------- 가짜 LLM

def call(name, **args):
    """tool_calls 하나짜리 응답 (OpenAI chat/completions 의 choices[0].message)."""
    return calls((name, args))


def calls(*pairs):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"srv_{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}}
        for i, (n, a) in enumerate(pairs)]}


def text(s):
    return {"role": "assistant", "content": s}


class Script:
    """차례대로 응답을 돌려주는 가짜 LLM — 받은 messages·tools 를 기록한다."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.seen = []

    def __call__(self, messages, tools):
        self.seen.append((json.loads(json.dumps(messages, ensure_ascii=False)), tools))
        if not self.responses:
            raise AssertionError("LLM 을 예상보다 많이 불렀다")
        return self.responses.pop(0)


# ---------------------------------------------------------------- 가짜 C1

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?)?$")
CATS = {"personal": "개인", "study": "학업", "team": "팀플", "etc": "기타"}


class FakeService:
    @staticmethod
    def date_error(start, end):
        for label, v in (("start", start), ("end", end)):
            if v is not None and not DATE_RE.match(v):
                return f"{label} 형식이 잘못됐습니다: {v!r}"
        if start and end and end <= start:
            return "종료가 시작보다 빠르거나 같습니다."
        return None

    @staticmethod
    def parse_user_id(event_id):
        return int(event_id) if event_id.isdigit() else None


class FakeStore:
    def __init__(self):
        self.rows = {}
        self.next_id = 1
        self.writes = []

    def _event(self, uid):
        r = self.rows[uid]
        return {"id": str(uid), "title": r["title"], "start": r["start"], "end": r["end"], "allDay": bool(r["all_day"]),
                "editable": True, "extendedProps": {"kind": "user", "category": r["category"],
                                                    "categoryLabel": CATS[r["category"]], "memo": r.get("memo", ""),
                                                    "isTodo": bool(r.get("is_todo")), "done": bool(r.get("done")),
                                                    "origin": r.get("origin")}}

    def list_events(self):
        return [self._event(u) for u in sorted(self.rows)]

    def get_event(self, uid):
        return self._event(uid) if uid in self.rows else None

    def create_event(self, **f):
        uid = self.next_id
        self.next_id += 1
        self.rows[uid] = dict(f)
        self.writes.append(("create", uid, dict(f)))
        return self._event(uid)

    def update_event(self, uid, fields):
        if uid not in self.rows:
            return None
        self.rows[uid].update(fields)
        self.writes.append(("update", uid, dict(fields)))
        return self._event(uid)

    def delete_event(self, uid):
        self.writes.append(("delete", uid, None))
        return self.rows.pop(uid, None) is not None


class UserEventIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    start: str
    end: Optional[str] = None
    all_day: bool = False
    category: Literal["personal", "study", "team", "etc"] = "personal"
    memo: str = ""
    is_todo: bool = False
    done: bool = False
    origin: Optional[str] = None


class FakeC1:
    def __init__(self):
        self.store = FakeStore()
        self.service = FakeService()
        self.UserEventIn = UserEventIn

    def add(self, title, start, end=None, all_day=False, category="personal", **kw):
        return self.store.create_event(title=title, start=start, end=end, all_day=all_day, category=category, **kw)


def ev(eid, title, start, kind="user", end=None, all_day=False, **props):
    """/api/events 모양 (C1 5절)."""
    return {"id": eid, "title": title, "start": start, "end": end, "allDay": all_day,
            "extendedProps": {"kind": kind, **props}}


def overlaps(e, start, end):
    s = e["start"][:10]
    t = (e.get("end") or e["start"])[:10]
    return (not end or s <= end) and (not start or t >= start)


class World:
    """가짜 백엔드 — 합친 캘린더(다른 기능 일정 + 가짜 C1 내 일정) · 읽기 값 · 수집."""

    def __init__(self):
        self.c1 = FakeC1()
        self.others = []
        self.now = NOW
        self.sync_calls = 0
        self.sync_result = {"running": True}
        self.grad = {"available": True, "hasData": True, "remaining": 32.0, "earned": 98.0, "required": 130,
                     "headline": "전공필수 2과목", "verdict": "진행 중"}
        self.courses = []
        self.ranked = []
        self.exam_view = {"exams": [], "past": []}
        self.loaded = []

    def load_events(self, start, end):
        self.loaded.append((start, end))
        return [e for e in self.others + self.c1.store.list_events() if overlaps(e, start, end)]

    def start_sync(self):
        self.sync_calls += 1
        return dict(self.sync_result)

    def deps(self, chat=None, **over):
        from chat.tools import Deps
        kw = dict(load_events=self.load_events, c1=self.c1, graduation=lambda: self.grad,
                  attendance=lambda: self.courses, assignments=lambda: self.ranked, exams=lambda: self.exam_view,
                  start_sync=self.start_sync, sync_state=lambda: {"running": False}, chat=chat, now=lambda: self.now)
        kw.update(over)
        return Deps(**kw)


@pytest.fixture
def world():
    return World()


def client(deps):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from chat import api
    app = FastAPI()
    app.include_router(api.build_router(deps))
    return TestClient(app)


@pytest.fixture
def make(world):
    """make(Script(...)) → TestClient — 가짜 LLM 을 끼운 앱."""
    def _make(script=None, **over):
        return client(world.deps(chat=script, **over))
    return _make
