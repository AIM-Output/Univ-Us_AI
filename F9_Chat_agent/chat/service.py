"""F9 대화 규칙 — 1턴 처리(LLM 루프) · 시간 검증 · 겹침 검사 · 미리보기 · 확인/취소 · 템플릿 답변.

fastapi 를 부르지 않는다 — 오류는 아래 예외로 올리고 HTTP 상태로 바꾸는 일은 api.py 가 한다.

1턴 (요구사항 F9 5절 · 계약 3절)
    system 문구는 학습 데이터와 똑같이 · [system, user] 단일 턴 · temperature 0 · max_tokens 256 · tools 11개
    텍스트만        → assistant (모델 문장 그대로 — 거절·범위 밖은 모델이 이미 그렇게 답한다)
    쓰기 도구       → 미리보기(PendingAction 10분) — 실행하지 않는다 (F9-R40)
    읽기 도구       → 서버 템플릿 문장 + 목록 카드 + 근거 링크 (숫자는 도구 값 그대로, F9-R23)
                      단 쓰기 의도 표지가 있으면 list_events 결과를 tool 메시지로 돌려주고 한 번 더 부른다(2-step) — 최대 3회
    run_sync · open_placement · navigate → 실행 기록 / 링크
    없는 도구 · arguments 가 JSON 이 아님 → "다시 말씀해 주세요"
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Optional
from urllib.parse import quote

from . import config as C
from . import llm, store
from . import tools as T
from .tools import Deps

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?)?$")      # C1 service.DATE_RE 와 같다
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
WRITE_TOOLS = ("create_event", "update_event", "delete_event")


class Invalid(Exception):        # 422
    pass


class NotFound(Exception):       # 404
    pass


class Conflict(Exception):       # 409
    pass


class Unavailable(Exception):    # 503
    pass


class _Bad(Exception):
    """미리보기를 만들 수 없는 값 — 사용자에게 보일 문장(또는 confirm 에서는 422)."""

    def __init__(self, what: str):
        super().__init__(what)
        self.what = what            # "title" | "date" | "category"


# ---------------------------------------------------------------- 시각 · 표시

def now_of(deps: Deps) -> datetime:
    return (deps.now() if deps.now else datetime.now()).replace(microsecond=0)


def stamp(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M")


def system_prompt(now: datetime) -> str:
    """학습 데이터(prepare_f9.system_message)와 한 글자도 다르지 않게."""
    return f"오늘은 {now:%Y-%m-%d}({C.WEEKDAY_KO[now.weekday()]}) {now:%H:%M} 입니다. 주 시작은 월요일입니다.\n{C.SYSTEM_RULES}"


def md(d: date) -> str:
    return f"{d.month}/{d.day}({C.WEEKDAY_KO[d.weekday()]})"


def _day_word(d: date, today: Optional[date]) -> str:
    if today is not None:
        if d == today:
            return "오늘"
        if d == today + timedelta(days=1):
            return "내일"
    return md(d)


def when(value: Optional[str], all_day: bool, today: Optional[date] = None) -> str:
    """표시 문자열 하나로 통일 — 종일 "11/6(금)" · 시각 "11/6(금) 18:00" · 오늘/내일이면 "오늘 23:59"·"내일 18:00".
    today=None 이면 오늘/내일로 바꾸지 않는다(실행 기록용)."""
    if not value:
        return ""
    try:
        t = datetime.fromisoformat(str(value)[:19])
    except ValueError:
        return str(value)
    if all_day or len(str(value)) <= 10:
        return _day_word(t.date(), today)
    return f"{_day_word(t.date(), today)} {t:%H:%M}"


def due_when(due: Optional[str], today: Optional[date]) -> str:
    """마감 시각 — 00:00 마감은 전날 24:00 으로 (F6 캘린더와 같은 읽기)."""
    if not due:
        return "마감 없음"
    try:
        t = datetime.fromisoformat(str(due)[:19])
    except ValueError:
        return str(due)
    if len(str(due)) > 10 and (t.hour, t.minute) == (0, 0):
        return f"{_day_word(t.date() - timedelta(days=1), today)} 24:00"
    return when(due, False, today)


def span_text(start: str, end: Optional[str], all_day: bool) -> str:
    """실행 기록의 일시 — "11/12(목) 20:00" · 종일 여러 날이면 "11/12(목)~11/13(금)"."""
    if all_day and end:
        try:
            s, e = date.fromisoformat(start[:10]), date.fromisoformat(end[:10]) - timedelta(days=1)
            if e > s:
                return f"{md(s)}~{md(e)}"
        except ValueError:
            pass
    return when(start, all_day)


def num(x: Any) -> str:
    if isinstance(x, bool) or x is None:
        return "—" if x is None else str(x)
    if isinstance(x, (int, float)):
        return str(int(x)) if float(x).is_integer() else f"{x:.1f}".rstrip("0").rstrip(".")
    return str(x)


# ---------------------------------------------------------------- 값 검증 (미리보기 · 확인 공통)

def _dt(v: Any) -> Optional[datetime]:
    if not isinstance(v, str) or not DATE_RE.match(v.strip()):
        return None
    try:
        return datetime.fromisoformat(v.strip())
    except ValueError:                                  # 2월 30일 같은 없는 날짜
        return None


def normalize(fields: dict, strict_category: bool = False) -> dict:
    """{title, start, end?, allDay?, category?} → 카드 값. 규칙 (계약 3-A):
    start 에 T 가 없으면 종일 · 종일 end 없으면 start+1일(exclusive) · 시각 있고 end 없으면 +1시간 · end <= start 면 다시 만든다."""
    title = str(fields.get("title") or "").strip()
    if not 1 <= len(title) <= 200:
        raise _Bad("title")
    s_raw = str(fields.get("start") or "").strip()
    e_raw = str(fields.get("end") or "").strip() or None
    s = _dt(s_raw)
    if s is None:
        raise _Bad("date")
    e = None
    if e_raw is not None:
        e = _dt(e_raw)
        if e is None:
            raise _Bad("date")
    all_day = "T" not in s_raw or fields.get("allDay") is True
    if all_day:
        sd = s.date()
        ed = e.date() if e is not None else None
        if ed is None or ed <= sd:
            ed = sd + timedelta(days=1)
        start, end = sd.isoformat(), ed.isoformat()
    else:
        if e is None or "T" not in (e_raw or "") or e <= s:
            e = s + timedelta(hours=1)
        start, end = stamp(s), stamp(e)
    cat = fields.get("category")
    if cat not in C.CATEGORIES:
        if strict_category and cat not in (None, ""):
            raise _Bad("category")
        cat = "personal"
    return {"title": title, "start": start, "end": end, "allDay": all_day, "category": cat}


def _interval(start: str, end: Optional[str], all_day: bool) -> tuple[datetime, datetime]:
    s = datetime.fromisoformat(str(start)[:19])
    if all_day:
        s = datetime(s.year, s.month, s.day)
    e = datetime.fromisoformat(str(end)[:19]) if end else None
    if e is None or e <= s:
        e = s + (timedelta(days=1) if all_day else timedelta(hours=1))
    return s, e


def _is_past(f: dict, now: datetime) -> bool:
    if f["allDay"]:
        return date.fromisoformat(f["start"]) < now.date()
    return datetime.fromisoformat(f["start"]) < now


def conflict_of(deps: Deps, f: dict, exclude_id: Optional[str] = None) -> Optional[str]:
    """같은 시간의 내 일정(kind=user, 할 일 제외) — 종일은 종일끼리, 시각은 시각끼리. 경고만 하고 막지 않는다."""
    s, e = _interval(f["start"], f["end"], f["allDay"])
    last = (e - timedelta(minutes=1)).date()
    hits = []
    for ev in T.safe(deps.load_events, [], s.date().isoformat(), last.isoformat()):
        p = ev.get("extendedProps") or {}
        if (p.get("kind") or "user") != "user" or p.get("isTodo"):
            continue
        if exclude_id is not None and str(ev.get("id")) == str(exclude_id):
            continue
        if bool(ev.get("allDay")) != f["allDay"]:
            continue
        try:
            es, ee = _interval(ev["start"], ev.get("end"), bool(ev.get("allDay")))
        except (KeyError, TypeError, ValueError):
            continue
        if es < e and s < ee:
            hits.append((es, ee, ev))
    if not hits:
        return None
    hits.sort(key=lambda h: h[0])
    parts = [f"{ev.get('title')} (종일)" if f["allDay"] else f"{ev.get('title')} {es:%H:%M}~{ee:%H:%M}"
             for es, ee, ev in hits[:3]]
    return f"겹치는 일정 {len(hits)}건 — " + ", ".join(parts)


def _event_fields(ev: dict) -> dict:
    """C1 일정 → 카드 값 (바꾸기 전 · 삭제 카드). 초는 뗀다. end 가 없던 일정은 ""."""
    all_day = bool(ev.get("allDay"))
    cut = 10 if all_day else 16
    p = ev.get("extendedProps") or {}
    return {"title": ev.get("title") or "", "start": str(ev.get("start") or "")[:cut],
            "end": str(ev.get("end") or "")[:cut], "allDay": all_day, "category": p.get("category") or "personal"}


def _apply_patch(before: dict, patch: dict) -> dict:
    m = dict(before)
    for k in ("title", "start", "end", "category"):
        if patch.get(k) is not None:
            m[k] = patch[k]
    if "allDay" in patch or "all_day" in patch:
        m["allDay"] = bool(patch.get("allDay", patch.get("all_day")))
    elif patch.get("start") is not None:
        m["allDay"] = "T" not in str(patch["start"])
    if patch.get("start") is not None and patch.get("end") is None:
        m["end"] = None
        ns = _dt(str(patch["start"]))
        if ns is not None and before.get("end") and m["allDay"] == before["allDay"]:
            try:                                        # 시작만 옮기면 길이는 그대로
                bs, be = _interval(before["start"], before["end"], before["allDay"])
                ne = ns + (be - bs)
                m["end"] = ne.date().isoformat() if m["allDay"] else stamp(ne)
            except ValueError:
                pass
    return m


# ---------------------------------------------------------------- 메시지

def _new_id(con: sqlite3.Connection, prefix: str) -> str:
    base = f"{prefix}_{int(time.time() * 1000)}"
    mid, k = base, 1
    while store.has_message(con, mid):
        k += 1
        mid = f"{base}_{k}"
    return mid


def _say(con: sqlite3.Connection, now: datetime, text: str, list_: Optional[list[dict]] = None,
         link: Optional[dict] = None, role: str = "assistant") -> dict:
    msg: dict[str, Any] = {"id": _new_id(con, "a" if role == "assistant" else "s"), "role": role, "text": text, "at": stamp(now)}
    if list_:
        msg["list"] = list_
    if link:
        msg["link"] = link
    return store.add_message(con, msg)


def _system(con: sqlite3.Connection, now: datetime, text: str) -> dict:
    return _say(con, now, text, role="system")


def _preview(con: sqlite3.Connection, now: datetime, kind: str, tool: str, f: dict, target_id: Optional[str] = None,
             before: Optional[dict] = None, warning: Optional[str] = None, conflict: Optional[str] = None) -> dict:
    aid = f"pa_{uuid.uuid4().hex[:8]}"
    while store.get_action(con, aid) or store.has_message(con, aid):
        aid = f"pa_{uuid.uuid4().hex[:8]}"
    expires = (now + timedelta(minutes=C.PENDING_MINUTES)).isoformat(timespec="seconds")
    msg: dict[str, Any] = {"id": aid, "role": "preview", "at": stamp(now), "kind": kind, **f,
                           "categoryLabel": C.CATEGORIES.get(f["category"], f["category"])}
    if target_id is not None:
        msg["targetId"] = target_id
    if before is not None:
        msg["before"] = before
    if conflict:
        msg["conflict"] = conflict
    if warning:
        msg["warning"] = warning
    msg.update(state="pending", expiresAt=expires)
    store.add_message(con, msg)
    store.add_action(con, {"id": aid, "tool": tool, "args": f, "targetId": target_id, "state": "pending",
                           "expiresAt": expires, "createdAt": now.isoformat(timespec="seconds"), "messageId": aid})
    return msg


def _expired(a: dict, now: datetime) -> bool:
    return datetime.fromisoformat(a["expiresAt"]) <= now


def view(msg: dict, now: datetime) -> dict:
    """내려줄 때 — 만료된 대기 미리보기는 state=expired 로 바꿔서."""
    if msg.get("role") == "preview" and msg.get("state") == "pending" and msg.get("expiresAt"):
        try:
            if datetime.fromisoformat(msg["expiresAt"]) <= now:
                return {**msg, "state": "expired"}
        except ValueError:
            pass
    return msg


# ---------------------------------------------------------------- LLM 응답 읽기

def _calls(resp: dict, kinds: dict[str, str]) -> list[dict]:
    out = []
    for tc in resp.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        name = fn.get("name")
        raw = fn.get("arguments")
        args: Any = None
        if isinstance(raw, dict):
            args = raw
        elif isinstance(raw, str):
            try:
                args = json.loads(raw) if raw.strip() else {}
            except ValueError:
                args = None
        elif raw is None:
            args = {}
        out.append({"name": name, "args": args if isinstance(args, dict) else {},
                    "bad": name not in kinds or not isinstance(args, dict)})
    return out


def _write_intent(text: str) -> bool:
    return any(w in text for w in C.WRITE_INTENT)


# ---------------------------------------------------------------- 1턴

def turn(con: sqlite3.Connection, deps: Deps, text: Any) -> dict:
    """POST /api/chat — {reply, messages(이번 턴에 새로 생긴 것, 사용자 echo 포함)}."""
    text = str(text or "").strip()
    if not text:
        raise Invalid(C.MSG["empty"])
    if len(text) > C.MAX_TEXT:
        raise Invalid(C.MSG["too_long"])
    now = now_of(deps)
    out = [store.add_message(con, {"id": _new_id(con, "u"), "role": "user", "text": text, "at": stamp(now)})]
    con.commit()                                        # LLM 이 실패해도 사용자 메시지는 남긴다
    chat = deps.chat
    if chat is None:
        if not C.llm_configured():
            raise llm.NotConfigured(C.MSG["not_configured"])
        chat = llm.complete
    tool_specs, kinds = T.load()
    msgs: list[dict] = [{"role": "system", "content": system_prompt(now)}, {"role": "user", "content": text}]
    resp = chat(msgs, tool_specs) or {}
    calls = _calls(resp, kinds)

    if not calls:
        content = str(resp.get("content") or "").strip()
        out.append(_say(con, now, content or C.MSG["again"]))
        return _turn_result(out)

    # 2-step — 쓰기 의도가 있는데 모델이 list_events 로 대상을 찾았다 → 결과를 돌려주고 한 번 더
    if _write_intent(text) and not any(kinds.get(c["name"]) == "write" for c in calls if not c["bad"]):
        n, k, cur = 1, 0, calls
        while n < C.MAX_LLM_CALLS:
            reads = [c for c in cur if not c["bad"] and c["name"] == "list_events"]
            if not reads:
                break
            tcs, results = [], []
            for c in reads:
                k += 1
                cid = f"call_{k}"
                tcs.append({"id": cid, "type": "function",
                            "function": {"name": c["name"], "arguments": json.dumps(c["args"], ensure_ascii=False)}})
                results.append({"role": "tool", "tool_call_id": cid,
                                "content": json.dumps(_list_result(deps, c["args"]), ensure_ascii=False)})
            msgs.append({"role": "assistant", "tool_calls": tcs})
            msgs.extend(results)
            nxt = _calls(chat(msgs, tool_specs) or {}, kinds)
            n += 1
            if any(not c["bad"] and kinds.get(c["name"]) == "write" for c in nxt):
                cur = nxt
                break
            if any(not c["bad"] and c["name"] == "list_events" for c in nxt):
                cur = nxt
                continue
            break
        calls = cur

    for c in calls:
        out.append(_dispatch(con, deps, now, c, kinds))
    return _turn_result(out)


def _turn_result(out: list[dict]) -> dict:
    texts = [m["text"] for m in out[1:] if m.get("role") in ("assistant", "system") and m.get("text")]
    return {"reply": texts[-1] if texts else C.MSG["preview_reply"], "messages": out}


def _list_result(deps: Deps, args: dict) -> dict:
    span = _span(args)
    if span is None:
        return {"items": []}
    frm, to, kinds = span
    return {"items": [T.event_brief(ev) for ev in T.events(deps, frm, to, kinds)]}


def _dispatch(con: sqlite3.Connection, deps: Deps, now: datetime, c: dict, kinds: dict[str, str]) -> dict:
    if c["bad"]:
        return _say(con, now, C.MSG["again"])
    name, args = c["name"], c["args"]
    if name == "create_event":
        return _create(con, deps, now, args)
    if name in ("update_event", "delete_event"):
        return _change(con, deps, now, name, args)
    if name == "list_events":
        return _answer_events(con, deps, now, args)
    if name == "get_graduation":
        return _answer_graduation(con, deps, now)
    if name == "get_attendance":
        return _answer_attendance(con, deps, now, args)
    if name == "list_assignments":
        return _answer_assignments(con, deps, now, args)
    if name == "get_exams":
        return _answer_exams(con, deps, now, args)
    if name == "run_sync":
        return _run_sync(con, deps, now, args)
    if name == "open_placement":
        return _say(con, now, "공강 배치 미리보기를 열어 둘게요", link={"label": "공강 배치 미리보기 열기", "href": "/?place=preview"})
    if name == "navigate":
        hit = T.route_of(args.get("route"))
        if hit is None:
            return _say(con, now, "그 화면은 없습니다")
        route, label = hit
        return _say(con, now, f"{label} 화면으로 이동할게요", link={"label": f"{label} 열기", "href": route})
    return _say(con, now, C.MSG["again"])


# ---------------------------------------------------------------- 쓰기 → 미리보기

def _bad_text(e: _Bad) -> str:
    return C.MSG["bad_date"] if e.what == "date" else C.MSG["again"]


def _warning(f: dict, now: datetime, repeat: Any = None) -> Optional[str]:
    w = []
    if _is_past(f, now):
        w.append(C.MSG["past"])
    if isinstance(repeat, str) and repeat.strip():
        w.append(C.MSG["repeat"])
    return " · ".join(w) or None


def _create(con: sqlite3.Connection, deps: Deps, now: datetime, args: dict) -> dict:
    try:
        f = normalize(args)
    except _Bad as e:
        return _say(con, now, _bad_text(e))
    return _preview(con, now, "create", "create_event", f, warning=_warning(f, now, args.get("repeat")),
                    conflict=conflict_of(deps, f))


def user_id_text(event_id: Any) -> str:
    """모델이 준 id — 학습 데이터의 'ev:1234' 꼴은 앱의 숫자 id('1234')로 읽는다. 다른 접두는 그대로."""
    eid = str(event_id).strip() if event_id is not None else ""
    if eid.startswith("ev:") and eid[3:].isdigit():
        eid = eid[3:]
    return eid


def refusal(event_id: Any) -> Optional[str]:
    """내 일정(숫자 id)이 아니면 거절 문장."""
    eid = user_id_text(event_id)
    if not eid:
        return C.MSG["no_target"]
    for prefix, msg in C.REFUSE_PREFIX.items():
        if eid.startswith(prefix):
            return msg
    return None if eid.isdigit() else C.MSG["no_target"]


def _get_event(c1: Any, uid: int) -> Optional[dict]:
    try:
        return c1.store.get_event(uid)
    except Exception:                                   # noqa: BLE001
        return None


def _change(con: sqlite3.Connection, deps: Deps, now: datetime, name: str, args: dict) -> dict:
    eid = args.get("id")
    if msg := refusal(eid):
        return _say(con, now, msg)
    if deps.c1 is None:
        return _say(con, now, C.MSG["no_calendar"])
    uid = int(user_id_text(eid))
    cur = _get_event(deps.c1, uid)
    if not cur:
        return _say(con, now, C.MSG["no_target"])
    before = _event_fields(cur)
    if name == "delete_event":
        return _preview(con, now, "delete", name, before, target_id=str(uid))
    patch = args.get("patch")
    if not isinstance(patch, dict) or not patch:
        return _say(con, now, C.MSG["again"])
    try:
        f = normalize(_apply_patch(before, patch))
    except _Bad as e:
        return _say(con, now, _bad_text(e))
    return _preview(con, now, "update", name, f, target_id=str(uid), before=before,
                    warning=_warning(f, now), conflict=conflict_of(deps, f, exclude_id=str(uid)))


# ---------------------------------------------------------------- 확인 · 취소

def _pending(con: sqlite3.Connection, aid: str, now: datetime) -> tuple[dict, dict]:
    a = store.get_action(con, aid)
    msg = store.get_message(con, a["messageId"]) if a else None
    if not a or not msg:
        raise NotFound(C.MSG["no_action"])
    if a["state"] == "expired":
        raise Conflict(C.MSG["expired"])
    if a["state"] != "pending":
        raise Conflict(C.MSG["done_already"])
    if _expired(a, now):
        store.set_action(con, aid, "expired")
        store.put_message(con, {**msg, "state": "expired"})
        con.commit()
        raise Conflict(C.MSG["expired"])
    return a, msg


def _edited(a: dict, edits: dict) -> dict:
    fields = dict(a["args"])
    for k in ("title", "start", "end", "allDay", "category"):
        if k in edits and edits[k] is not None:
            fields[k] = edits[k]
    if edits.get("start") is not None and "allDay" not in edits:
        fields["allDay"] = "T" not in str(edits["start"])
    if edits.get("start") is not None and edits.get("end") is None:
        fields["end"] = None
    if "allDay" in edits and edits["allDay"] is not None and "start" not in edits:
        fields["start"] = str(fields.get("start") or "")[:10] if edits["allDay"] else fields.get("start")
    return fields


def confirm(con: sqlite3.Connection, deps: Deps, aid: str, edits: Optional[dict]) -> dict:
    """POST /api/chat/actions/{id}/confirm — 고친 값을 다시 검사하고 C1 에 쓴다. {messages: [preview(confirmed), system]}"""
    now = now_of(deps)
    a, msg = _pending(con, aid, now)
    c1 = deps.c1
    if c1 is None:
        raise Unavailable(C.MSG["no_calendar"])
    edits = edits if isinstance(edits, dict) else {}
    f = dict(a["args"])
    if a["tool"] in ("create_event", "update_event"):
        try:
            f = normalize(_edited(a, edits), strict_category=True)
        except _Bad as e:
            raise Invalid({"title": "제목은 1~200자입니다",
                           "category": "분류는 personal·study·team·etc 중 하나입니다"}.get(
                e.what, "날짜를 알아듣지 못했습니다 — YYYY-MM-DD 또는 YYYY-MM-DDTHH:MM")) from e
        svc = getattr(c1, "service", None)
        if svc is not None and (err := svc.date_error(f["start"], f["end"])):
            raise Invalid(err)
    data = {"title": f["title"], "start": f["start"], "end": f["end"] or None, "all_day": f["allDay"],
            "category": f["category"]}
    if a["tool"] == "create_event":
        data.update(memo="", is_todo=False, done=False, origin=None)
        model = getattr(c1, "UserEventIn", None)
        if model is not None:                            # C1 라우터 post_event 와 같은 검사
            try:
                data = model(**data).model_dump()
            except Exception as e:                       # noqa: BLE001 — pydantic ValidationError
                raise Invalid(f"일정 값이 올바르지 않습니다: {e}") from e
        data["title"] = data["title"].strip()
        data["memo"] = (data.get("memo") or "").strip()

    # 한 문장으로 pending → running 을 잡는다 — 두 번 누른 확인(동시 요청)은 하나만 실행된다
    if not store.claim_action(con, aid, now.isoformat(timespec="seconds")):
        con.rollback()
        raise Conflict(C.MSG["done_already"])
    con.commit()
    try:
        done = _execute(c1, a, data)
    except BaseException:
        store.set_action(con, aid, "pending")              # 실패하면 다시 누를 수 있게 되돌린다
        con.commit()
        raise
    store.set_action(con, aid, "confirmed", f)
    card = {**msg, **f, "categoryLabel": C.CATEGORIES.get(f["category"], f["category"]), "state": "confirmed"}
    store.put_message(con, card)
    sysmsg = _system(con, now, f"{done} — {f['title']} {span_text(f['start'], f['end'], f['allDay'])}")
    return {"messages": [card, sysmsg]}


def _execute(c1: Any, a: dict, data: dict) -> str:
    """확인된 쓰기 하나를 C1 에 — 실행 기록 문구의 앞부분을 돌려준다."""
    if a["tool"] == "create_event":
        c1.store.create_event(**data)
        return "일정을 등록했습니다"
    svc = getattr(c1, "service", None)
    uid = svc.parse_user_id(str(a["targetId"])) if svc is not None else (
        int(a["targetId"]) if str(a["targetId"] or "").isdigit() else None)
    if uid is None or not c1.store.get_event(uid):
        raise Conflict(C.MSG["gone"])
    if a["tool"] == "update_event":
        if not c1.store.update_event(uid, data):
            raise Conflict(C.MSG["gone"])
        return "일정을 바꿨습니다"
    if not c1.store.delete_event(uid):
        raise Conflict(C.MSG["gone"])
    return "일정을 삭제했습니다"


def cancel(con: sqlite3.Connection, deps: Deps, aid: str) -> dict:
    now = now_of(deps)
    a, msg = _pending(con, aid, now)
    if not store.cancel_action(con, aid):                  # 그 사이 확인이 잡았다
        raise Conflict(C.MSG["done_already"])
    card = store.put_message(con, {**msg, "state": "canceled"})
    return {"messages": [card]}


def messages(con: sqlite3.Connection, deps: Deps, before: Optional[str] = None, limit: int = C.HISTORY_LIMIT) -> dict:
    now = now_of(deps)
    limit = max(1, min(int(limit or C.HISTORY_LIMIT), C.HISTORY_MAX))
    return {"messages": [view(m, now) for m in store.list_messages(con, before, limit)]}


def clear(con: sqlite3.Connection) -> None:
    store.clear(con)


# ---------------------------------------------------------------- 읽기 → 템플릿 답변 (LLM 재호출 없음)

def _span(args: dict) -> Optional[tuple[str, str, Optional[list[str]]]]:
    frm = str(args.get("from") or "").strip()[:10]
    to = str(args.get("to") or frm).strip()[:10]
    if not DAY_RE.match(frm) or not DAY_RE.match(to):
        return None
    try:
        date.fromisoformat(frm)
        date.fromisoformat(to)
    except ValueError:
        return None
    if to < frm:
        frm, to = to, frm
    kinds = args.get("kinds")
    kinds = [k for k in kinds if isinstance(k, str)] if isinstance(kinds, list) else None
    return frm, to, (kinds or None)


def _period(frm: str, to: str) -> str:
    a, b = date.fromisoformat(frm), date.fromisoformat(to)
    return md(a) if a == b else f"{md(a)}~{md(b)}"


def event_href(eid: str) -> str:
    if eid.startswith("dl:"):
        return "/assignments"
    if eid.startswith("ex:"):
        return "/exams"
    if eid.startswith("ac:"):
        return "/academic"
    return f"/?event={quote(eid, safe=':')}"


def _event_item(ev: dict, today: date) -> dict:
    p = ev.get("extendedProps") or {}
    kind = p.get("kind") or "user"
    if kind == "deadline":
        w = due_when(p.get("due") or ev.get("end") or ev.get("start"), today)
    else:
        w = when(ev.get("start"), bool(ev.get("allDay")), today)
    course = p.get("courseShort") or p.get("course") or p.get("courseName")
    label = C.KIND_LABEL.get(kind, kind)
    eid = str(ev.get("id"))
    return {"title": str(ev.get("title") or ""), "when": w, "sub": f"{label} · {course}" if course else label,
            "href": event_href(eid)}


def _answer_events(con: sqlite3.Connection, deps: Deps, now: datetime, args: dict) -> dict:
    span = _span(args)
    if span is None:
        return _say(con, now, C.MSG["bad_date"])
    frm, to, kinds = span
    evs = T.events(deps, frm, to, kinds)
    period = _period(frm, to)
    noun = "마감" if kinds == ["deadline"] else "일정"
    if kinds == ["deadline"]:
        link = {"label": "과제 목록 열기", "href": "/assignments"}
    elif kinds == ["exam"]:
        link = {"label": "시험 열기", "href": "/exams"}
    elif kinds == ["academic"]:
        link = {"label": "학사일정 열기", "href": "/academic"}
    else:
        link = {"label": "캘린더 열기", "href": "/"}
    if not evs:
        return _say(con, now, f"{period}에는 {noun}이 없습니다", link=link)
    items = [_event_item(ev, now.date()) for ev in evs[:C.LIST_MAX]]
    return _say(con, now, f"{period} {noun} {len(evs)}건", items, link)


def _answer_graduation(con: sqlite3.Connection, deps: Deps, now: datetime) -> dict:
    g = T.graduation(deps)
    link = {"label": "졸업요건 열기", "href": "/graduation"}
    if not g.get("available") or g.get("hasData") is False or g.get("remaining") is None:
        return _say(con, now, "졸업요건 데이터가 아직 없습니다. 졸업요건 화면에서 성적표를 불러오세요", link=link)
    text = f"졸업까지 {num(g['remaining'])}학점 남았습니다 ({num(g.get('earned'))}/{num(g.get('required'))})"
    if g.get("headline"):
        text += f" — {g['headline']}"
    return _say(con, now, text, link=link)


def _norm(s: Any) -> str:
    return re.sub(r"\s+", "", str(s or "")).lower()


def _answer_attendance(con: sqlite3.Connection, deps: Deps, now: datetime, args: dict) -> dict:
    courses = T.attendance_courses(deps)
    link = {"label": "출결 열기", "href": "/attendance"}
    want = _norm(args.get("course"))
    if want:
        courses = [c for c in courses if want in _norm(c.get("name")) or want in _norm(c.get("short"))]
        if not courses:
            return _say(con, now, f"'{str(args.get('course')).strip()}' 과목을 찾지 못했습니다", link=link)
    elif not courses:
        return _say(con, now, "출결 데이터가 아직 없습니다. 출결 화면에서 시간표를 입력하세요", link=link)
    items = []
    for c in courses:
        s = c.get("summary") or {}
        rem = s.get("remaining")
        rem_txt = num(max(0, rem)) if isinstance(rem, (int, float)) and not isinstance(rem, bool) else num(rem)
        item = {"title": str(c.get("short") or c.get("name") or ""), "when": str(s.get("levelLabel") or "—"),
                "sub": f"남은 결석 {rem_txt}회"}
        if c.get("id"):
            item["href"] = f"/attendance?course={quote(str(c['id']), safe=':')}"
        items.append(item)
    if len(items) == 1:
        text = f"{items[0]['title']} 출결 — {items[0]['when']} · {items[0]['sub']}"
    else:
        text = f"출결 {len(items)}과목"
    return _say(con, now, text, items[:C.LIST_MAX], link)


def _answer_assignments(con: sqlite3.Connection, deps: Deps, now: datetime, args: dict) -> dict:
    rows = T.assignments(deps)
    g = C.ASSIGNMENT_GROUPS.get(str(args.get("group") or ""))
    link = {"label": "과제 목록 열기", "href": "/assignments"}
    label = "급한 순"
    if g:
        rows = [r for r in rows if r.get("group") == g]
        label = C.GROUP_LABEL[g]
    if not rows:
        return _say(con, now, f"{label} 과제가 없습니다", link=link)
    items = []
    for r in rows[:C.LIST_MAX]:
        sub = " · ".join(str(x) for x in (r.get("courseShort"), r.get("kindLabel")) if x)
        items.append({"title": str(r.get("title") or ""), "when": due_when(r.get("due"), now.date()), "sub": sub,
                      "href": "/assignments"})
    return _say(con, now, f"{label} {len(rows)}건", items, link)


def _answer_exams(con: sqlite3.Connection, deps: Deps, now: datetime, args: dict) -> dict:
    ov = T.exams(deps)
    upcoming = args.get("upcoming") is not False
    rows = list(ov.get("exams" if upcoming else "past") or [])
    link = {"label": "시험 열기", "href": "/exams"}
    noun = "다가오는 시험" if upcoming else "지난 시험"
    if not rows:
        return _say(con, now, f"{noun}이 없습니다", link=link)
    items = []
    for e in rows[:C.LIST_MAX]:
        d = str(e.get("date") or "")
        t = str(e.get("time") or "")
        w = when(f"{d}T{t[:5]}" if d and t else d, not t, now.date()) if d else "날짜 미정"
        item = {"title": str(e.get("title") or e.get("typeLabel") or "시험"), "when": w,
                "sub": str(e.get("course") or e.get("courseName") or "")}
        if e.get("id") is not None:
            item["href"] = f"/exams?exam={quote(str(e['id']), safe=':')}"
        items.append(item)
    return _say(con, now, f"{noun} {len(rows)}건", items, link)


def _run_sync(con: sqlite3.Connection, deps: Deps, now: datetime, args: dict) -> dict:
    if args.get("source") != "eclass":
        return _say(con, now, "지원하지 않는 동기화 대상입니다")
    st = T.start_sync(deps)
    if st is None:
        return _system(con, now, "e클래스 수집을 시작할 수 없습니다")
    if st.get("already_running"):
        return _system(con, now, "이미 수집 중입니다")
    if st.get("error"):
        return _system(con, now, f"e클래스 수집을 시작하지 못했습니다 — {st['error']}")
    return _system(con, now, "e클래스 수집을 시작했습니다")
