"""쓰기 — 미리보기 · 겹침 · 지난 날짜 · 거절 · 확인 · 취소 · 만료 · 이력 · 지우기."""
from datetime import timedelta

import pytest

from conftest import Script, call, ev

REHEARSAL = {"title": "발표 리허설", "start": "2026-11-12T20:00", "end": "2026-11-12T22:00", "allDay": False,
             "category": "personal"}


def preview_client(make, **over):
    args = {k: v for k, v in {**REHEARSAL, **over}.items() if v is not None}
    c = make(Script(call("create_event", **args)))
    body = c.post("/api/chat", json={"text": "다음 주 목요일 발표 리허설 저녁 8시"}).json()
    return c, body["messages"][1], body


# ---------------------------------------------------------------- 미리보기

def test_create_preview_does_not_write(world, make):
    c, p, body = preview_client(make)
    assert p["role"] == "preview" and p["kind"] == "create" and p["id"].startswith("pa_")
    assert (p["title"], p["start"], p["end"], p["allDay"]) == ("발표 리허설", "2026-11-12T20:00", "2026-11-12T22:00", False)
    assert p["category"] == "personal" and p["categoryLabel"] == "개인"
    assert p["state"] == "pending" and p["expiresAt"] == "2026-11-02T10:10:00"
    assert "warning" not in p and "conflict" not in p and "targetId" not in p
    assert body["reply"] == "확인을 누르면 실행합니다"
    assert world.c1.store.writes == []


def test_create_default_one_hour(make):
    _, p, _ = preview_client(make, end=None)
    assert (p["start"], p["end"]) == ("2026-11-12T20:00", "2026-11-12T21:00")


def test_create_all_day_end_exclusive(make):
    _, p, _ = preview_client(make, start="2026-11-12", end="2026-11-12", allDay=True)
    assert (p["start"], p["end"], p["allDay"]) == ("2026-11-12", "2026-11-13", True)
    _, p, _ = preview_client(make, start="2026-11-12", end=None, allDay=None)
    assert (p["start"], p["end"], p["allDay"]) == ("2026-11-12", "2026-11-13", True)


def test_create_unknown_category_falls_back(make):
    _, p, _ = preview_client(make, category="hobby")
    assert p["category"] == "personal"


def test_past_date_warning_and_repeat(make):
    _, p, _ = preview_client(make, start="2026-11-02T09:00", end="2026-11-02T09:30")
    assert p["warning"] == "지난 날짜입니다"
    _, p, _ = preview_client(make, start="2026-10-30", end="2026-10-31", allDay=True)
    assert p["warning"] == "지난 날짜입니다"
    _, p, _ = preview_client(make, repeat="weekly:TH")
    assert p["warning"] == "반복은 아직 지원하지 않아 1회로 넣습니다"
    _, p, _ = preview_client(make, start="2026-11-02", end=None, allDay=True)      # 오늘 종일은 지난 날짜가 아니다
    assert "warning" not in p


def test_nonexistent_date_asks_again(world, make):
    _, m, _ = preview_client(make, start="2026-02-30T20:00", end=None)
    assert m["role"] == "assistant" and m["text"] == "다시 말씀해 주세요 — 날짜를 알아듣지 못했습니다"


def test_conflict_with_my_events_only(world, make):
    world.c1.add("팀 회의", "2026-11-12T20:30:00", "2026-11-12T21:30:00", category="team")
    world.c1.add("할 일", "2026-11-12T20:00:00", "2026-11-12T21:00:00", is_todo=True)
    world.c1.add("하루 종일 행사", "2026-11-12", "2026-11-13", all_day=True)
    world.c1.add("끝난 뒤", "2026-11-12T22:00:00", "2026-11-12T23:00:00")
    world.others = [ev("dl:1", "과제 마감", "2026-11-12T20:00:00", "deadline", end="2026-11-12T20:30:00"),
                    ev("cl:x", "수업", "2026-11-12T20:00:00", "class", end="2026-11-12T21:00:00")]
    _, p, _ = preview_client(make)
    assert p["conflict"] == "겹치는 일정 1건 — 팀 회의 20:30~21:30"
    assert ("2026-11-12", "2026-11-12") in world.loaded


def test_conflict_all_day(world, make):
    world.c1.add("MT", "2026-11-11", "2026-11-14", all_day=True)
    _, p, _ = preview_client(make, start="2026-11-13", end="2026-11-13", allDay=True)
    assert p["conflict"] == "겹치는 일정 1건 — MT (종일)"


@pytest.mark.parametrize("eid, msg", [
    ("dl:abc", "e클래스에서 가져온 일정은 바꿀 수 없습니다"),
    ("ac:2026-1", "학사일정·수업은 바꿀 수 없습니다"),
    ("cl:os:2026-11-03:09:00", "학사일정·수업은 바꿀 수 없습니다"),
    ("ex:3", "시험 일정은 바꿀 수 없습니다"),
    ("", "어느 일정인지 찾지 못했습니다. 날짜와 제목을 함께 말씀해 주세요"),
    ("999", "어느 일정인지 찾지 못했습니다. 날짜와 제목을 함께 말씀해 주세요"),
])
@pytest.mark.parametrize("tool", ["update_event", "delete_event"])
def test_refusals(world, make, tool, eid, msg):
    args = {"id": eid, "patch": {"title": "x"}} if tool == "update_event" else {"id": eid}
    m = make(Script(call(tool, **args))).post("/api/chat", json={"text": "그거 바꿔줘"}).json()["messages"]
    assert m[-1]["role"] == "assistant" and m[-1]["text"] == msg
    assert world.c1.store.writes == []


def test_missing_id_refused(world, make):
    m = make(Script(call("delete_event"))).post("/api/chat", json={"text": "지워줘"}).json()["messages"][1]
    assert m["text"] == "어느 일정인지 찾지 못했습니다. 날짜와 제목을 함께 말씀해 주세요"


def test_write_without_calendar(make):
    m = make(Script(call("delete_event", id="1")), c1=None).post("/api/chat", json={"text": "지워줘"}).json()["messages"][1]
    assert m["text"] == "캘린더를 쓸 수 없습니다"


def test_update_moves_start_keeps_length(world, make):
    world.c1.add("스터디", "2026-11-05T19:00:00", "2026-11-05T21:00:00", category="study")
    m = make(Script(call("update_event", id="1", patch={"start": "2026-11-06T18:00"}))).post(
        "/api/chat", json={"text": "스터디 금요일 6시로 옮겨줘"}).json()["messages"][1]
    assert (m["start"], m["end"], m["category"]) == ("2026-11-06T18:00", "2026-11-06T20:00", "study")


# ---------------------------------------------------------------- 확인 · 취소

def test_confirm_create_with_edits(world, make):
    c, p, _ = preview_client(make)
    r = c.post(f"/api/chat/actions/{p['id']}/confirm", json={"title": "  발표 리허설 2차 ", "category": "team"})
    assert r.status_code == 200
    card, sysmsg = r.json()["messages"]
    assert card["id"] == p["id"] and card["state"] == "confirmed"
    assert card["title"] == "발표 리허설 2차" and card["category"] == "team" and card["categoryLabel"] == "팀플"
    assert sysmsg["role"] == "system" and sysmsg["text"] == "일정을 등록했습니다 — 발표 리허설 2차 11/12(목) 20:00"
    (op, uid, fields), = world.c1.store.writes
    assert op == "create"
    assert fields == {"title": "발표 리허설 2차", "start": "2026-11-12T20:00", "end": "2026-11-12T22:00", "all_day": False,
                      "category": "team", "memo": "", "is_todo": False, "done": False, "origin": None}
    msgs = c.get("/api/chat/messages").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "preview", "system"]
    assert msgs[1]["state"] == "confirmed"
    again = c.post(f"/api/chat/actions/{p['id']}/confirm", json={})
    assert again.status_code == 409 and again.json()["detail"] == "이미 처리된 미리보기입니다"
    assert len(world.c1.store.writes) == 1


def test_confirm_edits_revalidated(world, make):
    c, p, _ = preview_client(make)
    r = c.post(f"/api/chat/actions/{p['id']}/confirm", json={"start": "2026-02-30T10:00"})
    assert r.status_code == 422
    r = c.post(f"/api/chat/actions/{p['id']}/confirm", json={"title": ""})
    assert r.status_code == 422 and r.json()["detail"] == "제목은 1~200자입니다"
    r = c.post(f"/api/chat/actions/{p['id']}/confirm", json={"category": "hobby"})
    assert r.status_code == 422
    assert world.c1.store.writes == []
    # 날짜만 고쳐 종일로 — end 는 규칙대로 다시 만든다
    r = c.post(f"/api/chat/actions/{p['id']}/confirm", json={"start": "2026-11-13", "end": None, "allDay": True})
    card, sysmsg = r.json()["messages"]
    assert (card["start"], card["end"], card["allDay"]) == ("2026-11-13", "2026-11-14", True)
    assert sysmsg["text"] == "일정을 등록했습니다 — 발표 리허설 11/13(금)"


def test_confirm_update_and_target_gone(world, make):
    world.c1.add("팀 회의", "2026-11-03T19:00:00", "2026-11-03T20:00:00", category="team", memo="비밀 메모")
    c = make(Script(call("update_event", id="1", patch={"title": "팀 회의(온라인)"}),
                    call("update_event", id="1", patch={"start": "2026-11-03T21:00"})))
    p = c.post("/api/chat", json={"text": "팀 회의 이름 바꿔줘"}).json()["messages"][1]
    assert "memo" not in p and "비밀 메모" not in str(p)
    r = c.post(f"/api/chat/actions/{p['id']}/confirm", json={})
    assert r.status_code == 200
    assert r.json()["messages"][1]["text"] == "일정을 바꿨습니다 — 팀 회의(온라인) 11/3(화) 19:00"
    assert world.c1.store.rows[1]["title"] == "팀 회의(온라인)" and world.c1.store.rows[1]["memo"] == "비밀 메모"
    p2 = c.post("/api/chat", json={"text": "9시로 바꿔줘"}).json()["messages"][1]
    world.c1.store.rows.clear()
    r = c.post(f"/api/chat/actions/{p2['id']}/confirm", json={})
    assert r.status_code == 409 and r.json()["detail"] == "대상 일정이 사라졌습니다"


def test_confirm_delete(world, make):
    world.c1.add("스터디", "2026-11-04", "2026-11-05", all_day=True, category="study")
    c = make(Script(call("delete_event", id="1")))
    p = c.post("/api/chat", json={"text": "스터디 지워줘"}).json()["messages"][1]
    r = c.post(f"/api/chat/actions/{p['id']}/confirm", json={"title": "무시됨"})
    assert r.status_code == 200
    assert r.json()["messages"][1]["text"] == "일정을 삭제했습니다 — 스터디 11/4(수)"
    assert world.c1.store.rows == {}


def test_confirm_without_calendar_503(world, make):
    c, p, _ = preview_client(make)
    from chat import api
    api.configure(world.deps(chat=None, c1=None))
    r = c.post(f"/api/chat/actions/{p['id']}/confirm", json={})
    assert r.status_code == 503 and r.json()["detail"] == "캘린더를 쓸 수 없습니다"


def test_cancel(world, make):
    c, p, _ = preview_client(make)
    r = c.post(f"/api/chat/actions/{p['id']}/cancel")
    assert r.status_code == 200
    (card,) = r.json()["messages"]
    assert card["id"] == p["id"] and card["state"] == "canceled"
    assert c.post(f"/api/chat/actions/{p['id']}/confirm", json={}).status_code == 409
    assert world.c1.store.writes == []
    assert c.post("/api/chat/actions/pa_nothere/cancel").status_code == 404
    assert c.post("/api/chat/actions/pa_nothere/confirm", json={}).status_code == 404


def test_expiry(world, make):
    c, p, _ = preview_client(make)
    world.now = world.now + timedelta(minutes=9)
    assert c.get("/api/chat/messages").json()["messages"][1]["state"] == "pending"
    world.now = world.now + timedelta(minutes=2)
    assert c.get("/api/chat/messages").json()["messages"][1]["state"] == "expired"
    r = c.post(f"/api/chat/actions/{p['id']}/confirm", json={})
    assert r.status_code == 409 and r.json()["detail"] == "만료되었습니다 — 다시 말씀해 주세요"
    assert world.c1.store.writes == []
    r = c.post(f"/api/chat/actions/{p['id']}/cancel")
    assert r.status_code == 409 and r.json()["detail"] == "만료되었습니다 — 다시 말씀해 주세요"
    assert c.get("/api/chat/messages").json()["messages"][1]["state"] == "expired"


# ---------------------------------------------------------------- 이력 · 지우기

def test_history_paging(make):
    c = make(Script(*[{"role": "assistant", "content": f"답{i}"} for i in range(3)]))
    for i in range(3):
        c.post("/api/chat", json={"text": f"질문{i}"})
    allm = c.get("/api/chat/messages").json()["messages"]
    assert [m["text"] for m in allm] == ["질문0", "답0", "질문1", "답1", "질문2", "답2"]
    assert len({m["id"] for m in allm}) == 6
    page = c.get("/api/chat/messages", params={"before": allm[4]["id"], "limit": 2}).json()["messages"]
    assert [m["text"] for m in page] == ["질문1", "답1"]
    assert c.get("/api/chat/messages", params={"limit": 0}).status_code == 422


def test_delete_clears_everything(world, make):
    c, p, _ = preview_client(make)
    r = c.delete("/api/chat/messages")
    assert r.status_code == 204 and r.content == b""
    assert c.get("/api/chat/messages").json() == {"messages": []}
    assert c.post(f"/api/chat/actions/{p['id']}/confirm", json={}).status_code == 404


# ---------------------------------------------------------------- 동시 확인 · 실패 되돌리기 (2026-11-02 아키텍트 검토)

def _state(aid):
    from chat import store
    with store.connect() as con:
        return store.get_action(con, aid)["state"]


def test_duplicate_confirm_while_running(world, make):
    """첫 확인이 C1 에 쓰는 동안 들어온 두 번째 확인은 409 — 일정은 1건만."""
    from chat import service, store
    c, p, _ = preview_client(make)
    seen = []
    orig = world.c1.store.create_event

    def create_and_race(**f):
        assert _state(p["id"]) == "running"
        with store.connect() as con2:
            try:
                service.confirm(con2, world.deps(), p["id"], {})
            except service.Conflict as e:
                seen.append(str(e))
        return orig(**f)
    world.c1.store.create_event = create_and_race
    r = c.post(f"/api/chat/actions/{p['id']}/confirm", json={})
    assert r.status_code == 200
    assert seen == ["이미 처리된 미리보기입니다"]
    assert len(world.c1.store.rows) == 1 and _state(p["id"]) == "confirmed"


def test_confirm_claim_is_atomic(world, make, monkeypatch):
    """두 요청이 모두 pending 을 읽은 뒤라도 상태를 잡는 UPDATE 는 하나만 이긴다."""
    from chat import service, store
    c, p, _ = preview_client(make)
    orig = service._pending

    def racing(con, aid, now):
        out = orig(con, aid, now)
        with store.connect() as other:                     # 다른 요청이 먼저 잡았다
            assert store.claim_action(other, aid, now.isoformat(timespec="seconds"))
            assert not store.claim_action(other, aid, now.isoformat(timespec="seconds"))
        return out
    monkeypatch.setattr(service, "_pending", racing)
    r = c.post(f"/api/chat/actions/{p['id']}/confirm", json={})
    assert r.status_code == 409 and r.json()["detail"] == "이미 처리된 미리보기입니다"
    assert world.c1.store.rows == {}


def test_claim_respects_expiry(make):
    from chat import store
    _, p, _ = preview_client(make)
    with store.connect() as con:
        assert not store.claim_action(con, p["id"], "2026-11-02T10:10:00")     # expiresAt 과 같으면 만료
        assert store.claim_action(con, p["id"], "2026-11-02T10:09:59")


def test_confirm_failure_returns_to_pending(world, make):
    c, p, _ = preview_client(make)

    def boom(**f):
        raise RuntimeError("database is locked")
    world.c1.store.create_event = boom
    with pytest.raises(RuntimeError):
        c.post(f"/api/chat/actions/{p['id']}/confirm", json={})
    assert _state(p["id"]) == "pending"
    del world.c1.store.create_event                        # 고쳐졌다 — 다시 누르면 된다
    r = c.post(f"/api/chat/actions/{p['id']}/confirm", json={})
    assert r.status_code == 200 and len(world.c1.store.rows) == 1


def test_target_gone_returns_to_pending(world, make):
    world.c1.add("팀 회의", "2026-11-03T19:00:00", "2026-11-03T20:00:00")
    c = make(Script(call("delete_event", id="1")))
    p = c.post("/api/chat", json={"text": "팀 회의 지워줘"}).json()["messages"][1]
    world.c1.store.rows.clear()
    r = c.post(f"/api/chat/actions/{p['id']}/confirm", json={})
    assert r.status_code == 409 and r.json()["detail"] == "대상 일정이 사라졌습니다"
    assert _state(p["id"]) == "pending"


def test_cancel_after_claim_is_409(world, make):
    from chat import store
    c, p, _ = preview_client(make)
    with store.connect() as con:
        store.claim_action(con, p["id"], "2026-11-02T10:00:00")
    assert c.post(f"/api/chat/actions/{p['id']}/cancel").status_code == 409


# ---------------------------------------------------------------- ev:<숫자> id (학습 데이터 꼴)

def test_ev_prefixed_id_is_my_event(world, make):
    world.c1.add("스터디", "2026-11-05T19:00:00", "2026-11-05T21:00:00", category="study")
    world.c1.add("팀 회의", "2026-11-06T19:00:00", "2026-11-06T20:00:00", category="team")
    c = make(Script(call("update_event", id="ev:1", patch={"title": "스터디(온라인)"}), call("delete_event", id="ev:2"),
                    call("delete_event", id="ev:dl:3")))
    up = c.post("/api/chat", json={"text": "스터디 이름 바꿔줘"}).json()["messages"][1]
    assert up["role"] == "preview" and up["kind"] == "update" and up["targetId"] == "1"
    de = c.post("/api/chat", json={"text": "팀 회의 지워줘"}).json()["messages"][1]
    assert de["role"] == "preview" and de["kind"] == "delete" and de["targetId"] == "2"
    bad = c.post("/api/chat", json={"text": "지워줘"}).json()["messages"][1]
    assert bad["text"] == "어느 일정인지 찾지 못했습니다. 날짜와 제목을 함께 말씀해 주세요"
    assert c.post(f"/api/chat/actions/{up['id']}/confirm", json={}).status_code == 200
    assert c.post(f"/api/chat/actions/{de['id']}/confirm", json={}).status_code == 200
    assert world.c1.store.rows[1]["title"] == "스터디(온라인)" and 2 not in world.c1.store.rows


@pytest.mark.parametrize("eid, msg", [
    ("dl:7", "e클래스에서 가져온 일정은 바꿀 수 없습니다"),
    ("ac:7", "학사일정·수업은 바꿀 수 없습니다"),
    ("cl:7", "학사일정·수업은 바꿀 수 없습니다"),
    ("ex:7", "시험 일정은 바꿀 수 없습니다"),
    ("pb:7", "학습 블록은 공강 배치 화면에서 바꿔 주세요"),
])
def test_other_prefixes_still_refused(world, make, eid, msg):
    world.c1.add("x", "2026-11-05T19:00:00")
    m = make(Script(call("delete_event", id=eid))).post("/api/chat", json={"text": "지워줘"}).json()["messages"][1]
    assert m["role"] == "assistant" and m["text"] == msg
