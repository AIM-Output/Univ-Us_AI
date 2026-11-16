"""POST /api/chat — LLM 루프 · 읽기 템플릿 · 실행 · 이동 · 오류."""
import json
import urllib.error

from conftest import NOW, Script, call, calls, ev, text


def post(c, t):
    return c.post("/api/chat", json={"text": t})


# ---------------------------------------------------------------- 텍스트 · 요청 모양

def test_text_only_reply_and_request_shape(make):
    llm = Script(text("그건 도와드릴 수 없습니다. 일정·과제·출결을 물어보세요."))
    r = post(make(llm), "오늘 날씨 어때?")
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "그건 도와드릴 수 없습니다. 일정·과제·출결을 물어보세요."
    user, bot = body["messages"]
    assert user["role"] == "user" and user["text"] == "오늘 날씨 어때?" and user["id"].startswith("u_")
    assert user["at"] == "2026-11-02T10:00"
    assert bot["role"] == "assistant" and bot["text"] == body["reply"]
    (msgs, tools), = llm.seen
    assert [m["role"] for m in msgs] == ["system", "user"]                     # 단일 턴
    assert msgs[0]["content"].startswith("오늘은 2026-11-02(월) 10:00 입니다. 주 시작은 월요일입니다.\n당신은")
    assert msgs[1]["content"] == "오늘 날씨 어때?"
    assert len(tools) == 11 and all("x-kind" not in t for t in tools)


def test_history_is_not_sent_to_llm(make):
    llm = Script(text("하나"), text("둘"))
    c = make(llm)
    post(c, "첫 질문")
    post(c, "두 번째 질문")
    assert [m["role"] for m in llm.seen[1][0]] == ["system", "user"]
    msgs = c.get("/api/chat/messages").json()["messages"]
    assert [m["text"] for m in msgs] == ["첫 질문", "하나", "두 번째 질문", "둘"]


def test_empty_text_422(make):
    c = make(Script())
    assert post(c, "   ").status_code == 422
    assert c.post("/api/chat", json={}).status_code == 422
    assert c.get("/api/chat/messages").json()["messages"] == []


def test_unknown_tool_or_bad_arguments_asks_again(make):
    bad_json = {"role": "assistant", "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "list_events", "arguments": "{from: 오늘"}}]}
    c = make(Script(call("delete_everything"), bad_json))
    assert post(c, "a").json()["messages"][1]["text"] == "다시 말씀해 주세요"
    assert post(c, "b").json()["messages"][1]["text"] == "다시 말씀해 주세요"


# ---------------------------------------------------------------- list_events 템플릿

def week_world(world):
    world.others = [
        ev("dl:1", "퀴즈 2회", "2026-11-02T23:29:00", "deadline", end="2026-11-02T23:59:00",
           due="2026-11-02T23:59:00", courseShort="컴퓨터네트워크"),
        ev("dl:2", "3주차 실습", "2026-11-03T23:29:00", "deadline", end="2026-11-03T23:59:00",
           due="2026-11-03T23:59:00", courseShort="운영체제"),
        ev("dl:3", "끝낸 과제", "2026-11-04T23:29:00", "deadline", end="2026-11-04T23:59:00",
           due="2026-11-04T23:59:00", courseShort="운영체제", done=True),
        ev("ac:9", "수강 정정", "2026-11-05", "academic", end="2026-11-07", all_day=True),
        ev("ex:4", "중간고사", "2026-11-06T10:00:00", "exam", end="2026-11-06T11:00:00", course="운영체제"),
        ev("ac:old", "어제 끝난 종일", "2026-11-01", "academic", end="2026-11-02", all_day=True),
    ]
    world.c1.add("팀 회의", "2026-11-04T20:00:00", "2026-11-04T21:00:00", category="team")


def test_list_events_deadlines_template(world, make):
    week_world(world)
    llm = Script(call("list_events", **{"from": "2026-11-02", "to": "2026-11-08", "kinds": ["deadline"]}))
    m = post(make(llm), "이번 주 마감 뭐 남았어?").json()["messages"][1]
    assert m["role"] == "assistant"
    assert m["text"] == "11/2(월)~11/8(일) 마감 2건"
    assert m["list"] == [
        {"title": "퀴즈 2회", "when": "오늘 23:59", "sub": "마감 · 컴퓨터네트워크", "href": "/assignments"},
        {"title": "3주차 실습", "when": "내일 23:59", "sub": "마감 · 운영체제", "href": "/assignments"},
    ]
    assert m["link"] == {"label": "과제 목록 열기", "href": "/assignments"}
    assert world.loaded[-1] == ("2026-11-02", "2026-11-08")


def test_list_events_all_kinds_exclusive_end_and_hrefs(world, make):
    week_world(world)
    llm = Script(call("list_events", **{"from": "2026-11-02", "to": "2026-11-08"}))
    m = post(make(llm), "이번 주 일정 보여줘").json()["messages"][1]
    titles = [x["title"] for x in m["list"]]
    assert "어제 끝난 종일" not in titles                         # 종일 end(11/2) 는 exclusive — from 11/2 에 걸리지 않는다
    assert titles == ["퀴즈 2회", "3주차 실습", "팀 회의", "수강 정정", "중간고사"]
    assert m["text"] == "11/2(월)~11/8(일) 일정 5건"
    by = {x["title"]: x for x in m["list"]}
    assert by["팀 회의"] == {"title": "팀 회의", "when": "11/4(수) 20:00", "sub": "내 일정", "href": "/?event=1"}
    assert by["수강 정정"]["href"] == "/academic" and by["수강 정정"]["when"] == "11/5(목)"
    assert by["수강 정정"]["sub"] == "학사일정"
    assert by["중간고사"]["href"] == "/exams" and by["중간고사"]["sub"] == "시험 · 운영체제"
    assert m["link"] == {"label": "캘린더 열기", "href": "/"}


def test_list_events_empty_and_single_day(world, make):
    llm = Script(call("list_events", **{"from": "2026-11-03", "to": "2026-11-03"}))
    m = post(make(llm), "내일 일정 뭐 있어?").json()["messages"][1]
    assert m["text"] == "11/3(화)에는 일정이 없습니다"
    assert "list" not in m and m["link"]["href"] == "/"


def test_list_events_bad_dates(make):
    m = post(make(Script(call("list_events", **{"from": "다음주", "to": "2026-11-03"}))), "x").json()["messages"][1]
    assert m["text"] == "다시 말씀해 주세요 — 날짜를 알아듣지 못했습니다"


def test_read_without_write_intent_calls_llm_once(world, make):
    week_world(world)
    llm = Script(call("list_events", **{"from": "2026-11-04", "to": "2026-11-04"}))
    post(make(llm), "수요일 일정 알려줘")
    assert len(llm.seen) == 1


# ---------------------------------------------------------------- 2-step (list_events → update/delete)

def test_two_step_update_feeds_tool_result(world, make):
    world.c1.add("팀 회의", "2026-11-03T19:00:00", "2026-11-03T20:00:00", category="team")
    llm = Script(call("list_events", **{"from": "2026-11-03", "to": "2026-11-03"}),
                 call("update_event", id="1", patch={"start": "2026-11-03T20:00", "end": "2026-11-03T21:00"}))
    body = post(make(llm), "내일 팀 회의 8시로 옮겨줘").json()
    assert len(llm.seen) == 2
    second = llm.seen[1][0]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    tc = second[2]["tool_calls"][0]
    assert tc["id"] == "call_1" and tc["function"]["name"] == "list_events"
    assert json.loads(tc["function"]["arguments"]) == {"from": "2026-11-03", "to": "2026-11-03"}
    assert second[3]["tool_call_id"] == "call_1"
    assert json.loads(second[3]["content"]) == {"items": [
        {"id": "1", "title": "팀 회의", "kind": "user", "start": "2026-11-03T19:00:00", "end": "2026-11-03T20:00:00"}]}
    p = body["messages"][1]
    assert p["role"] == "preview" and p["kind"] == "update" and p["targetId"] == "1"
    assert (p["start"], p["end"], p["title"], p["category"]) == ("2026-11-03T20:00", "2026-11-03T21:00", "팀 회의", "team")
    assert p["before"] == {"title": "팀 회의", "start": "2026-11-03T19:00", "end": "2026-11-03T20:00",
                           "allDay": False, "category": "team"}
    assert "conflict" not in p                                  # 자기 자신과는 겹치지 않는다
    assert [w[0] for w in world.c1.store.writes] == ["create"]     # 테스트 준비로 넣은 1건뿐 — 미리보기는 쓰지 않는다


def test_two_step_delete(world, make):
    world.c1.add("스터디", "2026-11-04", "2026-11-05", all_day=True, category="study")
    llm = Script(call("list_events", **{"from": "2026-11-04", "to": "2026-11-04"}), call("delete_event", id="1"))
    p = post(make(llm), "수요일 스터디 취소해줘").json()["messages"][1]
    assert p["kind"] == "delete" and p["targetId"] == "1"
    assert (p["title"], p["start"], p["end"], p["allDay"], p["categoryLabel"]) == ("스터디", "2026-11-04", "2026-11-05", True, "학업")


def test_two_step_text_falls_back_to_template(world, make):
    week_world(world)
    llm = Script(call("list_events", **{"from": "2026-11-04", "to": "2026-11-04"}), text("찾지 못했습니다"))
    m = post(make(llm), "수요일 회의 지워줘").json()["messages"][1]
    assert len(llm.seen) == 2
    assert m["role"] == "assistant" and m["text"] == "11/4(수) 일정 1건"


def test_two_step_caps_llm_calls_at_three(world, make):
    rd = call("list_events", **{"from": "2026-11-04", "to": "2026-11-04"})
    llm = Script(rd, rd, rd, rd)
    post(make(llm), "수요일 회의 삭제")
    assert len(llm.seen) == 3


def test_multiple_writes_stack_cards(make):
    llm = Script(calls(("create_event", {"title": "A", "start": "2026-11-10T10:00", "end": "2026-11-10T11:00",
                                         "allDay": False, "category": "etc"}),
                       ("delete_event", {"id": "dl:5"})))
    out = post(make(llm), "A 넣고 과제 마감 지워줘").json()["messages"]
    assert [m["role"] for m in out] == ["user", "preview", "assistant"]
    assert out[2]["text"] == "e클래스에서 가져온 일정은 바꿀 수 없습니다"


# ---------------------------------------------------------------- 다른 읽기 도구

def test_graduation(world, make):
    m = post(make(Script(call("get_graduation"))), "졸업까지 몇 학점?").json()["messages"][1]
    assert m["text"] == "졸업까지 32학점 남았습니다 (98/130) — 전공필수 2과목"
    assert m["link"] == {"label": "졸업요건 열기", "href": "/graduation"}
    world.grad = {"available": False}
    m = post(make(Script(call("get_graduation"))), "졸업?").json()["messages"][1]
    assert m["text"].startswith("졸업요건 데이터가 아직 없습니다")


def test_attendance(world, make):
    world.courses = [
        {"id": "c1", "name": "운영체제(01)", "short": "운영체제", "excluded": False,
         "summary": {"remaining": 2.0, "level": "caution", "levelLabel": "주의"}},
        {"id": "c2", "name": "컴퓨터네트워크", "short": "컴퓨터네트워크", "excluded": False,
         "summary": {"remaining": -1, "level": "over", "levelLabel": "초과"}},
        {"id": "c3", "name": "제외 과목", "short": "제외", "excluded": True, "summary": {}},
    ]
    m = post(make(Script(call("get_attendance", course="운영 체제"))), "운영체제 결석").json()["messages"][1]
    assert m["text"] == "운영체제 출결 — 주의 · 남은 결석 2회"
    assert m["list"] == [{"title": "운영체제", "when": "주의", "sub": "남은 결석 2회", "href": "/attendance?course=c1"}]
    m = post(make(Script(call("get_attendance"))), "출결").json()["messages"][1]
    assert m["text"] == "출결 2과목" and m["list"][1]["sub"] == "남은 결석 0회"
    m = post(make(Script(call("get_attendance", course="미적분"))), "미적분").json()["messages"][1]
    assert m["text"] == "'미적분' 과목을 찾지 못했습니다" and m["link"]["href"] == "/attendance"


def test_assignments(world, make):
    world.ranked = [
        {"id": "a1", "title": "보고서", "courseShort": "소공", "kindLabel": "과제", "due": "2026-11-02T23:59:00", "group": "now"},
        {"id": "a2", "title": "퀴즈", "courseShort": "OS", "kindLabel": "퀴즈", "due": "2026-11-05T18:00:00", "group": "week"},
        {"id": "a3", "title": "오래 지난", "group": "overdue", "stale": True},
    ]
    m = post(make(Script(call("list_assignments", group="this_week"))), "이번 주 과제").json()["messages"][1]
    assert m["text"] == "이번 주 1건"
    assert m["list"] == [{"title": "퀴즈", "when": "11/5(목) 18:00", "sub": "OS · 퀴즈", "href": "/assignments"}]
    m = post(make(Script(call("list_assignments"))), "과제").json()["messages"][1]
    assert m["text"] == "급한 순 2건" and m["list"][0]["when"] == "오늘 23:59"


def test_exams(world, make):
    world.exam_view = {"exams": [{"id": 3, "title": "중간고사", "course": "운영체제", "date": "2026-11-06", "time": "10:00"},
                                 {"id": 4, "title": "기말고사", "course": "소공", "date": "2026-12-15", "time": ""}],
                       "past": []}
    m = post(make(Script(call("get_exams", upcoming=True))), "시험 언제야").json()["messages"][1]
    assert m["text"] == "다가오는 시험 2건"
    assert m["list"][0] == {"title": "중간고사", "when": "11/6(금) 10:00", "sub": "운영체제", "href": "/exams?exam=3"}
    assert m["list"][1]["when"] == "12/15(화)"
    m = post(make(Script(call("get_exams", upcoming=False))), "지난 시험").json()["messages"][1]
    assert m["text"] == "지난 시험이 없습니다"


# ---------------------------------------------------------------- 실행 · 이동

def test_run_sync(world, make):
    out = post(make(Script(call("run_sync", source="eclass"))), "e클래스 동기화해줘").json()
    m = out["messages"][1]
    assert m["role"] == "system" and m["text"] == "e클래스 수집을 시작했습니다" and world.sync_calls == 1
    assert out["reply"] == "e클래스 수집을 시작했습니다"
    world.sync_result = {"running": True, "already_running": True}
    m = post(make(Script(call("run_sync", source="eclass"))), "동기화").json()["messages"][1]
    assert m["text"] == "이미 수집 중입니다"
    m = post(make(Script(call("run_sync", source="academic"))), "학사 동기화").json()["messages"][1]
    assert m["role"] == "assistant" and m["text"] == "지원하지 않는 동기화 대상입니다"
    assert world.sync_calls == 2


def test_open_placement(make):
    m = post(make(Script(call("open_placement", range="next_week"))), "공강에 배치해줘").json()["messages"][1]
    assert m["text"] == "공강 배치 미리보기를 열어 둘게요"
    assert m["link"] == {"label": "공강 배치 미리보기 열기", "href": "/?place=preview"}


def test_navigate_whitelist(make):
    m = post(make(Script(call("navigate", route="/academic"))), "학사일정 보여줘").json()["messages"][1]
    assert m["text"] == "학사일정 화면으로 이동할게요"
    assert m["link"] == {"label": "학사일정 열기", "href": "/academic"}
    m = post(make(Script(call("navigate", route="/settings/profile/"))), "프로필").json()["messages"][1]
    assert m["link"]["href"] == "/settings/profile"
    for bad in ("/admin", "https://evil.example", "//evil"):
        m = post(make(Script(call("navigate", route=bad))), "x").json()["messages"][1]
        assert m["text"] == "그 화면은 없습니다" and "link" not in m


# ---------------------------------------------------------------- LLM 오류

def test_503_when_llm_not_configured(make):
    c = make(None)                                          # Deps.chat 없음 → llm.complete, 슬롯 비어 있음
    r = post(c, "내일 일정")
    assert r.status_code == 503 and r.json()["detail"] == "분석 서비스가 설정되지 않았습니다"
    assert c.get("/api/chat/status").json() == {"available": True, "llmConfigured": False}
    msgs = c.get("/api/chat/messages").json()["messages"]
    assert [m["text"] for m in msgs] == ["내일 일정"]            # 사용자 메시지는 남는다


def _configure(monkeypatch):
    from chat import config as C
    monkeypatch.setattr(C, "LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setattr(C, "LLM_API_KEY", "k")
    monkeypatch.setattr(C, "LLM_MODEL", "univus-f9")


def test_502_on_url_error(make, monkeypatch):
    from chat import llm
    _configure(monkeypatch)

    def boom(req, timeout):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(llm.urllib.request, "urlopen", boom)
    c = make(None)
    r = post(c, "내일 일정")
    assert r.status_code == 502
    assert r.json()["detail"] == "분석 서비스에 연결할 수 없습니다 — 터널(8600)과 모델 서버를 확인하세요"
    assert c.get("/api/chat/status").json() == {"available": True, "llmConfigured": True, "model": "univus-f9"}


def test_502_on_auth_failure(make, monkeypatch):
    from chat import llm
    _configure(monkeypatch)

    def denied(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)
    monkeypatch.setattr(llm.urllib.request, "urlopen", denied)
    r = post(make(None), "내일 일정")
    assert r.status_code == 502 and r.json()["detail"] == "분석 서비스 인증에 실패했습니다 — 키를 확인하세요"


def test_llm_complete_request_and_parse(monkeypatch):
    """실제 호출 경로 — URL · 헤더 · 본문 · 타임아웃 · choices[0].message."""
    from chat import config as C
    from chat import llm
    _configure(monkeypatch)
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"finish_reason": "tool_calls", "message": call("get_graduation")}]}).encode()

    def fake(req, timeout):
        seen.update(url=req.full_url, auth=req.get_header("Authorization"), body=json.loads(req.data), timeout=timeout)
        return Resp()
    monkeypatch.setattr(llm.urllib.request, "urlopen", fake)
    msg = llm.complete([{"role": "user", "content": "x"}], [{"type": "function"}])
    assert msg["tool_calls"][0]["function"]["name"] == "get_graduation"
    assert seen["url"] == "http://127.0.0.1:9/v1/chat/completions" and seen["auth"] == "Bearer k"
    assert seen["body"]["temperature"] == 0 and seen["body"]["max_tokens"] == 256 and seen["body"]["model"] == "univus-f9"
    assert seen["body"]["tools"] == [{"type": "function"}]
    assert seen["timeout"] == C.LLM_TIMEOUT == 5.0


def test_llm_off_switch(monkeypatch):
    from chat import config as C
    _configure(monkeypatch)
    assert C.llm_configured()
    monkeypatch.setenv("F9_LLM", "off")
    assert not C.llm_configured()


def test_now_is_used_for_system_prompt(world, make):
    from datetime import datetime
    world.now = datetime(2026, 11, 8, 21, 5)
    llm = Script(text("네"))
    post(make(llm), "안녕")
    assert llm.seen[0][0][0]["content"].startswith("오늘은 2026-11-08(일) 21:05 입니다.")
    assert NOW.weekday() == 0
