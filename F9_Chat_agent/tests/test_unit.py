"""순수 함수 — system 문구 · 도구 목록 · 값 검증 · 표시."""
from datetime import date, datetime

import pytest

from conftest import NOW


def test_system_prompt_matches_training_text():
    from chat import service
    expected = ("오늘은 2026-11-02(월) 10:00 입니다. 주 시작은 월요일입니다.\n"
                "당신은 유니버스(Univ-Us) 앱의 일정 도우미입니다. 일정 쓰기(등록·수정·삭제)는 미리보기만 만들고 "
                "실행은 사용자 확인 뒤 서버가 합니다. 숫자는 직접 계산하지 않고 도구 결과에 있는 값만 말합니다. "
                "모르면 모른다고 답합니다. e클래스 마감·학사일정·수업은 바꿀 수 없습니다.")
    assert service.system_prompt(NOW) == expected
    assert service.system_prompt(datetime(2026, 11, 8, 21, 5)).startswith("오늘은 2026-11-08(일) 21:05 입니다.")


def test_tools_loader_strips_extension_fields():
    from chat import tools as T
    specs, kinds = T.load()
    assert len(specs) == 11
    assert all("x-kind" not in t and "_note" not in t for t in specs)
    assert all(t["type"] == "function" and "name" in t["function"] for t in specs)
    assert kinds == {"list_events": "read", "get_graduation": "read", "get_attendance": "read",
                     "list_assignments": "read", "get_exams": "read", "create_event": "write",
                     "update_event": "write", "delete_event": "write", "run_sync": "run",
                     "open_placement": "run", "navigate": "navigate"}
    # 원본(학습에 쓴 파일)은 그대로 남아 있다
    assert "x-kind" in T._raw()["tools"][0]


@pytest.mark.parametrize("fields, want", [
    ({"title": " 발표 ", "start": "2026-11-12T20:00"}, ("발표", "2026-11-12T20:00", "2026-11-12T21:00", False)),
    ({"title": "a", "start": "2026-11-12T20:00:00", "end": "2026-11-12T22:00"}, ("a", "2026-11-12T20:00", "2026-11-12T22:00", False)),
    ({"title": "a", "start": "2026-11-12T20:00", "end": "2026-11-12T19:00"}, ("a", "2026-11-12T20:00", "2026-11-12T21:00", False)),
    ({"title": "a", "start": "2026-11-12T23:30"}, ("a", "2026-11-12T23:30", "2026-11-13T00:30", False)),
    ({"title": "a", "start": "2026-11-12"}, ("a", "2026-11-12", "2026-11-13", True)),
    ({"title": "a", "start": "2026-11-12", "end": "2026-11-12"}, ("a", "2026-11-12", "2026-11-13", True)),
    ({"title": "a", "start": "2026-11-12", "end": "2026-11-14"}, ("a", "2026-11-12", "2026-11-14", True)),
    ({"title": "a", "start": "2026-11-12T09:00", "allDay": True}, ("a", "2026-11-12", "2026-11-13", True)),
])
def test_normalize_rules(fields, want):
    from chat import service
    f = service.normalize(fields)
    assert (f["title"], f["start"], f["end"], f["allDay"]) == want
    assert f["category"] == "personal"


@pytest.mark.parametrize("fields, what", [
    ({"title": "a", "start": "2026-02-30"}, "date"),
    ({"title": "a", "start": "내일"}, "date"),
    ({"title": "a", "start": "2026-11-12T25:00"}, "date"),
    ({"title": "a", "start": "2026-11-12", "end": "모레"}, "date"),
    ({"title": "  ", "start": "2026-11-12"}, "title"),
    ({"title": "x" * 201, "start": "2026-11-12"}, "title"),
])
def test_normalize_rejects(fields, what):
    from chat import service
    with pytest.raises(service._Bad) as e:
        service.normalize(fields)
    assert e.value.what == what


def test_when_strings():
    from chat import service
    t = date(2026, 11, 2)
    assert service.when("2026-11-06", True, t) == "11/6(금)"
    assert service.when("2026-11-06T18:00:00", False, t) == "11/6(금) 18:00"
    assert service.when("2026-11-02T23:59", False, t) == "오늘 23:59"
    assert service.when("2026-11-03T18:00", False, t) == "내일 18:00"
    assert service.when("2026-11-02", True, t) == "오늘"
    assert service.due_when("2026-11-04T00:00:00", t) == "내일 24:00"
    assert service.due_when(None, t) == "마감 없음"
    assert service.span_text("2026-11-12", "2026-11-14", True) == "11/12(목)~11/13(금)"
    assert service.span_text("2026-11-12T20:00", "2026-11-12T21:00", False) == "11/12(목) 20:00"


def test_refusal_by_prefix():
    from chat import service
    assert service.refusal("dl:123") == "e클래스에서 가져온 일정은 바꿀 수 없습니다"
    assert service.refusal("ac:2026-1") == "학사일정·수업은 바꿀 수 없습니다"
    assert service.refusal("cl:os:2026-11-02:09:00") == "학사일정·수업은 바꿀 수 없습니다"
    assert service.refusal("ex:7") == "시험 일정은 바꿀 수 없습니다"
    assert service.refusal(None) == service.refusal("") == "어느 일정인지 찾지 못했습니다. 날짜와 제목을 함께 말씀해 주세요"
    assert service.refusal("ev:1234") is None                      # 학습 데이터 꼴 — 숫자 id 로 읽는다
    assert service.refusal("ev:abc") == "어느 일정인지 찾지 못했습니다. 날짜와 제목을 함께 말씀해 주세요"
    assert service.refusal("ev:dl:1") == "어느 일정인지 찾지 못했습니다. 날짜와 제목을 함께 말씀해 주세요"
    assert service.user_id_text(" ev:42 ") == "42" and service.user_id_text("dl:42") == "dl:42"
    assert service.refusal("12") is None
