# -*- coding: utf-8 -*-
"""날짜 규칙(scripts/date_rule.py) 자가 점검 — 모델·GPU·네트워크 없이 돈다.

  python -m unittest discover -s tests -v

규칙을 앱(F9_Chat_agent)으로 옮길 때 같은 경우를 그대로 시험하면 된다.
"""
from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import date_rule as dr  # noqa: E402


def D(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def create(start: str, end: str, all_day: bool = False, **more) -> dict:
    return {"name": "create_event", "arguments": {"title": "회의", "start": start, "end": end, "allDay": all_day, "category": "team", **more}}


class CreateEvent(unittest.TestCase):
    def test_n_days_fixes_month_end(self):
        # 10-22(목) 기준 '9일 뒤' = 10-31. 모델은 11-01 로 답했다(f9-r1 의 실제 오답)
        new, what = dr.apply_rule("9일 뒤 오전 10시 멘토링 일정 넣어 1시간", D("2026-10-22"), create("2026-11-01T10:00", "2026-11-01T11:00"))
        self.assertEqual(what, "changed")
        self.assertEqual((new["arguments"]["start"], new["arguments"]["end"]), ("2026-10-31T10:00", "2026-10-31T11:00"))
        self.assertEqual(new["arguments"]["title"], "회의")          # 날짜 말고는 건드리지 않는다

    def test_next_week_weekday_from_sunday(self):
        # 10-25(일) 기준 '다음 주 토요일' = 10-31 (주 시작은 월요일). 모델은 11-07
        new, what = dr.apply_rule("다음 주 토요일 저녁 8시 멘토링 일정 넣어", D("2026-10-25"), create("2026-11-07T20:00", "2026-11-07T21:00"))
        self.assertEqual(what, "changed")
        self.assertEqual(new["arguments"]["start"], "2026-10-31T20:00")

    def test_two_weeks_later_weekday(self):
        new, what = dr.apply_rule("다다음 주 화요일 아침 열 시에 학과 세미나 있어", D("2026-11-16"), create("2026-11-30T10:00", "2026-11-30T11:00"))
        self.assertEqual((what, new["arguments"]["start"], new["arguments"]["end"]), ("changed", "2026-12-01T10:00", "2026-12-01T11:00"))

    def test_date_not_on_calendar(self):
        # 모델이 달력에 없는 11월 31일을 답해도 고친다
        new, what = dr.apply_rule("4일 뒤 오전 11시 알바 일정 넣어", D("2026-11-27"), create("2026-11-31T11:00", "2026-11-31T12:00"))
        self.assertEqual((what, new["arguments"]["start"], new["arguments"]["end"]), ("changed", "2026-12-01T11:00", "2026-12-01T12:00"))

    def test_all_day_keeps_span(self):
        # 종일 일정의 end 는 다음 날(exclusive) — 간격을 그대로 옮긴다
        new, what = dr.apply_rule("내일 종일 워크숍", D("2026-11-18"), create("2026-11-20", "2026-11-21", all_day=True))
        self.assertEqual((what, new["arguments"]["start"], new["arguments"]["end"]), ("changed", "2026-11-19", "2026-11-20"))

    def test_already_right_is_same(self):
        call = create("2026-11-19T15:00", "2026-11-19T17:00")
        new, what = dr.apply_rule("내일 오후 3시 스터디 2시간", D("2026-11-18"), call)
        self.assertEqual(what, "same")
        self.assertEqual(new, call)

    def test_untouched_cases(self):
        today = D("2026-11-18")
        for user, want in (("매주 화요일 저녁 7시 스터디 반복으로 넣어줘", "skip:반복"),
                           ("수요일마다 점심 12시에 팀플 회의 있어", "skip:반복"),
                           ("다음 주 월요일부터 수요일까지 종일 학회 출장", "skip:날짜 2개 이상"),
                           ("저녁 8시에 회의 넣어줘", "none")):
            call = create("2026-11-24T19:00", "2026-11-24T20:00")
            new, what = dr.apply_rule(user, today, call)
            self.assertEqual(what, want, user)
            self.assertEqual(new, call, user)


class ListEvents(unittest.TestCase):
    def test_single_day(self):
        call = {"name": "list_events", "arguments": {"from": "2026-11-20", "to": "2026-11-20", "kinds": ["deadline"]}}
        new, what = dr.apply_rule("내일 마감 뭐 있어?", D("2026-11-18"), call)
        self.assertEqual(what, "changed")
        self.assertEqual(new["arguments"], {"from": "2026-11-19", "to": "2026-11-19", "kinds": ["deadline"]})

    def test_week_range(self):
        # 11-16(월) 기준 '다다음 주' = 11-30 ~ 12-06. 모델은 한 주 더 뒤로 잡았다(f9-r1 의 실제 오답)
        call = {"name": "list_events", "arguments": {"from": "2026-12-07", "to": "2026-12-13", "kinds": ["exam"]}}
        new, what = dr.apply_rule("다다음 주 시험 뭐 있어?", D("2026-11-16"), call)
        self.assertEqual((what, new["arguments"]["from"], new["arguments"]["to"]), ("changed", "2026-11-30", "2026-12-06"))

    def test_month_range(self):
        call = {"name": "list_events", "arguments": {"from": "2026-12-01", "to": "2026-12-30"}}
        new, what = dr.apply_rule("다음 달 일정 보여줘", D("2026-11-18"), call)
        self.assertEqual((what, new["arguments"]["from"], new["arguments"]["to"]), ("changed", "2026-12-01", "2026-12-31"))

    def test_no_date_words(self):
        call = {"name": "list_events", "arguments": {"from": "2026-11-18", "to": "2026-11-24"}}
        self.assertEqual(dr.apply_rule("일정 뭐 있어?", D("2026-11-18"), call), (call, "none"))


class UpdateAndOthers(unittest.TestCase):
    def test_move_to_another_day(self):
        today = D("2026-11-18")
        user = "내일 스터디를 모레로 옮겨줘"
        # 첫 단계: 지금 일정이 있는 날(첫 날짜)을 찾는다
        new, what = dr.apply_rule(user, today, {"name": "list_events", "arguments": {"from": "2026-11-20", "to": "2026-11-20"}})
        self.assertEqual((what, new["arguments"]["from"]), ("changed", "2026-11-19"))
        # 둘째 단계: 옮겨 갈 날(둘째 날짜)로 patch 를 고친다
        call = {"name": "update_event", "arguments": {"id": "ev:1", "patch": {"start": "2026-11-21T19:00", "end": "2026-11-21T21:00"}}}
        new, what = dr.apply_rule(user, today, call)
        self.assertEqual(what, "changed")
        self.assertEqual(new["arguments"], {"id": "ev:1", "patch": {"start": "2026-11-20T19:00", "end": "2026-11-20T21:00"}})

    def test_time_change_is_left_alone(self):
        # 날짜 표현이 1개(시각만 바꾸는 수정)면 patch 는 건드리지 않는다 — '하루 미뤄줘' 같은 문장에서 틀리지 않게
        call = {"name": "update_event", "arguments": {"id": "ev:1", "patch": {"start": "2026-11-19T20:00", "end": "2026-11-19T21:00"}}}
        self.assertEqual(dr.apply_rule("내일 동아리 모임 시간을 저녁 8시로 바꿔줘", D("2026-11-18"), call), (call, "none"))

    def test_other_tools(self):
        for call in ({"name": "navigate", "arguments": {"route": "/academic"}},
                     {"name": "open_placement", "arguments": {}},
                     {"name": "delete_event", "arguments": {"id": "ev:1"}}):
            self.assertEqual(dr.apply_rule("내일 학사일정 보여줘", D("2026-11-18"), call), (call, "none"))


class Scoring(unittest.TestCase):
    def test_today_of(self):
        msgs = [{"role": "system", "content": "오늘은 2026-11-18(수) 14:00 입니다. 주 시작은 월요일입니다.\n규칙…"}, {"role": "user", "content": "안녕"}]
        self.assertEqual(dr.today_of(msgs), D("2026-11-18"))

    def test_rescored_turn(self):
        exp = {"name": "create_event", "arguments": {"title": "멘토링", "start": "2026-10-31T10:00", "end": "2026-10-31T11:00", "allDay": False, "category": "etc"}}
        pred = '<tool_call>{"name": "create_event", "arguments": {"allDay": false, "category": "etc", "end": "2026-11-01T11:00", "start": "2026-11-01T10:00", "title": "멘토링"}}</tool_call>'
        s = {"expected": exp, "pred_text": pred, "turn_type": "tool", "exact_ok": False, "text_ok": None}
        ok, what, text = dr.rescored(s, "9일 뒤 오전 10시 멘토링 일정 넣어 1시간", D("2026-10-22"))
        self.assertEqual((ok, what), (True, "changed"))
        self.assertIn("2026-10-31T10:00", text)
        # 글로 답한 턴(거절)은 그대로
        s2 = {"expected": "거절", "pred_text": "학사일정·수업은 바꿀 수 없습니다.", "turn_type": "refusal", "exact_ok": False, "text_ok": True}
        self.assertEqual(dr.rescored(s2, "다음 주 수강신청 기간 삭제해줘", D("2026-10-22"))[:2], (True, "none"))

    def test_real_test_is_always_hidden(self):
        self.assertTrue(dr.is_test_file("data/f9/real_test.jsonl"))
        self.assertFalse(dr.is_test_file("data/f9/real_dev.jsonl"))


if __name__ == "__main__":
    unittest.main()
