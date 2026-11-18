# -*- coding: utf-8 -*-
"""data_src 도구(make_real · prepare_f9_v3 · peek_*) 자가 점검 — 모델·GPU·네트워크 없이 돈다.

  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import common  # noqa: E402
import eval as f9_eval  # noqa: E402  — score_turn 만 쓴다(모델은 main 안에서만 불러온다)
import make_real as mr  # noqa: E402
import prepare_f9_v2 as v2  # noqa: E402
import prepare_f9_v3 as v3  # noqa: E402
from datasrc import Report, describe_call, row_user  # noqa: E402

N = "2026-11-18 14:00"
TOOLS, _ = common.load_tools(ROOT / "tools" / "f9_tools.json")
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
EMPTY = v3.Blocks(set(), set(), set())


def write_tsv(folder: Path, name: str, header: list[str], rows: list[tuple]) -> Path:
    p = folder / name
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    return p


def check_real(rows: list[tuple], name: str = "subin.tsv") -> tuple[list, Report]:
    with tempfile.TemporaryDirectory() as d:
        p = write_tsv(Path(d), name, mr.HEADER, rows)
        rep = Report()
        return mr.load_items([p], SCHEMAS, rep), rep


def check_frames(rows: list[tuple], name: str = "subin.tsv") -> tuple[list, Report]:
    with tempfile.TemporaryDirectory() as d:
        p = write_tsv(Path(d), name, v3.HEADER, rows)
        rep = Report()
        frames = v3.load_frames([p], rep)
        v3.trial(frames, TOOLS, SCHEMAS, EMPTY, 11, rep)
        return frames, rep


class RealSentences(unittest.TestCase):
    def test_example_file_passes(self):
        rep = Report()
        items = mr.load_items([ROOT / "data_src" / "examples" / "real_example.tsv"], SCHEMAS, rep)
        self.assertEqual(rep.errors, [])
        self.assertEqual(rep.warnings, [])
        self.assertEqual(len(items), 34)

    def test_empty_personal_files_are_valid(self):
        rep = Report()
        files = sorted((ROOT / "data_src" / "real").glob("*.tsv"))
        self.assertEqual([f.stem for f in files], ["haram", "jiuk", "subin"])
        mr.load_items(files, SCHEMAS, rep)
        self.assertEqual(rep.errors, [])

    def test_errors_are_caught(self):
        cases = [
            ("내일 3시에 치과 예약 넣어줘", 'create_event {"title": "치과 예약", "start": "2026-11-19T15:00", "end": "2026-11-19T16:00", "allDay": false, "category": "personal"}', "오전·오후 없는"),
            ("졸업요건 열어줘", 'navigate {"route": "/graduate"}', "쓸 수 없습니다"),
            ("금요일 하루 종일 MT", 'create_event {"title": "MT", "start": "2026-11-20", "end": "2026-11-20", "allDay": true, "category": "personal"}', "다음 날"),
            ("매주 화요일 저녁 7시 스터디", 'create_event {"title": "스터디", "start": "2026-11-25T19:00", "end": "2026-11-25T20:00", "allDay": false, "category": "study", "repeat": "weekly:TU"}', "요일을 맞춰"),
            ("내일 병원 취소해줘", 'delete_event {"id": "1"}', "첫 단계"),
            ("어제 오후 3시 회의 넣어줘", 'create_event {"title": "회의", "start": "2026-11-17T15:00", "end": "2026-11-17T16:00", "allDay": false, "category": "team"}', "기준시각"),
            ("다음 시험 언제야", 'get_exams {"upcoming": "true"}', "따옴표 없이"),
            ("이번 주 시험 뭐 있어", 'list_events {"from": "2026-11-16", "to": "2026-11-22", "kinds": []}', "하나 이상"),
            ("내일 오후 3시 치과", "create_event {'title': '치과'}", "작은따옴표"),
            ("출결 보여줘", "navigate /attendance", "도구이름 {JSON 인자}"),
            ("내일 오후 3시 회의", 'create_event {"title": "회의", "start": "2026-11-19 15:00", "end": "2026-11-19 16:00", "allDay": false, "category": "team"}', "YYYY-MM-DDTHH:MM"),
            ("오늘 종일 워크숍", 'create_event {"title": "워크숍", "start": "2026-11-18", "end": "2026-11-19", "allDay": true, "category": "etc"}', "기준시각"),
            ("내 번호 010-1234-5678 로 알림 줘", "거절", "개인정보"),
        ]
        for i, (user, answer, needle) in enumerate(cases, 1):
            items, rep = check_real([(f"sb-{i:03d}", N, user, answer, "")])
            self.assertEqual(items, [], user)
            self.assertTrue(any(needle in e for e in rep.errors), f"{user!r}: {needle!r} not in {rep.errors}")

    def test_format_errors(self):
        _, rep = check_real([("sb-001", "2026.11.18 오후 2:00", "출결 보여줘", 'navigate {"route": "/attendance"}', "")])
        self.assertTrue(any("일반 텍스트" in e for e in rep.errors))
        _, rep = check_real([("ju-001", N, "출결 보여줘", 'navigate {"route": "/attendance"}', "")])
        self.assertTrue(any("sb-" in e for e in rep.errors))
        _, rep = check_real([("sb-001", N, "출결 보여줘", 'navigate {"route": "/attendance"}', ""),
                             ("sb-001", N, "브리핑 열어", 'navigate {"route": "/briefing"}', "")])
        self.assertTrue(any("이미 있습니다" in e for e in rep.errors))

    def test_warnings_use_the_generators_date_rules(self):
        cases = [
            ("내일 오후 3시에 치과 예약 넣어줘", 'create_event {"title": "치과 예약", "start": "2026-11-20T15:00", "end": "2026-11-20T16:00", "allDay": false, "category": "personal"}', "읽은 날짜"),
            ("내일 오후 2시 팀 회의 30분", 'create_event {"title": "팀 회의", "start": "2026-11-19T14:00", "end": "2026-11-19T15:00", "allDay": false, "category": "team"}', "30분"),
            ("이번 주 마감 뭐 있어", 'list_events {"from": "2026-11-16", "to": "2026-11-22"}', "kinds"),
            ("다음 주 일정 알려줘", 'list_events {"from": "2026-11-16", "to": "2026-11-22"}', "읽은 기간"),
            ("운영체제 결석 몇 번이야", 'get_attendance {"course": "운체"}', "문장에 그대로"),
        ]
        for i, (user, answer, needle) in enumerate(cases, 1):
            items, rep = check_real([(f"sb-{i:03d}", N, user, answer, "")])
            self.assertEqual(rep.errors, [], user)
            self.assertEqual(len(items), 1)
            self.assertTrue(any(needle in w for w in rep.warnings), f"{user!r}: {needle!r} not in {rep.warnings}")

    def test_split_never_changes(self):
        # id → dev/test 가 바뀌면 모의고사 문제가 최종 시험으로 넘어간다. 이 값은 절대 바뀌면 안 된다.
        want = {"sb-001": "test", "sb-002": "dev", "sb-003": "dev", "ju-001": "dev", "ju-002": "dev",
                "hr-001": "test", "hr-002": "dev", "ex-001": "dev"}
        self.assertEqual({k: mr.split_of(k) for k in want}, want)

    def test_rows_are_scored_by_eval(self):
        rep = Report()
        items = mr.load_items([ROOT / "data_src" / "examples" / "real_example.tsv"], SCHEMAS, rep)
        for it in items:
            row = mr.to_row(it, TOOLS)
            self.assertEqual(list(row), ["id", "messages", "tools", "meta"])
            self.assertTrue(row["messages"][0]["content"].startswith("오늘은 2026-11-18(수) 14:00 입니다."))
            last = row["messages"][-1]
            exp = common.expected_call(last)
            if it.tool is None:
                self.assertIsNone(exp)
                ok = f9_eval.score_turn(None, "그건 제가 알 수 없습니다.", refusal=True)
                self.assertTrue(ok["text_ok"])
                self.assertFalse(f9_eval.score_turn(None, "네, 도와드리겠습니다.", refusal=True)["text_ok"])
            else:
                perfect = common.tool_call_text(it.tool, it.args)
                self.assertTrue(f9_eval.score_turn(exp, perfect)["exact_ok"], it.user)

    def test_describe_shows_weekdays(self):
        self.assertEqual(describe_call("list_events", {"from": "2026-11-16", "to": "2026-11-22", "kinds": ["deadline"]}),
                         "list_events 11/16(월)~11/22(일) · deadline")
        self.assertIn("11/26(목) 20:00~22:00", describe_call("create_event", {
            "title": "발표 리허설", "start": "2026-11-26T20:00", "end": "2026-11-26T22:00", "allDay": False, "category": "personal"}))


class Frames(unittest.TestCase):
    def test_example_frames_pass(self):
        rep = Report()
        frames = v3.load_frames([ROOT / "data_src" / "examples" / "frames_example.tsv"], rep)
        v3.trial(frames, TOOLS, SCHEMAS, EMPTY, 11, rep)
        self.assertEqual(rep.errors, [])
        self.assertEqual(len(frames), 25)

    def test_every_kind_has_an_example(self):
        rep = Report()
        frames = v3.load_frames([ROOT / "data_src" / "examples" / "frames_example.tsv"], rep)
        self.assertEqual({f.kind for f in frames}, set(v3.KINDS))

    def test_syntax_errors(self):
        cases = [
            ("grad", "졸업 언제 해?", "없습니다"),
            ("gr", "{C} 졸업 가능해?", "쓸 수 없는 빈칸"),
            ("at_course", "결석 몇 번 했어?", "꼭 있어야"),
            ("nav", "{S|를} 열어줘", "쓸 수 없는 조사"),
            ("nav", "{S} [열어줘|켜줘", "닫는 ]"),
            ("sync", "{S} 받아와|가져와", "세로줄"),
            ("pl", "{RG} 공강 [채워줘]", "둘 이상"),
            ("ce_weekly", "매주 {W} {X} [넣어줘|등록]", "{T}"),
            ("gr", "졸업까지 몇 학점 남았[어|지|더라]?", "이미 똑같은 틀"),
        ]
        for i, (kind, tpl, needle) in enumerate(cases, 1):
            frames, rep = check_frames([(f"sb-f{i:03d}", kind, tpl, "")])
            self.assertEqual(frames, [], tpl)
            self.assertTrue(any(needle in e for e in rep.errors), f"{tpl!r}: {needle!r} not in {rep.errors}")

    def test_frames_that_make_wrong_data_are_rejected(self):
        frames, rep = check_frames([("sb-f001", "ce_timed", "내일 말고 {D} {T|에} {X} [넣어줘|추가해줘]", "")])
        self.assertEqual(frames, [])
        self.assertTrue(any("다시 읽은" in e for e in rep.errors), rep.errors)
        frames, rep = check_frames([("sb-f002", "ce_timed", "{D} {T|에} {X} 3시까지 [할 거야|한다] 넣어줘", "")])
        self.assertEqual(frames, [])
        self.assertTrue(any("오전·오후 없는" in e for e in rep.errors), rep.errors)

    def test_generation_is_deterministic_and_clean(self):
        rep = Report()
        frames = v3.load_frames([ROOT / "data_src" / "examples" / "frames_example.tsv"], rep)
        v3.trial(frames, TOOLS, SCHEMAS, EMPTY, 11, rep)
        a_rows, a_tags, _ = v3.build(frames, 300, 11, TOOLS, SCHEMAS, EMPTY)
        b_rows, _b, _ = v3.build(frames, 300, 11, TOOLS, SCHEMAS, EMPTY)
        self.assertEqual(json.dumps(a_rows, ensure_ascii=False), json.dumps(b_rows, ensure_ascii=False))
        self.assertGreaterEqual(len(a_rows), 280)
        for r, t in zip(a_rows, a_tags):
            self.assertEqual(v3.verify_row(r, t, SCHEMAS), [])
            self.assertEqual(r["meta"]["source"], "synthetic-v3")
            self.assertEqual(r["tools"], TOOLS)
            self.assertFalse(v2.bare_hour_hits(row_user(r)))
        self.assertEqual(a_rows[0]["id"], "f9v3-00001")

    def test_generated_sentences_avoid_test_sentences(self):
        rep = Report()
        frames, rep = check_frames([("sb-f001", "gr", "졸업하려면 얼마나 더 [들어야 해|들어야 돼]?", "")])
        self.assertEqual(rep.errors, [])
        block = {"졸업하려면 얼마나 더 들어야 해?"}
        blocks = v3.Blocks(block, {v2.norm_key(u) for u in block}, set())
        rows, _t, _s = v3.build(frames, 40, 11, TOOLS, SCHEMAS, blocks)
        self.assertTrue(rows)
        for r in rows:
            self.assertNotIn(v2.norm_key(row_user(r)), blocks.norm)


class PeekTools(unittest.TestCase):
    def test_peek_fail_reads_eval_json(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import peek_fail
        good = {"turn_type": "tool", "expected_tool": "navigate", "exact_ok": True, "text_ok": None, "id": "a#t0", "user": "출결 열어줘",
                "expected": {"name": "navigate", "arguments": {"route": "/attendance"}},
                "pred_text": '<tool_call>{"name": "navigate", "arguments": {"route": "/attendance"}}</tool_call>'}
        bad = dict(good, id="b#t0", exact_ok=False, user="내일 뭐 있어",
                   expected={"name": "list_events", "arguments": {"from": "2026-11-19", "to": "2026-11-19"}}, expected_tool="list_events",
                   pred_text='<tool_call>{"name": "list_events", "arguments": {"from": "2026-11-18", "to": "2026-11-19"}}</tool_call>')
        refusal = {"turn_type": "refusal", "expected_tool": None, "exact_ok": False, "text_ok": False, "id": "c#t0", "user": "수업 지워줘",
                   "expected": "(거절)", "pred_text": '<tool_call>{"name": "list_events", "arguments": {"from": "2026-11-18", "to": "2026-11-18"}}</tool_call>'}
        self.assertFalse(peek_fail.failed(good))
        self.assertTrue(peek_fail.failed(bad) and peek_fail.failed(refusal))
        self.assertIn("from", peek_fail.diff_fields(bad))
        self.assertEqual(peek_fail.expected_text(refusal), "거절")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "eval_real_dev.json"
            p.write_text(json.dumps({"summary": {"label": "t", "turn_acc": 33.3}, "results": [good, bad, refusal]}, ensure_ascii=False), encoding="utf-8")
            buf = io.StringIO()
            old = sys.argv
            sys.argv = ["peek_fail.py", str(p)]
            try:
                with redirect_stdout(buf):
                    self.assertEqual(peek_fail.main(), 0)
            finally:
                sys.argv = old
            text = buf.getvalue()
            self.assertIn("틀린 턴 2개", text)
            self.assertIn("| 거절 | list_events | 1 |", text)


if __name__ == "__main__":
    unittest.main()
