#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F9 자연어 대화(Function Calling) 학습 데이터 생성 — 합성(결정적) + 외부 JSONL 병합.

  python src/prepare_f9.py                      # data/f9/train.jsonl · eval.jsonl · README.md (1500건, seed 42)
  python src/prepare_f9.py --n 3000 --seed 7
  python src/prepare_f9.py --extra 공개데이터.jsonl   # 외부 행은 train 에만 들어간다 (평가셋 고정)
  python src/prepare_f9.py --smoke              # 60건 — 파이프라인 점검용

도구 11개와 시간 해석 규칙은 WE-Meet 요구사항정의서 F9 §5 를 따른다.
  · 주 시작은 월요일 · '다음 주 X요일' = 다음 주의 X요일 · 시각 없음 → 종일 · 길이 없음 → 1시간
  · '8시' 처럼 오전/오후가 없는 시각은 애매하므로 **만들지 않는다** (백엔드가 물음표 처리)
  · 쓰기 도구는 미리보기만 · 모델은 계산하지 않고 도구 결과의 숫자만 말한다
평가셋은 **템플릿 패밀리 단위**로 떼어 낸다 — 같은 말투가 학습·평가에 함께 들어가지 않게.
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_tools, write_jsonl, ensure_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOOLS_PATH = ROOT / "tools" / "f9_tools.json"
WD = "월화수목금토일"
WD_CODE = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]

RULES = ("당신은 유니버스(Univ-Us) 앱의 일정 도우미입니다. "
         "일정 쓰기(등록·수정·삭제)는 미리보기만 만들고 실행은 사용자 확인 뒤 서버가 합니다. "
         "숫자는 직접 계산하지 않고 도구 결과에 있는 값만 말합니다. 모르면 모른다고 답합니다. "
         "e클래스 마감·학사일정·수업은 바꿀 수 없습니다.")

TITLES = [("발표 리허설", "personal"), ("스터디", "study"), ("팀 회의", "team"), ("팀플 미팅", "team"),
          ("병원", "personal"), ("치과 예약", "personal"), ("알바", "etc"), ("동아리 모임", "personal"),
          ("과외", "personal"), ("도서관 공부", "study"), ("시험 공부", "study"), ("캡스톤 회의", "team"),
          ("멘토링", "etc"), ("운동", "personal"), ("친구 약속", "personal"), ("세미나", "study")]
COURSES = ["운영체제", "자료구조", "데이터베이스", "컴퓨터네트워크", "알고리즘", "인공지능개론",
           "소프트웨어공학", "선형대수", "확률과통계", "기계학습"]
TIMES = [("아침 9시", 9, 0), ("오전 10시", 10, 0), ("오전 11시", 11, 0), ("점심 12시", 12, 0), ("오후 1시", 13, 0),
         ("오후 2시", 14, 0), ("오후 2시 반", 14, 30), ("오후 3시", 15, 0), ("오후 4시", 16, 0), ("오후 5시", 17, 0),
         ("저녁 6시", 18, 0), ("저녁 7시", 19, 0), ("저녁 7시 반", 19, 30), ("저녁 8시", 20, 0), ("밤 9시", 21, 0),
         ("밤 10시", 22, 0)]
DURATIONS = [("1시간", 60), ("2시간", 120), ("3시간", 180), ("30분", 30), ("1시간 반", 90)]
NAV = [("학사일정", "/academic"), ("졸업요건", "/graduation"), ("출결", "/attendance"), ("강의자료", "/courses"),
       ("시험", "/exams"), ("장학금", "/scholarships"), ("대외활동", "/opportunities"), ("브리핑", "/briefing"),
       ("과제", "/assignments"), ("대시보드", "/"), ("프로필 설정", "/settings/profile"), ("수집 원천 설정", "/settings/sources"),
       ("알림 설정", "/settings/notifications"), ("가용 시간 설정", "/settings/availability"), ("졸업요건 기준 설정", "/settings/requirements")]
SYNC = [("e클래스", "eclass"), ("이클래스", "eclass"), ("학사일정", "academic"), ("장학 공지", "scholarship"), ("대외활동", "activity")]
OOS = ["오늘 점심 뭐 먹을까?", "내일 날씨 어때?", "비트코인 시세 알려줘", "파이썬 코드 좀 짜줘", "요즘 볼 만한 영화 추천해줘",
       "학식 메뉴 뭐야?", "교수님 연락처 알려줘", "버스 몇 분 뒤에 와?"]

# 평가 전용 패밀리 — 여기 있는 말투는 train 에 들어가지 않는다 (가중치 기준 전체의 약 21%).
# 도구 11개가 전부 평가에 나온다 — update_event/delete_event 는 up_B/del_B 행의 **2번째 턴**(list_events 로 대상을 찾은 뒤)이고,
# eval.py 가 모든 assistant 턴을 채점하므로 도구별 표에 잡힌다. 첫 턴만 보는 집계(data/f9/README.md 의 도구 열)에는 list_events 로 센다.
EVAL_FAMILIES = {"ce_F", "le_today", "nav_goto", "gr_ask2", "at_ask2", "as_eval", "ex_ask2", "sync_refresh",
                 "pl_eval", "up_B", "del_B", "ref_class"}

# ── 날짜·시각 ────────────────────────────────────────────────

def fmt_dt(d: dt.datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M")


def fmt_d(d: dt.date) -> str:
    return d.strftime("%Y-%m-%d")


def kr_md(d: dt.date) -> str:
    return f"{d.month}/{d.day}({WD[d.weekday()]})"


def system_message(now: dt.datetime) -> str:
    return f"오늘은 {fmt_d(now.date())}({WD[now.weekday()]}) {now:%H:%M} 입니다. 주 시작은 월요일입니다.\n{RULES}"


def rand_now(rng: random.Random) -> dt.datetime:
    start = dt.date(2026, 9, 1)
    d = start + dt.timedelta(days=rng.randint(0, 110))          # ~12/20 까지
    return dt.datetime(d.year, d.month, d.day, rng.randint(8, 22), rng.choice([0, 15, 30, 45]))


def week_monday(d: dt.date) -> dt.date:
    return d - dt.timedelta(days=d.weekday())


def rel_date(rng: random.Random, now: dt.datetime) -> tuple[str, dt.date]:
    """상대 날짜 표현과 그 해석. 과거는 만들지 않는다."""
    today = now.date()
    kind = rng.choice(["today", "tomorrow", "dayafter", "this_wd", "next_wd", "n_days", "next_month", "abs"])
    if kind == "today":
        return "오늘", today
    if kind == "tomorrow":
        return "내일", today + dt.timedelta(days=1)
    if kind == "dayafter":
        return "모레", today + dt.timedelta(days=2)
    if kind == "this_wd":
        idx = rng.randint(today.weekday(), 6)
        return f"이번 주 {WD[idx]}요일", week_monday(today) + dt.timedelta(days=idx)
    if kind == "next_wd":
        idx = rng.randint(0, 6)
        return f"다음 주 {WD[idx]}요일", week_monday(today) + dt.timedelta(days=7 + idx)
    if kind == "n_days":
        n = rng.randint(2, 10)
        return f"{n}일 뒤", today + dt.timedelta(days=n)
    if kind == "next_month":
        y, m = (today.year + (today.month // 12), today.month % 12 + 1)
        day = rng.randint(1, 28)
        return f"다음 달 {day}일", dt.date(y, m, day)
    d = today + dt.timedelta(days=rng.randint(1, 60))
    return f"{d.month}월 {d.day}일", d


# ── 생성기 — (user, steps) 를 돌려준다. steps: ("call",name,args) · ("tool",result) · ("text",answer) ──
Gen = Callable[[random.Random, dt.datetime], tuple[str, list]]


def _event_args(title: str, cat: str, d: dt.date, t: tuple | None, dur: int | None, repeat: str | None = None) -> dict:
    if t is None:
        # 종일 end 는 exclusive — 하루짜리 10/20 → end 2026-10-21 (WE-Meet C1-R14)
        args = {"title": title, "start": fmt_d(d), "end": fmt_d(d + dt.timedelta(days=1)), "allDay": True, "category": cat}
    else:
        s = dt.datetime(d.year, d.month, d.day, t[1], t[2])
        args = {"title": title, "start": fmt_dt(s), "end": fmt_dt(s + dt.timedelta(minutes=dur or 60)),
                "allDay": False, "category": cat}
    if repeat:
        args["repeat"] = repeat
    return args


def future_slot(rng: random.Random, now: dt.datetime) -> tuple[str, dt.date, tuple]:
    """(날짜 표현, 날짜, 시각) — 시작 시각이 now 보다 뒤가 될 때까지 다시 뽑는다. 백엔드는 과거 날짜를 거부한다(F9 8절)."""
    for _ in range(100):
        phrase, d = rel_date(rng, now)
        t = rng.choice(TIMES)
        if dt.datetime(d.year, d.month, d.day, t[1], t[2]) > now:
            return phrase, d, t
    d = now.date() + dt.timedelta(days=1)
    return "내일", d, TIMES[-1]


def gen_create(fam: str) -> Gen:
    def g(rng, now):
        title, cat = rng.choice(TITLES)
        phrase, d, t = future_slot(rng, now)
        durp, dur = rng.choice(DURATIONS + [("", None)])
        dur_s = f" {durp}" if durp else ""
        if fam == "ce_A":
            u = f"{phrase} {title} {t[0]}에 넣어줘"
            dur = None                      # 발화에 길이가 없으면 기본 1시간 (F9 시간 해석 규칙)
        elif fam == "ce_B":
            u = f"{phrase} {t[0]} {title}{dur_s} 등록해줘"
        elif fam == "ce_C":
            u = f"{title} {phrase} {t[0]}로 잡아줘{dur_s}"
        elif fam == "ce_D":
            return f"{phrase} {title} 일정 추가해줘", [("call", "create_event", _event_args(title, cat, d, None, None))]
        elif fam == "ce_E":
            u = f"{phrase}에 {title} 있어, {t[0]}부터{dur_s}"
        elif fam == "ce_F":
            u = f"{phrase} {t[0]} {title} 일정 넣어{dur_s}"
        else:  # ce_G 반복
            idx = rng.randint(0, 6)
            first = week_monday(now.date()) + dt.timedelta(days=idx)
            if first <= now.date():
                first += dt.timedelta(days=7)
            u = f"매주 {WD[idx]}요일 {t[0]} {title}{dur_s} 넣어줘"
            return u, [("call", "create_event", _event_args(title, cat, first, t, dur, f"weekly:{WD_CODE[idx]}"))]
        return u, [("call", "create_event", _event_args(title, cat, d, t, dur))]
    return g


def gen_list_events(fam: str, two_step: bool = False) -> Gen:
    def g(rng, now):
        today = now.date()
        mon = week_monday(today)
        if fam in ("le_week_deadline", "ts_le_A", "ts_le_B"):
            u = rng.choice(["이번 주 마감 뭐 남았어?", "이번 주에 내야 할 거 뭐 있어?"]) if fam != "ts_le_B" else "이번 주 마감 알려줘"
            args = {"from": fmt_d(mon), "to": fmt_d(mon + dt.timedelta(days=6)), "kinds": ["deadline"]}
        elif fam == "le_tomorrow":
            u = "내일 일정 뭐 있어?"
            args = {"from": fmt_d(today + dt.timedelta(days=1)), "to": fmt_d(today + dt.timedelta(days=1))}
        elif fam == "le_next_week":
            u = rng.choice(["다음 주 일정 보여줘", "다음 주에 뭐 있지?"])
            args = {"from": fmt_d(mon + dt.timedelta(days=7)), "to": fmt_d(mon + dt.timedelta(days=13))}
        elif fam == "le_next_deadline":
            u = "다음 주 마감 있어?"
            args = {"from": fmt_d(mon + dt.timedelta(days=7)), "to": fmt_d(mon + dt.timedelta(days=13)), "kinds": ["deadline"]}
        else:  # le_today
            u = rng.choice(["오늘 뭐 해야 돼?", "오늘 일정 알려줘"])
            args = {"from": fmt_d(today), "to": fmt_d(today)}
        steps = [("call", "list_events", args)]
        if two_step:
            items = []
            for i in range(rng.randint(0, 3)):
                c = rng.choice(COURSES)
                d = mon + dt.timedelta(days=rng.randint(max(0, today.weekday()), 6))
                items.append({"id": f"dl:{rng.randint(100, 999)}", "title": f"{c} 과제 {rng.randint(1, 5)}", "kind": "deadline",
                              "due": fmt_dt(dt.datetime(d.year, d.month, d.day, 23, 59))})
            steps.append(("tool", {"items": items}))
            if not items:
                ans = "이번 주에 남은 마감이 없습니다. 캘린더 열기(/)"
            else:
                ans = f"이번 주 마감 {len(items)}건: " + ", ".join(
                    f"{it['title']}({kr_md(dt.date.fromisoformat(it['due'][:10]))} 23:59)" for it in items) + ". 캘린더 열기(/)"
            steps.append(("text", ans))
        return u, steps
    return g


def gen_graduation(fam: str, two_step: bool = False) -> Gen:
    def g(rng, now):
        u = {"gr_ask": rng.choice(["졸업까지 몇 학점 남았어?", "남은 학점 알려줘"]),
             "gr_ask2": rng.choice(["졸업요건 얼마나 채웠어?", "나 졸업 가능해?"]),
             "ts_gr_A": "졸업까지 몇 학점 남았어?", "ts_gr_B": "졸업 학점 얼마나 남았지?"}[fam]
        steps = [("call", "get_graduation", {})]
        if two_step:
            total = rng.choice([130, 140])
            earned = rng.randint(60, total - 3)
            steps.append(("tool", {"earned": earned, "totalRequired": total, "remaining": total - earned, "status": "진행 중"}))
            steps.append(("text", f"졸업까지 {total - earned}학점 남았습니다 (이수 {earned} / {total}). 졸업요건 열기(/graduation)"))
        return u, steps
    return g


def gen_attendance(fam: str, two_step: bool = False) -> Gen:
    def g(rng, now):
        c = rng.choice(COURSES)
        if fam == "at_all":
            return "출결 위험한 과목 있어?", [("call", "get_attendance", {})]
        u = {"at_ask": f"{c} 결석 몇 번 했어?", "at_ask2": f"{c} 출석 상태 어때?",
             "ts_at_A": f"{c} 결석 몇 번이야?", "ts_at_B": f"{c} 몇 번 더 빠지면 위험해?"}[fam]
        steps = [("call", "get_attendance", {"course": c})]
        if two_step:
            limit = rng.choice([4, 5])
            absent = rng.randint(0, limit - 1)
            risky = absent >= limit - 1
            steps.append(("tool", {"course": c, "absent": absent, "limit": limit, "risky": risky}))
            ans = f"{c}는 결석 {absent}회, 한도 {limit}회입니다." + (" 한도에 가까워 주의가 필요합니다." if risky else "") + " 출결 열기(/attendance)"
            steps.append(("text", ans))
        return u, steps
    return g


def gen_assignments(fam: str, two_step: bool = False) -> Gen:
    def g(rng, now):
        spec = {"as_now": ("지금 해야 할 과제 뭐야?", {"group": "now"}), "as_week": ("이번 주 과제 뭐 있어?", {"group": "this_week"}),
                "as_all": ("과제 우선순위 보여줘", {}), "as_ask2": ("나중에 해도 되는 과제는?", {"group": "later"}),
                "as_eval": (rng.choice(["급한 과제부터 보여줘", "제일 먼저 할 과제가 뭐야?"]), {"group": "now"}),
                "ts_as_A": ("지금 해야 할 과제 알려줘", {"group": "now"}), "ts_as_B": ("당장 급한 과제 뭐야?", {"group": "now"})}
        u, args = spec[fam]
        steps = [("call", "list_assignments", args)]
        if two_step:
            items = []
            for _ in range(rng.randint(1, 3)):
                d = now + dt.timedelta(days=rng.randint(1, 6))
                items.append({"title": f"{rng.choice(COURSES)} 과제 {rng.randint(1, 5)}",
                              "due": fmt_dt(d.replace(hour=23, minute=59)), "hours": rng.randint(1, 5)})
            steps.append(("tool", {"group": "now", "items": items}))
            ans = f"지금 해야 할 과제 {len(items)}건: " + ", ".join(
                f"{it['title']}({kr_md(dt.date.fromisoformat(it['due'][:10]))} 23:59 마감, 예상 {it['hours']}시간)" for it in items) + ". 과제 열기(/assignments)"
            steps.append(("text", ans))
        return u, steps
    return g


def gen_exams(fam: str, two_step: bool = False) -> Gen:
    def g(rng, now):
        spec = {"ex_next": ("다음 시험 언제야?", {"upcoming": True}), "ex_plan": ("시험 공부 계획 어떻게 돼?", {}),
                "ex_ask2": ("다가오는 시험 알려줘", {"upcoming": True}), "ts_ex_A": ("다음 시험 언제야?", {"upcoming": True}),
                "ts_ex_B": ("시험 뭐 남았어?", {"upcoming": True})}
        u, args = spec[fam]
        steps = [("call", "get_exams", args)]
        if two_step:
            c, kind = rng.choice(COURSES), rng.choice(["중간고사", "기말고사"])
            d = now.date() + dt.timedelta(days=rng.randint(3, 30))
            steps.append(("tool", {"items": [{"course": c, "title": kind, "date": fmt_d(d)}]}))
            steps.append(("text", f"다음 시험은 {c} {kind}, {kr_md(d)}입니다. 시험 열기(/exams)"))
        return u, steps
    return g


def gen_sync(fam: str, two_step: bool = False) -> Gen:
    def g(rng, now):
        name, src = rng.choice(SYNC)
        u = {"sync_do": f"{name} 동기화해줘", "sync_refresh": f"{name} 새로 받아와", "sync_again": f"{name} 다시 수집해줘",
             "ts_sync_A": f"{name} 동기화해줘", "ts_sync_B": f"{name} 지금 받아줘"}[fam]
        steps = [("call", "run_sync", {"source": src})]
        if two_step:
            state = rng.choice(["started", "started", "running"])
            steps.append(("tool", {"source": src, "state": state}))
            steps.append(("text", f"{name} 수집을 시작했습니다." if state == "started" else f"{name} 수집이 이미 진행 중입니다."))
        return u, steps
    return g


def gen_placement(fam: str) -> Gen:
    def g(rng, now):
        if fam == "pl_this":
            return rng.choice(["공강에 배치해줘", "이번 주 공강에 공부 계획 짜줘"]), [("call", "open_placement", {"range": "this_week"})]
        if fam == "pl_plain":
            return "공강 학습 플랜 열어줘", [("call", "open_placement", {})]
        if fam == "pl_eval":
            return rng.choice(["비는 시간에 공부 계획 넣어줘", "이번 주 빈 시간 채워줘"]), [("call", "open_placement", {"range": "this_week"})]
        return "다음 주 공강 배치 열어줘", [("call", "open_placement", {"range": "next_week"})]
    return g


def gen_navigate(fam: str) -> Gen:
    def g(rng, now):
        name, route = rng.choice(NAV)
        u = {"nav_show": f"{name} 보여줘", "nav_open": f"{name} 화면 열어줘", "nav_goto": f"{name} 페이지로 가줘"}[fam]
        return u, [("call", "navigate", {"route": route})]
    return g


def gen_update(fam: str) -> Gen:
    def g(rng, now):
        title, cat = rng.choice(TITLES)
        phrase, d, t = future_slot(rng, now)                      # 옮길 시각도 미래여야 한다
        u = f"{phrase} {title} {t[0]}로 옮겨줘" if fam == "up_A" else f"{phrase} {title} 시간을 {t[0]}로 바꿔줘"
        ev = {"id": f"ev:{rng.randint(1000, 9999)}", "title": title, "kind": "user",
              "start": fmt_dt(dt.datetime(d.year, d.month, d.day, rng.choice(TIMES)[1], 0))}
        s = dt.datetime(d.year, d.month, d.day, t[1], t[2])
        return u, [("call", "list_events", {"from": fmt_d(d), "to": fmt_d(d)}), ("tool", {"items": [ev]}),
                   ("call", "update_event", {"id": ev["id"], "patch": {"start": fmt_dt(s), "end": fmt_dt(s + dt.timedelta(hours=1))}})]
    return g


def gen_delete(fam: str) -> Gen:
    def g(rng, now):
        title, _ = rng.choice(TITLES)
        phrase, d = rel_date(rng, now)
        u = f"{phrase} {title} 취소해줘" if fam == "del_A" else f"{phrase} {title} 일정 지워줘"
        ev = {"id": f"ev:{rng.randint(1000, 9999)}", "title": title, "kind": "user",
              "start": fmt_dt(dt.datetime(d.year, d.month, d.day, rng.choice(TIMES)[1], 0))}
        return u, [("call", "list_events", {"from": fmt_d(d), "to": fmt_d(d)}), ("tool", {"items": [ev]}),
                   ("call", "delete_event", {"id": ev["id"]})]
    return g


def gen_refusal(fam: str) -> Gen:
    def g(rng, now):
        c = rng.choice(COURSES)
        if fam == "ref_eclass":
            u = rng.choice([f"{c} 과제 마감 {rel_date(rng, now)[0]}로 바꿔줘", f"{c} 과제 마감일 좀 미뤄줘", "e클래스 퀴즈 마감 지워줘"])
            return u, [("text", "e클래스에서 가져온 일정은 바꿀 수 없습니다. 마감은 e클래스에서 직접 확인해 주세요.")]
        if fam == "ref_class":
            u = rng.choice([f"{c} 수업 {rng.choice(TIMES)[0]}로 옮겨줘", "다음 주 수강신청 기간 삭제해줘", f"{c} 수업 일정 지워줘"])
            return u, [("text", "학사일정·수업은 바꿀 수 없습니다. 학교에서 가져온 일정이라 원본을 따릅니다.")]
        return rng.choice(OOS), [("text", "그건 제가 알 수 없습니다. 일정 등록·조회, 졸업요건·출결·과제 확인, 화면 이동을 도와드릴 수 있어요.")]
    return g


# (가중치, 패밀리, 생성기) — 가중치 합 ≈ 1.0
GENERATORS: list[tuple[float, str, Gen]] = []


def _add(w: float, fams: list[str], make: Callable[[str], Gen]) -> None:
    for f in fams:
        GENERATORS.append((w / len(fams), f, make(f)))


_add(0.22, ["ce_A", "ce_B", "ce_C", "ce_D", "ce_E", "ce_F", "ce_G"], gen_create)
_add(0.08, ["le_week_deadline", "le_tomorrow", "le_next_week", "le_next_deadline", "le_today"], lambda f: gen_list_events(f))
_add(0.08, ["nav_show", "nav_open", "nav_goto"], gen_navigate)
_add(0.03, ["gr_ask", "gr_ask2"], lambda f: gen_graduation(f))
_add(0.04, ["at_ask", "at_all", "at_ask2"], lambda f: gen_attendance(f))
_add(0.04, ["as_now", "as_week", "as_all", "as_ask2", "as_eval"], lambda f: gen_assignments(f))
_add(0.03, ["ex_next", "ex_plan", "ex_ask2"], lambda f: gen_exams(f))
_add(0.04, ["sync_do", "sync_refresh", "sync_again"], lambda f: gen_sync(f))
_add(0.04, ["pl_this", "pl_plain", "pl_next", "pl_eval"], gen_placement)
_add(0.04, ["up_A", "up_B"], gen_update)
_add(0.03, ["del_A", "del_B"], gen_delete)
_add(0.10, ["ref_eclass", "ref_class", "ref_oos"], gen_refusal)
_add(0.04, ["ts_gr_A", "ts_gr_B"], lambda f: gen_graduation(f, True))
_add(0.04, ["ts_at_A", "ts_at_B"], lambda f: gen_attendance(f, True))
_add(0.04, ["ts_as_A", "ts_as_B"], lambda f: gen_assignments(f, True))
_add(0.03, ["ts_ex_A", "ts_ex_B"], lambda f: gen_exams(f, True))
_add(0.04, ["ts_le_A", "ts_le_B"], lambda f: gen_list_events(f, True))
_add(0.04, ["ts_sync_A", "ts_sync_B"], lambda f: gen_sync(f, True))


def build_row(idx: int, fam: str, gen: Gen, rng: random.Random, tools: list[dict]) -> dict:
    now = rand_now(rng)
    user, steps = gen(rng, now)
    msgs: list[dict] = [{"role": "system", "content": system_message(now)}, {"role": "user", "content": user}]
    k = 0
    first_tool = "none"
    for s in steps:
        if s[0] == "call":
            k += 1
            if first_tool == "none":
                first_tool = s[1]
            msgs.append({"role": "assistant", "tool_calls": [{"id": f"call_{k}", "type": "function",
                                                              "function": {"name": s[1], "arguments": json.dumps(s[2], ensure_ascii=False)}}]})
        elif s[0] == "tool":
            msgs.append({"role": "tool", "tool_call_id": f"call_{k}", "content": json.dumps(s[1], ensure_ascii=False)})
        else:
            msgs.append({"role": "assistant", "content": s[1]})
    kind = "refusal" if steps[0][0] == "text" else ("two_step" if len(steps) > 1 else "single")
    return {"id": f"f9-{idx:05d}", "messages": msgs, "tools": tools,
            "meta": {"tool": first_tool, "template": fam, "kind": kind, "source": "synthetic"}}


def load_extra(paths: list[str], tools: list[dict]) -> list[dict]:
    """외부 JSONL → 우리 행 형식. 형식: ours / {messages,tools?} / {instruction,input,output}."""
    rows: list[dict] = []
    ours = {t["function"]["name"] for t in tools}
    for p in paths:
        for i, line in enumerate(Path(p).read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            r = json.loads(line)
            if isinstance(r.get("messages"), list):
                msgs = r["messages"]
                tl = r.get("tools") or tools
                names = {tc["function"]["name"] for m in msgs for tc in (m.get("tool_calls") or []) if "function" in tc}
                if not r.get("tools") and not names <= ours:
                    continue          # 우리 도구 밖을 부르는데 도구 정의가 없으면 쓸 수 없다
                rows.append({"id": f"extra-{Path(p).stem}-{i:05d}", "messages": msgs, "tools": tl,
                             "meta": {"tool": next(iter(names), "none"), "template": "extra", "kind": "extra", "source": str(p)}})
            elif r.get("instruction") and r.get("output"):
                user = r["instruction"] + (f"\n\n{r['input']}" if r.get("input") else "")
                rows.append({"id": f"extra-{Path(p).stem}-{i:05d}", "tools": tools,
                             "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": str(r["output"])}],
                             "meta": {"tool": "none", "template": "extra", "kind": "extra", "source": str(p)}})
    return rows


def write_readme(out: Path, train: list[dict], ev: list[dict], seed: int, n: int, extra: int) -> None:
    def counts(rows):
        c: dict[str, int] = {}
        for r in rows:
            c[r["meta"]["tool"]] = c.get(r["meta"]["tool"], 0) + 1
        return c
    ct, ce = counts(train), counts(ev)
    kinds = {}
    for r in train + ev:
        kinds[r["meta"]["kind"]] = kinds.get(r["meta"]["kind"], 0) + 1
    lines = ["# data/f9 — F9 Function Calling 학습 데이터", "",
             f"> 생성: {dt.datetime.now():%Y-%m-%d %H:%M} · seed {seed} · 합성 {n}건 + 외부 {extra}건", "",
             "**합성 데이터**입니다. 공개 데이터를 합치려면 `python src/prepare_f9.py --extra 파일.jsonl` (train 에만 들어감).",
             "평가셋은 템플릿 패밀리 단위로 분리돼 있으며 **학습에 쓰지 않습니다.** seed 를 바꾸면 이전 실험과 비교할 수 없습니다.", "",
             "| 도구 | train | eval |", "| --- | ---: | ---: |"]
    for k in sorted(set(ct) | set(ce)):
        lines.append(f"| {k} | {ct.get(k, 0)} | {ce.get(k, 0)} |")
    lines += [f"| **합계** | **{len(train)}** | **{len(ev)}** |", "",
              "| 종류 | 건수 |", "| --- | ---: |"] + [f"| {k} | {v} |" for k, v in sorted(kinds.items())]
    lines += ["", f"평가 전용 패밀리: {', '.join(sorted(EVAL_FAMILIES))}"]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "f9"))
    ap.add_argument("--n", type=int, default=1500, help="합성 데이터 총 수 (train+eval)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra", nargs="*", default=[], help="외부 JSONL — train 에만 병합")
    ap.add_argument("--tools", default=str(TOOLS_PATH))
    ap.add_argument("--smoke", action="store_true", help="60건만 빠르게")
    args = ap.parse_args()
    if args.smoke:
        args.n = 60

    tools, _ = load_tools(args.tools)
    rng = random.Random(args.seed)
    weights = [w for w, _, _ in GENERATORS]
    train, ev = [], []
    for i in range(args.n):
        w_idx = rng.choices(range(len(GENERATORS)), weights=weights, k=1)[0]
        _, fam, gen = GENERATORS[w_idx]
        row = build_row(i, fam, gen, rng, tools)
        (ev if fam in EVAL_FAMILIES else train).append(row)
    extra = load_extra(args.extra, tools)
    train += extra
    rng.shuffle(train)

    out = ensure_dir(args.out)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "eval.jsonl", ev)
    write_readme(out, train, ev, args.seed, args.n, len(extra))
    print(f"생성 완료 → {out}  train {len(train)}건 (외부 {len(extra)}) · eval {len(ev)}건 · 도구 {len(tools)}개")


if __name__ == "__main__":
    main()
