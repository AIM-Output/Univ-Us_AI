#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F9 추가 학습 데이터 v2 — 대학생 말투 수백 종 × 날짜·시각·제목 슬롯을 결정적으로 조합한다.

  python src/prepare_f9_v2.py                                   # data/f9/extra_v2.jsonl + extra_v2.README.md (10,000행, seed 7)
  python src/prepare_f9_v2.py --n 10000 --seed 7 --out data/f9
  python src/prepare_f9.py --n 1500 --seed 42 --extra data/f9/extra_v2.jsonl   # train 에 병합 (eval.jsonl 은 그대로)

행 형식·system 문구·거절 문장은 prepare_f9 와 바이트 단위로 같다(system_message 등을 import 해서 쓴다).
정답 날짜·시각은 전부 now 에서 코드로 계산한다. 쓰기(create/update)의 start 는 항상 now 보다 뒤.
'1~12시' 는 반드시 오전/오후/아침/점심/낮/저녁/밤 과 함께만 나온다(끝 시각은 앞 시각의 시간대를 상속할 수 있음).
끝에서 validate() 가 품질 게이트(사양 §6)를 assert 로 확인하고, 그 수치를 README 에 적는다.
표준 라이브러리만 쓰고, 난수는 random.Random(seed) 하나뿐이다.
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, NamedTuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REFUSAL_MARKERS, load_tools, write_jsonl  # noqa: E402
import prepare_f9  # noqa: E402
from prepare_f9 import (RULES, TOOLS_PATH, WD, WD_CODE, fmt_d, fmt_dt, kr_md,  # noqa: E402
                        system_message, week_monday)

ROOT = Path(__file__).resolve().parents[1]
D1 = dt.timedelta(days=1)

# ── 고정 답 문장 (거절 3종은 prepare_f9 와 글자 하나 다르지 않게 — validate 에서 원본 소스와 대조) ──
REF_ECLASS = "e클래스에서 가져온 일정은 바꿀 수 없습니다. 마감은 e클래스에서 직접 확인해 주세요."
REF_CLASS = "학사일정·수업은 바꿀 수 없습니다. 학교에서 가져온 일정이라 원본을 따릅니다."
REF_OOS = "그건 제가 알 수 없습니다. 일정 등록·조회, 졸업요건·출결·과제 확인, 화면 이동을 도와드릴 수 있어요."
CHIT_HELLO = "안녕하세요! 일정 등록·조회, 졸업요건·출결·과제 확인, 화면 이동을 도와드릴 수 있어요."
CHIT_THANKS = "네, 더 필요한 게 있으면 말씀해 주세요."


def not_found_text(d: dt.date, title: str) -> str:
    return f"{kr_md(d)}에 '{title}' 일정을 찾지 못했습니다. 날짜와 제목을 함께 말씀해 주세요."


# ── 의도별 목표 건수 (n=10,000 기준, 사양 §4) ──────────────────
TARGETS = {"create": 2200, "list_events": 1500, "navigate": 800, "run_sync": 400, "get_graduation": 300,
           "get_attendance": 400, "get_exams": 300, "list_assignments": 450, "open_placement": 600,
           "update": 500, "delete": 400, "refusal_eclass": 400, "refusal_class": 600, "refusal_oos": 600,
           "chit": 150, "two_step_read": 400}
MIN_FAMILIES = {"create": 45, "list_events": 40, "navigate": 25, "run_sync": 20, "get_graduation": 25,
                "get_attendance": 25, "get_exams": 20, "list_assignments": 25, "open_placement": 35,
                "update": 25, "delete": 20, "refusal_eclass": 50, "refusal_class": 60, "refusal_oos": 60}
SURFACE_P = 0.35           # §5 — 문장의 약 35% 에 표면 변형 1~2개
MAX_DUP = 5                # §6-2 — 같은 사용자 문장 최대 빈도

# ════════════════════════════════════════════════════════════════
# 1. 슬롯 뱅크 — 과목·제목(분류 고정)·학사 행사
# ════════════════════════════════════════════════════════════════

COURSES = ["운영체제", "자료구조", "데이터베이스", "컴퓨터네트워크", "알고리즘", "인공지능개론", "소프트웨어공학",
           "선형대수", "확률과통계", "기계학습", "컴퓨터구조", "이산수학", "프로그래밍언어론", "웹프로그래밍",
           "모바일프로그래밍", "캡스톤디자인", "정보보호개론", "영어회화", "경영학원론", "미시경제학", "회계원리",
           "심리학개론", "일반물리학", "미적분학", "데이터마이닝", "컴파일러"]

# (제목, 분류, 플래그)  w=매주 반복에 어울림 · m=여러 날 종일에 어울림 · o=한 번짜리(반복 X)
FIXED_TITLES = [
    # study
    ("스터디", "study", "w"), ("알고리즘 스터디", "study", "w"), ("도서관 공부", "study", "w"), ("시험 공부", "study", "w"),
    ("코딩 테스트 준비", "study", "w"), ("토익 공부", "study", "w"), ("토익 스피킹 연습", "study", "w"),
    ("논문 읽기", "study", "w"), ("세미나", "study", "w"), ("특강", "study", "o"), ("자격증 공부", "study", "w"),
    ("정보처리기사 공부", "study", "w"), ("영어 단어 암기", "study", "w"), ("노트 정리", "study", "w"),
    ("코딩 연습", "study", "w"), ("학회", "study", "m"), ("독서실 공부", "study", "w"), ("오픽 준비", "study", "w"),
    ("전공 공부", "study", "w"), ("코테 스터디", "study", "w"), ("영어 회화 연습", "study", "w"), ("CS 스터디", "study", "w"),
    # team
    ("팀 회의", "team", "w"), ("팀플 미팅", "team", "w"), ("캡스톤 회의", "team", "w"), ("조별 과제 모임", "team", "w"),
    ("프로젝트 회의", "team", "w"), ("팀 발표 준비", "team", "o"), ("해커톤 준비 회의", "team", "o"),
    ("졸업 프로젝트 미팅", "team", "w"), ("팀플 자료 조사", "team", "o"), ("팀플 PPT 작업", "team", "o"),
    ("해커톤", "team", "m"), ("팀 코드 리뷰", "team", "w"), ("스프린트 회의", "team", "w"), ("팀플 발표 연습", "team", "o"),
    # personal
    ("발표 리허설", "personal", "o"), ("병원", "personal", "o"), ("치과 예약", "personal", "o"), ("동아리 모임", "personal", "w"),
    ("과외", "personal", "w"), ("운동", "personal", "w"), ("헬스", "personal", "w"), ("친구 약속", "personal", "o"),
    ("생일 파티", "personal", "o"), ("미용실", "personal", "o"), ("데이트", "personal", "o"), ("가족 모임", "personal", "o"),
    ("영화 보기", "personal", "o"), ("농구", "personal", "w"), ("밴드 합주", "personal", "w"), ("교수님 면담", "personal", "o"),
    ("상담 예약", "personal", "o"), ("봉사활동", "personal", "wm"), ("운전면허 학원", "personal", "w"),
    ("건강검진", "personal", "o"), ("안경 맞추기", "personal", "o"), ("본가 가기", "personal", "m"), ("독서", "personal", "w"),
    ("러닝", "personal", "w"), ("요가", "personal", "w"), ("MT", "personal", "m"), ("동아리 MT", "personal", "m"),
    ("가족 여행", "personal", "m"), ("제주도 여행", "personal", "m"), ("수영", "personal", "w"), ("필라테스", "personal", "w"),
    ("피부과", "personal", "o"), ("축구", "personal", "w"), ("동기 모임", "personal", "o"), ("고향 친구 만나기", "personal", "o"),
    ("콘서트", "personal", "o"), ("전시회 보기", "personal", "o"),
    # etc
    ("알바", "etc", "w"), ("카페 알바", "etc", "w"), ("편의점 알바", "etc", "w"), ("멘토링", "etc", "w"), ("면접", "etc", "o"),
    ("인턴 면접", "etc", "o"), ("기업 설명회", "etc", "o"), ("채용 설명회", "etc", "o"), ("공모전 발표", "etc", "o"),
    ("장학금 면접", "etc", "o"), ("은행 가기", "etc", "o"), ("택배 받기", "etc", "o"), ("세탁", "etc", "w"), ("이사", "etc", "m"),
    ("워크숍", "etc", "m"), ("주민센터 가기", "etc", "o"), ("휴대폰 수리", "etc", "o"), ("우체국 가기", "etc", "o"),
]
COURSE_TITLE_FORMS = [("{c} 스터디", "study", "w"), ("{c} 복습", "study", "w"), ("{c} 예습", "study", "w"),
                      ("{c} 과제 하기", "study", "o"), ("{c} 시험 공부", "study", "o"), ("{c} 팀플 회의", "team", "w")]
TITLE_CAT: dict[str, str] = {}
TITLE_FLAGS: dict[str, str] = {}
for _t, _c, _f in FIXED_TITLES:
    TITLE_CAT[_t], TITLE_FLAGS[_t] = _c, _f
for _form, _c, _f in COURSE_TITLE_FORMS:
    for _co in COURSES:
        TITLE_CAT[_form.format(c=_co)], TITLE_FLAGS[_form.format(c=_co)] = _c, _f
ALL_TITLES = list(TITLE_CAT)
# §3 — 사용자 일정 제목에 쓰면 안 되는 말 ('시험 공부'·'과제 하기' 는 허용)
FORBIDDEN_TITLE_WORDS = ("수업", "강의", "퀴즈", "수강신청", "마감")

ACADEMIC_EVENTS = ["수강신청 기간", "수강 정정 기간", "중간고사 기간", "기말고사 기간", "개강일", "종강일", "등록금 납부 기간",
                   "성적 열람 기간", "성적 이의신청 기간", "휴학 신청 기간", "복학 신청 기간", "보강 주간", "개교기념일",
                   "축제 기간", "졸업식", "계절학기 수강신청", "강의 평가 기간", "학위 청구 기간"]


def related(a: str, b: str) -> bool:
    return a in b or b in a


def pick_title(rng: random.Random, need: str = "any") -> tuple[str, str]:
    """need: any · weekly(w) · multi(m) · single(m 전용 제목 제외)."""
    for _ in range(100):
        if need == "multi" or rng.random() < 0.55:
            t, c, f = rng.choice(FIXED_TITLES)
        else:
            form, c, f = rng.choice(COURSE_TITLE_FORMS)
            t = form.format(c=rng.choice(COURSES))
        if need == "weekly" and "w" not in f:
            continue
        if need == "multi" and "m" not in f:
            continue
        if need == "single" and f == "m":
            continue
        return t, c
    raise RuntimeError("pick_title")


# ════════════════════════════════════════════════════════════════
# 2. 조사·템플릿 렌더러
#    {SLOT} · {SLOT|을} (받침 따라 을/를 …) · [a|b|c] (같은 틀 안의 동의어 고르기, 빈 선택지 허용)
# ════════════════════════════════════════════════════════════════

JOSA = {"을": ("을", "를"), "이": ("이", "가"), "은": ("은", "는"), "과": ("과", "와"), "이랑": ("이랑", "랑"),
        "이야": ("이야", "야"), "이고": ("이고", "고"), "이라": ("이라", "라")}
NO_E = {"오늘", "내일", "모레", "내일모레"}       # '내일에' 처럼 어색한 '에' 를 붙이지 않는 날짜 말


def _batchim(ch: str) -> tuple[bool, bool]:
    """(받침 있음, ㄹ 받침)."""
    if "가" <= ch <= "힣":
        b = (ord(ch) - 0xAC00) % 28
        return b != 0, b == 8
    if ch.isdigit():
        return ch in "013678", ch in "178"
    up = ch.upper()
    if up in "LMNR":
        return True, up in "LR"
    return False, False


def josa(word: str, j: str) -> str:
    if not word:
        return ""
    if j == "에":
        return word if word in NO_E else word + "에"
    if word.endswith("거") and j in ("이", "은", "을"):           # 내야 할 거 → 내야 할 게/건/걸
        return word[:-1] + {"이": "게", "은": "건", "을": "걸"}[j]
    has, rieul = _batchim(word[-1])
    if j == "으로":
        return word + ("으로" if has and not rieul else "로")
    a, b = JOSA[j]
    return word + (a if has else b)


def _split_top(body: str) -> list[str]:
    out, cur, depth = [], "", 0
    for ch in body:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "|" and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


_ALT = re.compile(r"\[([^\[\]]*)\]")
_SLOT = re.compile(r"\{(\w+)(?:\|(\w+))?\}")


def render(rng: random.Random, tpl: str, slots: dict[str, str]) -> str:
    s = _ALT.sub(lambda m: rng.choice(_split_top(m.group(1))), tpl)
    s = _SLOT.sub(lambda m: josa(slots[m.group(1)], m.group(2)) if m.group(2) else slots[m.group(1)], s)
    s = re.sub(r"\s+", " ", s).strip()
    for a, b in ((" ?", "?"), (" ,", ","), (" .", "."), ("?.", "?"), ("??", "?")):
        s = s.replace(a, b)
    return s


# ════════════════════════════════════════════════════════════════
# 3. 날짜·시각 (사양 §2) — 표현과 정답을 함께 만든다
# ════════════════════════════════════════════════════════════════

def rand_now(rng: random.Random) -> dt.datetime:
    """2026-09-01 ~ 2027-08-31 균등, 시 7~23, 분 5분 단위."""
    d = dt.date(2026, 9, 1) + dt.timedelta(days=rng.randint(0, 364))
    return dt.datetime(d.year, d.month, d.day, rng.randint(7, 23), rng.choice(range(0, 60, 5)))


def month_bounds(y: int, m: int) -> tuple[dt.date, dt.date]:
    return dt.date(y, m, 1), dt.date(y, m, calendar.monthrange(y, m)[1])


def next_month_of(d: dt.date) -> tuple[int, int]:
    return (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)


def _md_ok(d: dt.date) -> bool:
    return not (d.month == 2 and d.day == 29)          # '2월 29일' 은 만들지 않는다


THIS_WD_WORDS = ["이번 주", "이번주", "이번"]
NEXT_WD_WORDS = ["다음 주", "다음주", "담주", "내주"]
NN_WORDS = ["다다음 주", "다다음주"]


def abs_phrase(rng: random.Random, d: dt.date) -> str:
    return f"{d.month}/{d.day}" if rng.random() < 0.25 else f"{d.month}월 {d.day}일"


def date_phrase(rng: random.Random, kind: str, now: dt.datetime, allday: bool = False) -> Optional[tuple[str, dt.date]]:
    """쓰기(create·update·delete)용 날짜 표현. 과거는 만들지 않는다 — 불가능하면 None (now 를 다시 뽑는다)."""
    today = now.date()
    mon, wd = week_monday(today), today.weekday()
    if kind == "today":
        return ("오늘", today) if not allday else None
    if kind == "tomorrow":
        return "내일", today + D1
    if kind == "dayafter":
        return rng.choice(["모레", "모레", "내일모레"]), today + 2 * D1
    if kind == "this_wd":
        lo = wd + 1 if allday else wd
        if lo > 6:
            return None
        i = rng.randint(lo, 6)
        return f"{rng.choice(THIS_WD_WORDS)} {WD[i]}요일", mon + dt.timedelta(days=i)
    if kind == "bare_wd":                              # 수식어 없는 'X요일' — 이번 주의 남은 요일만
        if wd >= 6:
            return None
        i = rng.randint(wd + 1, 6)
        return f"{WD[i]}요일", mon + dt.timedelta(days=i)
    if kind == "next_wd":                              # 오늘이 무슨 요일이든 항상 다음 주
        i = rng.randint(0, 6)
        return f"{rng.choice(NEXT_WD_WORDS)} {WD[i]}요일", mon + dt.timedelta(days=7 + i)
    if kind == "nn_wd":
        i = rng.randint(0, 6)
        return f"{rng.choice(NN_WORDS)} {WD[i]}요일", mon + dt.timedelta(days=14 + i)
    if kind == "n_days":
        n = rng.randint(2, 14)
        return f"{n}일 {rng.choice(['뒤', '후'])}", today + dt.timedelta(days=n)
    if kind == "next_month":
        y, m = next_month_of(today)
        n = rng.randint(1, calendar.monthrange(y, m)[1])
        return f"{rng.choice(['다음 달', '다음달'])} {n}일", dt.date(y, m, n)
    if kind == "abs":
        d = today + dt.timedelta(days=rng.randint(1, 200))
        return (abs_phrase(rng, d), d) if _md_ok(d) else None
    raise ValueError(kind)


def range_phrase(rng: random.Random, kind: str, now: dt.datetime) -> Optional[tuple[str, dt.date, dt.date]]:
    """조회(list_events)용 범위 표현 → (표현, from, to). 과거 요일 조회도 허용."""
    today = now.date()
    mon, wd = week_monday(today), today.weekday()
    day = dt.timedelta(days=1)
    if kind == "today":
        return "오늘", today, today
    if kind == "tomorrow":
        return "내일", today + day, today + day
    if kind == "dayafter":
        d = today + 2 * day
        return rng.choice(["모레", "내일모레"]), d, d
    if kind == "this_week":
        return rng.choice(["이번 주", "이번 주", "이번주"]), mon, mon + 6 * day
    if kind == "next_week":
        return rng.choice(["다음 주", "다음 주", "다음주", "담주"]), mon + 7 * day, mon + 13 * day
    if kind == "nn_week":
        return rng.choice(NN_WORDS), mon + 14 * day, mon + 20 * day
    if kind == "this_weekend":
        return rng.choice(["이번 주말", "이번 주말", "이번주말"]), mon + 5 * day, mon + 6 * day
    if kind == "next_weekend":
        return rng.choice(["다음 주말", "다음 주말", "다음주말"]), mon + 12 * day, mon + 13 * day
    if kind == "this_month":
        a, b = month_bounds(today.year, today.month)
        return rng.choice(["이번 달", "이번달"]), a, b
    if kind == "next_month":
        a, b = month_bounds(*next_month_of(today))
        return rng.choice(["다음 달", "다음달"]), a, b
    if kind == "bare_wd":
        if wd >= 6:
            return None
        i = rng.randint(wd + 1, 6)
        d = mon + dt.timedelta(days=i)
        return f"{WD[i]}요일", d, d
    if kind == "this_wd":
        i = rng.randint(0, 6)
        d = mon + dt.timedelta(days=i)
        return f"{rng.choice(THIS_WD_WORDS)} {WD[i]}요일", d, d
    if kind == "next_wd":
        i = rng.randint(0, 6)
        d = mon + dt.timedelta(days=7 + i)
        return f"{rng.choice(NEXT_WD_WORDS)} {WD[i]}요일", d, d
    if kind == "abs":
        d = today + dt.timedelta(days=rng.randint(0, 150))
        return (abs_phrase(rng, d), d, d) if _md_ok(d) else None
    if kind == "n_days":
        n = rng.randint(2, 14)
        d = today + dt.timedelta(days=n)
        return f"{n}일 {rng.choice(['뒤', '후'])}", d, d
    raise ValueError(kind)


DAY_ONLY_RANGES = {"today", "tomorrow", "dayafter", "bare_wd", "this_wd", "next_wd", "abs", "n_days"}


class Tm(NamedTuple):
    phrase: str
    h: int
    m: int
    period: str
    style: str        # std · 24h · kor


KN = {1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯", 7: "일곱", 8: "여덟", 9: "아홉", 10: "열", 11: "열한", 12: "열두"}
# (시간대, 12시간제 시, 분 선택 코드)
STD_SLOTS = [("아침", 7, "h"), ("아침", 8, "h"), ("아침", 9, "h"), ("오전", 9, "hs"), ("오전", 10, "hs"), ("오전", 11, "hs"),
             ("점심", 12, "n"), ("낮", 12, "n"), ("정오", 12, "z"), ("낮", 1, "p"), ("낮", 2, "p"),
             ("오후", 1, "q"), ("오후", 2, "q"), ("오후", 3, "q"), ("오후", 4, "q"), ("오후", 5, "q"),
             ("저녁", 6, "h"), ("저녁", 7, "h"), ("저녁", 8, "h"), ("밤", 9, "h"), ("밤", 10, "h"), ("밤", 11, "h")]
MIN_OPTS = {"h": [(0, "", 75), (30, " 반", 25)], "hs": [(0, "", 60), (30, " 반", 25), (30, " 30분", 15)],
            "n": [(0, "", 80), (30, " 반", 20)], "p": [(0, "", 100)],
            "q": [(0, "", 55), (30, " 반", 20), (30, " 30분", 10), (15, " 15분", 8), (45, " 45분", 7)]}
KOR_SLOTS = [("아침", 7), ("아침", 8), ("아침", 9), ("오전", 9), ("오전", 10), ("오전", 11), ("점심", 12), ("낮", 1), ("낮", 2),
             ("오후", 1), ("오후", 2), ("오후", 3), ("오후", 4), ("오후", 5), ("저녁", 6), ("저녁", 7), ("저녁", 8),
             ("밤", 9), ("밤", 10), ("밤", 11)]
PM_PERIODS = {"오후", "낮", "저녁", "밤"}


def to24(period: str, h12: int) -> int:
    if h12 == 12:
        return 12
    return h12 + 12 if period in PM_PERIODS else h12


def pick_time(rng: random.Random, style: str, half_only: bool = False) -> Tm:
    if style == "24h":
        h = rng.randint(13, 23)
        m = 0 if rng.random() < 0.65 else 30
        return Tm(f"{h}시" + ("" if m == 0 else rng.choice([" 30분", " 30분", " 반"])), h, m, "", "24h")
    if style == "kor":
        per, h12 = rng.choice(KOR_SLOTS)
        m = 30 if rng.random() < 0.25 else 0
        return Tm(f"{per} {KN[h12]} 시" + (" 반" if m else ""), to24(per, h12), m, per, "kor")
    per, h12, code = rng.choice(STD_SLOTS)
    if code == "z":
        return Tm("정오", 12, 0, "정오", "std")
    opts = [o for o in MIN_OPTS[code] if not half_only or o[0] in (0, 30)]
    m, suf, _ = rng.choices(opts, weights=[o[2] for o in opts], k=1)[0]
    return Tm(f"{per} {h12}시{suf}", to24(per, h12), m, per, "std")


def period_for(rng: random.Random, h: int) -> str:
    if h <= 8:
        return rng.choice(["아침", "오전"])
    if h <= 11:
        return "오전"
    if h == 12:
        return rng.choice(["점심", "낮"])
    if h <= 14:
        return rng.choice(["오후", "오후", "낮"])
    if h <= 17:
        return "오후"
    if h <= 20:
        return "저녁"
    return "밤"


# 끝 시각이 시간대 말을 생략할 때 앞 시각의 시간대를 상속할 수 있는 범위 (24시 기준 끝 시각)
INHERIT_LIM = {"아침": (8, 11), "오전": (10, 11), "오후": (14, 19), "낮": (14, 17), "저녁": (19, 22), "밤": (22, 23)}


def end_phrase(rng: random.Random, t1: Tm, h2: int, m2: int) -> str:
    half = "" if m2 == 0 else " 반"
    if t1.style == "24h":
        return f"{h2}시" + ("" if m2 == 0 else rng.choice([" 30분", " 반"]))
    lim = INHERIT_LIM.get(t1.period)
    n12 = h2 if h2 <= 12 else h2 - 12
    num = f"{KN[n12]} 시" if t1.style == "kor" else f"{n12}시"
    if lim and lim[0] <= h2 <= lim[1] and h2 != 12 and rng.random() < 0.55:
        return f"{num}{half}"                                   # 오후 2시부터 4시까지 → 16:00 (상속)
    return f"{period_for(rng, h2)} {num}{half}"


def make_range(rng: random.Random, t1: Tm) -> Optional[tuple[str, int, int]]:
    """'{t1}부터 {t2}까지' 류 → (표현, 끝 시, 끝 분). t2 ≤ t1 이 되거나 자정을 넘으면 None."""
    dur = rng.choice([60, 90, 120, 120, 150, 180, 180, 240, 300])
    tot = t1.h * 60 + t1.m + dur
    h2, m2 = divmod(tot, 60)
    if h2 > 23 or m2 not in (0, 30):
        return None
    t2 = end_phrase(rng, t1, h2, m2)
    form = rng.choices(["부터", "~", "에서"], weights=[55, 25, 20], k=1)[0]
    if form == "부터":
        p = f"{t1.phrase}부터 {t2}까지"
    elif form == "~":
        p = f"{t1.phrase}~{t2}"
    else:
        p = f"{t1.phrase}에서 {t2}까지"
    return p, h2, m2


DURS = [("1시간", 60), ("한 시간", 60), ("2시간", 120), ("두 시간", 120), ("3시간", 180), ("세 시간", 180), ("4시간", 240),
        ("30분", 30), ("반 시간", 30), ("45분", 45), ("1시간 반", 90), ("한 시간 반", 90), ("90분", 90), ("2시간 반", 150)]


def at(d: dt.date, h: int, m: int) -> dt.datetime:
    return dt.datetime(d.year, d.month, d.day, h, m)


def ev_args(title: str, cat: str, start, end, all_day: bool, repeat: Optional[str] = None) -> dict:
    a = {"title": title, "start": fmt_d(start) if all_day else fmt_dt(start),
         "end": fmt_d(end) if all_day else fmt_dt(end), "allDay": all_day, "category": cat}
    if repeat:
        a["repeat"] = repeat
    return a


# ════════════════════════════════════════════════════════════════
# 4. 문장 틀 뱅크 — 의도별. 한 줄 = 한 틀(패밀리). [a|b] 는 같은 틀 안의 동의어.
# ════════════════════════════════════════════════════════════════

# ── 4-1 create_event ────────────────────────────────────────────
# 시각 있는 하루 일정. {D} 날짜 · {T} 시각 · {X} 제목 · {L} 길이(있으면 그 길이, 없으면 60분)
CE_TIMED = [
    "{D} {T|에} {X} 넣어줘",
    "{D} {X} {T|에} 넣어줘",
    "{X} 잡아줘 {D} {T}",
    "일정 하나 추가: {D} {T} {X}",
    "{D} {T} {X} 있어 캘린더에 넣어줘",
    "{D|에} {X} 있는데 {T|이야}. 등록 좀",
    "{D} {T|으로} {X} 일정 만들어줘",
    "나 {D} {T|에} {X} 있어. 추가해줘",
    "{D} {T|에} {X|이} 생겼어 일정에 넣어줘",
    "{X} 일정 {D} {T|으로} 등록해 주세요",
    "{D} {T|에} {X} 있을 예정이야 넣어둬",
    "캘린더에 {D} {T} {X} 추가",
    "{D} {T} {X} 일정 추가",
    "{X} {D} {T|에} 있음 저장해줘",
    "{D} {T|에} {X} 잡혔어 기록해줘",
    "{X} 넣어줘. 날짜는 {D|이고} 시간은 {T}",
    "{D} {T} {X} 넣어놔",
    "혹시 {D} {T|에} {X} 등록해줄 수 있어?",
    "{D} 일정 추가할게 {T} {X}",
    "{D} {T}부터 {X}인데 일정에 넣어줘",
    "{X} 추가해줘 {D} {T}",
    "{D} {T|에} {X} 하나만 넣자",
    "{D} {T} {X} 스케줄 등록",
    "{X|을} {D} {T|에} 잡아놨어 캘린더에 올려줘",
    "{D} {T|에} {X} 박아줘",
    "{D} {T} 시작으로 {X} 일정 잡아줄래?",
    # 길이 명시 ({L} / {L} 동안 / {L}짜리 / {L}만)
    "{D} {T} {X} 잡아줘 {L}",
    "{D} {X} {L} 넣어줄래? {T}부터",
    "{D} {T|에} {L}만 {X} 넣어줘",
    "{D} {T}부터 {L} 동안 {X} 일정 넣어줘",
    "{X} {L}짜리 {D} {T|에} 등록해줘",
    "{D} {T} {X}, {L}",
    "{D} {T|에} {X} {L}만 잡아줘",
    "{X} 일정 만들어줘. {D} {T} 시작이고 {L} 걸려",
    "{D} {T|에} {X} 있어 {L} 동안",
    "{D} {X} {T} 시작 {L}",
    "{X} {D} {T}부터 {L} 잡아놔",
]
# 끝 시각 명시. {R} = '{t1}부터 {t2}까지' · '{t1}~{t2}' · '{t1}에서 {t2}까지'
CE_RANGE = [
    "{D} {R} {X}",
    "{D|에} {X} 있어 {R}",
    "{X} {D} {R|으로} 잡아줘",
    "{D} {R} {X} 일정 등록해줘",
    "{D} {X} 시간은 {R}. 추가해줘",
    "캘린더에 {X} 넣어줘 {D} {R}",
    "{D} {R} {X} 잡혔어 넣어둬",
    "{X} 일정 추가 부탁해 {D} {R}",
]
# 시각 없음 → 종일
CE_ALLDAY = [
    "{D} 하루 종일 {X}",
    "{D} 종일 {X} 넣어줘",
    "{D|에} {X} 일정 추가해줘",
    "{D} {X} 있어 종일이야",
    "{X} {D|으로} 등록해줘 시간은 아직 안 정해졌어",
    "{D|은} {X} 있는 날이야 캘린더에 표시해줘",
    "{D} {X} 종일 일정으로 넣어줘",
    "시간 없이 {D|에} {X} 넣어줘",
    "{X} 일정 하나 만들어줘 {D}, 종일로",
]
# 여러 날 종일 — {A} 시작 날짜, {B} 끝 날짜
CE_MULTI = [
    "{A}부터 {B}까지 {X}",
    "{A}부터 {B}까지 {X} 일정 종일로 넣어줘",
    "{X} {A}부터 {B}까지야 캘린더에 넣어줘",
    "{A}에서 {B}까지 {X} 잡아줘",
    "{A}~{B} {X} 있어서 그 기간 종일로 막아줘",
]
# 매주 반복 — {W} 요일
CE_WEEKLY = [
    "매주 {W} {T} {X} 등록",
    "{W}마다 {T|에} {X} 있어 반복으로 넣어줘",
    "매주 {W} {T}부터 {X} {L} 반복 일정으로 넣어줘",
    "{X} 매주 {W} {T|에} 고정이야 반복 등록해줘",
    "앞으로 매주 {W} {T|에} {X} 넣어줘",
    "매주 {W} {T} {X} 반복으로 잡아줘 {L}",
    "{X} 반복 일정 만들어줘 매주 {W} {T}",
    "매주 {W} {R} {X} 반복으로 넣어줘",
]

# ── 4-2 list_events ─────────────────────────────────────────────
# (틀, 허용 종류어 등급, 묶음 전용[, 여러 날 범위 전용])  {P} 범위 · {K} 종류어.  등급 n=명사형만 · c=절('내야 할 거')까지
LE_KINDED = [
    ("{P} {K} 뭐 있어?", "nc", False),
    ("{P} {K} 뭐 남았어?", "nc", False),
    ("{P|에} {K} 있나?", "n", False),
    ("{P} {K} 알려줘", "nc", False),
    ("{P} {K} 보여줘", "nc", False),
    ("{P} {K} 몇 개야?", "n", False),
    ("{K} {P} 거 뭐 있지", "n", False),
    ("{P|에} 잡혀 있는 {K} 확인해줘", "n", False),
    ("{P} {K} 정리해서 보여줄래?", "n", False),
    ("혹시 {P|에} {K} 있어?", "nc", False),
    ("{P} {K} 어떻게 돼?", "n", False),
    ("{P|에} {K|이} 몇 개나 있어?", "nc", False),
    ("{P} {K} 목록 좀", "n", False),
    ("{P} {K} 쭉 보여줘", "n", False),
    ("{P|에} 내가 챙겨야 할 {K} 뭐야", "n", False),
    ("{P} {K} 있는지 봐줘", "nc", False),
    ("{P} {K|이} 언제 언제야?", "n", False, True),
    ("{P} {K}만 따로 보여줘", "n", False),
    ("{P} {K} 위주로 알려줘", "n", False),
    ("{P} 일정 중에 {K}만 뽑아줘", "n", False),
    ("{P} {K} 같이 보여줘", "n", True),
    ("{P} {K} 언제야", "n", False, True),
    ("{P} {K} 확인 좀 해줄래?", "nc", False),
]
# kinds 생략 (일정·스케줄·뭐 있어·뭐 해야 돼·바빠?)  (틀, 하루 범위 전용)
LE_PLAIN = [
    ("{P|에} 뭐 있지", False),
    ("{P} 일정 알려줘", False),
    ("{P} 스케줄 어때", False),
    ("{P} 뭐 해야 돼?", False),
    ("{P} 바빠?", False),
    ("{P|에} 나 뭐 있어?", False),
    ("{P} 일정 쭉 보여줘", False),
    ("{P} 일정 어떻게 돼?", False),
    ("{P|에} 잡힌 거 있어?", False),
    ("{P} 스케줄 정리해줘", False),
    ("{P} 나 시간 돼? 일정 좀 봐줘", False),
    ("{P} 전체 일정 보여줘", False),
    ("{P} 일정 뭐뭐 있어", False),
    ("{P|에} 무슨 일정 있더라", False),
    ("{P} 일정 확인", False),
    ("{P|은} 뭐 하는 날이야?", True),
    ("{P} 일정 몇 개야", False),
    ("{P} 남은 일정 보여줘", False),
    ("{P} 뭐 있는지 한번 봐줘", False),
    ("{P|에} 일정 있는 거 다 알려줘", False),
    ("{P} 캘린더에 뭐 들어 있어?", False),
]
# 종류어 → kinds  (명사형, 절)
K_WORDS = {
    "deadline": (["마감", "마감", "과제 마감", "데드라인", "마감 일정"], ["제출할 거", "내야 할 거"]),
    "exam": (["시험", "시험", "시험 일정"], ["시험 범위"]),
    "class": (["수업", "강의", "수업 일정"], []),
    "academic": (["학사일정", "학교 행사", "학사 일정"], []),
    "user": (["내 일정", "개인 일정", "약속", "내가 넣은 일정"], []),
    "study": (["공부 블록", "학습 블록", "공부 시간"], []),
    "deadline_exam": (["시험이랑 마감", "마감이랑 시험", "마감하고 시험"], []),
    "class_exam": (["수업이랑 시험", "수업하고 시험"], []),
}
K_KINDS = {"deadline": ["deadline"], "exam": ["exam"], "class": ["class"], "academic": ["academic"], "user": ["user"],
           "study": ["study"], "deadline_exam": ["deadline", "exam"], "class_exam": ["class", "exam"]}

# ── 4-3 navigate ───────────────────────────────────────────────
# (틀, 화면말 붙음) — 화면말(화면·페이지·탭·메뉴·창)이 붙는 틀에는 '출석 화면' 같은 이름을 넣지 않는다
NAV_FRAMES = [
    ("{S} 열어줘", False), ("{S} 화면 띄워줘", True), ("{S} 페이지로 이동해줘", True), ("{S} 탭 열어", True),
    ("{S} 화면으로 가자", True), ("{S|으로} 바로 가기", False), ("{S} 좀 켜줘", False), ("{S} 화면 보여줘", True),
    ("{S} 들어가줘", False), ("{S} 화면으로 넘어가줘", True), ("{S} 페이지 띄워줄래?", True), ("{S|으로} 이동", False),
    ("{S} 메뉴 열어줘", True), ("{S} 창 좀 열어봐", True), ("앱에서 {S} 화면 열어줘", True),
    ("{S} 화면 어디 있어? 거기로 가줘", True), ("지금 {S} 화면으로 바꿔줘", True), ("{S} 페이지 들어가고 싶어", True),
    ("{S} 띄워봐", False), ("{S} 쪽으로 이동해줘", False), ("{S} 보러 가자", False), ("{S} 페이지 열어주세요", True),
    ("{S|에} 들어가고 싶어 열어줘", False), ("{S} 바로 열어", False), ("{S|으로} 가줘", False), ("{S} 화면 켜줄래?", True),
    ("{S} 확인하게 열어줘", False),
]
# route → [(이름, 이름에 화면말 포함)]
NAV_NAMES = {
    "/": [("대시보드", False), ("홈", False), ("메인", False), ("처음 화면", True), ("메인 화면", True), ("홈 화면", True)],
    "/academic": [("학사일정", False), ("학교 일정", False), ("학사 캘린더", False)],
    "/graduation": [("졸업요건", False), ("졸업 체크", False), ("졸업요건 화면", True)],
    "/attendance": [("출결", False), ("출석 화면", True), ("출결 화면", True)],
    "/courses": [("강의자료", False), ("수업 자료", False), ("자료실", False)],
    "/exams": [("시험 화면", True), ("시험 일정 탭", True), ("시험 탭", True)],
    "/scholarships": [("장학금", False), ("장학 공지", False)],
    "/opportunities": [("대외활동", False), ("공모전", False), ("사업단", False)],
    "/briefing": [("브리핑", False), ("오늘 요약", False)],
    "/assignments": [("과제 화면", True), ("e클래스 과제 화면", True), ("과제 탭", True)],
    "/chat": [("대화 화면", True), ("채팅", False), ("채팅 화면", True)],
    "/settings/profile": [("프로필", False), ("내 정보 설정", False), ("프로필 설정", False)],
    "/settings/sources": [("수집 원천 설정", False), ("연동 설정", False), ("계정 연동", False)],
    "/settings/notifications": [("알림 설정", False)],
    "/settings/availability": [("가용 시간 설정", False), ("공강 시간 설정", False)],
    "/settings/requirements": [("졸업요건 기준 설정", False)],
}

# ── 4-4 run_sync ───────────────────────────────────────────────
SYNC_FRAMES = [
    "{S} 동기화해줘", "{S} 새로 받아줘", "{S} 다시 수집해", "{S} 갱신 좀 해줘", "{S} 업데이트해줘", "{S} 새로고침",
    "{S} 최신으로 불러와", "{S} 긁어와줘", "{S} 최신화 부탁해", "{S} 다시 긁어와", "{S} 리프레시[해줘| 좀 해줘]",
    "{S} 지금 다시 가져와", "{S|이} 안 바뀐 거 같은데 다시 받아와", "{S} 최신 거 반영해줘",
    "방금 올라온 {S} 반영되게 동기화 돌려줘", "{S} 수집 다시 돌려줘", "{S} 새로 올라온 거 있나 갱신해봐",
    "{S} 동기화 한 번만 더 해줘", "{S} 오래된 거 같아 새로 받아", "{S} 지금 바로 업데이트", "{S} 싹 다시 불러와줘",
    "[혹시 |]{S} 동기화 돌릴 수 있어?",
]
SYNC_NAMES = {"eclass": ["e클래스", "이클래스", "LMS", "과제 목록", "e클래스 과제"],
              "academic": ["학사일정", "학사 일정", "학교 일정"],
              "scholarship": ["장학 공지", "장학금 정보", "장학 공지사항"],
              "activity": ["대외활동", "공모전 정보", "대외활동 공고"]}
# 2-step 답에 이름이 그대로 들어가므로 prepare_f9 와 같은 결의 이름만
TS_SYNC_NAMES = {"eclass": ["e클래스", "이클래스", "LMS"], "academic": ["학사일정"],
                 "scholarship": ["장학 공지", "장학금 정보"], "activity": ["대외활동", "공모전 정보"]}

# ── 4-5 get_graduation ─────────────────────────────────────────
GR_FRAMES = [
    "졸업까지 몇 학점 남았[어|지|더라]?",
    "[나|내가] [졸업 가능해|졸업할 수 있어|졸업 가능한 거야]?",
    "졸업요건 [얼마나|어디까지] 채웠어?",
    "전공 학점 [몇 학점 들었어|얼마나 채웠어]?",
    "교양 학점 [다 채웠나|충분해|모자라]?",
    "남은 이수 영역 [뭐야|알려줘|뭐 남았어]",
    "졸업 사정 [결과|상태] 어때?",
    "지금까지 이수한 학점 [몇이야|얼마야|총 몇 학점이야]",
    "졸업하려면 [뭐 더 들어야 돼|뭐가 더 필요해]?",
    "[전공필수|전필] 다 들었나?",
    "교양 [영역|과목] 뭐 남았는지 알려줘",
    "졸업 [못 하는|안 되는] 거 아니지? [확인해줘|확인 좀]",
    "학점 [현황|상황] 좀 알려줘",
    "졸업까지 얼마나 남았는지 [봐줘|확인해줘]",
    "[이번 학기 기준으로|현재 기준] 졸업요건 [어때|어떻게 돼]?",
    "[내년에|이번 학기 끝나면] 졸업 가능할까?",
    "졸업요건 [충족했어|다 맞췄어]?",
    "부족한 학점 [있어|어디서 부족해]?",
    "[복수전공|부전공] 학점은 어떻게 돼?",
    "졸업 판정 [나왔어|어떻게 나와]?",
    "졸업 가능 여부 [확인해줘|알려줘]",
    "[나|내가] 들은 학점 [정리해줘|합계 알려줘]",
    "졸업 학점 [기준에서|요건에서] 얼마나 모자라?",
    "전공 선택 학점 [몇 남았지|얼마나 남았어]",
    "남은 학점 [얼마야|몇이야]",
    "졸업요건 어떻게 돼?",
    "영역별로 [얼마나 들었는지|이수 현황] 알려줘",
    "필수 과목 [빠진 거|안 들은 거] 있어?",
    "졸업 [조건|기준] 중에 [못 채운|남은] 거 뭐야",
    "학점 몇 개 더 들어야 졸업이야?",
    "[졸업 논문|졸업 작품] 말고 학점은 다 됐어?",
    "[전공|교양|일반선택] 중에 뭐가 제일 부족해?",
]
TS_GR_IDX = [0, 7, 12, 13, 22, 24, 29]          # '남은 학점' 숫자 답과 맞는 틀만 2-step 에

# ── 4-6 get_attendance ─────────────────────────────────────────
AT_COURSE = [
    "{C} 결석 몇 번 했[어|지|더라]?",
    "{C} 출석 [어때|괜찮아]?",
    "{C} 몇 번 더 빠지면 [F야|F 나와|위험해]?",
    "{C} 지각 [몇 번이야|몇 번 했어]?",
    "{C} 출결 [상태|현황] 알려줘",
    "나 {C} F 위험[해|한 거 아니지]?",
    "{C} 이번 학기에 [몇 번|얼마나] 빠졌지",
    "{C} 결석 한도 [얼마나 남았어|몇 번이야]",
    "{C} 출석 [몇 번|몇 회] 인정됐어?",
    "{C} 더 빠져도 [돼|괜찮아]?",
    "{C|은} 출석 문제 없지?",
    "{C} 출결 좀 봐줘",
    "{C} 결석 [누적|쌓인 거] 확인해줘",
    "나 {C} 너무 많이 빠진 거 같은데 [몇 번이야|확인 좀 해줘]",
    "{C} [결석|지각] 기록 [보여줘|알려줘]",
]
TS_AT_EXCLUDE = {3, 8, 14}      # 지각·출석 인정·[결석|지각] 기록 — 답이 '결석 N회' 라 2-step 에서 뺀다
AT_ALL = [
    "출결 [어때|괜찮아]?",
    "결석 [많은|위험한] 과목 있어?",
    "F 위험한 과목 [있나|있어]?",
    "전체 [출석|출결] 현황 알려줘",
    "나 결석 [몇 번|얼마나] 했어? [과목별로|전부]",
    "지각 많은 [과목|수업] 있나?",
    "[빠진|결석한] 수업 [정리해줘|다 알려줘]",
    "출석 [경고|위험] 뜬 과목 있어?",
    "이번 학기 출석 [괜찮은지|문제없는지] 봐줘",
    "[어느|무슨] 과목 결석 한도 다 돼 가?",
    "몇 번 더 빠지면 위험한 과목 [있어|알려줘]",
    "과목별 결석 횟수 [보여줘|알려줘]",
]

# ── 4-7 get_exams ──────────────────────────────────────────────
EX_UP = [
    "다음 시험 [언제야|언제지|언제더라]?",
    "다가오는 시험 [있어|뭐 있어]?",
    "시험까지 며칠 남았어?",
    "제일 가까운 시험 [뭐야|언제야]",
    "[중간고사|기말고사] 언제야?",
    "곧 볼 시험 [알려줘|있어?]",
    "앞으로 남은 시험 [다|전부] 알려줘",
    "시험 언제부터야?",
    "아직 안 본 시험 [뭐 있어|몇 개야]",
    "시험 일정 [다가오는 것만|남은 것만] 보여줘",
    "{C} 시험 언제야?",
    "다음 시험 무슨 과목이야?",
    "시험 [얼마나|며칠이나] 남았지",
    "이제 시험 몇 개 남았어?",
]
TS_EX_IDX = [0, 1, 2, 3, 5, 7, 11, 12]
EX_PLAN = [
    "시험 공부 계획 [어떻게 돼?|보여줘]",
    "시험 계획 [짜놓은 거|세운 거] 보여줘",
    "시험 대비 [공부 계획|스케줄] 알려줘",
    "지난 시험까지 [전부|다] 보여줘",
    "시험별 공부 계획 [어떻게 돼?|정리해줘]",
    "[중간|기말] 대비 공부 계획 확인해줘",
    "시험 공부 계획 [얼마나 진행됐어?|진행 상황 알려줘]",
]

# ── 4-8 list_assignments ('과제' 로 물으면 여기, '마감' 으로 물으면 list_events) ─────
AS_FRAMES = {
    "now": ["지금 [해야 할|당장 해야 할] 과제 뭐야?", "당장 [해야 하는|급한] 과제 [있어?|알려줘]", "급한 과제[부터|만] 보여줘",
            "먼저 해야 할 과제 [뭐야|뭐지]?", "제일 급한 과제 [하나만 알려줘|뭐야?]", "지금 손대야 하는 과제 [뭐 있어?|알려줘]",
            "과제 뭐부터 [해야 돼|할까]?", "급하게 [끝내야|처리해야] 할 과제 [목록|있어?]", "발등에 불 떨어진 과제 [뭐야|뭐 있어?|뭔지 알려줘]"],
    "this_week": ["이번 주 과제 [뭐 있어?|알려줘]", "이번 주 안에 해야 하는 과제 [있어?|뭐야?]",
                  "이번 주까지 끝낼 과제 [보여줘|정리해줘]", "이번주 과제 [몇 개야?|몇 개 있어?]",
                  "이번 주에 처리할 과제 목록 [보여줘|좀]"],
    "later": ["나중에 해도 되는 과제 [있어?|뭐야?]", "여유 있는 과제 [뭐 있어?|보여줘]", "급하지 않은 과제 [알려줘|목록]",
              "천천히 해도 되는 과제 [뭐야?|있어?]"],
    "all": ["과제 [뭐 있어|뭐 있지]?", "과제 우선순위 [보여줘|알려줘]", "과제 목록 [전체|다] 보여줘",
            "남은 과제 [전부|싹 다] 알려줘", "해야 할 과제 [다|전부] 정리해줘", "과제 [순서|우선순위] 어떻게 돼?",
            "e클래스 과제 [뭐 있어?|목록 보여줘]", "밀린 과제 [있어?|몇 개야?]", "과제 [몇 개|얼마나] 남았어?"],
}

# ── 4-9 open_placement — {RG} 는 이번 주/다음 주/빈칸 ──────────
PL_FRAMES = [
    "{RG} 공강 배치해줘",
    "{RG} 공강에 공부 [배치해줘|넣어줘]",
    "{RG} 공강 [좀 |]채워줘",
    "{RG} 빈 시간 [채워줘|좀 채워줄래?|공부로 메워줘|다 채워줘]",
    "{RG} 비는 시간에 공부 넣어줘",
    "{RG} 공강 학습 플랜 [열어줘|보여줘]",
    "{RG} 공부 블록 자동으로 [배치해줘|채워줘]",
    "{RG} 학습 계획 짜줘",
    "{RG} 공부 계획 세워줘",
    "시간표 빈칸에 공부 넣어줘 {RG}",
    "{RG} 자투리 시간에 공부 배치해줘",
    "{RG} 공강 시간 활용 플랜 [짜줘|만들어줘]",
    "{RG} 배치 미리보기 열어줘",
    "{RG} 공부 시간 자동으로 잡아줘",
    "{RG} 공부 플랜 짜줘",
    "{RG|에} 공강 [몇 개 있는데|많은데] 공부로 채워줘",
    "{RG} 남는 시간에 과제랑 공부 [배치해줘|채워줘]",
    "공강 활용해서 {RG} 공부 계획 [잡아줘|만들어줘]",
    "{RG} 빈 시간 활용 계획 [보여줘|세워줘]",
    "{RG} 공강마다 공부 넣어줄래?",
    "{RG} 공부 배치 [돌려줘|한번 해줘]",
    "{RG} 수업 사이 빈 시간에 공부 [넣어줘|채워줘]",
    "{RG} 시간표 보고 공부 시간 짜줘",
    "{RG} 학습 블록 배치해줘",
    "{RG} 공강 시간에 뭐 공부할지 [짜줘|배치해줘]",
    "{RG} 공부 일정 자동 배치[ 부탁해|해줘]",
    "{RG} 비는 시간 다 공부로 [돌려줘|채워줘]",
    "{RG} 공강 플랜 [미리보기 |]보여줘",
    "{RG} 공강 학습 [계획|플랜] [다시 |]짜줘",
    "{RG} 틈새 시간 공부 배치해줘",
    "{RG} 시험 공부 공강에 [넣어줘|배치해줘]",
    "{RG|에} 공강 비어 있는 거 공부로 채우자",
    "{RG} 공강 [자동 배치|자동 채우기] 실행",
    "{RG} 공부할 시간 알아서 배치해줘",
    "{RG} 과제할 시간 공강에 [잡아줘|넣어줘]",
    "{RG} 공강 [놀리지 말고|그냥 두지 말고] 공부 넣어줘",
    "{RG} 빈 시간에 맞춰서 학습 계획 [세워줘|짜줘]",
    "{RG} 공강에 [뭐라도|공부라도] 넣어서 플랜 만들어줘",
]
PL_RANGE_WORDS = {"this_week": ["이번 주", "이번 주", "이번주"], "next_week": ["다음 주", "다음 주", "다음주", "담주"], "none": [""]}

# ── 4-10 update_event (2-step) — (patch 종류, 방향 조건, 틀) ────
#   time=시각 이동 · date=날짜 이동 · rel=상대 이동({Δ}) · title=제목 변경({Y}) · rel_assign='과제 하기' 대비쌍
UP_FRAMES = [
    ("time", "any", "{D} {X} {T|으로} 옮겨줘"),
    ("time", "any", "{D} {X} 시간 {T|으로} 변경해줘"),
    ("time", "any", "{X} {D} {T|으로} 변경[ 부탁해|해줘]"),
    ("time", "any", "{D} {X|이} {T|으로} 바뀌었어 수정해줘"),
    ("time", "any", "{D|에} 있는 {X} {T|으로} 옮길래"),
    ("time", "any", "{D} {X} {T} 시작으로 고쳐줘"),
    ("time", "any", "{D} {X} 시간 좀 {T|으로} 옮겨줄래?"),
    ("time", "any", "{X} {D} 일정 {T|으로} 바꿔놔"),
    ("time", "later", "{D} {X} {T|으로} 늦춰줘"),
    ("time", "earlier", "{D} {X} {T|으로} 당겨줘"),
    ("time", "any", "{D} {X} 시간 변경: {T}"),
    ("time", "any", "[아 |]{D} {X} 시간 잘못 넣었어 {T|이야}"),
    ("date", "any", "{D} {X} {E|으로} 바꿔줘"),
    ("date", "any", "{D} {X|을} {E|으로} 옮겨줘"),
    ("date", "later", "{X} {D}에서 {E|으로} 미뤄줘"),
    ("date", "any", "{D|에} 잡힌 {X} 날짜 {E|으로} 변경해줘"),
    ("date", "any", "{D} {X} 같은 시간에 {E|으로} 옮겨줘"),
    ("rel", "later", "{D} {X} {Δ} 뒤로 미뤄줘"),
    ("rel", "earlier", "{D} {X} {Δ} 당겨줘"),
    ("rel", "later", "{D} {X} {Δ} 늦춰줘"),
    ("rel", "earlier", "{X} {D} 거 {Δ}만 앞당겨줘"),
    ("rel", "later", "{D} {X} {Δ} 뒤로 밀어줘"),
    ("rel", "later", "{D} {X} {Δ}만 미뤄 줄 수 있어?"),
    ("title", "any", "{D} {X} 제목 {Y|으로} 바꿔"),
    ("title", "any", "{D} {X|을} {Y|으로} 이름 바꿔줘"),
    ("title", "any", "{D|에} 있는 {X} 일정 이름 {Y|으로} 수정해줘"),
    ("title", "any", "{D} {X|이} 아니라 {Y|이야} 제목 고쳐줘"),
    ("rel_assign", "later", "{D} {X} 일정 {Δ} 미뤄줘"),        # X = '{과목} 과제 하기' ↔ 거절 A '과제 마감 미뤄줘'
]
DELTA_LATER = [("1시간", 60), ("한 시간", 60), ("30분", 30), ("2시간", 120), ("두 시간", 120), ("1시간 반", 90),
               ("하루", 1440), ("이틀", 2880)]
DELTA_EARLIER = [("30분", 30), ("1시간", 60), ("한 시간", 60), ("2시간", 120), ("하루", 1440)]

# ── 4-11 delete_event (2-step) ─────────────────────────────────
DEL_FRAMES = [
    "{D} {X} [취소해줘|취소해 주세요|취소]",
    "{D} {X} 일정 [지워줘|삭제해줘|삭제]",
    "{D|에} 있는 {X} [빼줘|빼 주실래요?]",
    "{X} {D} 거 [없애줘|날려줘]",
    "{D} {X} 취소됐어 일정에서 [빼줘|지워줘]",
    "{D} {X|은} 없던 걸로 해줘",
    "캘린더에서 {D} {X} [지워|삭제해]",
    "{D} {X} 일정 [필요 없어졌어|이제 필요 없어] 삭제해줘",
    "{D|에} 잡아둔 {X} 취소할래",
    "{X} 일정 삭제 {D} 거",
    "{D} {X} 펑크났어 일정 [지워줘|없애줘]",
    "{D} {X} 일정 [좀 |]빼줄래?",
    "{D} {X} [삭제 부탁해|삭제 좀 해줘]",
    "{D} {X} 없어졌어 [지워줘|일정에서 빼줘]",
    "{D} 일정 중에 {X} [삭제|빼줘]",
    "{X} 지워줘 {D} 거",
    "{D} {X} 캘린더에서 내려줘",
    "{D|에} 넣어둔 {X} [취소해줘|없애줘]",
    "{D} {X} 일정 날려줘",
    "{D} {X} 취소할게 [지워줘|빼줘]",
    "실수로 넣은 {D} {X} 지워줘",
    "{X} [지워줘|삭제해줘] {D} 거",              # del_contrast: X = '{과목} 스터디' ↔ 거절 B '{과목} 수업 지워줘'
]
DEL_CONTRAST_IDX = len(DEL_FRAMES) - 1

# ── 4-12 거절 A — e클래스 마감 변경·삭제 ─────────────────────────
REF_A = [
    "{C} 과제 마감 {D|으로} 바꿔줘", "퀴즈 기한 [늘려줘|좀 늘려줘|늘려주면 안 돼?]", "e클래스 과제 [지워|지워줘]",
    "과제 제출 기한 하루만 늦춰줘", "레포트 마감 삭제", "이클래스 과제 데드라인 수정해줘", "과제 마감 [없애줘|없애 주세요]",
    "{C} 퀴즈 마감 다음 주로 미뤄", "{C} 과제 마감일 {DD} 미뤄줘", "{C} 레포트 제출일 {D|으로} 옮겨줘",
    "e클래스 마감 {D|으로} 변경해줘", "{C} 과제 마감 시간 {T|으로} 바꿔줘", "{C} 과제 마감 [캘린더에서 |]빼줘",
    "이클래스에서 온 마감 [다 지워줘|전부 삭제해줘]", "{C} 과제 데드라인 연장해줘", "{C} 퀴즈 [마감 지워|마감 없애]",
    "마감 하나만 미뤄주면 안 돼? {C} 과제", "{C} 과제 제출 마감 {D}까지로 늘려줘", "{C} 과제 기한 [연장 좀 부탁해|좀 연장해줘]",
    "{C} 레포트 마감 [하루만|이틀만] 늦춰줄 수 있어?", "e클래스 과제 마감 [수정|변경] 좀 해줘",
    "{D} 마감인 {C} 과제 [지워줘|없애줘]", "{D}까지인 {C} 과제 마감 {E|으로} 바꿔",
    "{C} 과제 마감 [알림 말고 |]날짜 자체를 바꿔줘", "과제 마감 날짜 [잘못 들어간 거 같은데|틀린 거 같은데] 고쳐줘",
    "{C} 온라인 퀴즈 기한 [늘려|늘려줘]", "{C} 과제 [제출일|제출 날짜] 수정해줘",
    "이클래스 퀴즈 마감 [다음 주|담주] [월요일|금요일]로 옮겨", "{C} 과제 마감 [캘린더에서|목록에서] 안 보이게 지워줘",
    "e클래스 과제 [하나|두 개] 삭제해줘", "{C} 과제 마감을 {T|으로} 늦춰줘", "{C} 레포트 데드라인 [없애|지워] 버려",
    "마감 [너무 빡센데|너무 촉박한데] {C} 과제 [좀 미뤄줘|기한 늘려줘]", "{C} 과제 마감 [다음 주로|다다음 주로] 넘겨줘",
    "{C} 과제 제출 기한 [{D}|주말]까지로 바꿔", "e클래스에 있는 {C} 퀴즈 날짜 바꿔줘", "{C} 과제 마감 [취소해줘|취소]",
    "이번 주 과제 마감 [다 미뤄줘|전부 다음 주로 미뤄]", "{C} 과제 마감 [한 시간만|두 시간만] 늘려줘",
    "{C} 팀플 과제 마감 [미뤄줘|변경해줘]", "LMS 과제 마감 [수정해줘|지워줘]",
    "{C} 과제 마감일 내 맘대로 바꿀 수 있어? {D|으로}", "{C} 과제 마감 일정 [삭제해줘|지워주세요]",
    "{C} 퀴즈 제출 기한 [연장|연장해줘]", "{C} 과제 마감 표시 [없애줘|지워줘]", "{D} 과제 마감 [지워줘|빼줘]",
    "{C} 과제 마감 {D}에서 {E|으로} 바꿔줘", "e클래스 마감 [자꾸 뜨는데|거슬리는데] [지워줘|안 보이게 해줘]",
    "{C} 과제 마감 [좀 |]늦춰주라", "{C} 과제 [마감 연장 신청|기한 연장 처리]해줘",
    "{C} 기말 레포트 마감 {D|으로} [변경|미뤄]",
]
# ── 4-12 거절 B — 학사일정·수업 변경 ────────────────────────────
REF_B = [
    "{C} 수업 {T|으로} 옮겨줘", "{C} 수업 [빼줘|지워줘|취소해줘]", "{D} {C} 수업 [휴강 처리해줘|휴강으로 바꿔줘]",
    "수강신청 기간 [삭제해줘|변경해줘]", "중간고사 기간 [옮겨줘|옮겨]", "기말고사 날짜 [바꿔줘|바꿔]", "개강 [미뤄줘|좀 미뤄줘]",
    "종강 [당겨줘|좀 당겨줘]", "학사일정 [지워|지워줘]", "{C} 강의 시간 변경해줘", "{C} 강의 {W|으로} 옮겨",
    "등록금 납부 기간 [바꿔줘|변경해줘]", "성적 열람 기간 [연장해줘|늘려줘]", "{C} 시험 날짜 [하루|이틀] 미뤄줘",
    "{D} {C} 수업 [지워줘|없애줘]", "{C} 수업 [캘린더에서|시간표에서] 빼줘", "{C} 수업 시간 {T}에서 {U|으로} 바꿔줘",
    "{W} {C} 수업 [없애줘|삭제해줘]", "{C} 중간고사 {D|으로} 옮겨줘", "{C} 기말 [일정|날짜] [변경해줘|바꿔줘]",
    "수강 정정 기간 [늘려줘|연장해줘]", "{C} 수업 [이번 주만|이번만] 빼줘", "{C} 강의실 [바꿔줘|변경해줘]",
    "학사일정에 있는 {AE} [삭제해줘|지워줘]", "{AE} 날짜 [바꿔줘|옮겨줘]", "{AE} [{D|으로}|좀] 미뤄줘",
    "{AE} 일정 [캘린더에서 지워줘|삭제]", "{AP} [늘려줘|연장해줘]", "{AE} 하루만 당겨줘", "{C} 수업 [{D|에}|] 보강 잡아줘",
    "{C} [수업|강의] [끝나는|시작하는] 시간 [바꿔줘|고쳐줘]", "{C} 수업 [{D}|다음 주] 거 휴강으로 [표시해줘|바꿔줘]",
    "{C} 수업 요일 {W|으로} 변경", "{C} 수업 분반 [옮겨줘|바꿔줘]", "{C} 수업 [시간표에서|캘린더에서] 삭제해 주세요",
    "{C} 수업 {T}부터 시작하게 바꿔줘", "다음 학기 개강일 [바꿔줘|좀 늦춰줘]", "{C} 수업 듣기 싫어 [지워줘|없애줘]ㅋㅋ",
    "{C} 강의 일정 [삭제|지워]", "{C} 시험 [{D|으로}|다음 주로] 미뤄줘", "{C} 수업 {D} 거만 {T|으로} 옮겨줘",
    "학사일정 [수정|변경]해줘 {AE} 틀렸어", "{C} 수업이랑 {C2} 수업 [시간 바꿔줘|자리 바꿔줘]",
    "{C} 수업 [이번 학기|앞으로] 전부 지워줘", "{C} 강의 [휴강|결강] 처리[해줘| 부탁해]", "{C} 수업 {W}에서 {V|으로} 옮겨줘",
    "학교 일정 중에 {AE} [빼줘|지워줘]", "{C} 수업 [1교시|2교시|3교시]로 [옮겨|바꿔]줘",
    "{C} 수업 [{N}분 일찍|{N}분 늦게] 시작하게 해줘", "중간고사 [일정|기간] [다음 주로|일주일] 미뤄줘",
    "기말고사 [{D|으로}|좀 일찍] [당겨줘|옮겨줘]", "{C} 수업 캘린더에 [안 뜨게|안 보이게] 지워줘", "{AE} [취소해줘|없애줘]",
    "{C} 실습 수업 [{D}|이번 주] 거 [빼줘|삭제해줘]", "개강 [일주일|하루만] 늦춰줘", "종강 [{D|으로}|좀 빨리] 당겨줘",
    "수강신청 날짜 [{D|으로}|좀] 바꿔줘", "{C} 수업 시간 [옮기고 싶어|바꾸고 싶어] {W} {T|으로}",
    "{C} 수업 [오늘|내일] 거 취소해줘", "{D} {AE} [지워|지워줘]", "{C} 강의 [시작 시간|끝나는 시간] {T|으로} 수정",
    "{C} 중간 시험 일정 [미뤄줘|바꿔줘]",
]
# ── 4-12 거절 C — 범위 밖 (주제 = 패밀리) ───────────────────────
REF_C = {
    "weather": ["[내일|오늘|이번 주말] 날씨 어때?", "[밖에|지금] 비 와?", "우산 챙겨야 돼?", "{CITY} 날씨 알려줘"],
    "food": ["[점심|저녁] 뭐 먹지?", "오늘 [야식|저녁] 추천 좀", "{FOOD} 먹을까 {FOOD2} 먹을까"],
    "code": ["파이썬으로 [정렬 코드|크롤러] 짜줘", "[자바|C언어] 과제 코드 대신 짜줘", "이 에러 왜 나는지 봐줘 [NullPointerException|IndexError]"],
    "translate": ["이 문장 [영어로|일본어로] 번역해줘", "'{PHRASE}' [영어로|중국어로|일본어로] 뭐야?"],
    "news": ["오늘 뉴스 [뭐 있어|요약해줘]", "요즘 [핫한|화제인] 이슈 알려줘"],
    "stock": ["삼성전자 주가 [얼마야|어때]?", "[주식|미장] 뭐 사야 돼?", "[테슬라|엔비디아] 지금 사도 돼?"],
    "coin": ["[비트코인|이더리움] [시세|지금 얼마]?", "코인 [오를까|떨어질까]?"],
    "movie": ["요즘 [볼 만한|재밌는] 영화 뭐 있어?", "넷플릭스 [드라마|시리즈] 추천해줘"],
    "music_reco": ["[공부할 때|잘 때] 들을 노래 추천해줘", "요즘 [노래|플레이리스트] 추천 좀"],
    "music_play": ["노래 [틀어줘|재생해줘]", "[아이유|뉴진스|데이식스] 노래 틀어"],
    "joke": ["웃긴 얘기 해줘", "[아재개그|농담] 하나 해봐"],
    "love": ["[썸|짝사랑] 상대한테 뭐라고 연락하지?", "[여자친구|남자친구]랑 싸웠는데 어떡해"],
    "game": ["[롤|발로란트] 티어 올리는 법 알려줘", "[요즘|방학에] 할 만한 게임 추천해줘"],
    "prof_contact": ["{C} 교수님 [연락처|이메일] 알려줘", "[교수님|조교님] [연구실|번호] 어디야?"],
    "cafeteria": ["[오늘|내일] 학식 [메뉴|뭐 나와]?", "[기숙사 식당|학생식당] 메뉴 알려줘"],
    "bus": ["[셔틀버스|마을버스] 언제 와?", "[지하철 막차|막차] 몇 시까지 있어?"],
    "parcel": ["택배 [어디쯤 왔어|언제 와]?", "[쿠팡|택배] 배송 조회해줘"],
    "shopping": ["[노트북|태블릿] 뭐 사는 게 나아?", "[무선 이어폰|키보드] 추천해줘"],
    "math": ["[적분|행렬] 문제 좀 풀어줘", "이 식 [풀어줘|계산해줘] {EXPR}"],
    "essay": ["[자소서|자기소개서] 써줘", "[에세이|독후감] 대신 써줘", "레포트 [서론|결론] 써줘"],
    "counsel": ["요즘 너무 [우울해|힘들어]", "[번아웃 온 것 같아|무기력해] 어떡하지"],
    "health": ["[머리가|배가] 아픈데 약 뭐 먹어?", "[감기|두통]에 좋은 거 알려줘"],
    "travel": ["[제주도|부산|강릉] 여행 코스 짜줘", "[방학에|이번 겨울에] 여행 어디 갈까?"],
    "restaurant": ["[학교 앞|근처] 맛집 추천해줘", "{CITY} 맛집 알려줘"],
    "lotto": ["로또 번호 [뽑아줘|추천해줘]", "이번 주 로또 [당첨 번호|1등 번호] 뭐야?"],
    "fortune": ["오늘 [운세|별자리 운세] 어때?", "내 [사주|타로] 봐줘"],
    "alarm": ["[내일 아침 7시|아침 6시 반]에 알람 맞춰줘", "[타이머|알람] [10분|30분] 뒤로 맞춰줘"],
    "messenger": ["[엄마한테|친구한테] 카톡 보내줘", "교수님께 메일 [보내줘|써서 보내줘]"],
    "call": ["[엄마|친구|동기]한테 전화 걸어줘", "[학과 사무실|조교실]에 전화해줘"],
    "gpa_calc": ["평점 계산해줘 [A+ 두 개랑 B+ 하나|A 세 개랑 B 두 개]", "내 평점 몇 점이야?", "직전 학기 평점 [몇이야|얼마야]?"],
    "dorm": ["기숙사 [신청|입사 신청]해줘", "기숙사 [추가 모집|입사] 언제야?"],
    "library_seat": ["도서관 좌석 [예약해줘|잡아줘]", "열람실 자리 [남았어|있어]?"],
    "parking": ["학교 주차 [어디 해|되는 데 알려줘]", "주차권 [어떻게 받아|신청해줘]"],
    "tuition": ["등록금 [얼마야|얼마 나와]?", "이번 학기 등록금 [얼마|고지서 보여줘]"],
    "exam_leak": ["{C} 시험 문제 [알려줘|뭐 나와?]", "[기출|족보] 구해줘"],
    "answer": ["이 문제 답 [알려줘|뭐야]?", "퀴즈 답 좀 알려줘"],
    "blind_date": ["소개팅 [해줘|시켜줘]", "소개팅 [첫 마디|대화 주제] 뭐 하지?"],
    "exchange": ["[달러|엔화|유로] 환율 얼마야?", "지금 환전하는 게 나아?"],
    "sports": ["어제 [축구|야구] 누가 이겼어?", "[손흥민|이강인] 골 넣었어?"],
    "recipe": ["[김치찌개|라면] 맛있게 끓이는 법", "[계란|자취] 요리 레시피 알려줘"],
    "dictionary": ["'{WORD}' 뜻이 뭐야?", "{WORD} 무슨 뜻이야?"],
    "spelling": ["'되'랑 '돼' 뭐가 맞아?", "이 문장 맞춤법 검사해줘"],
    "summary": ["이 [논문|기사] 요약해줘", "[이 PDF|이 글] 세 줄로 요약해"],
    "ppt": ["[발표 PPT|발표 자료] 만들어줘", "PPT 템플릿 추천해줘"],
    "job": ["[취업|대기업] 준비 어떻게 해?", "[이력서|포트폴리오] 봐줘"],
    "military": ["[군대|입대] 언제 가는 게 좋아?", "[공군|해군] 지원 어떻게 해?"],
    "housing": ["[원룸|자취방] 구하는 법 알려줘", "[월세|전세] 뭐가 나아?"],
    "job_search": ["[알바|과외] 자리 구해줘", "시급 [높은|센] 알바 추천해줘"],
    "calc": ["[1500 곱하기 12|3의 10제곱|17 나누기 3]는 얼마야?", "[이거|팁] 계산 좀 해줘 [3만 원 나누기 4|5만 원의 15%]"],
    "wifi": ["학교 와이파이 [비번|비밀번호] 뭐야?", "[와이파이|인터넷] 안 되는데 어떡해"],
    "printing": ["[학교에서|도서관에서] 프린트 어디서 해?", "[인쇄|프린트]해줘"],
    "lost": ["[지갑|학생증] 잃어버렸어 어떡해", "분실물 센터 어디야?"],
    "club_reco": ["[동아리|중앙동아리] 추천해줘", "무슨 동아리 들까?"],
    "prof_review": ["{C} 교수님 [어때|깐깐해]?", "{C} 교수님 [성격|스타일] 알려줘"],
    "email_write": ["교수님께 보낼 메일 [써줘|문구 다듬어줘]", "[결석 사유서|사과 메일] 대신 써줘"],
    "pet": ["[강아지|고양이] 키우고 싶은데 뭐 필요해?", "[강아지|고양이] 이름 추천해줘"],
    "diet": ["[다이어트|벌크업] 식단 짜줘", "[살 빼는|근육 키우는] 법 알려줘"],
    "sleep": ["잠이 안 와 어떡하지", "[밤새고|벼락치기하고] 졸릴 때 꿀팁"],
    "cafe": ["근처 카페 [어디 있어|추천해줘]", "[공부하기 좋은|조용한] 카페 알려줘"],
    "fashion": ["[면접|소개팅] 옷 뭐 입지?", "[요즘|올겨울] 유행하는 옷 알려줘"],
    "gift": ["[여친|남친|엄마] 생일 선물 추천해줘", "[3만 원대|5만 원대] 선물 뭐가 좋아?"],
    "taxi": ["택시 [불러줘|잡아줘]", "[카카오택시|택시] 지금 호출해줘"],
    "delivery": ["[치킨|피자|떡볶이] 시켜줘", "배달 [주문해줘|시켜줘]"],
    "ticket": ["[KTX|기차] 표 [예매해줘|끊어줘]", "[영화|콘서트] 예매해줘"],
    "youtube": ["[유튜브|쇼츠] 재밌는 거 [추천해줘|틀어줘]", "[공부 브이로그|강의 영상] 틀어줘"],
    "philosophy": ["인생의 의미가 뭘까?", "[삶이|사는 게] 왜 이렇게 힘들까?"],
    "trivia": ["[임진왜란|한국전쟁] 언제 일어났어?", "[세종대왕|이순신] 업적 알려줘"],
    "election": ["[학생회 선거|총학 선거] 누구 뽑지?", "[총학|학생회] 공약 알려줘"],
    "drawing": ["[고양이|강아지] 그림 그려줘", "[프로필 사진|로고] 만들어줘"],
}
REF_C_SLOTS = {
    "CITY": ["서울", "부산", "대전", "광주", "대구", "인천", "수원", "춘천", "전주", "강릉"],
    "FOOD": ["짜장면", "짬뽕", "돈까스", "마라탕", "떡볶이", "국밥", "햄버거", "초밥", "쌀국수", "비빔밥"],
    "WORD": ["세렌디피티", "아포리즘", "패러다임", "레트로", "휴리스틱", "메타인지", "아이러니"],
    "EXPR": ["x^2+3x+2=0", "2x+5=17", "3x-7=11", "x^2-9=0"],
    "PHRASE": ["잘 부탁드립니다", "오늘 수고했어", "배고파", "보고 싶어", "다음에 또 봐"],
}

# ── 4-13 인사·기능 안내 ─────────────────────────────────────────
CHIT = {
    "chit_hi": (CHIT_HELLO, ["안녕[|!|~]", "하이[|~]", "ㅎㅇ[|ㅎㅇ]", "안녕하세요[|!]", "헬로[|~]", "반가워[|!]",
                             "[안녕|하이] 반가워", "안녕하세요 처음이에요", "하이 하이", "[안녕|하이] 오랜만이야"]),
    "chit_help": (CHIT_HELLO, ["뭐 할 수 있어?", "사용법 [알려줘|좀]", "도움말[| 보여줘]", "뭘 도와줄 수 있어?",
                               "처음 써보는데 뭐 할 수 있어?", "기능 뭐 있어?", "이거 어떻게 써?", "[하이|안녕] 뭐 해줄 수 있어?",
                               "어떤 거 물어보면 돼?", "할 수 있는 거 [알려줘|다 말해줘]", "무슨 기능 있어?", "너 뭐까지 돼?"]),
    "chit_who": (CHIT_HELLO, ["너 누구야?", "넌 뭐 하는 애야?", "자기소개 [해봐|좀 해줘]", "누구세요?", "너 이름 뭐야?",
                              "넌 정체가 뭐야", "너 뭐야?", "여기 챗봇이야?"]),
    "chit_thanks": (CHIT_THANKS, ["고마워[|!|~]", "감사[|합니다|해요]", "고마워요", "[오|와] 고마워",
                                  "[좋아|알겠어|ㅇㅋ|오케이] 고마워", "고마워 [잘했어|최고야]", "덕분에 살았다[|ㅋㅋ]", "정말 고마워"]),
    "chit_slang": (CHIT_THANKS, ["ㄳ", "ㄱㅅ", "땡큐[|~]", "[와|오] 땡큐", "땡큐 땡큐", "[굿|오키] ㄳ", "ㄳㄳ", "땡스"]),
    "chit_bye": (CHIT_THANKS, ["수고[|해]", "수고했어[|!]", "수고하셨습니다", "오늘도 수고", "수고해 고마워",
                               "[됐어|오케이] 수고했어", "알겠어 수고"]),
}


# ════════════════════════════════════════════════════════════════
# 5. 표면 변형 (§5) — 슬롯 의미는 바꾸지 않는다
# ════════════════════════════════════════════════════════════════

PREFIXES = ["아 맞다 ", "음 ", "야 ", "저기 ", "그리고 ", "혹시 "]
_JUL = re.compile(r"([가-힣]+)줘(?![가-힣])")


def _op_hon(rng, s):          # 존댓말·다른 어미
    ms = list(_JUL.finditer(s))
    if not ms:
        return s
    m = ms[-1]
    stem = m.group(1)
    opts = [f"{stem} 주세요", f"{stem}주세요", f"{stem}줄래?", f"{stem}줄 수 있어?", f"{stem}주실래요?", f"{stem}줘요"]
    if stem.endswith("해") and len(stem) >= 3:
        opts.append(f"{stem[:-1]} 부탁해")
    out = s[:m.start()] + rng.choice(opts) + s[m.end():]
    return out.replace("??", "?").replace("?.", "?")


def _op_space(rng, s):        # 다음 주↔다음주↔담주 · 이번 주↔이번주 · 다다음 주↔다다음주 (해석 동일)
    if "다다음 주" in s:
        return s.replace("다다음 주", "다다음주")
    if re.search(r"(?<!다)다음 주(?!말)", s):          # 한 문장 안에서는 같은 표기로 (다음 주 … 담주 섞지 않음)
        if re.search(r"(?<![가-힣])(?:다음주|담주|내주)", s):
            return s
        return re.sub(r"(?<!다)다음 주(?!말)", rng.choice(["다음주", "담주"]), s)
    if "이번 주" in s:
        return s.replace("이번 주", "이번주")
    return s


def _op_drop(rng, s):         # 조사 '에' 생략 — 바로 뒤가 숫자·길이면 건드리지 않는다(시각과 길이가 붙지 않게)
    return re.sub(r"(?<=[일뒤후시반분오])에 (?![\d한두세네반])", " ", s, count=1)


def _op_jom(rng, s):          # '좀' 삽입
    if "좀" in s:
        return s
    toks = s.split(" ")
    for i in range(len(toks) - 1, max(0, len(toks) - 3), -1):
        if i > 0 and not toks[i].startswith("주") and re.search(r"[가-힣](줘|줄래|주세요|주실래요)[?!.~]*$", toks[i]):
            return " ".join(toks[:i] + ["좀"] + toks[i:])
    return s


_QEND = re.compile(r"(어때|돼|있어|있나|있지|뭐야|뭐지|언제야|몇이야|했어|했지|남았어|남았지|거야|아니지|할까|됐어|나와|"
                   r"모자라|충분해|가능해|괜찮아|채웠어|채웠나|들었나|얼마야|있더라|더라|뭘까|힘들까|이겼어|넣었어|와)$")
EMOTIONAL_FAMS = {"refC_counsel", "refC_philosophy", "refC_lost"}


def _op_tail(rng, s, was_q=False, fam=""):   # ㅋㅋ ㅠㅠ ~ ! .. — 질문에는 ㅋㅋ/ㅠㅠ/.. 만, 감정 문장엔 ㅋㅋ 금지
    if s.endswith("..") or s.endswith("ㅋ") or s.endswith("ㅠ"):
        return s
    if fam in EMOTIONAL_FAMS:
        return s + ("ㅠㅠ" if s.endswith("?") else rng.choice(["ㅠㅠ", ".."]))
    if s.endswith("?"):
        return s + rng.choice(["ㅋㅋ", "ㅠㅠ"])
    if was_q or _QEND.search(s):
        return s + rng.choice(["ㅋㅋ", "ㅠㅠ", ".."])
    return s + rng.choice(["ㅋㅋ", "ㅠㅠ", "~", "!", ".."])


def _op_noq(rng, s):          # 물음표 삭제
    return s[:-1] if s.endswith("?") else s


def _op_prefix(rng, s, intent=""):       # 앞 채움말 — 인사·감사엔 안 붙이고, 범위 밖 첫 질문엔 '그리고'·'아 맞다' 금지
    if intent == "chit":
        return s
    opts = [p for p in PREFIXES
            if not (intent == "refusal_oos" and p in ("그리고 ", "아 맞다 "))
            and not (p == "야 " and re.search(r"(요|세요|니다)[?!.~]*$|주세요|주실래요|줘요|니다", s))
            and not (p == "혹시 " and not (s.endswith("?") or "수 있어" in s or "줄래" in s))
            and not s.startswith(p.strip())
            and not (p == "혹시 " and re.match(r"(안녕|하이|헬로|ㅎㅇ)", s))
            and not s.startswith("아 ")]
    return rng.choice(opts) + s if opts else s


SURFACE_OPS: list[tuple[str, Callable]] = [("hon", _op_hon), ("space", _op_space), ("drop", _op_drop), ("jom", _op_jom),
                                           ("noq", _op_noq), ("tail", _op_tail), ("prefix", _op_prefix)]


def surface(rng: random.Random, s: str, intent: str = "", fam: str = "") -> tuple[str, bool]:
    if rng.random() >= SURFACE_P:
        return s, False
    k = rng.choice((1, 1, 2))
    chosen = set(rng.sample([n for n, _ in SURFACE_OPS], k))
    out = s
    was_q = s.endswith("?")
    for name, fn in SURFACE_OPS:          # 고정 순서 — 어미를 바꾼 뒤에 앞말을 붙인다
        if name in chosen:
            if name == "tail":
                out = _op_tail(rng, out, was_q, fam)
            elif name == "prefix":
                out = _op_prefix(rng, out, intent)
            else:
                out = fn(rng, out)
    return out, out != s


# ════════════════════════════════════════════════════════════════
# 6. 생성기 — (rng, now, spec) → (user, steps, kind, tags) | None(불가능 → now 다시 뽑기)
#    steps: ("call", name, args) · ("tool", result) · ("text", answer)
# ════════════════════════════════════════════════════════════════

def g_create(rng, now, sp):
    grp, tpl = sp["group"], sp["tpl"]
    today = now.date()
    title, cat = pick_title(rng, {"weekly": "weekly", "multi": "multi"}.get(grp, "single"))
    slots = {"X": title}
    tags = {"group": grp, "date": sp.get("date"), "time": sp.get("time")}
    if grp == "multi":
        kind = sp["date"]
        span = rng.randint(1, 3)
        if kind == "abs":
            a = today + dt.timedelta(days=rng.randint(1, 200))
            b = a + dt.timedelta(days=span)
            if not (_md_ok(a) and _md_ok(b)):
                return None
            pa = f"{a.month}월 {a.day}일"
            pb = f"{b.day}일" if (b.month == a.month and rng.random() < 0.6) else f"{b.month}월 {b.day}일"
        elif kind in ("next_wd", "nn_wd", "this_wd"):
            mon = week_monday(today)
            off = {"next_wd": 7, "nn_wd": 14, "this_wd": 0}[kind]
            lo = today.weekday() + 1 if kind == "this_wd" else 0
            if lo + span > 6:
                return None
            i = rng.randint(lo, 6 - span)
            a, b = mon + dt.timedelta(days=off + i), mon + dt.timedelta(days=off + i + span)
            word = rng.choice({"next_wd": NEXT_WD_WORDS[:3], "nn_wd": NN_WORDS, "this_wd": THIS_WD_WORDS[:2]}[kind])
            pa, pb = f"{word} {WD[a.weekday()]}요일", f"{WD[b.weekday()]}요일"
        elif kind == "next_month":
            y, m = next_month_of(today)
            last = calendar.monthrange(y, m)[1]
            n = rng.randint(1, last - span)
            a, b = dt.date(y, m, n), dt.date(y, m, n + span)
            pa, pb = f"{rng.choice(['다음 달', '다음달'])} {n}일", f"{n + span}일"
        elif kind == "tomorrow":
            a, b = today + D1, today + 2 * D1
            pa, pb = "내일", "모레"
        else:
            raise ValueError(kind)
        slots.update(A=pa, B=pb)
        args = ev_args(title, cat, a, b + D1, True)
    elif grp == "weekly":
        idx = rng.randint(0, 6)
        first = week_monday(today) + dt.timedelta(days=idx)
        if first <= today:
            first += dt.timedelta(days=7)            # 오늘 이후 첫 X요일 (오늘과 같은 요일이면 +7)
        t = pick_time(rng, sp["time"], half_only="{R}" in tpl)
        slots.update(W=f"{WD[idx]}요일", T=t.phrase)
        s = at(first, t.h, t.m)
        if "{R}" in tpl:
            r = make_range(rng, t)
            if r is None:
                return None
            slots["R"] = r[0]
            e = at(first, r[1], r[2])
        else:
            dur = 60
            if "{L}" in tpl:
                slots["L"], dur = rng.choice(DURS)
            e = s + dt.timedelta(minutes=dur)
            if e.date() != s.date():
                return None
        args = ev_args(title, cat, s, e, False, f"weekly:{WD_CODE[idx]}")
    else:
        dp = date_phrase(rng, sp["date"], now, allday=(grp == "allday"))
        if dp is None:
            return None
        slots["D"], d = dp
        if grp == "allday":
            if at(d, 0, 0) <= now:
                return None
            args = ev_args(title, cat, d, d + D1, True)
        else:
            t = pick_time(rng, sp["time"], half_only=(grp == "range"))
            slots["T"] = t.phrase
            s = at(d, t.h, t.m)
            if s <= now:
                return None
            if grp == "range":
                r = make_range(rng, t)
                if r is None:
                    return None
                slots["R"] = r[0]
                e = at(d, r[1], r[2])
            else:
                dur = 60
                if "{L}" in tpl:
                    slots["L"], dur = rng.choice(DURS)
                e = s + dt.timedelta(minutes=dur)
                if e.date() != s.date():
                    return None
            args = ev_args(title, cat, s, e, False)
    return render(rng, tpl, slots), [("call", "create_event", args)], "single", tags


def g_list(rng, now, sp):
    rp = range_phrase(rng, sp["range"], now)
    if rp is None:
        return None
    phrase, a, b = rp
    slots = {"P": phrase}
    args = {"from": fmt_d(a), "to": fmt_d(b)}
    kk = sp["kinds"]
    if kk:
        noun, clause = K_WORDS[kk]
        slots["K"] = rng.choice(noun + (clause if "c" in sp["tier"] else []))
        args["kinds"] = list(K_KINDS[kk])
    tags = {"range": sp["range"], "kinds": kk, "weekday": now.weekday()}
    return render(rng, sp["tpl"], slots), [("call", "list_events", args)], "single", tags


def g_navigate(rng, now, sp):
    names = [n for n, scr in NAV_NAMES[sp["route"]] if not (sp["needs_bare"] and scr)]
    return render(rng, sp["tpl"], {"S": rng.choice(names)}), [("call", "navigate", {"route": sp["route"]})], "single", {}


def g_sync(rng, now, sp):
    u = render(rng, sp["tpl"], {"S": rng.choice(SYNC_NAMES[sp["source"]])})
    return u, [("call", "run_sync", {"source": sp["source"]})], "single", {}


def g_simple(rng, now, sp):
    """슬롯이 과목뿐인 읽기 도구 (graduation · attendance · exams · assignments · placement)."""
    slots = {"C": rng.choice(COURSES), "RG": rng.choice(PL_RANGE_WORDS[sp.get("range", "none")])}
    return render(rng, sp["tpl"], slots), [("call", sp["tool"], dict(sp["args"]))], "single", {}


def _item_id(rng, used: set) -> str:
    while True:
        n = rng.randint(1000, 99999)
        i = str(n) if rng.random() < 0.5 else f"ev:{n}"
        if i not in used:
            used.add(i)
            return i


def _distractors(rng, d: dt.date, target_title: str, k: int, used: set) -> list[dict]:
    seen_t = {target_title}
    uniq = []
    for x in _distractors_raw(rng, d, target_title, k, used):     # 한 결과 안에서 제목 중복 없게
        if x["title"] not in seen_t:
            seen_t.add(x["title"])
            uniq.append(x)
    return uniq


def _distractors_raw(rng, d: dt.date, target_title: str, k: int, used: set) -> list[dict]:
    out = []
    for _ in range(k):
        typ = rng.choice(["user", "user", "allday", "class", "deadline"])
        if typ in ("user", "allday"):
            for _ in range(50):
                t, _c = pick_title(rng, "single")
                if not related(t, target_title):
                    break
            if typ == "user":
                s = at(d, rng.randint(8, 20), rng.choice([0, 30]))
                e = s + dt.timedelta(minutes=rng.choice([60, 90, 120]))
                out.append({"id": _item_id(rng, used), "title": t, "kind": "user", "start": fmt_dt(s), "end": fmt_dt(e)})
            else:
                out.append({"id": _item_id(rng, used), "title": t, "kind": "user", "start": fmt_d(d), "end": fmt_d(d + D1)})
        elif typ == "class":
            c = rng.choice([c for c in COURSES if not related(c, target_title)])
            s = at(d, rng.choice([9, 10, 12, 13, 15, 16]), rng.choice([0, 30]))
            out.append({"id": _item_id(rng, used), "title": c, "kind": "class", "start": fmt_dt(s),
                        "end": fmt_dt(s + dt.timedelta(minutes=75))})
        else:
            c = rng.choice([c for c in COURSES if not related(c, target_title)])
            due = fmt_dt(at(d, 23, 59))
            out.append({"id": _item_id(rng, used), "title": f"{c} 과제 {rng.randint(1, 5)}", "kind": "deadline",
                        "start": due, "end": due})
    return out


WRITE_DATE_KINDS = ["today", "tomorrow", "tomorrow", "dayafter", "this_wd", "bare_wd", "next_wd", "next_wd", "abs", "n_days"]


def g_update(rng, now, sp):
    ptype, cond, tpl = sp["ptype"], sp["cond"], sp["tpl"]
    today = now.date()
    if ptype == "rel_assign":
        title = f"{rng.choice(COURSES)} 과제 하기"
        cat = TITLE_CAT[title]
    else:
        title, cat = pick_title(rng, "single")
    dp = date_phrase(rng, rng.choice(WRITE_DATE_KINDS), now)
    if dp is None:
        return None
    dphr, d = dp
    dur0 = rng.choice([60, 60, 90, 120, 180])
    s0 = at(d, rng.randint(9, 20), rng.choice([0, 0, 30]))
    e0 = s0 + dt.timedelta(minutes=dur0)
    if s0 <= now or e0.date() != d:
        return None
    slots = {"D": dphr, "X": title}
    patch: dict
    if ptype == "time":
        t = pick_time(rng, rng.choices(["std", "24h", "kor"], weights=[80, 10, 10], k=1)[0])
        s1 = at(d, t.h, t.m)
        if s1 == s0 or s1 <= now or (cond == "later" and s1 <= s0) or (cond == "earlier" and s1 >= s0):
            return None
        e1 = s1 + dt.timedelta(minutes=dur0)
        if e1.date() != d:
            return None
        slots["T"] = t.phrase
        patch = {"start": fmt_dt(s1), "end": fmt_dt(e1)}
    elif ptype == "date":
        dp2 = date_phrase(rng, rng.choice(["tomorrow", "dayafter", "this_wd", "bare_wd", "next_wd", "next_wd", "abs", "n_days"]), now)
        if dp2 is None:
            return None
        ephr, d2 = dp2
        if d2 == d or (cond == "later" and d2 <= d) or ephr == dphr:
            return None
        if re.fullmatch(r"[월화수목금토일]요일", ephr) and d >= week_monday(today) + dt.timedelta(days=7):
            return None          # '다음 주 화요일 … 금요일로' 의 맨 '금요일' 은 애매 — 만들지 않는다
        m1 = re.match(r"(다음 주|다음주|담주|내주) ", dphr)
        m2 = re.match(r"(다음 주|다음주|담주|내주) ", ephr)
        if m1 and m2:            # 출발·도착이 둘 다 다음 주면 같은 표기로
            ephr = m1.group(1) + ephr[len(m2.group(1)):]
            if ephr == dphr:
                return None
        s1 = at(d2, s0.hour, s0.minute)
        if s1 <= now:
            return None
        slots["E"] = ephr
        patch = {"start": fmt_dt(s1), "end": fmt_dt(s1 + dt.timedelta(minutes=dur0))}
    elif ptype in ("rel", "rel_assign"):
        dphrase, mins = rng.choice(DELTA_LATER if cond == "later" else DELTA_EARLIER)
        delta = dt.timedelta(minutes=mins if cond == "later" else -mins)
        s1 = s0 + delta
        e1 = e0 + delta
        if s1 <= now or (mins < 1440 and (s1.date() != d or e1.date() != d)):
            return None
        slots["Δ"] = dphrase
        patch = {"start": fmt_dt(s1), "end": fmt_dt(e1)}
    else:  # title
        for _ in range(80):                  # 바꿀 제목은 같은 분류에서 고른다 (분류가 튀지 않게)
            y, yc = pick_title(rng, "single")
            if yc == cat and not related(y, title):
                break
        else:
            return None
        slots["Y"] = y
        patch = {"title": y}
    user = render(rng, tpl, slots)
    used: set = set()
    if sp["zero"]:
        steps = [("call", "list_events", {"from": fmt_d(d), "to": fmt_d(d)}), ("tool", {"items": []}),
                 ("text", not_found_text(d, title))]
        return user, steps, "two_step", {"ptype": ptype, "zero": True}
    target = {"id": _item_id(rng, used), "title": title, "kind": "user", "start": fmt_dt(s0), "end": fmt_dt(e0)}
    items = sorted([target] + _distractors(rng, d, title, rng.choice([0, 1, 1, 2]), used), key=lambda x: (x["start"], x["id"]))
    steps = [("call", "list_events", {"from": fmt_d(d), "to": fmt_d(d)}), ("tool", {"items": items}),
             ("call", "update_event", {"id": target["id"], "patch": patch})]
    return user, steps, "two_step", {"ptype": ptype, "zero": False}


def g_delete(rng, now, sp):
    tpl = sp["tpl"]
    if sp["fidx"] == DEL_CONTRAST_IDX:
        title = f"{rng.choice(COURSES)} 스터디"
    else:
        title, _c = pick_title(rng, "any")
    dp = date_phrase(rng, rng.choice(WRITE_DATE_KINDS), now)
    if dp is None:
        return None
    dphr, d = dp
    all_day = TITLE_FLAGS[title] == "m" or rng.random() < 0.2
    if all_day:
        if at(d, 0, 0) <= now:
            return None
        st, en = fmt_d(d), fmt_d(d + D1)
    else:
        s0 = at(d, rng.randint(8, 21), rng.choice([0, 0, 30]))
        e0 = s0 + dt.timedelta(minutes=rng.choice([60, 90, 120]))
        if s0 <= now or e0.date() != d:
            return None
        st, en = fmt_dt(s0), fmt_dt(e0)
    user = render(rng, tpl, {"D": dphr, "X": title})
    if sp["zero"]:
        steps = [("call", "list_events", {"from": fmt_d(d), "to": fmt_d(d)}), ("tool", {"items": []}),
                 ("text", not_found_text(d, title))]
        return user, steps, "two_step", {"zero": True}
    used: set = set()
    target = {"id": _item_id(rng, used), "title": title, "kind": "user", "start": st, "end": en}
    items = sorted([target] + _distractors(rng, d, title, rng.choice([0, 1, 1, 2]), used), key=lambda x: (x["start"], x["id"]))
    steps = [("call", "list_events", {"from": fmt_d(d), "to": fmt_d(d)}), ("tool", {"items": items}),
             ("call", "delete_event", {"id": target["id"]})]
    return user, steps, "two_step", {"zero": False}


def _ref_slots(rng, now) -> dict:
    """거절 문장용 슬롯 — 날짜는 그럴듯하게만(정답에 안 쓰임)."""
    c, c2 = rng.sample(COURSES, 2)
    dps = []
    for k in ("tomorrow", "dayafter", "next_wd", "abs", "n_days", "next_month"):
        dps.append(date_phrase(rng, k, now, allday=True))
    dps = [x for x in dps if x]
    da, db = sorted(rng.sample(dps, 2), key=lambda x: x[1])
    w, w2 = rng.sample(range(7), 2)
    t, u = pick_time(rng, "std"), pick_time(rng, "std")
    s = {"C": c, "C2": c2, "D": da[0], "E": db[0], "W": f"{WD[w]}요일", "V": f"{WD[w2]}요일", "T": t.phrase,
         "U": u.phrase, "AE": rng.choice(ACADEMIC_EVENTS), "AP": rng.choice([a for a in ACADEMIC_EVENTS if a.endswith(("기간", "주간"))]),
         "N": str(rng.choice([10, 15, 20, 30])),
         "DD": rng.choice(["하루", "이틀", "사흘", "일주일"])}
    for k, v in REF_C_SLOTS.items():
        s[k] = rng.choice(v)
    s["FOOD2"] = rng.choice([f for f in REF_C_SLOTS["FOOD"] if f != s["FOOD"]])
    return s


def g_refusal(rng, now, sp):
    return render(rng, sp["tpl"], _ref_slots(rng, now)), [("text", sp["answer"])], "refusal", {}


def g_chit(rng, now, sp):
    return render(rng, sp["tpl"], {}), [("text", sp["answer"])], "text", {}


def g_two_step(rng, now, sp):
    """도구 결과 뒤 답변 — 답 템플릿은 prepare_f9 의 ts_* 와 같다."""
    sub = sp["sub"]
    today = now.date()
    mon = week_monday(today)
    c = rng.choice(COURSES)
    if sub == "le":
        u = render(rng, sp["tpl"], {"P": rng.choice(["이번 주", "이번 주", "이번주"]),
                                    "K": rng.choice(K_WORDS["deadline"][0] + (K_WORDS["deadline"][1] if "c" in sp["tier"] else []))})
        args = {"from": fmt_d(mon), "to": fmt_d(mon + 6 * D1), "kinds": ["deadline"]}
        items, used, titles = [], set(), set()
        for _ in range(rng.randint(0, 3)):        # 백엔드 모양 {"id","title","kind","start","end"} — 마감은 start = end = 기한
            cc = rng.choice(COURSES)
            d = mon + dt.timedelta(days=rng.randint(max(0, today.weekday()), 6))
            title = f"{cc} 과제 {rng.randint(1, 5)}"
            if title in titles:
                continue
            titles.add(title)
            due = fmt_dt(at(d, 23, 59))
            items.append({"id": _item_id(rng, used), "title": title, "kind": "deadline", "start": due, "end": due})
        items.sort(key=lambda x: (x["start"], x["id"]))
        if not items:
            ans = "이번 주에 남은 마감이 없습니다. 캘린더 열기(/)"
        else:
            ans = f"이번 주 마감 {len(items)}건: " + ", ".join(
                f"{it['title']}({kr_md(dt.date.fromisoformat(it['start'][:10]))} {it['start'][11:]})" for it in items) + ". 캘린더 열기(/)"
        return u, [("call", "list_events", args), ("tool", {"items": items}), ("text", ans)], "two_step", {}
    if sub == "gr":
        total = rng.choice([130, 140])
        earned = rng.randint(60, total - 3)
        return render(rng, sp["tpl"], {}), [
            ("call", "get_graduation", {}),
            ("tool", {"earned": earned, "totalRequired": total, "remaining": total - earned, "status": "진행 중"}),
            ("text", f"졸업까지 {total - earned}학점 남았습니다 (이수 {earned} / {total}). 졸업요건 열기(/graduation)")], "two_step", {}
    if sub == "at":
        limit = rng.choice([4, 5])
        absent = rng.randint(0, limit - 1)
        risky = absent >= limit - 1
        ans = f"{josa(c, '은')} 결석 {absent}회, 한도 {limit}회입니다." + (" 한도에 가까워 주의가 필요합니다." if risky else "") + " 출결 열기(/attendance)"
        return render(rng, sp["tpl"], {"C": c}), [("call", "get_attendance", {"course": c}),
                                                   ("tool", {"course": c, "absent": absent, "limit": limit, "risky": risky}),
                                                   ("text", ans)], "two_step", {}
    if sub == "as":
        items = []
        for _ in range(rng.randint(1, 3)):
            d = now + dt.timedelta(days=rng.randint(1, 6))
            items.append({"title": f"{rng.choice(COURSES)} 과제 {rng.randint(1, 5)}",
                          "due": fmt_dt(d.replace(hour=23, minute=59)), "hours": rng.randint(1, 5)})
        ans = f"지금 해야 할 과제 {len(items)}건: " + ", ".join(
            f"{it['title']}({kr_md(dt.date.fromisoformat(it['due'][:10]))} 23:59 마감, 예상 {it['hours']}시간)" for it in items) + ". 과제 열기(/assignments)"
        return render(rng, sp["tpl"], {}), [("call", "list_assignments", {"group": "now"}),
                                             ("tool", {"group": "now", "items": items}), ("text", ans)], "two_step", {}
    if sub == "ex":
        kind = rng.choice(["중간고사", "기말고사"])
        d = today + dt.timedelta(days=rng.randint(3, 30))
        return render(rng, sp["tpl"], {}), [("call", "get_exams", {"upcoming": True}),
                                             ("tool", {"items": [{"course": c, "title": kind, "date": fmt_d(d)}]}),
                                             ("text", f"다음 시험은 {c} {kind}, {kr_md(d)}입니다. 시험 열기(/exams)")], "two_step", {}
    if sub == "sync":
        name = rng.choice(TS_SYNC_NAMES[sp["source"]])
        state = rng.choice(["started", "started", "running"])
        return render(rng, sp["tpl"], {"S": name}), [
            ("call", "run_sync", {"source": sp["source"]}), ("tool", {"source": sp["source"], "state": state}),
            ("text", f"{name} 수집을 시작했습니다." if state == "started" else f"{name} 수집이 이미 진행 중입니다.")], "two_step", {}
    raise ValueError(sub)


# ════════════════════════════════════════════════════════════════
# 7. 배분(plan) — 의도별 건수·하위 배분을 정확히 맞추고, 틀은 순환(cycler)으로 고르게 쓴다
# ════════════════════════════════════════════════════════════════

def split(total: int, weights: dict) -> dict:
    """최대 나머지 방식 — 합이 정확히 total."""
    keys = list(weights)
    tot = sum(weights.values())
    raw = [total * weights[k] / tot for k in keys]
    base = [int(x) for x in raw]
    for i in sorted(range(len(keys)), key=lambda i: -(raw[i] - base[i]))[: total - sum(base)]:
        base[i] += 1
    return dict(zip(keys, base))


def quota(rng: random.Random, total: int, weights: dict) -> list:
    out = [k for k, c in split(total, weights).items() for _ in range(c)]
    rng.shuffle(out)
    return out


class Cycler:
    """항목을 섞어 한 바퀴씩 돈다 — 모든 틀이 고르게 쓰인다. pred 로 호환되는 것만 고를 수 있다."""

    def __init__(self, rng: random.Random, items: list):
        self.rng, self.items, self.q = rng, list(items), []

    def take(self, pred=lambda x: True):
        for _ in range(3):
            for i, x in enumerate(self.q):
                if pred(x):
                    return self.q.pop(i)
            self.q = list(self.items)
            self.rng.shuffle(self.q)
        raise RuntimeError("Cycler: 맞는 항목 없음")


def fam_list(prefix: str, tpls: list) -> list[tuple[str, object]]:
    return [(f"{prefix}{i + 1:02d}", t) for i, t in enumerate(tpls)]


CE_FAMS = {"timed": fam_list("ce_t", CE_TIMED), "range": fam_list("ce_r", CE_RANGE), "allday": fam_list("ce_a", CE_ALLDAY),
           "multi": fam_list("ce_m", CE_MULTI), "weekly": fam_list("ce_w", CE_WEEKLY)}
CE_GROUPS = {"timed": 1130, "range": 330, "allday": 420, "multi": 110, "weekly": 210}
CE_DATES = {
    "timed": {"next_wd": 390, "nn_wd": 55, "abs": 190, "n_days": 70, "next_month": 65, "today": 70, "tomorrow": 110,
              "dayafter": 60, "this_wd": 60, "bare_wd": 60},
    "range": {"next_wd": 110, "nn_wd": 15, "abs": 50, "n_days": 20, "next_month": 20, "today": 20, "tomorrow": 45,
              "dayafter": 20, "this_wd": 15, "bare_wd": 15},
    "allday": {"next_wd": 130, "nn_wd": 20, "abs": 110, "n_days": 30, "next_month": 30, "tomorrow": 45, "dayafter": 25,
               "this_wd": 15, "bare_wd": 15},
    "multi": {"abs": 45, "next_wd": 25, "next_month": 20, "tomorrow": 10, "this_wd": 5, "nn_wd": 5},
}
CE_TIMES = {"timed": {"std": 870, "24h": 130, "kor": 130}, "range": {"std": 270, "24h": 30, "kor": 30},
            "weekly": {"std": 160, "24h": 25, "kor": 25}}
LE_RANGES = {"this_week": 18, "next_week": 13, "nn_week": 4, "today": 8, "tomorrow": 8, "dayafter": 4, "this_weekend": 5,
             "next_weekend": 4, "this_month": 5, "next_month": 4, "bare_wd": 5, "this_wd": 5, "next_wd": 9, "abs": 5,
             "n_days": 3}
LE_KINDS = {None: 525, "deadline": 300, "exam": 140, "class": 140, "user": 120, "study": 110, "academic": 75,
            "deadline_exam": 50, "class_exam": 40}


def build_plan(rng: random.Random, n: int) -> list[dict]:
    counts = split(n, TARGETS)
    plan: list[dict] = []

    # create
    nc = counts["create"]
    for grp, gc in split(nc, CE_GROUPS).items():
        dates = quota(rng, gc, CE_DATES[grp]) if grp in CE_DATES else [None] * gc
        times = quota(rng, gc, CE_TIMES[grp]) if grp in CE_TIMES else [None] * gc
        cyc = Cycler(rng, CE_FAMS[grp])
        for dk, tk in zip(dates, times):
            fam, tpl = cyc.take()
            plan.append({"intent": "create", "gen": g_create, "group": grp, "date": dk, "time": tk, "fam": fam, "tpl": tpl})

    # list_events
    nl = counts["list_events"]
    ranges = quota(rng, nl, LE_RANGES)
    kinds = quota(rng, nl, {k if k else "_none": v for k, v in LE_KINDS.items()})
    kc, pc = Cycler(rng, fam_list("le_k", LE_KINDED)), Cycler(rng, fam_list("le_p", LE_PLAIN))
    for rk, kk in zip(ranges, kinds):
        kk = None if kk == "_none" else kk
        if kk:
            combo = kk in ("deadline_exam", "class_exam")
            multi_ok = rk not in DAY_ONLY_RANGES
            fam, (tpl, tier, *_flags) = kc.take(lambda f: (combo or not f[1][2]) and (multi_ok or len(f[1]) < 4))
        else:
            fam, (tpl, day_only) = pc.take(lambda f: rk in DAY_ONLY_RANGES or not f[1][1])
            tier = ""
        plan.append({"intent": "list_events", "gen": g_list, "range": rk, "kinds": kk, "tier": tier, "fam": fam, "tpl": tpl})

    # navigate
    routes = quota(rng, counts["navigate"], {r: 1 for r in NAV_NAMES})
    nav_cyc = Cycler(rng, fam_list("nav_", NAV_FRAMES))
    for r in routes:
        bare_ok = any(not scr for _, scr in NAV_NAMES[r])
        fam, (tpl, needs_bare) = nav_cyc.take(lambda f: bare_ok or not f[1][1])
        plan.append({"intent": "navigate", "gen": g_navigate, "route": r, "needs_bare": needs_bare, "fam": fam, "tpl": tpl})

    # run_sync
    sync_cyc = Cycler(rng, fam_list("sync_", SYNC_FRAMES))
    for src in quota(rng, counts["run_sync"], {s: 1 for s in SYNC_NAMES}):
        fam, tpl = sync_cyc.take()
        plan.append({"intent": "run_sync", "gen": g_sync, "source": src, "fam": fam, "tpl": tpl})

    def simple(intent, tool, groups: dict, fams_by_group: dict, argmap: dict, extra=None):
        cycs = {g: Cycler(rng, fams_by_group[g]) for g in groups}
        for g in quota(rng, counts[intent], groups):
            fam, tpl = cycs[g].take()
            sp = {"intent": intent, "gen": g_simple, "tool": tool, "args": argmap[g], "fam": fam, "tpl": tpl}
            if extra:
                sp.update(extra(g))
            plan.append(sp)

    simple("get_graduation", "get_graduation", {"all": 1}, {"all": fam_list("gr_", GR_FRAMES)}, {"all": {}})
    simple("get_attendance", "get_attendance", {"course": 60, "all": 40},
           {"course": fam_list("at_c", AT_COURSE), "all": fam_list("at_a", AT_ALL)}, {"course": None, "all": {}})
    # get_attendance 과목 지정은 슬롯 값이 인자에 들어가므로 g_simple 대신 아래에서 감싼다
    for sp in plan:
        if sp["intent"] == "get_attendance" and sp["args"] is None:
            sp["gen"] = g_attendance_course
    simple("get_exams", "get_exams", {"up": 70, "plan": 30},
           {"up": fam_list("ex_u", EX_UP), "plan": fam_list("ex_p", EX_PLAN)}, {"up": {"upcoming": True}, "plan": {}})
    simple("list_assignments", "list_assignments", {"now": 40, "this_week": 25, "later": 15, "all": 20},
           {g: fam_list(f"as_{g}_", AS_FRAMES[g]) for g in AS_FRAMES},
           {"now": {"group": "now"}, "this_week": {"group": "this_week"}, "later": {"group": "later"}, "all": {}})
    pl_fams = fam_list("pl_", PL_FRAMES)
    simple("open_placement", "open_placement", {"this_week": 40, "next_week": 30, "none": 30},
           {"this_week": pl_fams, "next_week": pl_fams, "none": pl_fams},
           {"this_week": {"range": "this_week"}, "next_week": {"range": "next_week"}, "none": {}}, extra=lambda g: {"range": g})

    # update — patch 종류 배분 · 0건 5%
    nu = counts["update"]
    ptypes = quota(rng, nu, {"time": 300, "date": 75, "rel": 60, "rel_assign": 15, "title": 50})
    zeros = quota(rng, nu, {True: 5, False: 95})
    up_cycs = {}
    for i, (pt, cond, tpl) in enumerate(UP_FRAMES):
        up_cycs.setdefault(pt, []).append((f"up_{i + 1:02d}", (cond, tpl)))
    up_cycs = {k: Cycler(rng, v) for k, v in up_cycs.items()}
    for pt, z in zip(ptypes, zeros):
        fam, (cond, tpl) = up_cycs[pt].take()
        plan.append({"intent": "update", "gen": g_update, "ptype": pt, "cond": cond, "zero": z, "fam": fam, "tpl": tpl})

    # delete
    nd = counts["delete"]
    del_cyc = Cycler(rng, [(f"del_{i + 1:02d}", (i, t)) for i, t in enumerate(DEL_FRAMES)])
    for z in quota(rng, nd, {True: 5, False: 95}):
        fam, (fi, tpl) = del_cyc.take()
        plan.append({"intent": "delete", "gen": g_delete, "fidx": fi, "zero": z, "fam": fam, "tpl": tpl})

    # 거절 A·B·C
    for intent, fams, ans in (("refusal_eclass", fam_list("refA_", REF_A), REF_ECLASS),
                              ("refusal_class", fam_list("refB_", REF_B), REF_CLASS)):
        cyc = Cycler(rng, fams)
        for _ in range(counts[intent]):
            fam, tpl = cyc.take()
            plan.append({"intent": intent, "gen": g_refusal, "answer": ans, "fam": fam, "tpl": tpl})
    topic_cyc = Cycler(rng, list(REF_C))
    for _ in range(counts["refusal_oos"]):
        topic = topic_cyc.take()
        plan.append({"intent": "refusal_oos", "gen": g_refusal, "answer": REF_OOS, "fam": f"refC_{topic}",
                     "tpl": rng.choice(REF_C[topic])})

    # 인사·감사
    chit_cyc = Cycler(rng, list(CHIT))
    for _ in range(counts["chit"]):
        fam = chit_cyc.take()
        ans, tpls = CHIT[fam]
        plan.append({"intent": "chit", "gen": g_chit, "answer": ans, "fam": fam, "tpl": rng.choice(tpls)})

    # 도구 결과 뒤 답변 (2-step 읽기)
    ts_fams = {
        "le": [(f"ts_le_k{i + 1:02d}", (f[0], f[1])) for i, f in enumerate(LE_KINDED) if not f[2]],
        "gr": [(f"ts_gr_{i + 1:02d}", (GR_FRAMES[i], "")) for i in TS_GR_IDX],
        "at": [(f"ts_at_c{i + 1:02d}", (t, "")) for i, t in enumerate(AT_COURSE) if i not in TS_AT_EXCLUDE],
        "as": [(f"ts_as_now_{i + 1:02d}", (t, "")) for i, t in enumerate(AS_FRAMES["now"])],
        "ex": [(f"ts_ex_u{i + 1:02d}", (EX_UP[i], "")) for i in TS_EX_IDX],
        "sync": [(f"ts_sync_{i + 1:02d}", (t, "")) for i, t in enumerate(SYNC_FRAMES)],
    }
    ts_cycs = {k: Cycler(rng, v) for k, v in ts_fams.items()}
    sync_src = Cycler(rng, list(TS_SYNC_NAMES))
    for sub in quota(rng, counts["two_step_read"], {"le": 80, "gr": 60, "at": 70, "as": 70, "ex": 60, "sync": 60}):
        fam, (tpl, tier) = ts_cycs[sub].take()
        sp = {"intent": "two_step_read", "gen": g_two_step, "sub": sub, "tier": tier, "fam": fam, "tpl": tpl}
        if sub == "sync":
            sp["source"] = sync_src.take()
        plan.append(sp)
    assert len(plan) == n, (len(plan), n)
    return plan


def g_attendance_course(rng, now, sp):
    c = rng.choice(COURSES)
    return render(rng, sp["tpl"], {"C": c}), [("call", "get_attendance", {"course": c})], "single", {}


# ════════════════════════════════════════════════════════════════
# 8. 행 만들기 — prepare_f9.build_row 와 같은 모양
# ════════════════════════════════════════════════════════════════

REJECT: dict[str, int] = {"exact": 0, "normalised": 0}     # 생성 중 평가셋 때문에 다시 뽑은 횟수


def make_row(now: dt.datetime, user: str, steps: list, fam: str, kind: str, tools: list[dict]) -> dict:
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
    return {"id": "", "messages": msgs, "tools": tools,
            "meta": {"tool": first_tool, "template": fam, "kind": kind, "source": "synthetic-v2"}}


_PREFIX_RE = re.compile(r"^(?:아 맞다|음|야|저기|그리고|혹시)\s+")
_TAIL_RE = re.compile(r"(?:ㅋ+|ㅠ+|~+|!+|\.+|\?+)$")


def norm_key(s: str) -> str:
    """평가셋 근접 사본 검사용 정규화 — §5 표면 변형(앞말·꼬리·좀·어미·띄어쓰기)을 걷어 낸다."""
    t = s.strip()
    while True:
        u = _TAIL_RE.sub("", _PREFIX_RE.sub("", t)).strip()
        if u == t:
            break
        t = u
    t = re.sub(r"(?<![가-힣])(?:다음주|담주|내주)", "다음 주", t)
    t = re.sub(r"(?<![가-힣])이번주", "이번 주", t)
    t = re.sub(r"(?<![가-힣])좀(?![가-힣])", " ", t)
    t = re.sub(r"\s*(?:주세요|주실래요|줄 수 있어|줄래|줘요)", "줘", t)
    t = re.sub(r"(\S+) 부탁해(?=$|[\s?!.~])", r"\1해줘", t)
    t = re.sub(r"[\s?!.~]", "", t)
    return t


def generate(rng: random.Random, n: int, tools: list[dict], eval_users: set) -> tuple[list[dict], list[dict]]:
    plan = build_plan(rng, n)
    seen: dict[str, int] = {}
    eval_norm = {norm_key(u) for u in eval_users}
    REJECT.update(exact=0, normalised=0)
    rows, tags = [], []
    for sp in plan:
        for attempt in range(800):
            if attempt == 300 and sp["intent"] == "list_events" and sp["range"] in DAY_ONLY_RANGES:
                # 이 틀×범위의 모든 변형이 평가셋 근접 사본이면('오늘 뭐 해야 돼?' 류) 같은 틀을 다른 하루 범위로
                sp = dict(sp, range="tomorrow" if sp["range"] != "tomorrow" else "dayafter")
            now = rand_now(rng)
            out = sp["gen"](rng, now, sp)
            if out is None:
                continue
            user, steps, kind, tg = out
            user, sv = surface(rng, user, sp["intent"], sp["fam"])
            if user in eval_users:                       # 평가셋 누출 0 (정확히 같은 문장)
                REJECT["exact"] += 1
                continue
            if norm_key(user) in eval_norm:              # 표면 변형만 다른 근접 사본도 거른다
                REJECT["normalised"] += 1
                continue
            c = seen.get(user, 0)
            if c >= MAX_DUP or (c and attempt < 120):     # 먼저 새 문장을 찾고, 끝내 없으면 5회까지만 중복 허용
                continue
            break
        else:
            raise RuntimeError(f"생성 실패: {sp['fam']}")
        seen[user] = c + 1
        rows.append(make_row(now, user, steps, sp["fam"], kind, tools))
        tags.append(dict(tg, intent=sp["intent"], sv=sv))
    order = list(range(n))
    rng.shuffle(order)
    rows, tags = [rows[i] for i in order], [tags[i] for i in order]
    for i, r in enumerate(rows):
        r["id"] = f"f9v2-{i + 1:05d}"
    return rows, tags


# ════════════════════════════════════════════════════════════════
# 9. 독립 검산기 — 사용자 문장에서 날짜·시각을 다시 읽어 정답과 대조한다 (생성 로직과 별도 구현)
# ════════════════════════════════════════════════════════════════

_WDC = "월화수목금토일"
_PER = "오전|오후|아침|점심|낮|저녁|밤"
_NUM = r"\d{1,2}|열한|열두|열|한|두|세|네|다섯|여섯|일곱|여덟|아홉"
_KNUM = {v: k for k, v in KN.items()}


def _num(s: str) -> int:
    return int(s) if s.isdigit() else _KNUM[s]


def parse_dates(text: str, today: dt.date) -> list[dt.date]:
    mon = today - dt.timedelta(days=today.weekday())
    pats: list[tuple[str, Callable]] = [
        (r"다다음\s?주\s?([월화수목금토일])요일", lambda m: mon + dt.timedelta(days=14 + _WDC.index(m.group(1)))),
        (r"(?<!다)(?:다음\s?주|담주|내주)\s?([월화수목금토일])요일", lambda m: mon + dt.timedelta(days=7 + _WDC.index(m.group(1)))),
        (r"(?:이번\s?주\s?|이번\s)([월화수목금토일])요일", lambda m: mon + dt.timedelta(days=_WDC.index(m.group(1)))),
        (r"다음\s?달\s?(\d{1,2})일", lambda m: dt.date(*next_month_of(today), int(m.group(1)))),
        (r"(\d{1,2})월\s?(\d{1,2})일", lambda m: _abs(today, int(m.group(1)), int(m.group(2)))),
        (r"(?<![\d/])(\d{1,2})/(\d{1,2})(?![\d/])", lambda m: _abs(today, int(m.group(1)), int(m.group(2)))),
        (r"(\d{1,2})일\s?(?:뒤|후)", lambda m: today + dt.timedelta(days=int(m.group(1)))),
        (r"내일모레", lambda m: today + 2 * D1),
        (r"(?<!내일)모레", lambda m: today + 2 * D1),
        (r"내일(?!모레)", lambda m: today + D1),
        (r"오늘", lambda m: today),
        (r"([월화수목금토일])요일", lambda m: mon + dt.timedelta(days=_WDC.index(m.group(1)))),
    ]
    hits = []
    for pi, (p, f) in enumerate(pats):
        for m in re.finditer(p, text):
            hits.append((m.start(), -(m.end() - m.start()), pi, m.end(), f(m)))
    hits.sort()
    out, last_end = [], -1
    for st, _neg, _pi, en, d in hits:
        if st >= last_end:
            out.append(d)
            last_end = en
    return out


def _abs(today: dt.date, m: int, d: int) -> dt.date:
    x = dt.date(today.year, m, d)
    return x if x >= today else dt.date(today.year + 1, m, d)


_RE_T = re.compile(rf"(?P<per>{_PER})\s?(?P<n>{_NUM})\s?시(?!간)(?:\s?(?P<mm>반|\d{{1,2}}분))?"
                   rf"|(?P<noon>정오)"
                   rf"|(?<!\d)(?P<h24>1[3-9]|2[0-3])시(?!간)(?:\s?(?P<mm24>반|30분))?"
                   rf"|(?:부터|에서|~)\s?(?P<n2>{_NUM})\s?시(?!간)(?:\s?(?P<mm2>반|\d{{1,2}}분))?")


def _mins(s: Optional[str]) -> int:
    if not s:
        return 0
    return 30 if s == "반" else int(s[:-1])


def parse_times(text: str) -> list[tuple[int, int]]:
    out, last_per = [], None
    for m in _RE_T.finditer(text):
        if m.group("per"):
            last_per = m.group("per")
            out.append((to24(last_per, _num(m.group("n"))), _mins(m.group("mm"))))
        elif m.group("noon"):
            last_per = None
            out.append((12, 0))
        elif m.group("h24"):
            last_per = None
            out.append((int(m.group("h24")), _mins(m.group("mm24"))))
        elif m.group("n2"):
            v = _num(m.group("n2"))
            if v >= 13:                                  # '19시~22시' — 24시간 표기 끝 시각
                out.append((v, _mins(m.group("mm2"))))
            elif last_per:                               # 끝 시각 — 앞 시간대 상속
                out.append((to24(last_per, v), _mins(m.group("mm2"))))
    return out


def parse_range(text: str, today: dt.date) -> Optional[tuple[dt.date, dt.date]]:
    ds = parse_dates(text, today)
    if ds:
        return ds[0], ds[0]
    mon = today - dt.timedelta(days=today.weekday())
    w = lambda a, b: (mon + dt.timedelta(days=a), mon + dt.timedelta(days=b))  # noqa: E731
    if re.search(r"이번\s?주말", text):
        return w(5, 6)
    if re.search(r"다음\s?주말", text):
        return w(12, 13)
    if re.search(r"다다음\s?주", text):
        return w(14, 20)
    if re.search(r"(?<!다)(?:다음\s?주|담주|내주)", text):
        return w(7, 13)
    if re.search(r"이번\s?주", text):
        return w(0, 6)
    if re.search(r"이번\s?달", text):
        return month_bounds(today.year, today.month)
    if re.search(r"다음\s?달", text):
        return month_bounds(*next_month_of(today))
    return None


# §6-9 — 시간대 말 없는 1~12시 찾기 (끝 시각 상속 꼴 '오후 2시부터 4시까지' 는 허용)
_TP = rf"(?:{_PER})\s?(?:{_NUM})\s?시(?!간)(?:\s?(?:반|\d{{1,2}}분))?"
_INHERIT = re.compile(_TP + rf"\s?(?:부터|에서|~)\s?(?:{_NUM})\s?시(?!간)")
_TP_RE = re.compile(_TP)
_BARE_DIGIT = re.compile(r"(?<!\d)(?:1[0-2]|[1-9])\s?시(?![간험작즌])")
_BARE_KOR = re.compile(r"(?:^|\s)(?:열한|열두|열|한|두|세|네|다섯|여섯|일곱|여덟|아홉)\s?시(?![간험작즌])")


def bare_hour_hits(s: str) -> list[str]:
    t = _INHERIT.sub(" ", s)
    t = _TP_RE.sub(" ", t)
    return _BARE_DIGIT.findall(t) + _BARE_KOR.findall(t)


# ════════════════════════════════════════════════════════════════
# 10. 품질 게이트 (§6) — 전부 assert. 수치는 README 로
# ════════════════════════════════════════════════════════════════

_SYS_NOW = re.compile(r"^오늘은 (\d{4}-\d{2}-\d{2})\([월화수목금토일]\) (\d{2}:\d{2}) 입니다\.")


def _parse_any(s: str) -> tuple[dt.datetime, bool]:
    """(값, 시각 포함 여부). 형식이 틀리면 ValueError."""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", s):
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M"), True
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return dt.datetime.strptime(s, "%Y-%m-%d"), False
    raise ValueError(s)


def _check_args(name: str, args: dict, schema: dict) -> None:
    props = schema["properties"]
    for r in schema.get("required", []):
        assert r in args, (name, "필수 인자 없음", r, args)
    for k, v in args.items():
        assert k in props, (name, "정의에 없는 인자", k)
        p = props[k]
        if "enum" in p:
            assert v in p["enum"], (name, k, v)
        if p.get("type") == "array":
            assert isinstance(v, list) and v and all(x in p["items"]["enum"] for x in v), (name, k, v)
        if p.get("type") == "boolean":
            assert isinstance(v, bool), (name, k, v)
        if p.get("type") == "string":
            assert isinstance(v, str) and v, (name, k, v)
        if p.get("type") == "object":
            assert isinstance(v, dict) and v, (name, k, v)


def intent_of(fam: str) -> str:
    for pre, it in (("ts_", "two_step_read"), ("ce_", "create"), ("le_", "list_events"), ("nav_", "navigate"),
                    ("sync_", "run_sync"), ("gr_", "get_graduation"), ("at_", "get_attendance"), ("ex_", "get_exams"),
                    ("as_", "list_assignments"), ("pl_", "open_placement"), ("up_", "update"), ("del_", "delete"),
                    ("refA_", "refusal_eclass"), ("refB_", "refusal_class"), ("refC_", "refusal_oos"), ("chit_", "chit")):
        if fam.startswith(pre):
            return it
    raise ValueError(fam)


def validate(rows: list[dict], tags: list[dict], tools: list[dict], eval_users: set, n: int) -> dict:
    M: dict = {}
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    scale = n / 10000

    # 1. 행 수 · 의도별 건수 ±5%
    assert len(rows) == n, len(rows)
    intents = Counter(intent_of(r["meta"]["template"]) for r in rows)
    M["intents"] = {}
    for k, tgt in TARGETS.items():
        exp = tgt * scale
        assert abs(intents[k] - exp) <= max(0.05 * exp, 1), (k, intents[k], exp)
        M["intents"][k] = (intents[k], round(exp))
    assert all(intent_of(r["meta"]["template"]) == t["intent"] for r, t in zip(rows, tags))

    # 2. 고유 문장 · 최대 중복
    users = [r["messages"][1]["content"] for r in rows]
    cnt = Counter(users)
    M["unique"], M["max_dup"] = len(cnt), max(cnt.values())
    assert M["unique"] >= 0.8 * n, M["unique"]
    assert M["max_dup"] <= MAX_DUP, cnt.most_common(3)

    # 3. 평가셋 누출
    M["eval_users"] = len(eval_users)
    M["leak"] = sum(u in eval_users for u in users)
    assert eval_users, "eval.jsonl 사용자 문장을 읽지 못했다"
    assert M["leak"] == 0, M["leak"]
    eval_norm = {norm_key(u) for u in eval_users}
    M["leak_norm"] = sum(norm_key(u) in eval_norm for u in users)
    assert M["leak_norm"] == 0, [u for u in users if norm_key(u) in eval_norm][:5]
    M["rejected"] = dict(REJECT)

    # 4·5·6·7·8 — 행 단위
    refusals = {REF_ECLASS, REF_CLASS, REF_OOS}
    src = Path(prepare_f9.__file__).read_text(encoding="utf-8")
    for s in refusals:
        assert f'"{s}"' in src, ("prepare_f9 거절 문장과 다름", s)            # 7. 바이트 동일
        assert any(m in s for m in REFUSAL_MARKERS), s
    assert RULES in system_message(dt.datetime(2026, 9, 1, 9, 0))
    ref_tools = load_tools(TOOLS_PATH)[0]
    n_write = n_dates = text_mismatch = 0
    weekday_this_week: set = set()
    mism: list = []
    for r, tg in zip(rows, tags):
        msgs = r["messages"]
        assert list(r) == ["id", "messages", "tools", "meta"] and r["meta"]["source"] == "synthetic-v2"
        assert r["meta"]["kind"] in ("single", "two_step", "refusal", "text")
        m = _SYS_NOW.match(msgs[0]["content"])
        now = dt.datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M")
        assert msgs[0]["content"] == system_message(now)                     # 8
        assert r["tools"] == ref_tools                                        # 8
        assert msgs[1]["role"] == "user" and msgs[1]["content"]
        today = now.date()
        user = msgs[1]["content"]
        last_items = None
        calls = []
        for mm in msgs[2:]:
            if mm["role"] == "assistant" and mm.get("tool_calls"):
                fn = mm["tool_calls"][0]["function"]
                name, args = fn["name"], json.loads(fn["arguments"])
                assert fn["arguments"] == json.dumps(args, ensure_ascii=False)
                assert name in schemas, name
                _check_args(name, args, schemas[name])                      # 5
                calls.append((name, args))
                if name == "list_events":
                    a, b = dt.date.fromisoformat(args["from"]), dt.date.fromisoformat(args["to"])
                    assert a <= b
                    n_dates += 1
                if name == "create_event":
                    n_write += 1
                    s, s_t = _parse_any(args["start"])
                    e, e_t = _parse_any(args["end"])
                    n_dates += 2
                    assert s > now, ("과거 create", user, args)                # 4
                    assert s_t == e_t == (not args["allDay"]), args
                    if args["allDay"]:
                        assert e > s and (tg.get("group") == "multi" or e - s == D1), args
                    else:
                        assert e > s, args
                    if "repeat" in args:
                        code = args["repeat"].split(":")[1]
                        assert args["repeat"].startswith("weekly:") and WD_CODE[s.weekday()] == code
                        assert today < s.date() <= today + dt.timedelta(days=7)
                if name == "update_event":
                    n_write += 1
                    p = args["patch"]
                    assert set(p) <= {"title", "start", "end", "allDay", "category"}, p
                    tgt = [it for it in last_items if it["id"] == args["id"]]
                    assert len(tgt) == 1 and tgt[0]["kind"] == "user", (args, last_items)      # 6
                    assert tgt[0]["title"] in user
                    if "start" in p:
                        s, _ = _parse_any(p["start"])
                        e, _ = _parse_any(p["end"])
                        n_dates += 2
                        assert s > now and e > s, (user, p)                              # 4
                        os_, _ = _parse_any(tgt[0]["start"])
                        oe, _ = _parse_any(tgt[0]["end"])
                        assert e - s == oe - os_, ("길이 유지", user, p)
                if name == "delete_event":
                    tgt = [it for it in last_items if it["id"] == args["id"]]
                    assert len(tgt) == 1 and tgt[0]["kind"] == "user", (args, last_items)      # 6
                    assert tgt[0]["title"] in user
            elif mm["role"] == "tool":
                res = json.loads(mm["content"])
                last_items = res.get("items")
                if isinstance(last_items, list) and calls and calls[-1][0] == "list_events":
                    assert len({it["title"] for it in last_items}) == len(last_items), last_items
                    for it in last_items:
                        assert list(it) == ["id", "title", "kind", "start", "end"], it
                        assert re.fullmatch(r"\d+|ev:\d+", it["id"]), it
                        _parse_any(it["start"])
                        _parse_any(it["end"])
            elif mm["role"] == "assistant":
                if r["meta"]["kind"] == "refusal":
                    assert mm["content"] in refusals                              # 7
                    assert any(k in mm["content"] for k in REFUSAL_MARKERS)
        if r["meta"]["kind"] == "refusal":
            assert len(msgs) == 3 and not calls
        # 독립 검산 — 문장에서 다시 읽은 날짜·시각 == 정답
        it = tg["intent"]
        if it == "create":
            a = calls[0][1]
            s, _ = _parse_any(a["start"])
            if tg["group"] == "weekly":
                ok = s.weekday() == _WDC.index(re.search(r"([월화수목금토일])요일", user).group(1))
                tms = parse_times(user)
                ok = ok and tms and tms[0] == (s.hour, s.minute)
            else:
                ds = parse_dates(user, today)
                ok = bool(ds) and ds[0] == s.date()
                if not a["allDay"]:
                    tms = parse_times(user)
                    ok = ok and bool(tms) and tms[0] == (s.hour, s.minute)
                    if tg["group"] == "range":
                        e, _ = _parse_any(a["end"])
                        ok = ok and len(tms) >= 2 and tms[1] == (e.hour, e.minute)
            if not ok:
                mism.append((user, now, a))
        elif it == "list_events":
            a = calls[0][1]
            pr = parse_range(user, today)
            if pr != (dt.date.fromisoformat(a["from"]), dt.date.fromisoformat(a["to"])):
                mism.append((user, now, a))
            if tg["range"] == "this_week":
                weekday_this_week.add(tg["weekday"])
        elif it in ("update", "delete"):
            ds = parse_dates(user, today)
            a0 = calls[0][1]
            ok = bool(ds) and ds[0] == dt.date.fromisoformat(a0["from"]) == dt.date.fromisoformat(a0["to"])
            if it == "update" and len(calls) > 1 and "start" in calls[1][1]["patch"]:
                p = calls[1][1]["patch"]
                s, _ = _parse_any(p["start"])
                if tg["ptype"] == "time":
                    tms = parse_times(user)
                    ok = ok and bool(tms) and tms[0] == (s.hour, s.minute)
                if tg["ptype"] == "date":
                    ok = ok and len(ds) >= 2 and ds[1] == s.date()
            if not ok:
                mism.append((user, now, calls))
    M["text_mismatch"] = len(mism)
    assert not mism, mism[:5]
    M["writes"], M["dates_parsed"] = n_write, n_dates

    # 9. 시간대 말 없는 1~12시
    bare = [u for u in users if bare_hour_hits(u)]
    M["bare_hour"] = len(bare)
    assert not bare, bare[:5]

    # 틀(패밀리) 최소 개수 — n 이 작으면 생략
    fams_by_intent: dict = {}
    for r in rows:
        fams_by_intent.setdefault(intent_of(r["meta"]["template"]), set()).add(r["meta"]["template"])
    M["families"] = {k: len(v) for k, v in fams_by_intent.items()}
    if n >= 5000:
        for k, mn in MIN_FAMILIES.items():
            assert M["families"].get(k, 0) >= mn, (k, M["families"].get(k), mn)

    # create 하위 배분 (§4-1)
    ct = [t for t in tags if t["intent"] == "create"]
    cnt_d = Counter(t["date"] for t in ct if t["group"] != "weekly")
    cnt_g = Counter(t["group"] for t in ct)
    cnt_t = Counter(t["time"] for t in ct if t["time"])
    M["create_mix"] = {
        "다음 주 X요일": cnt_d["next_wd"], "다다음 주": cnt_d["nn_wd"], "종일(하루)": cnt_g["allday"],
        "여러 날 종일": cnt_g["multi"], "끝 시각 명시": cnt_g["range"], "매주 반복": cnt_g["weekly"], "24시간 표기": cnt_t["24h"],
        "한국어 수사 시각": cnt_t["kor"], "절대 날짜 N월 N일": cnt_d["abs"], "N일 뒤": cnt_d["n_days"], "다음 달": cnt_d["next_month"]}
    mins = {"다음 주 X요일": 600, "다다음 주": 80, "종일(하루)": 400, "여러 날 종일": 100, "끝 시각 명시": 300, "매주 반복": 200,
            "24시간 표기": 150, "한국어 수사 시각": 150, "절대 날짜 N월 N일": 300, "N일 뒤": 100, "다음 달": 100}
    for k, mn in mins.items():
        assert M["create_mix"][k] >= mn * scale * 0.999, (k, M["create_mix"][k], mn * scale)
    M["create_mix_min"] = mins
    # '이번 주' 조회가 월~일 모든 요일의 now 에서 나왔는가
    M["this_week_weekdays"] = sorted(weekday_this_week)
    if n >= 2000:
        assert len(weekday_this_week) == 7, weekday_this_week
    # 기타 수치
    M["kinds"] = Counter(r["meta"]["kind"] for r in rows)
    M["tools_first"] = Counter(r["meta"]["tool"] for r in rows)
    M["surface_rate"] = sum(t["sv"] for t in tags) / n
    M["zero_result"] = sum(1 for t in tags if t.get("zero"))
    nows = [_SYS_NOW.match(r["messages"][0]["content"]).group(1) for r in rows]
    M["now_range"] = (min(nows), max(nows))
    M["now_weekdays"] = Counter(dt.date.fromisoformat(x).weekday() for x in nows)
    return M


# ════════════════════════════════════════════════════════════════
# 11. README · main
# ════════════════════════════════════════════════════════════════

def write_readme(path: Path, rows: list[dict], M: dict, seed: int, n: int, out_name: str) -> None:
    fam_cnt = Counter(r["meta"]["template"] for r in rows)
    L = ["# data/f9/extra_v2 — F9 Function Calling 추가 학습 데이터 v2", "",
         f"> 생성: {dt.datetime.now():%Y-%m-%d %H:%M} · seed {seed} · {n}행 · 생성기 `src/prepare_f9_v2.py` · source `synthetic-v2`", "",
         "**합성 데이터**. train 에만 병합한다 (평가셋 `eval.jsonl` 은 prepare_f9 seed 42 그대로).", "",
         "## 재생성 · 병합", "", "```",
         f"python src/prepare_f9_v2.py --n {n} --seed {seed}",
         f"python src/prepare_f9.py --n 1500 --seed 42 --extra data/f9/{out_name}", "```", "",
         "## 의도별 건수 (목표 ±5%)", "", "| 의도 | 건수 | 목표 | 틀(패밀리) 수 | 최소 틀 |", "| --- | ---: | ---: | ---: | ---: |"]
    for k, (got, exp) in M["intents"].items():
        L.append(f"| {k} | {got} | {exp} | {M['families'].get(k, 0)} | {MIN_FAMILIES.get(k, '-')} |")
    L += ["", "## 첫 도구별 (meta.tool)", "", "| 도구 | 건수 |", "| --- | ---: |"]
    L += [f"| {k} | {v} |" for k, v in sorted(M["tools_first"].items())]
    L += ["", "## kind 별", "", "| kind | 건수 |", "| --- | ---: |"] + [f"| {k} | {v} |" for k, v in sorted(M["kinds"].items())]
    L += ["", "## create_event 하위 배분 (§4-1)", "", "| 항목 | 건수 | 최소 |", "| --- | ---: | ---: |"]
    L += [f"| {k} | {v} | {M['create_mix_min'][k]} |" for k, v in M["create_mix"].items()]
    L += ["", "## 품질 게이트 (§6 — 모두 assert 통과)", "",
          f"1. 행 수 {n} · 의도별 건수 목표 ±5% 안 (위 표)",
          f"2. 고유 사용자 문장 **{M['unique']}** / {n} ({M['unique'] / n:.1%}) · 최대 중복 {M['max_dup']}",
          f"3. eval.jsonl 사용자 문장 {M['eval_users']}개와 겹침 **{M['leak']}** · 정규화(앞말·꼬리·좀·어미·띄어쓰기 제거) 후 겹침 **{M['leak_norm']}** "
          f"(생성 중 다시 뽑음: 정확 일치 {M['rejected']['exact']} · 근접 사본 {M['rejected']['normalised']})",
          f"4. 쓰기 호출 {M['writes']}건 start > now · 날짜 {M['dates_parsed']}개 파싱 · 종일 end = start+1일(여러 날은 end > start) · allDay↔형식 일치",
          "5. category·kinds·route·source·range·group enum, 필수 인자 존재, 정의 밖 인자 없음",
          f"6. update/delete id = tool 결과 항목 id (kind user) · 0건 결과 {M['zero_result']}행",
          "7. 거절 3종 = prepare_f9 문자열(소스 대조) · REFUSAL_MARKERS 포함",
          "8. system == prepare_f9.system_message(now) · tools == load_tools(tools/f9_tools.json)[0]",
          f"9. 시간대 말 없는 1~12시 사용자 문장 **{M['bare_hour']}**건",
          f"- 독립 검산(문장에서 날짜·시각을 다시 읽어 정답과 대조 — create·list_events·update·delete) 불일치 **{M['text_mismatch']}**",
          f"- '이번 주' 조회의 now 요일: {', '.join(WD[i] for i in M['this_week_weekdays'])}",
          f"- 표면 변형(§5) 적용 비율 {M['surface_rate']:.1%}", "",
          "## 날짜 범위", "",
          f"- now: {M['now_range'][0]} ~ {M['now_range'][1]} (시 7~23, 분 5분 단위) · 요일별 " +
          " ".join(f"{WD[k]} {v}" for k, v in sorted(M["now_weekdays"].items())), "",
          "## 해석 규칙 요약 (§2)", "",
          "| 표현 | 해석 |", "| --- | --- |",
          "| 오늘·내일·모레(내일모레) | today · +1 · +2 |",
          "| 이번 주 X요일 · 이번주 · 이번 X요일 | 이번 주 월요일 + idx (쓰기는 미래만) |",
          "| X요일 (수식어 없음) | 이번 주의 그 요일 — 오늘 이후 요일일 때만 생성 |",
          "| 다음 주·다음주·담주·내주 X요일 | 다음 주 월요일 + idx (오늘 요일 무관) |",
          "| 다다음 주 X요일 | +14 + idx |",
          "| N일 뒤/후 (2~14) · 다음 달 N일 · N월 N일 · N/N | today+N · 다음 달 N일 · 절대 날짜(지났으면 다음 해) |",
          "| 이번 주 · 다음 주 · 다다음 주 · 이번/다음 주말 · 이번/다음 달 | 월~일 · +7~+13 · +14~+20 · 토~일 · 1일~말일 |",
          "| 시각 | 1~12시는 반드시 오전·오후·아침·점심·낮·저녁·밤과 함께. 13~23시 표기 허용. 끝 시각은 앞 시간대 상속 가능 |",
          "| 길이 없음 / 시각 없음 | 60분 / allDay, end = start+1일 (여러 날은 마지막 날+1) |",
          "| 매주 X요일 | repeat weekly:XX, start = 오늘 이후 첫 X요일 |", "",
          "## 패밀리별 건수", "", "| 패밀리 | 건수 |", "| --- | ---: |"]
    L += [f"| {k} | {v} |" for k, v in sorted(fam_cnt.items())]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def load_eval_users(path: Path) -> set:
    users = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                for m in json.loads(line)["messages"]:
                    if m["role"] == "user":
                        users.add(m["content"])
    return users


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "data" / "f9"))
    ap.add_argument("--eval", default=str(ROOT / "data" / "f9" / "eval.jsonl"), help="누출 검사용 평가셋")
    args = ap.parse_args()

    # 제목 뱅크 자체 점검 (§3)
    assert len(ALL_TITLES) >= 90 and len(COURSES) >= 20
    for t in ALL_TITLES:
        assert not any(w in t for w in FORBIDDEN_TITLE_WORDS), t
        assert "시험" not in t or t.endswith("시험 공부"), t
        assert TITLE_CAT[t] in ("personal", "study", "team", "etc")

    tools, _ = load_tools(TOOLS_PATH)
    eval_users = load_eval_users(Path(args.eval))
    rng = random.Random(args.seed)
    rows, tags = generate(rng, args.n, tools, eval_users)
    M = validate(rows, tags, tools, eval_users, args.n)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "extra_v2.jsonl", rows)
    write_readme(out / "extra_v2.README.md", rows, M, args.seed, args.n, "extra_v2.jsonl")
    print(f"생성 완료 → {out / 'extra_v2.jsonl'}  {len(rows)}행 · 고유 문장 {M['unique']} · 최대 중복 {M['max_dup']} · "
          f"누출 {M['leak']} (정규화 {M['leak_norm']}, 재추첨 정확 {M['rejected']['exact']}·근접 {M['rejected']['normalised']}) · 맨 1~12시 {M['bare_hour']} · 검산 불일치 {M['text_mismatch']} · 표면 변형 {M['surface_rate']:.1%}")
    print("의도별:", ", ".join(f"{k} {v[0]}" for k, v in M["intents"].items()))
    print("틀 수:", ", ".join(f"{k} {v}" for k, v in sorted(M["families"].items())))
    print("create 배분:", ", ".join(f"{k} {v}" for k, v in M["create_mix"].items()))


if __name__ == "__main__":
    main()
