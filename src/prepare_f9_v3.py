#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F9 추가 학습 데이터 v3 — 사람이 쓴 말투 틀(data_src/frames/*.tsv)로 문장을 만든다.

  python src/prepare_f9_v3.py --check              # 틀 검사만 (오류가 있으면 종료 코드 1)
  python src/prepare_f9_v3.py --show               # 틀마다 만들어지는 문장 5개
  python src/prepare_f9_v3.py --n 3000 --seed 11   # data/f9/extra_v3.jsonl + extra_v3.README.md
  python src/prepare_f9.py --n 1500 --seed 42 --extra data/f9/extra_v2.jsonl data/f9/extra_v3.jsonl   # train 에 병합
  python src/prepare_f9_v3.py --check data_src/examples/frames_example.tsv    # 파일을 직접 지정

TSV 칸: id · 종류 · 틀 · 메모   (docs/F9_학습_가이드.md 7-6)

틀의 문법과 정답 계산은 prepare_f9_v2 를 그대로 쓴다 — 빈칸을 채우는 것도, 날짜·시각 정답을 내는 것도 v2 의 함수다.
행마다 v2 와 같은 품질 검사를 한다: 도구 정의와 일치 · 쓰기는 미래 · 종일 end = 다음 날 · 문장에서 날짜·시각을 다시 읽어
정답과 대조 · 오전·오후 없는 1~12시 없음 · 시험 문장(eval.jsonl · data_src/real)과 겹치지 않음.
표준 라이브러리만 쓴다. 난수는 random.Random(seed) 하나뿐이라, 같은 틀 · 같은 seed · 같은 eval/v2/실제 문장이면 같은 파일이 나온다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_jsonl, load_tools, write_jsonl  # noqa: E402
from datasrc import (ROOT, Line, Report, cell, list_sources, owner_prefix, read_tsv, real_users, rel,  # noqa: E402
                     row_answer, row_now, row_user)
import prepare_f9_v2 as v2  # noqa: E402

HEADER = ["id", "종류", "틀", "메모"]
_FID = re.compile(r"^([a-z]{2,3})-f(\d{3})$")
JOSA_OK = sorted(set(v2.JOSA) | {"으로", "에"})
REF_SLOTS = frozenset({"C", "C2", "D", "E", "W", "V", "T", "U", "AE", "AP", "N", "DD", "FOOD2"} | set(v2.REF_C_SLOTS))
D1 = dt.timedelta(days=1)


@dataclass(frozen=True)
class Kind:
    label: str                       # 사람이 읽는 설명
    intent: str                      # v2 의 의도 이름 (표면 변형 규칙이 본다)
    req: frozenset = frozenset()     # 꼭 있어야 하는 빈칸
    opt: frozenset = frozenset()     # 써도 되는 빈칸
    weight: float = 1.0              # 틀 하나가 받는 행 수의 비중 — 빈칸이 많을수록 다양한 문장이 나온다


def _k(label, intent, req="", opt="", weight=1.0) -> Kind:
    return Kind(label, intent, frozenset(req.split()), frozenset(opt.split()), weight)


KINDS: dict[str, Kind] = {
    # ── 쉬운 종류 (빈칸이 없거나 하나) ──
    "gr":        _k("졸업요건 질문", "get_graduation", weight=0.4),
    "at_all":    _k("전체 출결 질문", "get_attendance", weight=0.4),
    "at_course": _k("과목 출결 질문", "get_attendance", "C", weight=1.2),
    "ex_up":     _k("다가오는 시험", "get_exams", opt="C", weight=0.4),
    "ex_plan":   _k("시험 공부 계획", "get_exams", weight=0.4),
    "as_now":    _k("지금 급한 과제", "list_assignments", weight=0.4),
    "as_week":   _k("이번 주 과제", "list_assignments", weight=0.4),
    "as_later":  _k("나중에 해도 되는 과제", "list_assignments", weight=0.4),
    "as_all":    _k("과제 전체", "list_assignments", weight=0.4),
    "sync":      _k("수집 실행", "run_sync", "S", weight=1.2),
    "nav":       _k("화면 이동", "navigate", "S", weight=1.5),
    "pl":        _k("공강 배치", "open_placement", opt="RG", weight=0.8),
    "ref_c":     _k("거절 — 앱이 못 하는 일", "refusal_oos", opt=" ".join(sorted(REF_SLOTS)), weight=0.4),
    # ── 어려운 종류 (날짜·시각·제목 빈칸) ──
    "ce_timed":  _k("시각이 있는 등록", "create", "D T X", "L", weight=3.0),
    "ce_range":  _k("끝 시각을 말하는 등록", "create", "D R X", weight=2.5),
    "ce_allday": _k("종일 등록", "create", "D X", weight=2.5),
    "ce_multi":  _k("여러 날 등록", "create", "A B X", weight=2.0),
    "ce_weekly": _k("매주 반복 등록", "create", "W X", "T R L", weight=2.0),
    "le_kinded": _k("종류를 말하는 조회", "list_events", "P K", weight=2.5),
    "le_plain":  _k("종류를 말하지 않는 조회", "list_events", "P", weight=2.0),
    "del":       _k("삭제 (두 단계)", "delete", "D X", weight=2.0),
    "up_time":   _k("시각을 옮기는 수정 (두 단계)", "update", "D X T", weight=2.0),
    "ref_a":     _k("거절 — e클래스 마감", "refusal_eclass", opt=" ".join(sorted(REF_SLOTS)), weight=1.2),
    "ref_b":     _k("거절 — 학사일정·수업", "refusal_class", opt=" ".join(sorted(REF_SLOTS)), weight=1.2),
}

BARE_ROUTES = [r for r, names in v2.NAV_NAMES.items() if any(not scr for _n, scr in names)]
_NAV_SCREEN_WORD = re.compile(r"\{S(?:\|\w+)?\}\s*(?:화면|페이지|탭|메뉴|창)")
LE_KINDS_ONLY = {k: w for k, w in v2.LE_KINDS.items() if k}
_SIMPLE = {     # 종류 → (도구, 인자)
    "gr": ("get_graduation", {}), "at_all": ("get_attendance", {}),
    "ex_up": ("get_exams", {"upcoming": True}), "ex_plan": ("get_exams", {}),
    "as_now": ("list_assignments", {"group": "now"}), "as_week": ("list_assignments", {"group": "this_week"}),
    "as_later": ("list_assignments", {"group": "later"}), "as_all": ("list_assignments", {}),
}
_REF_ANSWER = {"ref_a": v2.REF_ECLASS, "ref_b": v2.REF_CLASS, "ref_c": v2.REF_OOS}


@dataclass
class Frame:
    line: Line
    id: str
    kind: str
    tpl: str
    memo: str

    @property
    def fam(self) -> str:
        return f"v3:{self.kind}:{self.id}"


class GenError(Exception):
    pass


def _sq(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _pick(rng: random.Random, weights: dict):
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def v2_frames() -> set[str]:
    """v2 에 이미 있는 틀(공백 무시) — 같은 틀을 또 쓰지 않게."""
    out: set[str] = set()

    def add(x) -> None:
        if isinstance(x, str):
            out.add(_sq(x))
        elif isinstance(x, dict):
            for y in x.values():
                add(y)
        elif isinstance(x, (list, tuple)):
            for y in x:
                add(y)

    for name in ("CE_TIMED", "CE_RANGE", "CE_ALLDAY", "CE_MULTI", "CE_WEEKLY", "LE_KINDED", "LE_PLAIN", "NAV_FRAMES",
                 "SYNC_FRAMES", "GR_FRAMES", "AT_COURSE", "AT_ALL", "EX_UP", "EX_PLAN", "AS_FRAMES", "PL_FRAMES",
                 "UP_FRAMES", "DEL_FRAMES", "REF_A", "REF_B", "REF_C"):
        add(getattr(v2, name))
    return out


# ── 틀 문법 검사 ─────────────────────────────────────────────

def check_syntax(kind_name: str, tpl: str) -> list[str]:
    errs: list[str] = []
    k = KINDS[kind_name]
    allowed = k.req | k.opt
    depth = 0
    for ch in tpl:
        if ch == "[":
            depth += 1
            if depth > 1:
                errs.append("[ ] 안에 [ ] 를 또 쓸 수 없습니다")
                break
        elif ch == "]":
            depth -= 1
            if depth < 0:
                errs.append("닫는 ] 가 여는 [ 보다 먼저 나옵니다")
                break
    if depth > 0:
        errs.append("[ 를 닫는 ] 가 없습니다")
    rest = v2._SLOT.sub("", tpl)
    if "{" in rest or "}" in rest:
        errs.append("{ } 가 맞지 않거나 빈칸 이름이 잘못됐습니다 — {D} 나 {X|을} 처럼 씁니다")
    names: set[str] = set()
    for m in v2._SLOT.finditer(tpl):
        name, j = m.group(1), m.group(2)
        names.add(name)
        if name not in allowed:
            can = " ".join("{" + a + "}" for a in sorted(allowed)) or "없음"
            errs.append(f"{{{name}}} 은(는) 종류 {kind_name} 에서 쓸 수 없는 빈칸입니다 (쓸 수 있는 빈칸: {can})")
        if j and j not in JOSA_OK:
            errs.append(f"{{{name}|{j}}} 의 '{j}' 는 쓸 수 없는 조사입니다 (가능: {' '.join(JOSA_OK)})")
    for r in sorted(k.req - names):
        errs.append(f"종류 {kind_name} 에는 빈칸 {{{r}}} 이(가) 꼭 있어야 합니다")
    if errs:
        return errs
    if "|" in re.sub(r"\[[^\[\]]*\]", "", rest):
        errs.append("세로줄 | 은 [ ] 안에서만 씁니다 — 예: [취소해줘|날려줘]")
    for m in v2._ALT.finditer(tpl):
        alts = v2._split_top(m.group(1))
        if len(alts) < 2:
            errs.append(f"[{m.group(1)}] 안에 | 가 없습니다 — 고를 말이 둘 이상이어야 합니다 (하나를 비워 [좀 |] 처럼 써도 됩니다)")
        elif len(set(alts)) != len(alts):
            errs.append(f"[{m.group(1)}] 안에 같은 말이 두 번 있습니다")
    if kind_name == "ce_weekly":
        if ("T" in names) == ("R" in names):
            errs.append("ce_weekly 에는 {T}(시작 시각)나 {R}(시각 범위) 중 하나가 꼭 있어야 하고, 둘을 같이 쓸 수는 없습니다")
        if "R" in names and "L" in names:
            errs.append("ce_weekly 에서 {R} 과 {L} 은 같이 쓸 수 없습니다")
    if not re.search(r"[가-힣]", tpl):
        errs.append("틀에 한글이 없습니다")
    if len(tpl) > 80:
        errs.append(f"틀이 {len(tpl)}자로 너무 깁니다 — 한 문장에 부탁 하나만 써 주세요")
    return errs


# ── 틀 → v2 생성기에 넘길 사양(spec) ─────────────────────────

def make_spec(fr: Frame, rng: random.Random) -> dict:
    kind, tpl = fr.kind, fr.tpl
    sp: dict = {"intent": KINDS[kind].intent, "fam": fr.fam, "tpl": tpl}
    if kind in ("ce_timed", "ce_range", "ce_allday", "ce_multi", "ce_weekly"):
        grp = kind[3:]
        sp.update(gen=v2.g_create, group=grp,
                  date=_pick(rng, v2.CE_DATES[grp]) if grp in v2.CE_DATES else None,
                  time=_pick(rng, v2.CE_TIMES[grp]) if grp in v2.CE_TIMES else None)
    elif kind == "le_kinded":
        sp.update(gen=v2.g_list, range=_pick(rng, v2.LE_RANGES), kinds=_pick(rng, LE_KINDS_ONLY), tier="n")
    elif kind == "le_plain":
        sp.update(gen=v2.g_list, range=_pick(rng, v2.LE_RANGES), kinds=None, tier="")
    elif kind == "nav":
        needs_bare = bool(_NAV_SCREEN_WORD.search(tpl))     # '{S} 화면' 꼴이면 '출석 화면' 같은 이름을 넣지 않는다
        sp.update(gen=v2.g_navigate, needs_bare=needs_bare,
                  route=rng.choice(BARE_ROUTES if needs_bare else list(v2.NAV_NAMES)))
    elif kind == "sync":
        sp.update(gen=v2.g_sync, source=rng.choice(list(v2.SYNC_NAMES)))
    elif kind == "at_course":
        sp.update(gen=v2.g_attendance_course)
    elif kind in _SIMPLE:
        tool, args = _SIMPLE[kind]
        sp.update(gen=v2.g_simple, tool=tool, args=dict(args))
    elif kind == "pl":
        g = _pick(rng, {"this_week": 40, "next_week": 30, "none": 30}) if "{RG" in tpl else "none"
        sp.update(gen=v2.g_simple, tool="open_placement", args={} if g == "none" else {"range": g}, range=g)
    elif kind == "del":
        sp.update(gen=v2.g_delete, fidx=-1, zero=rng.random() < 0.05)
    elif kind == "up_time":
        sp.update(gen=v2.g_update, ptype="time", cond="any", zero=rng.random() < 0.05)
    elif kind in _REF_ANSWER:
        sp.update(gen=v2.g_refusal, answer=_REF_ANSWER[kind])
    else:                                                    # pragma: no cover — KINDS 와 어긋나면 바로 알게
        raise KeyError(kind)
    return sp


@dataclass
class Blocks:
    exact: set            # 시험 문장 (eval + data_src/real) — 똑같이 만들면 안 된다
    norm: set             # 위의 근접 사본(어미·띄어쓰기만 다른 것)
    known: set            # v2 에 이미 있는 문장 — 되도록 새 문장을 만든다


def gen_rows(fr: Frame, count: int, rng: random.Random, tools: list[dict], blocks: Blocks, seen: dict[str, int],
             strict_dup: bool = True) -> tuple[list[dict], list[dict]]:
    """틀 하나로 count 행. 더 만들 새 문장이 없으면 만든 만큼만 돌려준다(한 행도 못 만들면 GenError)."""
    rows, tags = [], []
    for _ in range(count):
        got = None
        for attempt in range(400):
            sp = make_spec(fr, rng)
            now = v2.rand_now(rng)
            try:
                out = sp["gen"](rng, now, sp)
            except (KeyError, IndexError, ValueError) as e:
                raise GenError(f"빈칸을 채우지 못했습니다 ({type(e).__name__}: {e})") from e
            if out is None:
                continue
            user, steps, kind, tg = out
            user, sv = v2.surface(rng, user, sp["intent"], fr.fam)
            if user in blocks.exact or v2.norm_key(user) in blocks.norm:
                continue
            c = seen.get(user, 0)
            if strict_dup and (c >= v2.MAX_DUP or ((c or user in blocks.known) and attempt < 120)):
                continue
            got = (now, user, steps, kind, tg, sp, sv, c)
            break
        if got is None:
            break
        now, user, steps, kind, tg, sp, sv, c = got
        seen[user] = c + 1
        row = v2.make_row(now, user, steps, fr.fam, kind, tools)
        row["meta"]["source"] = "synthetic-v3"
        rows.append(row)
        tags.append(dict(tg, intent=sp["intent"], sv=sv, frame=fr.id, kind=fr.kind))
    if not rows:
        raise GenError("이 틀로 문장을 하나도 만들지 못했습니다 — 시험 문장과 겹치거나 빈칸 조합이 맞지 않습니다")
    return rows, tags


# ── 행 검사 — prepare_f9_v2.validate 의 행 단위 검사와 같은 규칙. 문제를 글로 모아 돌려준다 ──────────

def verify_row(row: dict, tg: dict, schemas: dict[str, dict]) -> list[str]:
    probs: list[str] = []
    msgs = row["messages"]
    m = v2._SYS_NOW.match(msgs[0]["content"])
    now = dt.datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M")
    today, user = now.date(), msgs[1]["content"]
    last_items, calls = None, []
    try:
        for mm in msgs[2:]:
            if mm["role"] == "assistant" and mm.get("tool_calls"):
                fn = mm["tool_calls"][0]["function"]
                name, args = fn["name"], json.loads(fn["arguments"])
                try:
                    v2._check_args(name, args, schemas[name])
                except AssertionError as e:
                    probs.append(f"도구 정의와 다릅니다: {e}")
                calls.append((name, args))
                if name == "list_events" and dt.date.fromisoformat(args["from"]) > dt.date.fromisoformat(args["to"]):
                    probs.append("from 이 to 보다 늦습니다")
                if name == "create_event":
                    s, s_t = v2._parse_any(args["start"])
                    e, e_t = v2._parse_any(args["end"])
                    if s <= now:
                        probs.append("등록 시각이 기준시각보다 앞입니다")
                    if not (s_t == e_t == (not args["allDay"])):
                        probs.append("allDay 와 날짜 형식이 맞지 않습니다")
                    if e <= s or (args["allDay"] and tg.get("group") != "multi" and e - s != D1):
                        probs.append("end 가 규칙과 다릅니다")
                    if "repeat" in args:
                        code = args["repeat"].split(":")[1]
                        if not (args["repeat"].startswith("weekly:") and v2.WD_CODE[s.weekday()] == code
                                and today < s.date() <= today + 7 * D1):
                            probs.append("반복 규칙과 start 가 맞지 않습니다")
                if name in ("update_event", "delete_event"):
                    tgt = [it for it in (last_items or []) if it["id"] == args["id"]]
                    if len(tgt) != 1 or tgt[0]["kind"] != "user" or tgt[0]["title"] not in user:
                        probs.append("수정·삭제 대상이 도구 결과의 내 일정과 맞지 않습니다")
                    elif name == "update_event" and "start" in args["patch"]:
                        s, _ = v2._parse_any(args["patch"]["start"])
                        e, _ = v2._parse_any(args["patch"]["end"])
                        os_, _ = v2._parse_any(tgt[0]["start"])
                        oe, _ = v2._parse_any(tgt[0]["end"])
                        if s <= now or e <= s or e - s != oe - os_:
                            probs.append("수정한 시각이 규칙(미래 · 길이 유지)과 다릅니다")
            elif mm["role"] == "tool":
                last_items = json.loads(mm["content"]).get("items")
        if row["meta"]["kind"] == "refusal" and (len(msgs) != 3 or calls):
            probs.append("거절 행에 도구 호출이 있습니다")

        # 독립 검산 — 문장에서 다시 읽은 날짜·시각 == 정답
        it = tg["intent"]
        if it == "create":
            a = calls[0][1]
            s, _ = v2._parse_any(a["start"])
            tms = v2.parse_times(user)
            if tg["group"] == "weekly":
                wm = re.search(r"([월화수목금토일])요일", user)
                ok = bool(wm) and s.weekday() == v2._WDC.index(wm.group(1)) and bool(tms) and tms[0] == (s.hour, s.minute)
            else:
                ds = v2.parse_dates(user, today)
                ok = bool(ds) and ds[0] == s.date()
                if not a["allDay"]:
                    ok = ok and bool(tms) and tms[0] == (s.hour, s.minute)
                    if tg["group"] == "range":
                        e, _ = v2._parse_any(a["end"])
                        ok = ok and len(tms) >= 2 and tms[1] == (e.hour, e.minute)
            if not ok:
                probs.append("문장에서 다시 읽은 날짜·시각이 정답과 다릅니다")
        elif it == "list_events":
            a = calls[0][1]
            if v2.parse_range(user, today) != (dt.date.fromisoformat(a["from"]), dt.date.fromisoformat(a["to"])):
                probs.append("문장에서 다시 읽은 기간이 정답과 다릅니다")
        elif it in ("update", "delete"):
            ds = v2.parse_dates(user, today)
            a0 = calls[0][1]
            ok = bool(ds) and ds[0] == dt.date.fromisoformat(a0["from"]) == dt.date.fromisoformat(a0["to"])
            if it == "update" and len(calls) > 1 and "start" in calls[1][1]["patch"]:
                s, _ = v2._parse_any(calls[1][1]["patch"]["start"])
                tms = v2.parse_times(user)
                ok = ok and bool(tms) and tms[0] == (s.hour, s.minute)
            if not ok:
                probs.append("문장에서 다시 읽은 날짜·시각이 정답과 다릅니다")
    except (KeyError, ValueError, TypeError, IndexError, AttributeError) as e:
        probs.append(f"행 모양이 예상과 다릅니다 ({type(e).__name__}: {e})")
    if v2.bare_hour_hits(user):
        probs.append("오전·오후 없는 1~12시가 문장에 있습니다")
    return probs


# ── TSV → Frame ──────────────────────────────────────────────

def load_frames(paths: list[Path], rep: Report) -> list[Frame]:
    frames: list[Frame] = []
    seen_id: dict[str, str] = {}
    seen_tpl: dict[str, str] = {}
    existing = v2_frames()
    for path in paths:
        if not path.exists():
            rep.error(rel(path), "파일이 없습니다")
            continue
        prefix = owner_prefix(path)
        for ln in read_tsv(path, HEADER, rep, min_cols=3):
            fid, kind, tpl, memo = ln.cells
            bad = False
            m = _FID.match(fid)
            if not m:
                rep.error(ln.where, f"id \"{fid}\" 는 `이니셜-f번호 3자리` 꼴이어야 합니다 (예: sb-f001)")
                bad = True
            elif prefix and m.group(1) != prefix:
                rep.error(ln.where, f"이 파일의 id 는 {prefix}-f 로 시작해야 합니다 (지금: {fid})")
                bad = True
            elif fid in seen_id:
                rep.error(ln.where, f"id {fid} 은(는) 이미 있습니다 ({seen_id[fid]})")
                bad = True
            else:
                seen_id[fid] = ln.where
            if kind not in KINDS:
                rep.error(ln.where, f"종류 \"{kind}\" 는 없습니다 (가능: {' · '.join(KINDS)})")
                continue
            if not tpl:
                rep.error(ln.where, "틀 칸이 비어 있습니다")
                continue
            errs = check_syntax(kind, tpl)
            for e in errs:
                rep.error(ln.where, e)
            key = _sq(tpl)
            if key in existing:
                rep.error(ln.where, "AI 팀 v2 에 이미 똑같은 틀이 있습니다 — 다른 말투로 써 주세요")
                bad = True
            elif key in seen_tpl:
                rep.error(ln.where, f"똑같은 틀이 이미 있습니다 ({seen_tpl[key]})")
                bad = True
            else:
                seen_tpl[key] = ln.where
            if not (bad or errs):
                frames.append(Frame(ln, fid, kind, tpl, memo))
    return frames


def trial(frames: list[Frame], tools: list[dict], schemas: dict, blocks: Blocks, seed: int, rep: Report,
          k: int = 8) -> dict[str, list[dict]]:
    """틀마다 k 문장을 시험 삼아 만들어 검사한다. 돌려주는 값은 --show 용 표본."""
    samples: dict[str, list[dict]] = {}
    ok_frames = []
    for fr in frames:
        rng = random.Random(f"{seed}:{fr.id}")
        try:
            rows, tags = gen_rows(fr, k, rng, tools, blocks, {}, strict_dup=False)
        except GenError as e:
            rep.error(fr.line.where, str(e))
            continue
        bad = [(r, p) for r, t in zip(rows, tags) for p in verify_row(r, t, schemas)]
        if bad:
            r, p = bad[0]
            rep.error(fr.line.where, f"{p} — 예: \"{row_user(r)}\" (기준 {row_now(r)}) → {row_answer(r)}")
            continue
        samples[fr.id] = rows
        ok_frames.append(fr)
    frames[:] = ok_frames
    return samples


def build(frames: list[Frame], n: int, seed: int, tools: list[dict], schemas: dict, blocks: Blocks
          ) -> tuple[list[dict], list[dict], dict]:
    """틀 전체로 n 행(목표)을 만든다 → (행, 태그, 목표보다 적게 만든 틀). 검사에 걸리면 GenError."""
    rng = random.Random(seed)
    quota = v2.split(n, {f.id: KINDS[f.kind].weight for f in frames})
    rows, tags, seen, short = [], [], {}, {}
    for fr in frames:
        try:
            r, t = gen_rows(fr, quota[fr.id], rng, tools, blocks, seen)
        except GenError as e:
            raise GenError(f"{fr.line.where}: {e}") from e
        if len(r) < quota[fr.id]:
            short[fr.id] = (len(r), quota[fr.id])
        rows += r
        tags += t
    bad = [(r, p) for r, t in zip(rows, tags) for p in verify_row(r, t, schemas)]
    if bad:
        lines = [f"{r['meta']['template']}: {p} — \"{row_user(r)}\" (기준 {row_now(r)}) → {row_answer(r)}" for r, p in bad[:10]]
        raise GenError(f"검사에 걸린 행 {len(bad)}개 — 해당 틀을 고친 뒤 다시 실행해 주세요.\n" + "\n".join(lines))
    if any(row_user(r) in blocks.exact or v2.norm_key(row_user(r)) in blocks.norm for r in rows):
        raise GenError("시험 문장과 겹치는 행이 있습니다")          # gen_rows 가 이미 거르므로 여기 오면 버그다
    order = list(range(len(rows)))
    rng.shuffle(order)
    rows, tags = [rows[i] for i in order], [tags[i] for i in order]
    for i, r in enumerate(rows):
        r["id"] = f"f9v3-{i + 1:05d}"
    return rows, tags, short


def write_readme(path: Path, rows: list[dict], tags: list[dict], frames: list[Frame], seed: int, n: int, short: dict) -> None:
    users = [row_user(r) for r in rows]
    cnt = Counter(users)
    by_kind = Counter(t["kind"] for t in tags)
    by_frame = Counter(t["frame"] for t in tags)
    by_owner = Counter(f.id.split("-")[0] for f in frames)
    L = ["# data/f9/extra_v3 — F9 추가 학습 데이터 v3 (우리 말투 틀)", "",
         f"> 생성: {dt.datetime.now():%Y-%m-%d %H:%M} · seed {seed} · 목표 {n}행 → {len(rows)}행 · 틀 {len(frames)}개 · "
         "생성기 `src/prepare_f9_v3.py` · source `synthetic-v3`", "",
         "**합성 데이터**. train 에만 병합한다. 틀은 `data_src/frames/*.tsv` 에 있고, 정답은 v2 생성기의 함수가 계산한다.", "",
         "```", f"python src/prepare_f9_v3.py --n {n} --seed {seed}",
         "python src/prepare_f9.py --n 1500 --seed 42 --extra data/f9/extra_v2.jsonl data/f9/extra_v3.jsonl", "```", "",
         f"- 고유 사용자 문장 **{len(cnt)}** / {len(rows)} · 최대 중복 {max(cnt.values())}",
         "- 행마다 v2 와 같은 검사 통과: 도구 정의 · 쓰기는 미래 · 종일 end · 문장에서 다시 읽은 날짜·시각 = 정답 · 맨 1~12시 없음",
         "- 시험 문장(eval.jsonl · data_src/real)과 정확히·정규화 후 겹침 0",
         f"- 틀을 쓴 사람: {', '.join(f'{k} {v}개' for k, v in sorted(by_owner.items()))}", "",
         "| 종류 | 설명 | 틀 수 | 행 수 |", "| --- | --- | ---: | ---: |"]
    for k, kd in KINDS.items():
        nf = sum(1 for f in frames if f.kind == k)
        if nf:
            L.append(f"| {k} | {kd.label} | {nf} | {by_kind[k]} |")
    if short:
        L += ["", "새 문장이 모자라 목표보다 적게 만든 틀: " + ", ".join(f"{k}({v[0]}/{v[1]})" for k, v in sorted(short.items()))]
    L += ["", "| 틀 id | 종류 | 행 수 | 틀 |", "| --- | --- | ---: | --- |"]
    L += [f"| {f.id} | {f.kind} | {by_frame[f.id]} | {cell(f.tpl)} |" for f in frames]
    path.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="틀 TSV. 비우면 --src 폴더의 *.tsv 전부")
    ap.add_argument("--src", default=str(ROOT / "data_src" / "frames"))
    ap.add_argument("--real", default=str(ROOT / "data_src" / "real"), help="시험 문장 폴더 — 겹치지 않게 막는다")
    ap.add_argument("--out", default=str(ROOT / "data" / "f9"))
    ap.add_argument("--eval", default=str(ROOT / "data" / "f9" / "eval.jsonl"), help="누출 검사용 평가셋")
    ap.add_argument("--n", type=int, default=3000, help="만들 행 수(목표)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--check", action="store_true", help="틀 검사만")
    ap.add_argument("--show", action="store_true", help="틀마다 문장 5개를 보여 준다")
    args = ap.parse_args()

    tools, _ = load_tools(v2.TOOLS_PATH)
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    rep = Report()
    paths = list_sources(Path(args.src), args.files)
    if not paths:
        print(f"TSV 가 없습니다: {args.src}")
        return 1
    frames = load_frames(paths, rep)

    out = Path(args.out)
    eval_users = v2.load_eval_users(Path(args.eval))
    v2_path = out / "extra_v2.jsonl"
    only_look = args.check or args.show
    if not eval_users or not v2_path.exists():
        missing = rel(Path(args.eval)) if not eval_users else rel(v2_path)
        if not only_look:
            print(f"{missing} 이 없습니다 — 먼저 prepare_f9.py 와 prepare_f9_v2.py 를 순서대로 돌려 주세요 (가이드 3-1)")
            return 1
        print(f"(알림) {missing} 이 없어 시험·v2 문장과의 겹침은 보지 못했습니다 — 데이터를 먼저 만들면 함께 검사합니다")
    exact = set(eval_users) | real_users(Path(args.real))
    known = {row_user(r) for r in load_jsonl(v2_path)} if v2_path.exists() else set()
    blocks = Blocks(exact, {v2.norm_key(u) for u in exact}, known)

    samples = trial(frames, tools, schemas, blocks, args.seed, rep)
    if args.show:
        for fr in frames:
            print(f"\n### {fr.id} · {fr.kind} ({KINDS[fr.kind].label})\n틀: {fr.tpl}")
            for r in samples.get(fr.id, [])[:5]:
                print(f"- {row_user(r)}  →  {row_answer(r)}   (기준 {row_now(r)})")
    rep.dump()
    by = Counter((f.id.split("-")[0], f.kind) for f in frames)
    if by:
        print("\n| 누가 | 종류 | 틀 수 |\n| --- | --- | ---: |")
        for (who, kind), c in sorted(by.items()):
            print(f"| {who} | {kind} | {c} |")
    n_err = len(rep.errors)
    print(f"\n파일 {len(paths)}개 · 통과한 틀 {len(frames)}개 · 오류 {n_err} · 경고 {len(rep.warnings)}")
    if n_err:
        print("오류를 고친 뒤 다시 실행해 주세요.")
        return 1
    if only_look:
        return 0
    if not frames:
        print("틀이 아직 없어 파일을 만들지 않았습니다.")
        return 0

    try:
        rows, tags, short = build(frames, args.n, args.seed, tools, schemas, blocks)
    except GenError as e:
        print(f"오류: {e}")
        return 1
    users = [row_user(r) for r in rows]
    out.mkdir(parents=True, exist_ok=True)
    p = out / "extra_v3.jsonl"
    write_jsonl(p, rows)
    write_readme(out / "extra_v3.README.md", rows, tags, frames, args.seed, args.n, short)
    cnt = Counter(users)
    print(f"생성 완료 → {rel(p)}  {len(rows)}행 (목표 {args.n}) · 고유 문장 {len(cnt)} · 최대 중복 {max(cnt.values())} · "
          f"sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:12]}")
    if short:
        print("새 문장이 모자라 적게 만든 틀: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(short.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
