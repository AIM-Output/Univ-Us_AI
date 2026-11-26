#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F9 날짜 보강 데이터 — 정답 날짜가 그 달의 말일·1일에 걸리는 문장을 만든다.

  python src/prepare_f9_dates.py --n 1000 --seed 23
      → data/f9/extra_dates.jsonl + extra_dates.README.md
  python src/prepare_f9_dates.py --n 1000 --seed 23 --mix 2000 --mix_text 500 --mix_out data/f9-r1-date
      → 위 파일 + data/f9-r1-date/train.jsonl (날짜 보강 행 + 기존 train.jsonl 에서 뽑은 복습 행) · README.md

왜 만드나 (2026-11-23 날짜 진단 · f9-r1):
  정답이 말일·1일이면 72/108(67%), 그 밖은 67/72(93%) 를 맞혔다. v1+v2 학습 데이터에서 계산이 필요한 날짜 문장
  4,261행 중 정답이 말일·1일인 행은 382행(9%)뿐이다 — v2 는 기준시각(now)을 1년에서 고르게 뽑기 때문이다.

어떻게 만드나:
  · 말투(틀) · 빈칸 채우기 · 정답 계산은 전부 prepare_f9_v2 의 것이다. 이 파일은 **기준시각만** 고른다 —
    정답이 말일·1일에 떨어질 만한 기준시각을 뽑고, 실제로 그렇게 나온 행만 남긴다.
  · 하루짜리 날짜(등록 · 하루 조회 · 수정 · 삭제)는 정답 날짜가 말일 또는 1일.
    여러 날(여러 날 등록 · 주 단위 조회)은 기간이 달을 넘기거나 양 끝이 말일·1일.
    하루짜리 날짜는 말일과 1일이 반반이 되게 번갈아 겨냥한다.
  · 'N일 뒤' 와 '다음 주 X요일' 을 많이 넣는다(진단에서 가장 약했다). '다음 주 X요일' 은 기준일이 토·일인 것이 절반.
  · 시험 문장(eval.jsonl · data_src/real · --hold 진단 파일)과 같은 문장은 만들지 않는다. 문장이 달라도
    같은 날짜 상황(같은 기준일 → 같은 정답 날짜)이면 만들지 않는다 — 시험·진단이 '학습에서 본 적 없는 계산' 을 재게 하려는 것.
  · 행마다 prepare_f9_v3.verify_row 로 검사한다(도구 정의 · 쓰기는 미래 · 문장에서 다시 읽은 날짜·시각 == 정답).

--mix: 다른 run 에 이어서 학습(train.py --init_adapter)할 때 쓴다. 날짜 문장만 넣으면 거절을 잊을 수 있어 기존 행을 함께 넣는다.
  기존 train.jsonl 에서 종류별 비율대로 --mix 행을 뽑되, 글로 답하는 행(거절 · 인사)은 --mix_text 행으로 맞춘다.

표준 라이브러리만 쓴다. 난수는 seed 에서만 나오므로 같은 입력 파일 · 같은 seed 면 같은 파일이 나온다.
"""
from __future__ import annotations

import argparse
import calendar
import contextlib
import datetime as dt
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_jsonl, load_tools, write_jsonl  # noqa: E402
from datasrc import ROOT, Report, list_sources, real_users, rel, row_answer, row_now, row_user  # noqa: E402
import prepare_f9_v2 as v2  # noqa: E402
import prepare_f9_v3 as v3  # noqa: E402

D1 = dt.timedelta(days=1)
LO, HI = dt.date(2026, 9, 1), dt.date(2027, 8, 31)          # 기준시각 범위 — v2.rand_now 와 같다
SOURCE = "synthetic-dates"
MAX_TRY = 50_000            # 틀 하나에서 기준시각을 다시 뽑는 횟수
MAX_EDGE_HITS = 300         # 말일·1일에는 걸렸는데 문장이 겹쳐 버린 횟수가 이만큼이면 그 틀을 포기한다
MAX_ALT = 40                # 틀을 바꿔 보는 횟수

# ── 배분 (n=1000 일 때의 행 수 · 다른 n 이면 같은 비율) ─────────────
SHARE = {"ce_timed": 260, "ce_range": 80, "ce_allday": 100, "ce_multi": 30, "ce_weekly": 30,
         "list": 250, "update": 130, "delete": 120}
# 날짜 표현 비중. 'N일 뒤' · '다음 주 X요일' 이 가장 약했다. 'N월 N일' · '다음 달 N일' 은 날짜를 그대로 말하므로 조금만
DAY_KINDS = {"n_days": 30, "next_wd": 30, "tomorrow": 9, "dayafter": 8, "this_wd": 7, "bare_wd": 6, "nn_wd": 5,
             "next_month": 3, "abs": 2}
MULTI_KINDS = {"next_wd": 10, "tomorrow": 6, "this_wd": 5, "nn_wd": 4, "abs": 3, "next_month": 2}
LIST_KINDS = {"n_days": 20, "next_wd": 20, "tomorrow": 8, "dayafter": 6, "this_wd": 6, "bare_wd": 5, "abs": 2,
              "this_week": 10, "next_week": 12, "nn_week": 4, "this_weekend": 3, "next_weekend": 4}
UPDATE_PTYPES = {"time": 55, "date": 35, "title": 10}       # 시각 수정 · 날짜 수정 · 제목 수정 (상대 이동 '1시간 뒤로' 는 뺀다)
NEXT_WD_TODAY = [1, 1, 1, 1, 1, 2.5, 2.5]                   # '다음 주 X요일' 의 기준 요일(월~일) 비중 — 토·일이 절반

KEY_KO = {"ce_timed": "등록 (시각)", "ce_range": "등록 (끝 시각)", "ce_allday": "등록 (종일)", "ce_multi": "등록 (여러 날)",
          "ce_weekly": "등록 (매주)", "list": "조회", "update": "수정", "delete": "삭제"}
KIND_KO = {"n_days": "N일 뒤", "next_wd": "다음 주 X요일", "tomorrow": "내일", "dayafter": "모레", "this_wd": "이번 주 X요일",
           "bare_wd": "X요일", "nn_wd": "다다음 주 X요일", "next_month": "다음 달 N일", "abs": "N월 N일",
           "this_week": "이번 주", "next_week": "다음 주", "nn_week": "다다음 주", "this_weekend": "이번 주말",
           "next_weekend": "다음 주말", "weekly": "매주 X요일"}


def is_edge(d: dt.date) -> bool:
    """그 달의 1일 또는 말일."""
    return d.day == 1 or d.day == calendar.monthrange(d.year, d.month)[1]


def edge_dates(lo: dt.date, hi: dt.date) -> dict[str, list[dt.date]]:
    """lo~hi 사이의 말일("last") · 1일("first") · 둘 다("any")."""
    days = [lo + i * D1 for i in range((hi - lo).days + 1)]
    out = {"last": [d for d in days if is_edge(d) and d.day != 1], "first": [d for d in days if d.day == 1]}
    out["any"] = sorted(out["last"] + out["first"])
    return out


def pick_today(rng: random.Random, kind: str, t: dt.date) -> dt.date | None:
    """날짜 표현 kind 로 말했을 때 정답이 t(말일·1일)에 떨어질 수 있는 기준일을 하나 고른다. 맞는지는 만든 뒤에 본다."""
    mon = v2.week_monday(t)
    if kind == "n_days":
        return t - rng.randint(2, 14) * D1
    if kind == "tomorrow":
        return t - D1
    if kind == "dayafter":
        return t - 2 * D1
    if kind == "next_wd":
        return mon - 7 * D1 + rng.choices(range(7), weights=NEXT_WD_TODAY, k=1)[0] * D1
    if kind in ("next_week", "next_weekend"):
        return mon - 7 * D1 + rng.randint(0, 6) * D1
    if kind in ("nn_wd", "nn_week"):
        return mon - 14 * D1 + rng.randint(0, 6) * D1
    if kind == "this_wd":
        return mon + rng.randint(0, t.weekday()) * D1
    if kind == "bare_wd":                                   # 수식어 없는 'X요일' 은 오늘보다 뒤 요일만
        return mon + rng.randint(0, t.weekday() - 1) * D1 if t.weekday() else None
    if kind in ("this_week", "this_weekend"):
        return mon + rng.randint(0, 6) * D1
    if kind == "next_month":
        prev_last = t.replace(day=1) - D1                   # 그 전 달의 아무 날
        return prev_last.replace(day=rng.randint(1, prev_last.day))
    if kind == "abs":
        return t - rng.randint(1, 200) * D1
    if kind == "weekly":                                    # 오늘 이후 첫 X요일
        return t - rng.randint(1, 7) * D1
    raise ValueError(kind)


@contextlib.contextmanager
def write_kind(kind: str | None):
    """g_update · g_delete 는 날짜 표현을 v2.WRITE_DATE_KINDS 에서 스스로 고른다 — 부르는 동안만 한 종류로 고정한다."""
    old = v2.WRITE_DATE_KINDS
    if kind:
        v2.WRITE_DATE_KINDS = [kind]
    try:
        yield
    finally:
        v2.WRITE_DATE_KINDS = old


# ── 무엇을 몇 행 만들지 ──────────────────────────────────────

def spec_pools(seed: int) -> dict[str, list[dict]]:
    """v2 의 배분(build_plan)에서 틀을 빌린다 — 틀이 고르게 돌고, 틀과 범위 종류의 궁합도 v2 규칙 그대로다."""
    pools: dict[str, list[dict]] = defaultdict(list)
    for sp in v2.build_plan(random.Random(f"{seed}:plan"), 10000):
        it = sp["intent"]
        if it == "create":
            pools["ce_" + sp["group"]].append(sp)
        elif it == "list_events":
            pools["list_day" if sp["range"] in v2.DAY_ONLY_RANGES else "list_span"].append(sp)
        elif it == "update":
            pools["update_" + sp["ptype"]].append(sp)
        elif it == "delete":
            pools["delete"].append(sp)
    return pools


def make_jobs(rng: random.Random, n: int) -> list[dict]:
    """행 하나 = job 하나. pool·idx 는 빌릴 틀의 자리, fix 는 그 사양에 덮어쓸 값(날짜 표현 종류 등)."""
    counts = v2.split(n, SHARE)
    used: Counter = Counter()
    jobs: list[dict] = []

    def add(pool: str, key: str, kind: str, mode: str | None = None, **fix) -> None:
        # want: 하루짜리 날짜는 말일과 1일을 번갈아 겨냥한다(반반). 여러 날 범위는 어느 쪽이든 된다
        span = key == "ce_multi" or (key == "list" and kind not in v2.DAY_ONLY_RANGES)
        want = "any" if span else ("last", "first")[used["want:" + key] % 2]
        used["want:" + key] += not span
        jobs.append({"pool": pool, "idx": used[pool], "key": key, "kind": kind, "mode": mode, "want": want, "fix": fix})
        used[pool] += 1

    for grp in ("timed", "range", "allday"):
        allowed = {k: w for k, w in DAY_KINDS.items() if k in v2.CE_DATES[grp]}
        for k in v2.quota(rng, counts["ce_" + grp], allowed):
            add("ce_" + grp, "ce_" + grp, k, date=k)
    for k in v2.quota(rng, counts["ce_multi"], MULTI_KINDS):
        add("ce_multi", "ce_multi", k, date=k)
    for _ in range(counts["ce_weekly"]):
        add("ce_weekly", "ce_weekly", "weekly")
    for k in v2.quota(rng, counts["list"], LIST_KINDS):
        add("list_day" if k in v2.DAY_ONLY_RANGES else "list_span", "list", k, range=k)
    write_kinds = {k: w for k, w in DAY_KINDS.items() if k in set(v2.WRITE_DATE_KINDS)}
    n_date = 0
    for pt, k in zip(v2.quota(rng, counts["update"], UPDATE_PTYPES), v2.quota(rng, counts["update"], write_kinds)):
        if pt == "date" and n_date % 2:                     # 날짜 수정의 절반은 옮겨 갈 날짜({E})가 말일·1일에 걸리게
            add("update_date", "update", k, "E", zero=False)    # '찾지 못했습니다' 행에는 옮겨 갈 날짜가 정답에 없다
        else:
            add("update_" + pt, "update", k, "D")
        n_date += pt == "date"
    for k in v2.quota(rng, counts["delete"], write_kinds):
        add("delete", "delete", k, "D")
    return jobs


# ── 정답에서 날짜 꺼내기 ─────────────────────────────────────

def _d(s: str) -> dt.date:
    return dt.date.fromisoformat(s[:10])


def arg_dates(args: dict) -> list[dt.date]:
    """도구 인자에 든 날짜 전부 (from · to · start · end · patch.start · patch.end)."""
    vals = [args.get(k) for k in ("from", "to", "start", "end")]
    patch = args.get("patch")
    if isinstance(patch, dict):
        vals += [patch.get("start"), patch.get("end")]
    out = []
    for v in vals:
        if isinstance(v, str) and re.match(r"\d{4}-\d{2}-\d{2}", v):
            out.append(_d(v))
    return out


def key_dates(intent: str, group: str | None, mode: str | None, steps: list) -> tuple[list[dt.date], bool]:
    """문장의 날짜 표현이 가리키는 날짜들 · 여러 날인가."""
    calls = [s for s in steps if s[0] == "call"]
    a = calls[0][2]
    if intent == "create":
        if group == "multi":
            s, e = _d(a["start"]), _d(a["end"])             # 종일의 end 는 마지막 날의 다음 날
            return [s, e - D1, e], True
        return [_d(a["start"])], False
    if intent == "list_events":
        f, t = _d(a["from"]), _d(a["to"])
        return ([f], False) if f == t else ([f, t], True)
    if mode == "E":                                         # 수정 — 옮겨 갈 날짜
        patch = calls[1][2].get("patch", {}) if len(calls) > 1 else {}
        return ([_d(patch["start"])] if "start" in patch else []), False
    return [_d(a["from"])], False                           # 수정 · 삭제 — 원래 날짜


def hits_edge(dates: list[dt.date], span: bool, want: str = "any") -> bool:
    """하루짜리: 날짜가 말일·1일(want 가 last/first 면 그쪽)인가. 여러 날: 끝이 말일·1일이거나 기간이 달을 넘기는가."""
    if want == "last":
        return any(is_edge(d) and d.day != 1 for d in dates)
    if want == "first":
        return any(d.day == 1 for d in dates)
    if any(is_edge(d) for d in dates):
        return True
    return span and bool(dates) and (dates[0].year, dates[0].month) != (dates[-1].year, dates[-1].month)


def edge_label(dates: list[dt.date]) -> str:
    labs = []
    if any(d.day != 1 and is_edge(d) for d in dates):
        labs.append("말일")
    if any(d.day == 1 for d in dates):
        labs.append("1일")
    return "+".join(labs) or "달 넘김"


# ── 시험 문장 · 시험과 같은 날짜 상황 ─────────────────────────

def _row_today(row: dict) -> dt.date | None:
    m = v2._SYS_NOW.match(row["messages"][0].get("content") or "")
    return dt.date.fromisoformat(m.group(1)) if m else None


def rows_pairs(rows: list[dict]) -> set[tuple[dt.date, dt.date]]:
    """(기준일, 정답에 든 날짜) 쌍 — 같은 쌍을 학습 데이터로 만들지 않으려고 모은다."""
    pairs: set[tuple[dt.date, dt.date]] = set()
    for r in rows:
        today = _row_today(r)
        if today is None:
            continue
        for mm in r["messages"]:
            if mm.get("role") == "assistant" and mm.get("tool_calls"):
                a = mm["tool_calls"][0]["function"].get("arguments") or {}
                a = json.loads(a) if isinstance(a, str) else a
                pairs.update((today, d) for d in arg_dates(a))
    return pairs


def real_pairs(folder: Path, schemas: dict) -> set[tuple[dt.date, dt.date]]:
    """data_src/real 의 (기준일, 정답 날짜) 쌍. 문장 내용은 쓰지도 내보내지도 않는다."""
    import make_real as mr
    items = mr.load_items(list_sources(folder, []), schemas, Report())
    return {(it.now.date(), d) for it in items if isinstance(it.args, dict) for d in arg_dates(it.args)}


# ── 검사 ────────────────────────────────────────────────────

def verify(row: dict, tg: dict, schemas: dict) -> list[str]:
    if tg["intent"] != "update" or tg.get("ptype") == "time":
        return v3.verify_row(row, tg, schemas)
    # 날짜 수정 · 제목 수정: 문장에 시각이 없어 verify_row 의 시각 대조는 맞지 않는다 →
    # 구조·대상·원래 날짜는 삭제 행과 같은 경로로 보고, 옮겨 갈 날짜는 v2.validate 와 같은 규칙으로 본다
    probs = v3.verify_row(row, dict(tg, intent="delete"), schemas)
    if tg.get("ptype") == "date" and not tg.get("zero"):
        msgs = row["messages"]
        ds = v2.parse_dates(msgs[1]["content"], _row_today(row))
        patch = [json.loads(m["tool_calls"][0]["function"]["arguments"]) for m in msgs
                 if m.get("role") == "assistant" and m.get("tool_calls")][-1].get("patch", {})
        if not (len(ds) >= 2 and "start" in patch and ds[1] == _d(patch["start"])):
            probs.append("문장에서 다시 읽은 새 날짜가 정답과 다릅니다")
    return probs


# ── 만들기 ──────────────────────────────────────────────────

def one_row(rng: random.Random, sp: dict, job: dict, targets: dict, tools: list[dict], schemas: dict,
            blocks: v3.Blocks, block_pairs: set, seen: dict, stats: Counter):
    """사양 sp(틀 하나)로 말일·1일에 걸리는 행 하나. 이 틀로는 새 문장이 안 나오면 None(호출부가 다른 틀로 바꾼다)."""
    kind, key, mode, want = job["kind"], job["key"], job["mode"], job["want"]
    fix = kind if key in ("update", "delete") and mode == "D" else None
    edge_hits = 0
    for _ in range(MAX_TRY):
        if edge_hits >= MAX_EDGE_HITS:
            return None
        t = rng.choice(targets[want])
        today = pick_today(rng, kind, t)
        if today is None or not (LO <= today <= HI):
            continue
        now = dt.datetime(today.year, today.month, today.day, rng.randint(7, 23), rng.choice(range(0, 60, 5)))
        with write_kind(fix):
            out = sp["gen"](rng, now, sp)
        if out is None:
            continue
        user, steps, k2, tg = out
        dates, span = key_dates(sp["intent"], sp.get("group"), mode, steps)
        if not hits_edge(dates, span, want):
            continue
        edge_hits += 1
        user, sv = v2.surface(rng, user, sp["intent"], sp["fam"])
        if user in blocks.exact or v2.norm_key(user) in blocks.norm:
            stats["시험 문장과 같아 다시 뽑음"] += 1
            continue
        if any((today, d) in block_pairs for s in steps if s[0] == "call" for d in arg_dates(s[2])):
            stats["시험과 같은 날짜 상황이라 다시 뽑음"] += 1
            continue
        c = seen.get(user, 0)
        if c >= v2.MAX_DUP or ((c or user in blocks.known) and edge_hits <= MAX_EDGE_HITS // 2):
            stats["이미 있는 문장이라 다시 뽑음"] += 1
            continue
        row = v2.make_row(now, user, steps, sp["fam"], k2, tools)
        tag = dict(tg, intent=sp["intent"], sv=sv, key=key, kind=kind, mode=mode, ptype=sp.get("ptype"),
                   today_wd=today.weekday(), edge=edge_label(dates), span=span,
                   offset=(dates[0] - today).days if dates else None)
        probs = verify(row, tag, schemas)
        if probs:
            stats["검사에 걸려 다시 뽑음"] += 1
            if stats["검사에 걸려 다시 뽑음"] <= 5:
                print(f"  (검사) {probs[0]} — \"{user}\" (기준 {row_now(row)}) → {row_answer(row)}")
            continue
        return row, tag, user, c
    return None


def generate(n: int, seed: int, tools: list[dict], schemas: dict, blocks: v3.Blocks,
             block_pairs: set) -> tuple[list[dict], list[dict], Counter]:
    rng = random.Random(seed)
    targets = edge_dates(LO + D1, HI + 200 * D1)
    pools = spec_pools(seed)
    rows, tags, seen, stats = [], [], {}, Counter()
    for job in make_jobs(rng, n):
        pool = pools[job["pool"]]
        got = None
        for alt in range(MAX_ALT):                          # 이 틀의 문장이 전부 시험 문장과 겹치면 같은 종류의 다음 틀로
            sp = dict(pool[(job["idx"] + alt) % len(pool)], **job["fix"])
            got = one_row(rng, sp, job, targets, tools, schemas, blocks, block_pairs, seen, stats)
            if got:
                break
            stats["틀을 바꿔 다시 만듦"] += 1
        if got is None:
            raise RuntimeError(f"만들지 못했습니다: {job['key']} · {job['kind']}")
        row, tag, user, c = got
        seen[user] = c + 1
        row["meta"]["source"] = SOURCE
        row["meta"]["date_kind"] = job["kind"]
        rows.append(row)
        tags.append(tag)
    order = list(range(len(rows)))
    rng.shuffle(order)
    rows, tags = [rows[i] for i in order], [tags[i] for i in order]
    for i, r in enumerate(rows):
        r["id"] = f"f9dt-{i + 1:05d}"
    return rows, tags, stats


# ── 복습 행 고르기 (--mix) ───────────────────────────────────

def is_text_row(r: dict) -> bool:
    """첫 답이 글인 행 — 거절 · 인사."""
    return r["meta"].get("kind") in ("refusal", "text")


def stratum(r: dict) -> tuple:
    t = r["meta"].get("template", "")
    for pre in ("refC_", "chit_"):                          # 주제별로 9~25행씩 흩어진 것은 한 묶음으로
        if t.startswith(pre):
            t = pre
            break
    else:
        t = re.sub(r"\d+$", "", t)
    return r["meta"].get("source", ""), r["meta"].get("kind", ""), t


def pick_replay(pool: list[dict], n_rows: int, n_text: int, rng: random.Random) -> list[dict]:
    def strat(rows: list[dict], k: int) -> list[dict]:
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            groups[stratum(r)].append(r)
        q = v2.split(k, {g: len(v) for g, v in sorted(groups.items())})
        out: list[dict] = []
        for g in sorted(groups):
            out += rng.sample(groups[g], q[g])
        return out

    text = [r for r in pool if is_text_row(r)]
    rest = [r for r in pool if not is_text_row(r)]
    if n_text < 0:                                          # 지정하지 않으면 원래 비율대로
        n_text = round(n_rows * len(text) / len(pool))
    if n_text > len(text) or n_rows - n_text > len(rest) or n_text > n_rows:
        raise SystemExit(f"--mix {n_rows} · --mix_text {n_text} 만큼 뽑을 행이 없습니다 (글 답 {len(text)} · 도구 답 {len(rest)})")
    return strat(text, n_text) + strat(rest, n_rows - n_text)


# ── 기록 ────────────────────────────────────────────────────

def sha12(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def table(tags: list[dict]) -> list[str]:
    kinds = [k for k in KIND_KO if any(t["kind"] == k for t in tags)]
    L = ["| 종류 | 행 | " + " | ".join(KIND_KO[k] for k in kinds) + " |", "| --- | ---: |" + " ---: |" * len(kinds)]
    for key in SHARE:
        c = Counter(t["kind"] for t in tags if t["key"] == key)
        L.append(f"| {KEY_KO[key]} | {sum(c.values())} | " + " | ".join(str(c[k]) if c[k] else "" for k in kinds) + " |")
    c = Counter(t["kind"] for t in tags)
    L.append(f"| **합계** | **{len(tags)}** | " + " | ".join(str(c[k]) for k in kinds) + " |")
    return L


def write_readme(path: Path, rows: list[dict], tags: list[dict], stats: Counter, args, inputs: dict, n_block: dict) -> None:
    users = Counter(row_user(r) for r in rows)
    edge = Counter(t["edge"] for t in tags)
    wd = Counter(t["today_wd"] for t in tags if t["kind"] == "next_wd")
    L = ["# data/f9/extra_dates — F9 날짜 보강 데이터 (정답이 말일·1일에 걸리는 문장)", "",
         f"> 생성: {dt.datetime.now():%Y-%m-%d %H:%M} · seed {args.seed} · {len(rows)}행 · 생성기 `src/prepare_f9_dates.py` · "
         f"source `{SOURCE}`", "",
         "**합성 데이터**. 말투(틀)와 정답 계산은 v2 생성기의 것이고, 기준시각만 정답이 말일·1일에 떨어지게 골랐다.", "",
         "```", f"python src/prepare_f9_dates.py --n {args.n} --seed {args.seed}", "```", "",
         f"- 고유 사용자 문장 **{len(users)}** / {len(rows)} · 최대 중복 {max(users.values())}",
         "- 행마다 검사 통과: 도구 정의 · 쓰기는 미래 · 종일 end · 문장에서 다시 읽은 날짜·시각 = 정답 · 맨 1~12시 없음",
         f"- 시험 문장과 겹침 0 — 막은 문장 {n_block['sentences']}개 (eval · data_src/real · 진단) · "
         f"막은 날짜 상황(기준일 → 정답 날짜) {n_block['pairs']}쌍",
         "- 정답 날짜가 걸리는 곳: " + " · ".join(f"{k} {v}" for k, v in edge.most_common()),
         "- '다음 주 X요일' 의 기준 요일: " + " · ".join(f"{'월화수목금토일'[i]} {wd[i]}" for i in range(7)),
         "- 다시 뽑은 횟수: " + (" · ".join(f"{k} {v}" for k, v in stats.items()) or "없음"), "",
         "## 종류 × 날짜 표현", ""] + table(tags) + ["",
         "수정 130행의 날짜 표현은 원래 날짜({D}) 기준이다. 날짜를 옮기는 수정의 절반은 옮겨 갈 날짜({E})가 말일·1일에 걸리게 만들었고, "
         "그 표현 종류는 v2 가 고른다.", "",
         "## 입력 파일", "", "| 파일 | sha256 앞 12자 |", "| --- | --- |"]
    L += [f"| {k} | {v} |" for k, v in inputs.items()]
    path.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


def write_mix_readme(path: Path, mix: list[dict], n_dates: int, replay: list[dict], args, shas: dict) -> None:
    text = sum(is_text_row(r) for r in replay)
    first = Counter(r["meta"].get("tool", "none") for r in replay)
    src = Counter(r["meta"].get("source", "") for r in replay)
    all_text = sum(is_text_row(r) for r in mix)
    L = [f"# {rel(path.parent)} — 이어서 학습용 데이터 (날짜 보강 + 복습 행)", "",
         f"> 생성: {dt.datetime.now():%Y-%m-%d %H:%M} · seed {args.seed} · {len(mix)}행 · 생성기 `src/prepare_f9_dates.py`", "",
         "다른 run 의 어댑터에서 시작하는 학습(`train.py --init_adapter`)에 쓴다. 날짜 문장만 넣으면 거절을 잊을 수 있어 "
         "기존 학습 행을 함께 넣었다.", "",
         "```", f"python src/prepare_f9_dates.py --n {args.n} --seed {args.seed} --mix {args.mix} --mix_text {args.mix_text} "
         f"--mix_out {rel(path.parent)}", "```", "",
         "| 구성 | 행 |", "| --- | ---: |",
         f"| 날짜 보강 (`{SOURCE}`) | {n_dates} |",
         f"| 복습 — 글로 답하는 행 (거절 · 인사) | {text} |",
         f"| 복습 — 도구로 답하는 행 | {len(replay) - text} |",
         f"| **합계** | **{len(mix)}** |", "",
         f"- 전체에서 글로 답하는 행의 비율: {all_text}/{len(mix)} = {100 * all_text / len(mix):.1f}% "
         f"(원래 학습 데이터 {shas['text_share']})",
         "- 복습 행은 기존 train.jsonl 에서 (출처 · 종류 · 틀 묶음)별 비율대로 뽑았다. 출처: "
         + " · ".join(f"{k} {v}" for k, v in sorted(src.items())),
         "- 복습 행의 첫 도구: " + " · ".join(f"{k} {v}" for k, v in first.most_common()), "",
         "| 파일 | sha256 앞 12자 |", "| --- | --- |"]
    L += [f"| {k} | {v} |" for k, v in shas.items() if k != "text_share"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000, help="만들 날짜 보강 행 수")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", default=str(ROOT / "data" / "f9"), help="extra_dates.jsonl 을 쓸 폴더")
    ap.add_argument("--train", default=str(ROOT / "data" / "f9" / "train.jsonl"), help="기존 학습 데이터 — 같은 문장을 피하고, --mix 의 복습 행을 뽑는다")
    ap.add_argument("--eval", default=str(ROOT / "data" / "f9" / "eval.jsonl"), help="시험지 — 겹치지 않게 막는다")
    ap.add_argument("--real", default=str(ROOT / "data_src" / "real"), help="실제 문장 폴더 — 겹치지 않게 막는다")
    ap.add_argument("--hold", nargs="*", default=None,
                    help="학습에 넣지 않을 진단 문장 jsonl. 비우면 outputs/date_probe/probe_dates.jsonl 이 있을 때 그것")
    ap.add_argument("--mix", type=int, default=0, help="기존 train.jsonl 에서 뽑아 섞을 복습 행 수 (0 = 섞지 않음)")
    ap.add_argument("--mix_text", type=int, default=-1, help="복습 행 중 글로 답하는 행(거절 · 인사) 수. -1 = 원래 비율대로")
    ap.add_argument("--mix_out", default="", help="섞은 train.jsonl 을 쓸 폴더 (--mix 와 함께)")
    args = ap.parse_args()
    if bool(args.mix) != bool(args.mix_out):
        print("--mix 와 --mix_out 은 함께 씁니다")
        return 1

    tools, _ = load_tools(v2.TOOLS_PATH)
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    train_p, eval_p, real_dir = Path(args.train), Path(args.eval), Path(args.real)
    if args.hold is None:
        default_hold = ROOT / "outputs" / "date_probe" / "probe_dates.jsonl"
        hold_ps = [default_hold] if default_hold.exists() else []
    else:
        hold_ps = [Path(h) for h in args.hold]
    for p in [train_p, eval_p] + hold_ps:
        if not p.exists():
            print(f"{rel(p)} 이 없습니다")
            return 1
    if not hold_ps:
        print("(알림) 진단 문장 파일(--hold)이 없어 진단 문장과의 겹침은 막지 못했습니다")

    train_rows = load_jsonl(train_p)
    eval_rows = load_jsonl(eval_p)
    hold_rows = [r for p in hold_ps for r in load_jsonl(p)]
    exact = {row_user(r) for r in eval_rows} | real_users(real_dir) | {row_user(r) for r in hold_rows}
    blocks = v3.Blocks(exact, {v2.norm_key(u) for u in exact}, {row_user(r) for r in train_rows})
    pairs = rows_pairs(eval_rows) | rows_pairs(hold_rows) | real_pairs(real_dir, schemas)

    rows, tags, stats = generate(args.n, args.seed, tools, schemas, blocks, pairs)

    # 마지막 확인 — 여기서 걸리면 버그다
    users = [row_user(r) for r in rows]
    assert not any(u in blocks.exact or v2.norm_key(u) in blocks.norm for u in users), "시험 문장과 겹친다"
    assert not (rows_pairs(rows) & pairs), "시험과 같은 날짜 상황이 있다"
    assert all(hits_edge(*key_dates_of(t, r)) for t, r in zip(tags, rows)), "말일·1일에 걸리지 않는 행이 있다"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "extra_dates.jsonl"
    write_jsonl(p, rows)
    inputs = {rel(train_p): sha12(train_p), rel(eval_p): sha12(eval_p)}
    inputs.update({rel(h): sha12(h) for h in hold_ps})
    inputs[rel(real_dir) + "/*.tsv"] = hashlib.sha256(b"".join(q.read_bytes() for q in list_sources(real_dir, []))).hexdigest()[:12]
    write_readme(out / "extra_dates.README.md", rows, tags, stats, args, inputs, {"sentences": len(exact), "pairs": len(pairs)})
    cnt = Counter(users)
    print(f"생성 완료 → {rel(p)}  {len(rows)}행 · 고유 문장 {len(cnt)} · 최대 중복 {max(cnt.values())} · sha256 {sha12(p)}")
    print("\n".join(table(tags)))
    print("정답 날짜가 걸리는 곳:", dict(Counter(t["edge"] for t in tags)))
    print("다시 뽑은 횟수:", dict(stats))

    if args.mix:
        rng = random.Random(f"{args.seed}:mix")
        replay = pick_replay(train_rows, args.mix, args.mix_text, rng)
        mix = rows + replay
        rng.shuffle(mix)
        assert len({r["id"] for r in mix}) == len(mix), "id 가 겹친다"
        mdir = Path(args.mix_out)
        mdir.mkdir(parents=True, exist_ok=True)
        mp = mdir / "train.jsonl"
        write_jsonl(mp, mix)
        n_text_all = sum(is_text_row(r) for r in train_rows)
        shas = {rel(mp): sha12(mp), rel(p): sha12(p), rel(train_p): sha12(train_p),
                "text_share": f"{n_text_all}/{len(train_rows)} = {100 * n_text_all / len(train_rows):.1f}%"}
        write_mix_readme(mdir / "README.md", mix, len(rows), replay, args, shas)
        text = sum(is_text_row(r) for r in replay)
        print(f"섞기 완료 → {rel(mp)}  {len(mix)}행 = 날짜 보강 {len(rows)} + 복습 {len(replay)} (글 답 {text} · 도구 답 {len(replay) - text}) · "
              f"sha256 {sha12(mp)}")
    return 0


def key_dates_of(tag: dict, row: dict) -> tuple[list[dt.date], bool]:
    """다 만든 행에서 key_dates 를 다시 구한다(마지막 확인용)."""
    steps = []
    for m in row["messages"]:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            fn = m["tool_calls"][0]["function"]
            steps.append(("call", fn["name"], json.loads(fn["arguments"])))
    return key_dates(tag["intent"], tag.get("group"), tag.get("mode"), steps)


if __name__ == "__main__":
    sys.exit(main())
