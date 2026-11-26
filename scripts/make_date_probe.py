#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""날짜 계산 진단용 문장 만들기 — "정답 날짜가 그 달의 말일·1일일 때 약한가?" 를 확인한다.

  python scripts/make_date_probe.py                                   # → outputs/date_probe/probe_dates.jsonl (180문장)
  python scripts/make_date_probe.py --out <파일> --seed 20261123

  · v2 생성기의 함수와 v2 의 말투 틀(CE_TIMED)을 그대로 쓴다 → 말투는 학습 때 본 것이고, 날짜만 새로 뽑는다.
  · 날짜 표현은 'N일 뒤' 와 '다음 주 X요일' 두 가지. 정답 날짜가 어디에 떨어지는지로 묶는다
    (meta.template = 표현|묶음|오늘 요일|며칠 뒤).
  · 학습 문장·시험 문장(eval · data_src/real)과 같은 문장은 넣지 않는다. **학습에는 쓰지 않는 진단용이다.**
  · 먼저 data/f9/train.jsonl · eval.jsonl 이 있어야 한다(가이드 8-4 의 데이터 만들기).
  · 같은 seed · 같은 train·eval·실제 문장이면 같은 파일이 나온다.
    2026-11-23(실제 문장 320개)과 11-27(380개) 모두 sha256 앞 12자리 93ba6efce050.

채점은 eval.py 로 한다 (MODEL_ID 는 config.env 의 값):
  python src/eval.py --model_id "$MODEL_ID" --adapter outputs/<run>/adapter --data outputs/date_probe/probe_dates.jsonl
      --out outputs/date_probe/<run>/eval_probe.md --label "<run> probe" --load_4bit 1 --compute_dtype bf16
그 결과는 scripts/analyze_probe.py · scripts/compare_runs.py 로 본다.
표준 라이브러리만 쓴다.
"""
import argparse
import calendar
import collections
import datetime as dt
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--out", default=str(ROOT / "outputs" / "date_probe" / "probe_dates.jsonl"))
_ap.add_argument("--seed", type=int, default=20261123)
_args = _ap.parse_args()
OUT, SEED = Path(_args.out), _args.seed
sys.path.insert(0, str(ROOT / "src"))

import prepare_f9_v2 as v2  # noqa: E402
import prepare_f9_v3 as v3  # noqa: E402
from common import load_jsonl, load_tools, write_jsonl  # noqa: E402
from datasrc import real_users  # noqa: E402

tools, _ = load_tools(v2.TOOLS_PATH)
schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
train_users = {m["content"] for r in load_jsonl(ROOT / "data" / "f9" / "train.jsonl") for m in r["messages"] if m["role"] == "user"}
blocked = set(v2.load_eval_users(ROOT / "data" / "f9" / "eval.jsonl")) | real_users(ROOT / "data_src" / "real")
blocked_norm = {v2.norm_key(u) for u in blocked}

LO, HI = dt.date(2026, 9, 1), dt.date(2027, 8, 31)          # v2 의 기준시각 범위와 같다
D1 = dt.timedelta(days=1)
mdays = lambda d: calendar.monthrange(d.year, d.month)[1]


def bucket(today: dt.date, d: dt.date):
    if d.day == 31:
        return "d31"
    if d.day == 30 and mdays(d) == 30:
        return "end30"
    if d.day == 30 and mdays(d) == 31:
        return "d30of31"
    if d.day == 1:
        return "d01"
    if d.month != today.month and 2 <= d.day <= 9:
        return "cross"
    if d.month == today.month and 8 <= d.day <= 24:
        return "mid"
    return None


WANT = {"d31": 30, "end30": 12, "d30of31": 12, "d01": 12, "cross": 12, "mid": 12}
KINDS = ("n_days", "next_wd")
all_days = [LO + i * D1 for i in range((HI - LO).days + 1)]
rng = random.Random(SEED)
fams = v2.CE_FAMS["timed"]
rows, used, stats = [], set(), collections.Counter()

for kind in KINDS:
    for bk, want in WANT.items():
        got = 0
        for _ in range(200000):
            if got >= want:
                break
            d_star = rng.choice(all_days)
            if kind == "n_days":
                today = d_star - rng.randint(2, 14) * D1
            else:
                today = v2.week_monday(d_star) - 7 * D1 + rng.randint(0, 6) * D1
            if not (LO <= today <= HI):
                continue
            # 뽑기 전에 가능한지 먼저 본다(헛돌기 줄이기): 목표 날짜가 그 묶음에 들어가야 한다
            if bucket(today, d_star) != bk:
                continue
            now = dt.datetime(today.year, today.month, today.day, rng.randint(7, 23), rng.choice(range(0, 60, 5)))
            fam, tpl = rng.choice(fams)
            sp = {"intent": "create", "fam": fam, "tpl": tpl, "gen": v2.g_create, "group": "timed", "date": kind,
                  "time": v3._pick(rng, v2.CE_TIMES["timed"])}
            out = v2.g_create(rng, now, sp)
            if out is None:
                continue
            user, steps, k2, tg = out
            user, _sv = v2.surface(rng, user, "create", fam)
            d = dt.date.fromisoformat(steps[0][2]["start"][:10])
            if bucket(today, d) != bk:
                continue
            if user in train_users or user in blocked or v2.norm_key(user) in blocked_norm or user in used:
                stats["skip_dup"] += 1
                continue
            row = v2.make_row(now, user, steps, fam, k2, tools)
            probs = v3.verify_row(row, dict(tg, intent="create"), schemas)
            if probs:
                stats["skip_verify"] += 1
                continue
            used.add(user)
            row["meta"]["source"] = "probe"
            row["meta"]["template"] = f"{kind}|{bk}|wd{today.weekday()}|+{(d - today).days}"
            rows.append(row)
            got += 1
        stats[f"{kind}|{bk}"] = got

rng.shuffle(rows)
for i, r in enumerate(rows, 1):
    r["id"] = f"probe-{i:04d}"
OUT.parent.mkdir(parents=True, exist_ok=True)
write_jsonl(OUT, rows)
print("행", len(rows), "→", OUT)
print({k: v for k, v in stats.items()})
for r in rows[:6]:
    fn = r["messages"][2]["tool_calls"][0]["function"]
    print(" ", r["meta"]["template"], "|", r["messages"][0]["content"].splitlines()[0][:26], "|", r["messages"][1]["content"], "→", json.loads(fn["arguments"])["start"])
