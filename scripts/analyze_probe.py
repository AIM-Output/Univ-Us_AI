#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""날짜 진단(scripts/make_date_probe.py) 채점 결과를 묶음별로 본다 — 정답 날짜가 어디에 떨어질 때 틀리는가 (GPU 없이).

  python scripts/analyze_probe.py outputs/date_probe/f9-r1/eval_probe.json f9-r1

보여 주는 것: 묶음(정답이 31일 · 30일 · 1일 · 달을 넘겨 2~9일 · 같은 달 8~24일) × 표현('N일 뒤' · '다음 주 X요일')별
날짜 정확도 / 틀린 날짜가 정답과 며칠 차이인지 / 달력에 없는 날짜를 답한 턴 / 오늘 요일별 정확도 / 틀린 턴 예.
표준 라이브러리만 쓴다.
"""
import collections
import datetime as dt
import json
import re
import sys
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit(__doc__)
res = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["results"]
label = sys.argv[2] if len(sys.argv) > 2 else ""
WD = "월화수목금토일"
NAME = {"d31": "정답이 31일", "end30": "정답이 30일(30일짜리 달의 말일)", "d30of31": "정답이 30일(31일짜리 달)", "d01": "정답이 1일",
        "cross": "달을 넘겨 2~9일", "mid": "같은 달 8~24일"}
KN = {"n_days": "N일 뒤", "next_wd": "다음 주 X요일"}


def pred_raw(s):
    m = re.search(r'"start"\s*:\s*"(\d{4}-\d{2}-\d{2})', s["pred_text"])
    return m.group(1) if m else None


def pred_start(s):
    """예측한 start 날짜. 달력에 없는 날짜(예: 9월 31일)면 None."""
    raw = pred_raw(s)
    try:
        return dt.date.fromisoformat(raw) if raw else None
    except ValueError:
        return None


def invalid_date(s):
    raw = pred_raw(s)
    if not raw:
        return False
    try:
        dt.date.fromisoformat(raw)
        return False
    except ValueError:
        return True


def exp_start(s):
    return dt.date.fromisoformat(s["expected"]["arguments"]["start"][:10])


print(f"== {label} · 턴 {len(res)} · 인자까지 맞음 {sum(bool(s['exact_ok']) for s in res)} · 도구 맞음 {sum(bool(s['name_ok']) for s in res)}")
date_ok = lambda s: pred_start(s) == exp_start(s)
print(f"날짜만 보면 맞음 {sum(date_ok(s) for s in res)}/{len(res)}")
tab = collections.defaultdict(lambda: [0, 0, 0])
for s in res:
    kind, bk, wd, off = s["template"].split("|")
    for key in ((kind, bk), ("합계", bk), (kind, "전체")):
        t = tab[key]
        t[0] += 1
        t[1] += date_ok(s)
        t[2] += bool(s["exact_ok"])
print("\n묶음별 날짜 정확도 (맞음/전체)")
print(f"  {'묶음':28s} {'N일 뒤':>10s} {'다음 주 X요일':>12s} {'합계':>10s}")
for bk in NAME:
    cells = [tab[(k, bk)] for k in ("n_days", "next_wd", "합계")]
    print(f"  {NAME[bk]:28s} " + " ".join(f"{c[1]:3d}/{c[0]:<3d} {100*c[1]/max(1,c[0]):5.1f}%" for c in cells))
for k in ("n_days", "next_wd"):
    c = tab[(k, "전체")]
    print(f"  {KN[k]} 전체: 날짜 {c[1]}/{c[0]} · 인자까지 {c[2]}/{c[0]}")

print("\n틀린 날짜 — 정답과의 차이(일)")
diffs = collections.Counter()
for s in res:
    if date_ok(s):
        continue
    kind, bk, wd, off = s["template"].split("|")
    p, e = pred_start(s), exp_start(s)
    d = (p - e).days if p else ("없는 날짜 " + pred_raw(s)[5:] if invalid_date(s) else "날짜 못 읽음")
    diffs[(kind, bk, d)] += 1
for (kind, bk, d), n in sorted(diffs.items(), key=lambda x: (x[0][0], x[0][1], str(x[0][2]))):
    print(f"  {KN[kind]:10s} {NAME[bk]:28s} " + (f"예측 − 정답 = {d:+d}일" if isinstance(d, int) else str(d)), f"× {n}")
print("\n달력에 없는 날짜를 답한 턴:", sum(invalid_date(s) for s in res), "→", collections.Counter(pred_raw(s)[5:] for s in res if invalid_date(s)).most_common())

print("\n'다음 주 X요일' — 오늘 요일별 날짜 정확도")
byw = collections.defaultdict(lambda: [0, 0])
for s in res:
    kind, bk, wd, off = s["template"].split("|")
    if kind == "next_wd":
        byw[int(wd[2:])][0] += 1
        byw[int(wd[2:])][1] += date_ok(s)
print("  " + " · ".join(f"{WD[w]} {byw[w][1]}/{byw[w][0]}" for w in sorted(byw)))

print("\n날짜는 맞고 다른 인자가 틀린 턴:", sum(date_ok(s) and not s["exact_ok"] for s in res))
print("\n틀린 턴 예 (최대 14)")
k = 0
for s in res:
    if date_ok(s):
        continue
    kind, bk, wd, off = s["template"].split("|")
    p, e = pred_raw(s), exp_start(s)
    print(f"  [{KN[kind]} · {NAME[bk]} · 오늘 {WD[int(wd[2:])]} · {off}일] {s['user']}  → 정답 {e} / 예측 {p}")
    k += 1
    if k >= 14:
        break
