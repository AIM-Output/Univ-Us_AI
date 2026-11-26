#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""두 채점 결과(eval.py 가 만든 .json)를 턴별로 견준다 — 새 run 이 무엇을 고치고 무엇을 망가뜨렸는지 본다 (GPU 없이).

  python scripts/compare_runs.py outputs/f9-r1/eval_adapter.json outputs/f9-r1-date/eval_adapter.json f9-r1 f9-r1-date
  python scripts/compare_runs.py <a.json> <b.json> [a 이름] [b 이름] [목록 최대 줄 수]

두 결과는 같은 시험 파일을 같은 순서로 채점한 것이어야 한다(다르면 멈춘다).
보여 주는 것: 턴 정확도 · 도구 선택 · 인자 일치 · 거절 · 과잉 거절 / 턴별 변화(고쳐짐 · 나빠짐 · 둘 다 틀림)와 그 문장 /
도구별 · 틀별로 달라진 곳 / 날짜 진단 파일(scripts/make_date_probe.py)이면 묶음별 날짜 정확도.
real_test 결과에는 쓰지 않는다 — 문장을 그대로 출력한다.
표준 라이브러리만 쓴다.
"""
import collections
import datetime as dt
import json
import re
import sys
from pathlib import Path


def load(p):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    return d["summary"], d["results"]


def ok(s):
    return bool(s["exact_ok"] or s["text_ok"])


def one(x, n=150):
    x = json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)
    return re.sub(r"\s+", " ", x)[:n]


if len(sys.argv) < 3:
    sys.exit(__doc__)
sa, ra = load(sys.argv[1])
sb, rb = load(sys.argv[2])
na = sys.argv[3] if len(sys.argv) > 3 else "a"
nb = sys.argv[4] if len(sys.argv) > 4 else "b"
cap = int(sys.argv[5]) if len(sys.argv) > 5 else 40
assert [s["id"] for s in ra] == [s["id"] for s in rb], "두 결과의 턴 순서·id 가 다르다"
assert [s["user"] for s in ra] == [s["user"] for s in rb], "두 결과의 문장이 다르다"
n = len(ra)

print(f"== {na} → {nb} · 턴 {n}")
print(f"어댑터: {sa.get('adapter')} → {sb.get('adapter')}")
for name, s, r in ((na, sa, ra), (nb, sb, rb)):
    tool = [x for x in r if x["turn_type"] == "tool"]
    ref = [x for x in r if x["turn_type"] == "refusal"]
    line = (f"  {name:12s} 턴 정확도 {sum(ok(x) for x in r)}/{n} ({100*sum(ok(x) for x in r)/n:.1f}%)"
            f" · 도구 선택 {sum(bool(x['name_ok']) for x in tool)}/{len(tool)}"
            f" · 인자 일치 {sum(bool(x['exact_ok']) for x in tool)}/{len(tool)}")
    if ref:
        line += f" · 거절 {sum(bool(x['text_ok']) for x in ref)}/{len(ref)}"
    over = sum(x["pred_tool"] is None for x in tool)                       # 도구를 불러야 하는데 글로 답함
    wrong = sum(x["pred_tool"] is not None and not x["name_ok"] for x in tool)
    under = sum(x["pred_tool"] is not None for x in ref)                   # 거절해야 하는데 도구를 부름
    line += f" · 과잉 거절(도구 턴에 글로 답) {over} · 다른 도구 {wrong}" + (f" · 거절 턴에 도구 호출 {under}" if ref else "")
    line += f" · 평균 지연 {s.get('latency_mean_s', 0):.2f}s"
    print(line)

trans = collections.Counter()
groups = collections.defaultdict(list)
for x, y in zip(ra, rb):
    key = ("둘 다 맞음", "나빠짐", "고쳐짐", "둘 다 틀림")[(0 if ok(x) else 2) + (0 if ok(y) else 1)]
    trans[key] += 1
    groups[key].append((x, y))
print("\n턴별 변화:", " · ".join(f"{k} {trans[k]}" for k in ("둘 다 맞음", "고쳐짐", "나빠짐", "둘 다 틀림")))

pt = collections.defaultdict(lambda: [0, 0, 0])
for x, y in zip(ra, rb):
    k = x["expected_tool"] or f"({x['turn_type']})"
    pt[k][0] += 1
    pt[k][1] += ok(x)
    pt[k][2] += ok(y)
print("\n도구(턴 종류)별 맞은 턴 — 달라진 것만")
diff = [(k, v) for k, v in sorted(pt.items()) if v[1] != v[2]]
for k, v in diff:
    print(f"  {k:18s} {v[1]}/{v[0]} → {v[2]}/{v[0]}")
if not diff:
    print("  (없음)")

tp = collections.defaultdict(lambda: [0, 0, 0])
for x, y in zip(ra, rb):
    t = (x.get("template") or "?").split("|")[0]
    tp[t][0] += 1
    tp[t][1] += ok(x)
    tp[t][2] += ok(y)
diff = [(k, v) for k, v in sorted(tp.items()) if v[1] != v[2]]
print("\n틀(template)별 맞은 턴 — 달라진 것만")
for k, v in diff:
    print(f"  {k:18s} {v[1]}/{v[0]} → {v[2]}/{v[0]}")
if not diff:
    print("  (없음)")

for key in ("나빠짐", "고쳐짐", "둘 다 틀림"):
    g = groups[key]
    print(f"\n[{key}] {len(g)}턴" + (f" (앞 {cap}개)" if len(g) > cap else ""))
    for x, y in g[:cap]:
        same = " (예측 같음)" if x["pred_text"] == y["pred_text"] else ""
        print(f"  {x['id']} `{x.get('template')}` [{x['turn_type']}] 사용자: {one(x['user'], 90)}")
        print(f"      기대 {one(x['expected'])}")
        print(f"      {na} {one(x['pred_text'])}")
        print(f"      {nb} {one(y['pred_text'])}{same}")

# 날짜 진단 파일이면(틀 이름이 kind|묶음|요일|차이) 묶음별 날짜 정확도를 나란히 본다
if all((x.get("template") or "").count("|") == 3 for x in ra):
    NAME = {"d31": "정답이 31일", "end30": "정답이 30일(30일짜리 달의 말일)", "d01": "정답이 1일",
            "d30of31": "정답이 30일(31일짜리 달)", "cross": "달을 넘겨 2~9일", "mid": "같은 달 8~24일"}
    EDGE = ("d31", "end30", "d01")
    KN = {"n_days": "N일 뒤", "next_wd": "다음 주 X요일"}

    def raw(s):
        m = re.search(r'"start"\s*:\s*"(\d{4}-\d{2}-\d{2})', s["pred_text"])
        return m.group(1) if m else None

    def date_ok(s):
        try:
            return dt.date.fromisoformat(raw(s) or "") == dt.date.fromisoformat(s["expected"]["arguments"]["start"][:10])
        except ValueError:
            return False

    def invalid(s):
        r_ = raw(s)
        if not r_:
            return False
        try:
            dt.date.fromisoformat(r_)
            return False
        except ValueError:
            return True

    tab = collections.defaultdict(lambda: [0, 0, 0])
    for x, y in zip(ra, rb):
        kind, bk, wd, off = x["template"].split("|")
        for k in ((kind, bk), ("all", bk), (kind, "edge" if bk in EDGE else "rest"), ("all", "edge" if bk in EDGE else "rest"),
                  (kind, "total"), ("all", "total"), ("wd", int(wd[2:])) if kind == "next_wd" else None,
                  ("wd_edge", int(wd[2:])) if kind == "next_wd" and bk in EDGE else None):
            if k is None:
                continue
            tab[k][0] += 1
            tab[k][1] += date_ok(x)
            tab[k][2] += date_ok(y)
    cell = lambda c: f"{c[1]:3d}→{c[2]:<3d}/{c[0]:<3d}"
    print(f"\n== 날짜만 본 정확도 ({na} → {nb} / 전체)")
    print(f"  {'묶음':30s} {'N일 뒤':>14s} {'다음 주 X요일':>14s} {'합계':>14s}")
    for bk in NAME:
        print(f"  {NAME[bk]:30s} " + "   ".join(cell(tab[(k, bk)]) for k in ("n_days", "next_wd", "all")))
    for bk, title in (("edge", "말일·1일 (위 세 묶음)"), ("rest", "그 밖 (아래 세 묶음)"), ("total", "전체")):
        c = tab[("all", bk)]
        print(f"  {title:30s} " + "   ".join(cell(tab[(k, bk)]) for k in ("n_days", "next_wd", "all"))
              + f"   {100*c[1]/c[0]:.1f}% → {100*c[2]/c[0]:.1f}%")
    WD = "월화수목금토일"
    print("  '다음 주 X요일' 오늘 요일별:        " + " · ".join(f"{WD[w]} {tab[('wd', w)][1]}→{tab[('wd', w)][2]}/{tab[('wd', w)][0]}" for w in range(7) if tab[('wd', w)][0]))
    print("  그중 정답이 말일·1일인 것:         " + " · ".join(f"{WD[w]} {tab[('wd_edge', w)][1]}→{tab[('wd_edge', w)][2]}/{tab[('wd_edge', w)][0]}" for w in range(7) if tab[('wd_edge', w)][0]))
    print(f"  달력에 없는 날짜를 답한 턴: {sum(invalid(x) for x in ra)} → {sum(invalid(y) for y in rb)}")
    print(f"  인자까지 맞음: {sum(bool(x['exact_ok']) for x in ra)} → {sum(bool(y['exact_ok']) for y in rb)} / {n}")
    print(f"  날짜는 맞고 다른 인자가 틀린 턴: {sum(date_ok(x) and not x['exact_ok'] for x in ra)} → {sum(date_ok(y) and not y['exact_ok'] for y in rb)}")
    off = collections.Counter()
    for y in rb:
        if date_ok(y):
            continue
        kind, bk, wd, o = y["template"].split("|")
        try:
            d = (dt.date.fromisoformat(raw(y)) - dt.date.fromisoformat(y["expected"]["arguments"]["start"][:10])).days
            d = f"{d:+d}일"
        except (ValueError, TypeError):
            d = f"없는 날짜 {raw(y)[5:]}" if invalid(y) else "날짜 못 읽음"
        off[(KN[kind], NAME[bk], d)] += 1
    print(f"  {nb} 가 틀린 날짜 — 예측 − 정답:")
    for (k, b_, d), c in sorted(off.items()):
        print(f"    {k:12s} {b_:30s} {d} × {c}")
    if not off:
        print("    (없음)")
