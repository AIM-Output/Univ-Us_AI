#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""실제 문장 시험(real_dev · real_test)의 채점 결과를 묶음별로 본다 + 날짜 규칙을 적용했을 때의 점수 (GPU 없이).

  python scripts/real_report.py outputs/f9-r1/eval_real_dev.json data/f9/real_dev.jsonl             # 오답 사례까지
  python scripts/real_report.py outputs/f9-r1/eval_real_test.json data/f9/real_test.jsonl           # 집계만 (자동)
  python scripts/real_report.py <eval 결과.json> <rows.jsonl> --aggregate --max 20

보여 주는 것: 턴 정확도 · 도구 선택 · 인자 일치 · 거절 · 과잉 거절 / 날짜 규칙(scripts/date_rule.py) 적용 뒤 점수 /
학습 문장과 겹치는 문장(meta.seen_in_train)과 처음 보는 문장을 나눈 점수 / 쓴 사람별 · 정답 종류별 점수.
--aggregate 가 없으면 오답의 종류와 사례(문장 · 정답 · 모델 답)도 낸다.

**real_test 파일은 항상 집계만 낸다** — 문장 · 정답 · 모델 답 · 오답 종류를 출력하지 않는다(가이드 3-7 · 9장: 최종 시험은
점수만 보고, 그 결과로 데이터를 고치지 않는다). 실패 분석은 real_dev 로 한다.
표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import date_rule as dr  # noqa: E402  — src 경로도 여기서 잡힌다
from common import load_jsonl, parse_tool_call  # noqa: E402


def ok(s: dict) -> bool:
    return bool(s["exact_ok"] or s["text_ok"])


def one(x, n: int = 170) -> str:
    x = json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)
    return re.sub(r"\s+", " ", x)[:n]


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.1f}%)" if b else "—"


def fail_kind(s: dict) -> str:
    if s["turn_type"] == "refusal":
        return "거절해야 하는데 도구 호출" if s["pred_tool"] else "거절 표현 없음"
    if s["pred_tool"] is None:
        return "도구를 불러야 하는데 글로 답함"
    if not s["name_ok"]:
        return f"다른 도구 ({s['expected_tool']} → {s['pred_tool']})"
    pred = (parse_tool_call(s["pred_text"]) or {}).get("arguments") or {}
    exp = s["expected"]["arguments"]
    bad = sorted(k for k in set(exp) | set(pred) if exp.get(k) != pred.get(k))
    return f"{s['expected_tool']} 인자: " + "·".join(bad)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", help="eval.py 가 만든 .json")
    ap.add_argument("rows", help="채점에 쓴 시험 파일 (real_dev.jsonl · real_test.jsonl)")
    ap.add_argument("--aggregate", action="store_true", help="집계만 — 문장·답·오답 종류를 내지 않는다")
    ap.add_argument("--max", type=int, default=60, help="오답 사례 최대 줄 수")
    args = ap.parse_args()
    aggregate = args.aggregate or dr.is_test_file(args.rows)

    res = json.loads(Path(args.result).read_text(encoding="utf-8"))
    rows = load_jsonl(args.rows)
    results = res["results"]
    meta = {t[0]: t for t in dr.turns_of(rows)}
    assert [s["id"] for s in results] == list(meta), "결과와 시험 파일의 턴이 다르다"
    seen = {r["id"] for r in rows if (r.get("meta") or {}).get("seen_in_train")}
    rid = lambda s: s["id"].split("#")[0]  # noqa: E731
    after = {}
    for s in results:
        _tid, today, user, _c, _t, _s = meta[s["id"]]
        after[s["id"]] = dr.rescored(s, user, today)

    n = len(results)
    summ = res["summary"]
    tool = [s for s in results if s["turn_type"] == "tool"]
    ref = [s for s in results if s["turn_type"] == "refusal"]
    print(f"== {summ.get('label')} · 어댑터 {summ.get('adapter')} · 턴 {n} · 끝난 시각 {summ.get('finished_at')}")
    print(f"턴 정확도 {pct(sum(ok(s) for s in results), n)}")
    print(f"도구 선택 {pct(sum(bool(s['name_ok']) for s in tool), len(tool))} · 인자 일치 {pct(sum(bool(s['exact_ok']) for s in tool), len(tool))}"
          f" · 거절 {pct(sum(bool(s['text_ok']) for s in ref), len(ref))}")
    print(f"과잉 거절(도구 턴에 글로 답) {sum(s['pred_tool'] is None for s in tool)}"
          f" · 다른 도구 {sum(s['pred_tool'] is not None and not s['name_ok'] for s in tool)}"
          f" · 거절해야 하는데 도구 호출 {sum(s['pred_tool'] is not None for s in ref)}")
    chg = [s for s in results if after[s["id"]][1] == "changed"]
    a_ok = sum(v[0] for v in after.values())
    print(f"날짜 규칙 적용 후 {pct(a_ok, n)} — 규칙이 바꾼 턴 {len(chg)} (고쳐짐 {sum(not ok(s) and after[s['id']][0] for s in chg)}"
          f" · 맞던 것을 틀리게 함 {sum(ok(s) and not after[s['id']][0] for s in chg)}"
          f" · 바꿨지만 여전히 틀림 {sum(not ok(s) and not after[s['id']][0] for s in chg)}"
          f" · 그대로 맞음 {sum(ok(s) and after[s['id']][0] for s in chg)})")

    print("\n학습 문장과의 겹침")
    for name, pick in (("학습 문장과 같거나 거의 같은 문장", lambda s: rid(s) in seen), ("처음 보는 문장", lambda s: rid(s) not in seen)):
        g = [s for s in results if pick(s)]
        if g:
            print(f"  {name}: {pct(sum(ok(s) for s in g), len(g))} → 규칙 후 {sum(after[s['id']][0] for s in g)}/{len(g)}")

    print("\n쓴 사람별")
    for who in sorted({rid(s).split("-")[0] for s in results}):
        g = [s for s in results if rid(s).startswith(who + "-")]
        print(f"  {who}: {pct(sum(ok(s) for s in g), len(g))} → 규칙 후 {sum(after[s['id']][0] for s in g)}/{len(g)}")

    print("\n정답 종류별 (맞음/전체 → 규칙 후 · 도구 또는 거절을 맞게 고른 수)")
    tab = collections.defaultdict(lambda: [0, 0, 0, 0])
    for s in results:
        t = tab[s["expected_tool"] or "(거절)"]
        t[0] += 1
        t[1] += ok(s)
        t[2] += after[s["id"]][0]
        t[3] += bool(s["name_ok"]) if s["turn_type"] == "tool" else bool(s["text_ok"])
    for k, t in sorted(tab.items(), key=lambda x: -x[1][0]):
        print(f"  {k:18s} {t[1]:3d}/{t[0]:<3d} ({100 * t[1] / t[0]:5.1f}%) → {t[2]:3d}/{t[0]:<3d} · 선택 {t[3]}/{t[0]}")
    if aggregate:
        print("\n(집계만 냈습니다 — 문장 · 답 · 오답 종류는 출력하지 않습니다)")
        return 0

    fails = [s for s in results if not ok(s)]
    print("\n오답 종류")
    for k, v in collections.Counter(fail_kind(s) for s in fails).most_common():
        print(f"  {v:3d}  {k}")
    print(f"\n오답 {len(fails)}턴" + (f" (앞 {args.max}개)" if len(fails) > args.max else ""))
    for s in fails[:args.max]:
        _tid, today, user, _c, _t, _s = meta[s["id"]]
        a = after[s["id"]]
        tag = " [규칙으로 고쳐짐]" if a[0] else (" [규칙이 바꿨지만 틀림]" if a[1] == "changed" else "")
        mark = " (학습 문장과 같음)" if rid(s) in seen else ""
        print(f"- {s['id']} 오늘 {today}({dr.WDN[today.weekday()]}){mark}{tag} | {user}")
        print(f"    정답 {one(s['expected'])}")
        print(f"    모델 {one(s['pred_text'])}")
        if a[1] == "changed":
            print(f"    규칙 {one(a[2])}")
    for s in [s for s in chg if ok(s) and not after[s["id"]][0]]:
        _tid, today, user, _c, _t, _s = meta[s["id"]]
        print(f"- [규칙이 틀리게 함] {s['id']} 오늘 {today} | {user}")
        print(f"    정답 {one(s['expected'])}\n    모델 {one(s['pred_text'])}\n    규칙 {one(after[s['id']][2])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
