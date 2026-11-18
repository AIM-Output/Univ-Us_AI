#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""평가 결과(eval.py 가 만든 .json)에서 틀린 턴만 뽑아 표로 본다 — 실패 분석용.

  python scripts/peek_fail.py docs/eval/f9-r1/eval_real_dev.json            # 틀린 턴 전부
  python scripts/peek_fail.py outputs/f9-r1/eval_adapter.json --n 30         # 무작위 30개
  python scripts/peek_fail.py docs/eval/f9-r1/eval_real_dev.json --tool list_events    # 기대 도구로 거르기

위쪽 표: 무엇을 무엇으로 틀렸는지(기대 → 예측) 많은 순.  아래 표: 문장 · 기대 · 예측 · 달랐던 값.
real_test 결과에는 쓰지 않는다 — 최종 시험은 점수만 본다.
표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import normalize_value, parse_tool_call, strip_think  # noqa: E402
from datasrc import cell, describe_call  # noqa: E402


def failed(r: dict) -> bool:
    return not (r.get("exact_ok") or r.get("text_ok"))


def expected_text(r: dict) -> str:
    e = r.get("expected")
    if isinstance(e, dict):
        return describe_call(e.get("name", "?"), e.get("arguments", {}))
    return "거절" if r.get("turn_type") == "refusal" else "글: " + str(e)[:60]


def predicted(r: dict) -> tuple[str, str]:
    """(예측 요약, 표에 쓸 짧은 이름)."""
    raw = r.get("pred_text") or ""
    p = parse_tool_call(raw)
    if p:
        return describe_call(p["name"], p["arguments"]), p["name"]
    text = re.sub(r"\s+", " ", strip_think(raw)).strip()
    return ("글: " + text[:80]) if text else "(빈 답)", "글" if text else "빈 답"


def diff_fields(r: dict) -> str:
    """도구는 맞았는데 값이 다른 경우 — 어느 인자가 달랐는지."""
    e = r.get("expected")
    p = parse_tool_call(r.get("pred_text") or "")
    if not (isinstance(e, dict) and p and p["name"] == e.get("name")):
        return ""
    ea, pa = normalize_value(e.get("arguments") or {}), normalize_value(p["arguments"])
    out = []
    for k in sorted(set(ea) | set(pa)):
        if ea.get(k, "(없음)") != pa.get(k, "(없음)"):
            out.append(f"{k}: 기대 {json.dumps(ea.get(k, '(없음)'), ensure_ascii=False)} / 예측 {json.dumps(pa.get(k, '(없음)'), ensure_ascii=False)}")
    return "; ".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="eval.py 가 만든 .json (eval_adapter.json · eval_real_dev.json …)")
    ap.add_argument("--n", type=int, default=0, help="몇 개 볼지. 0 이면 전부")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tool", default="", help="기대 도구 이름으로 거르기 (거절은 none)")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"파일이 없습니다: {path}")
        return 1
    if "real_test" in path.name:
        print("real_test 결과는 분석하지 않습니다 — 최종 시험은 점수만 봅니다 (가이드 3-7).")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results") or []
    fails = [r for r in results if failed(r)]
    s = data.get("summary") or {}
    print(f"`{path.as_posix()}` · {s.get('label', '')} · 턴 {len(results)}개 중 틀린 턴 {len(fails)}개"
          + (f" · 턴 정확도 {s['turn_acc']:.1f}%" if isinstance(s.get("turn_acc"), (int, float)) else "") + "\n")

    pairs = Counter()
    for r in fails:
        exp = r.get("expected_tool") or ("거절" if r.get("turn_type") == "refusal" else "글")
        _, pname = predicted(r)
        pairs[(exp, pname if pname != exp else f"{pname} (값이 다름)")] += 1
    if pairs:
        print("| 기대 | 예측 | 턴 수 |\n| --- | --- | ---: |")
        for (exp, pname), c in pairs.most_common(20):
            print(f"| {exp} | {pname} | {c} |")
        print()

    if args.tool:
        want = None if args.tool == "none" else args.tool
        fails = [r for r in fails if r.get("expected_tool") == want]
    if args.n and len(fails) > args.n:
        fails = random.Random(args.seed).sample(fails, args.n)
    print("| # | id | 종류 | 문장 | 기대 | 예측 | 달랐던 값 |")
    print("| ---: | --- | --- | --- | --- | --- | --- |")
    for i, r in enumerate(fails, 1):
        ptxt, _ = predicted(r)
        print(f"| {i} | {r.get('id', '')} | {r.get('turn_type', '')} | {cell(r.get('user', ''))} | {cell(expected_text(r))} | "
              f"{cell(ptxt)} | {cell(diff_fields(r))} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
