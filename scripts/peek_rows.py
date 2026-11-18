#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""데이터 파일(JSONL)에서 문장을 뽑아 표로 본다 — 생성 문장 검수용.

  python scripts/peek_rows.py data/f9/extra_v2.jsonl --n 100 --seed 1125     # 무작위 100개 (seed 는 그날 날짜)
  python scripts/peek_rows.py data/f9/extra_v3.jsonl --tool create_event     # 첫 도구가 create_event 인 행만
  python scripts/peek_rows.py data/f9/extra_v3.jsonl --frame sb-f003         # 그 틀로 만든 문장만 (v3)
  python scripts/peek_rows.py data/f9/extra_v2.jsonl --find 과제              # 문장에 '과제' 가 든 행만

표의 칸: id · 기준시각 · 문장 · 정답(날짜 옆에 요일). 어색한 문장은 id 를 적어 이슈(data 라벨)로 남긴다.
표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import load_jsonl  # noqa: E402  — Windows 콘솔에서 한글이 깨지지 않게 하는 설정도 여기서 된다
from datasrc import cell, row_answer, row_now, row_user  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="JSONL 파일 (train · eval · extra_v2 · extra_v3 · real_dev …)")
    ap.add_argument("--n", type=int, default=100, help="몇 개 볼지. 0 이면 전부")
    ap.add_argument("--seed", type=int, default=0, help="같은 숫자면 같은 문장이 나온다")
    ap.add_argument("--tool", default="", help="첫 도구 이름으로 거르기 (거절·인사는 none)")
    ap.add_argument("--frame", default="", help="틀 id 나 패밀리 이름의 일부로 거르기 (meta.template)")
    ap.add_argument("--find", default="", help="문장에 이 말이 든 행만")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"파일이 없습니다: {path} — 먼저 데이터를 만드세요 (가이드 3-1)")
        return 1
    rows = load_jsonl(path)
    total = len(rows)
    if args.tool:
        rows = [r for r in rows if (r.get("meta") or {}).get("tool") == args.tool]
    if args.frame:
        rows = [r for r in rows if args.frame in str((r.get("meta") or {}).get("template", ""))]
    if args.find:
        rows = [r for r in rows if args.find in row_user(r)]
    matched = len(rows)
    if args.n and matched > args.n:
        rows = random.Random(args.seed).sample(rows, args.n)
    print(f"`{path.as_posix()}` · 전체 {total}행 · 조건에 맞는 {matched}행 중 {len(rows)}행 (seed {args.seed})\n")
    print("| # | id | 기준시각 | 문장 | 정답 |")
    print("| ---: | --- | --- | --- | --- |")
    for i, r in enumerate(rows, 1):
        print(f"| {i} | {r.get('id', '')} | {row_now(r)} | {cell(row_user(r))} | {cell(row_answer(r))} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
