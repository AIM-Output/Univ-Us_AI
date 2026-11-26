#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""저장된 채점 결과에서 일정 길이(끝 시각 − 시작 시각)가 정답과 같은지 센다 — 길이 계산이 약점인지 볼 때 (GPU 없이).

  python scripts/dur_check.py outputs/f9-r1/eval_adapter.json outputs/date_probe/f9-r1/eval_probe.json
  python scripts/dur_check.py <eval 결과.json> ... --train data/f9/train.jsonl     # + 학습 데이터의 길이 표현 개수

시각이 있는 등록(create_event · allDay 아님) 턴만 본다. 문장의 길이 표현('1시간 반' · '두 시간' …, v2 의 DURS)별로
맞은 수를 낸다. 날짜를 못 읽는 답(달력에 없는 날짜 등)은 '모델 None 분' 으로 나온다 — 길이가 아니라 날짜 오류다.
real_test 결과에는 쓰지 않는다 — 틀린 문장을 그대로 출력한다.
표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare_f9_v2 as v2  # noqa: E402
from common import load_jsonl, parse_tool_call  # noqa: E402

DURS = sorted(v2.DURS, key=lambda x: -len(x[0]))          # 긴 표현부터 ('1시간 반' 이 '1시간' 보다 먼저)


def dur_in(user: str):
    for w, m in DURS:
        if w in user:
            return w, m
    return None


def mins(a: dict):
    try:
        s, e = dt.datetime.fromisoformat(a["start"]), dt.datetime.fromisoformat(a["end"])
        return int((e - s).total_seconds() // 60)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="eval.py 가 만든 .json (여러 개)")
    ap.add_argument("--train", default="", help="학습 데이터 — 길이 표현이 든 등록 문장 수를 센다")
    args = ap.parse_args()
    print("길이 표현 목록:", ", ".join(f"{w}={m}분" for w, m in v2.DURS))
    for path in args.results:
        if Path(path).name.startswith("eval_real_test"):
            print(f"== {path}: real_test 결과는 보지 않습니다 (가이드 3-7)")
            continue
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        tab = collections.defaultdict(lambda: [0, 0])
        nodur, bad = [0, 0], []
        for s in d["results"]:
            if not isinstance(s["expected"], dict) or s["expected"]["name"] != "create_event":
                continue
            ea = s["expected"]["arguments"]
            if ea.get("allDay"):
                continue
            pa = (parse_tool_call(s["pred_text"]) or {}).get("arguments") or {}
            em, pm = mins(ea), mins(pa)
            hit = dur_in(s["user"])
            cell = tab[hit[0]] if hit else nodur
            cell[0] += 1
            cell[1] += (em == pm)
            if em != pm:
                bad.append((s["id"], s["user"], em, pm))
        print(f"== {d['summary'].get('label')} ({path}) — 시각 있는 등록 턴의 길이(분)가 정답과 같은가")
        print(f"  길이 표현 없음(기본 60분 또는 끝 시각을 말함): {nodur[1]}/{nodur[0]}")
        for w, c in sorted(tab.items(), key=lambda x: -x[1][0]):
            print(f"  '{w}': {c[1]}/{c[0]}")
        for b in bad:
            print("   틀림:", b[0], "|", b[1], "| 정답", b[2], "분 · 모델", b[3], "분")
    if args.train:
        c = collections.Counter()
        for r in load_jsonl(args.train):
            if not any(tc["function"]["name"] == "create_event" for x in r["messages"] for tc in (x.get("tool_calls") or [])):
                continue
            for m in r["messages"]:
                if m["role"] == "user":
                    hit = dur_in(m["content"])
                    if hit:
                        c[hit[0]] += 1
        print(f"== 학습 데이터({args.train})에서 길이 표현이 든 등록 문장 {sum(c.values())}개:",
              " · ".join(f"{k} {v}" for k, v in c.most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
