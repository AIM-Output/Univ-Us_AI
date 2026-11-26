#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""실제 문장 검사 — make_real.py 의 검사를 그대로 돌리되, real_test 로 갈 행은 내용을 가리고 종류·개수만 보여 준다.

  python scripts/real_check_masked.py              # 전체 통계: 파일별 문장 수 · dev/test 로 나뉜 수 · 종류별 수 · 오류·경고 종류
  python scripts/real_check_masked.py haram        # + 그 파일의 오류·경고 줄 (dev 행은 그대로, test 행은 종류만)

`make_real.py --check` 는 오류·경고 줄을 그대로 출력한다. 문장을 쓴 사람이 자기 파일을 고칠 때는 그것을 쓴다.
이 도구는 **남의 문장을 확인하는 사람**(리뷰어 · 학습 담당)이 real_test 로 갈 문장을 보지 않고 전체 상태를 볼 때 쓴다(가이드 3-7).
먼저 data/f9/train.jsonl 이 있어야 학습 문장과의 겹침까지 센다.
표준 라이브러리만 쓴다.
"""
import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOCUS = sys.argv[1] if len(sys.argv) > 1 else ""
sys.path.insert(0, str(ROOT / "src"))
import make_real as mr  # noqa: E402
from common import load_tools  # noqa: E402
from datasrc import Report, list_sources  # noqa: E402
import prepare_f9  # noqa: E402

tools, _ = load_tools(prepare_f9.TOOLS_PATH)
schemas = {t["function"]["name"]: t["function"].get("parameters") or {} for t in tools}
rep = Report()
paths = list_sources(ROOT / "data_src" / "real", [])
items = mr.load_items(paths, schemas, rep)
mr.check_overlap(items, ROOT / "data" / "f9", rep)

# 파일:줄 → id (내용은 출력하지 않는다)
id_at = {}
n_rows = {}
for p in paths:
    lines = p.read_text(encoding="utf-8-sig").splitlines()
    n_rows[p.stem] = sum(1 for ln in lines[1:] if ln.strip())
    for i, ln in enumerate(lines, 1):
        id_at[(p.name, i)] = ln.split("\t")[0].strip()


def mask(msg: str) -> str:
    msg = re.sub(r'"[^"]*"', '"…"', msg)
    msg = re.sub(r"\([^()]*\)", "(…)", msg)
    return re.sub(r"\d+", "N", msg)


print(f"파일 {len(paths)}개 · 문장 {sum(n_rows.values())}개 ({' · '.join(f'{k} {v}' for k, v in n_rows.items())})"
      f" · 검사를 통과한 문장 {len(items)}개 · 오류 {len(rep.errors)} · 경고 {len(rep.warnings)}")

by_owner = collections.defaultdict(lambda: collections.Counter())
split = collections.defaultdict(lambda: collections.Counter())
kinds = collections.defaultdict(lambda: collections.Counter())
seen = collections.defaultdict(lambda: collections.Counter())
for it in items:
    who = it.id.split("-")[0]
    sp = mr.split_of(it.id)
    split[who][sp] += 1
    if it.tool is None:
        k = "거절"
    elif it.tool == "list_events" and mr._WRITE_VERB.search(it.user):
        k = "수정·삭제(첫 단계 list_events)"
    else:
        k = it.tool
    kinds[who][k] += 1
    by_owner[sp][k] += 1
    if it.seen_in_train:
        seen[who][sp] += 1
print("\n누가 · dev/test 로 나뉜 수 · 학습 문장과 같거나 거의 같은 문장")
for who in sorted(split):
    print(f"  {who}: dev {split[who]['dev']} · test {split[who]['test']} · 학습 문장과 겹침 dev {seen[who]['dev']} / test {seen[who]['test']}")
tot = collections.Counter()
for who in split:
    tot.update(split[who])
print(f"  합계: dev {tot['dev']} · test {tot['test']}")
print("\n종류별 문장 수")
names = sorted({k for c in kinds.values() for k in c})
for who in sorted(kinds):
    print(f"  {who}: " + " · ".join(f"{k} {kinds[who][k]}" for k in names if kinds[who][k]))
print("  dev 묶음: " + " · ".join(f"{k} {by_owner['dev'][k]}" for k in names if by_owner['dev'][k]))
print("  test 묶음: " + " · ".join(f"{k} {by_owner['test'][k]}" for k in names if by_owner['test'][k]))

print("\n오류·경고 (파일별 · 종류별 개수 — test 로 갈 행은 내용을 가림)")
agg = collections.defaultdict(collections.Counter)
detail = []
for f, n, lv, text in sorted(rep._all, key=lambda x: x[:3]):
    name = Path(f).name
    rid = id_at.get((name, n), "?")
    sp = mr.split_of(rid) if mr._ID.match(rid or "") else "?"
    body = text.split(": ", 2)[-1] if text.count(": ") >= 2 else text
    level = "오류" if lv == 0 else "경고"
    agg[(Path(f).stem, level)][mask(body)[:70]] += 1
    if FOCUS and Path(f).stem == FOCUS:
        detail.append((rid, sp, level, body))
for (stem, level), c in sorted(agg.items()):
    print(f"  [{stem} · {level}] 합 {sum(c.values())}")
    for k, v in c.most_common():
        print(f"      {v:3d}  {k}")
if FOCUS:
    print(f"\n{FOCUS} 파일의 오류·경고 줄 (dev 행은 그대로 · test 행은 종류만)")
    for rid, sp, level, body in detail:
        print(f"  {rid} [{sp}] {level}: " + (body if sp == "dev" else mask(body)[:70]))
    if not detail:
        print("  (없음)")
