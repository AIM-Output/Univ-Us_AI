#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train.log 의 loss 줄을 모아 구간별로 본다 — 학습이 정상으로 끝났는지 확인할 때.

  python scripts/loss_summary.py outputs/f9-r1/train.log            # 40 step 묶음
  python scripts/loss_summary.py outputs/f9-r1/train.log 200        # 200 step 묶음

보여 주는 것: loss 줄 수 · NaN 이나 튀는 값 · 처음과 끝의 loss · 묶음별 평균 loss · 토큰 정확도 · 학습률 /
로그에서 Traceback · out of memory · Warning 등이 든 줄.
표준 라이브러리만 쓴다.
"""
import ast
import re
import sys
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit(__doc__)
text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
step_per = int(sys.argv[2]) if len(sys.argv) > 2 else 40
rows = []
for m in re.finditer(r"\{'loss':[^{}]*\}", text):
    try:
        d = ast.literal_eval(m.group(0))
    except Exception:
        continue
    rows.append({k: float(v) for k, v in d.items()})
print("loss 줄", len(rows), "(5 step 마다 한 줄)")
bad = [i for i, r in enumerate(rows) if r["loss"] != r["loss"] or r["loss"] > 1e3]
print("NaN·튀는 값", len(bad))
print("처음 5줄 loss:", [round(r["loss"], 5) for r in rows[:5]])
print("끝 5줄 loss  :", [round(r["loss"], 5) for r in rows[-5:]])
mx = max(range(len(rows)), key=lambda i: rows[i]["loss"])
print(f"가장 큰 loss {rows[mx]['loss']:.5f} (step {(mx + 1) * 5} 근처) · 가장 작은 loss {min(r['loss'] for r in rows):.6f}")
print(f"mean_token_accuracy 최소 {min(r['mean_token_accuracy'] for r in rows):.4f} · 평균 {sum(r['mean_token_accuracy'] for r in rows)/len(rows):.4f}")
print(f"grad_norm 최대 {max(r['grad_norm'] for r in rows):.4f}")
per = step_per // 5
print(f"\n{step_per} step 묶음별 평균 loss · 학습률(묶음 끝)")
for i in range(0, len(rows), per):
    seg = rows[i:i + per]
    print(f"  step {i*5+1:3d}~{(i+len(seg))*5:3d}: loss {sum(r['loss'] for r in seg)/len(seg):.5f} · 토큰 정확도 {sum(r['mean_token_accuracy'] for r in seg)/len(seg):.4f} · lr {seg[-1]['learning_rate']:.2e}")
for pat in ("Traceback", "out of memory", "CUDA error", "nan", "Error", "Warning", "경고", "중단"):
    hits = [ln for ln in re.split(r"[\r\n]+", text) if pat in ln]
    if hits:
        print(f"\n'{pat}' 가 든 줄 {len(hits)}:")
        for ln in hits[:6]:
            print("   ", ln[:220])
