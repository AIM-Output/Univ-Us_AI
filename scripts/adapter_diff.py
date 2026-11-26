#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""끝난 어댑터 점검 (CPU): 값이 모두 유한한지 · 시작 어댑터에서 얼마나 움직였는지 · 마지막 체크포인트와 같은지.

  python scripts/adapter_diff.py outputs/f9-r1/adapter outputs/f9-r1-date/adapter outputs/f9-r1-date/checkpoint-440

train.py --init_adapter 로 다른 run 에 이어서 학습했을 때 쓴다 — 가중치가 시작점에서 얼마나 움직였는지가
"가볍게 덧칠했는가, 크게 바뀌었는가" 를 말해 준다.
학습 환경(.venv — torch · safetensors)에서 돌린다. GPU 는 쓰지 않는다.
"""
import sys

import torch
from safetensors.torch import load_file

if len(sys.argv) != 4:
    sys.exit(__doc__)
a = load_file(sys.argv[1] + "/adapter_model.safetensors")
b = load_file(sys.argv[2] + "/adapter_model.safetensors")
c = load_file(sys.argv[3] + "/adapter_model.safetensors")
print("텐서 수", len(a), len(b), "· 이름 같음", set(a) == set(b), "· 모양 같음", all(a[k].shape == b[k].shape for k in a))
print("파라미터 수", sum(v.numel() for v in b.values()))
print("유한하지 않은 값이 든 텐서", sum(not torch.isfinite(v.float()).all().item() for v in b.values()))
rel, same = [], 0
na = nb = 0.0
for k in a:
    x, y = b[k].float(), a[k].float()
    d = (x - y).norm().item()
    rel.append(d / max(y.norm().item(), 1e-12))
    same += d == 0
    if "lora_B" in k:
        na += y.norm().item() ** 2
        nb += x.norm().item() ** 2
rel.sort()
print(f"시작 어댑터와의 상대 차이: 최소 {rel[0]:.2e} · 중앙 {rel[len(rel)//2]:.2e} · 최대 {rel[-1]:.2e} · 그대로인 텐서 {same}/{len(a)}")
print(f"lora_B 전체 크기: 시작 {na ** 0.5:.4f} → 끝 {nb ** 0.5:.4f}")
print("마지막 체크포인트와 완전히 같음:", set(b) == set(c) and all(torch.equal(b[k], c[k]) for k in b))
