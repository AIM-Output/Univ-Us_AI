#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""그래픽 메모리 8GB 카드(Windows)에서 본 학습이 들어가게 하는 스위치 두 개 — train.py --low_vram 1

같은 코드·같은 설정인데 Windows 에서만 그래픽 메모리가 수 GiB 더 드는 곳이 두 군데 있다.
(2026-11-22 RTX 5060 8GB · torch 2.9.0+cu129 · transformers 5.22.1 · trl 1.17.1 에서 확인)

1) LM head — 어느 위치를 계산하나
   trl 의 SFTTrainer 는 손실을 "fused LM head" 로 계산한다([위치 × 단어 32,768] 조각으로 나눠 계산).
     · Triton 있음(Linux)   → 라벨이 -100 이 아닌 위치만 (정답 토큰 수십 개)
     · Triton 없음(Windows) → 모든 위치 (문장 길이 ≈ 2,300) — 조각마다 fp32 임시 텐서가 여러 장 생긴다
   → Triton 이 없어도 라벨 위치만 계산하게 한다. 위치별 계산은 서로 독립이라 손실·기울기는 같다.

2) 어텐션 — 어느 커널로 계산하나
   key/value head(8개)가 query head(32개)보다 적은 모델(GQA)은 transformers 가 SDPA 에 enable_gqa=True 로 넘긴다.
   Windows 용 torch 에는 flash 커널이 없고, mem-efficient 커널은 head 수가 다르면 쓰지 못한다.
   그래서 math 경로로 떨어져 [head 32 × 길이 × 길이] 텐서를 여러 장 만든다 (길이 2,322 에서 +3.5GiB 이상).
   → key/value 를 32 head 로 펼쳐서 넘긴다(transformers 에 원래 있는 repeat_kv 경로). mem-efficient 커널이 쓰인다.
     수식은 같고 커널만 다르다. bf16 반올림 수준의 차이는 생긴다(한 번의 어텐션 출력에서 평균 1e-4 안팎).
     그래서 이 스위치를 켠 run 은 AI 팀 재현(f9-r0)으로 점수가 같은지 확인한다.

실측 (가장 긴 학습 문장 2,322 토큰 · 4-bit QLoRA · 모델만 올렸을 때 3.34GiB)
   스위치 없음: 7GiB 를 넘겨 OOM / 둘 다 켬: 최고 4.97GiB

기본값(--low_vram 0)에서는 이 파일을 읽지 않는다 — 아무것도 바뀌지 않는다.
"""
from __future__ import annotations

_HEAD_MARK = "compact = _use_triton("       # trl.trainer.utils.add_fused_lm_head 에서 위치 선택을 정하는 줄
_GQA_MARK = "use_gqa_in_sdpa("              # transformers.integrations.sdpa_attention.sdpa_attention_forward 에서 GQA 를 정하는 줄


def _source(fn) -> str:
    import inspect
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return ""


def _labels_only_head() -> str:
    """fused LM head 가 라벨 위치만 계산하게 한다. "on"=켰다 · "triton"=원래 그렇게 동작(바꾼 것 없음)."""
    try:
        from trl.kernels import chunked_logprob as kernel
        from trl.trainer import utils as trainer_utils
    except ImportError as e:
        raise SystemExit(f"[중단] --low_vram: 이 trl 에는 fused LM head 가 없다 ({e}). "
                         "requirements.lock.txt 의 버전으로 맞출 것") from e
    if getattr(kernel, "_TRITON_AVAILABLE", False):
        return "triton"
    if not hasattr(trainer_utils, "_use_triton") or _HEAD_MARK not in _source(getattr(trainer_utils, "add_fused_lm_head", None)):
        raise SystemExit("[중단] --low_vram: 이 trl 버전은 LM head 의 위치 선택 방식이 달라 적용할 수 없다. "
                         "requirements.lock.txt 의 버전으로 맞추거나 --low_vram 을 빼고 실행할 것")
    # trainer_utils 안의 이름만 바꾼다 → add_fused_lm_head 의 `compact`(라벨 위치만 고르기)만 켜진다.
    # kernels.chunked_logprob 는 자기 모듈의 _use_triton 을 보므로 조각 계산은 그대로 torch 로 한다.
    trainer_utils._use_triton = lambda device: True
    return "on"


def _same_heads_for_sdpa(model) -> str:
    """SDPA 에 key/value 를 query 와 같은 head 수로 펼쳐서 넘긴다. "on"=켰다 · "skip(...)"=해당 없음(바꾼 것 없음)."""
    impl = getattr(getattr(model, "config", None), "_attn_implementation", None)
    if impl != "sdpa":
        return f"skip(attn={impl})"
    try:
        from transformers.integrations import sdpa_attention
    except ImportError as e:
        raise SystemExit(f"[중단] --low_vram: transformers.integrations.sdpa_attention 이 없다 ({e})") from e
    src = _source(getattr(sdpa_attention, "sdpa_attention_forward", None))
    if not hasattr(sdpa_attention, "use_gqa_in_sdpa") or _GQA_MARK not in src or "repeat_kv(" not in src:
        raise SystemExit("[중단] --low_vram: 이 transformers 버전은 SDPA 의 GQA 처리 방식이 달라 적용할 수 없다. "
                         "requirements.lock.txt 의 버전으로 맞추거나 --low_vram 을 빼고 실행할 것")
    # False 를 돌려주면 sdpa_attention_forward 가 repeat_kv 로 key/value 를 펼친 뒤 enable_gqa 없이 SDPA 를 부른다.
    sdpa_attention.use_gqa_in_sdpa = lambda attention_mask, key, value: False
    return "on"


def enable(model) -> dict[str, str]:
    """두 스위치를 켠다. 적용할 수 없는 버전이면 조용히 넘어가지 않고 멈춘다(SystemExit).

    model: 학습할 모델(PEFT 로 감싼 것이어도 된다) — 어텐션 구현이 sdpa 인지 본다.
    반환: {"labels_only_head": "on"|"triton", "sdpa_same_heads": "on"|"skip(...)"} — run.json 에 적는다.
    """
    return {"labels_only_head": _labels_only_head(), "sdpa_same_heads": _same_heads_for_sdpa(model)}
