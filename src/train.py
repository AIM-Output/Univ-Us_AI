#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F9 Function Calling LoRA/QLoRA 학습 — 모델 이름은 여기 없다. config.env → train.sh → 인자로 들어온다.

  python src/train.py --model_id <HF id> --data_dir data/f9 --output_dir outputs/run1 \
      --load_4bit 1 --compute_dtype bf16 --max_seq 2560 --batch 1 --grad_accum 8 --lora_r 16 --epochs 3 --lr 2e-4

환경 차이는 런타임에 맞춘다.
  · CUDA 없음(스모크) → fp32 · 4-bit 끔 · gradient checkpointing 끔
  · transformers 4.5x~5.x: from_pretrained(dtype=) 실패 시 torch_dtype= 로 재시도
  · trl 0.17~1.x: SFTConfig 필드를 dataclass 에서 읽어 있는 것만 넘긴다 (max_length / max_seq_length, eval_strategy …)
손실(--loss):
  · completion(기본) — assistant 턴마다 (prompt, completion) 쌍을 만들어 **정답 토큰에만** 손실. 두 단계 행은 쌍 2개
  · full             — 대화 전체 텍스트에 손실
토큰 예산: system + 도구 11개 정의만 약 1,950 토큰이라 --max_seq 는 2304 이상이어야 한다.
  렌더 길이가 --max_seq 를 넘는 행은 **버린다**(잘린 정답으로 학습하지 않는다). 남는 행이 없으면 멈춘다.
산출물: <output_dir>/adapter/ (LoRA 어댑터 + 토크나이저) · <output_dir>/run.json
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import detect_render_mode, ensure_dir, lib_versions, load_jsonl, load_tools, render_chat  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", required=True, help="HF 모델 id — config.env 의 MODEL_ID")
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "f9"))
    ap.add_argument("--tools", default=str(ROOT / "tools" / "f9_tools.json"))
    ap.add_argument("--output_dir", default=str(ROOT / "outputs" / "run1"))
    ap.add_argument("--load_4bit", type=int, default=1, help="1=QLoRA(nf4) 0=LoRA")
    ap.add_argument("--compute_dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max_seq", type=int, default=2560, help="system+도구 정의 ≈1,950 토큰 → 2304 이상")
    ap.add_argument("--loss", default="completion", choices=["completion", "full"],
                    help="completion=assistant 정답 토큰에만 손실(기본) · full=전체 텍스트")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=-1, help="양수면 epochs 대신 step 수로 (스모크)")
    ap.add_argument("--max_train_rows", type=int, default=0, help="0=전부")
    ap.add_argument("--resume", action="store_true", help="output_dir 의 마지막 체크포인트에서 이어서")
    ap.add_argument("--push_to_hub", default="", help="비우지 않으면 어댑터를 이 HF repo(private) 에 올린다")
    return ap.parse_args()


def pick_dtype(torch, name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def from_pretrained_compat(cls, model_id: str, dtype, **kw):
    """transformers 5.x 는 dtype=, 4.x 는 torch_dtype=."""
    try:
        return cls.from_pretrained(model_id, dtype=dtype, **kw)
    except TypeError:
        return cls.from_pretrained(model_id, torch_dtype=dtype, **kw)


def strip_turn_newline(text: str, eos: str | None) -> str:
    """턴 구분용 개행(eos 뒤 '\\n')은 정답이 아니다 — 떼어 두어야 trl 이 eos 를 다시 붙여 eos 가 두 번 나오지 않는다."""
    if eos and text.endswith(eos + "\n"):
        return text[:-1]
    return text


def build_pairs(tok, messages: list[dict], tools: list[dict], eos: str | None) -> list[tuple[str, str]] | None:
    """assistant 턴마다 (prompt, completion). 템플릿이 prefix 성질을 안 지키면 None(호출부가 full 로 폴백)."""
    pairs = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        prompt = render_chat(tok, messages[:i], tools, add_generation_prompt=True)
        full = render_chat(tok, messages[:i + 1], tools)
        if not full.startswith(prompt):
            return None
        pairs.append((prompt, strip_turn_newline(full[len(prompt):], eos)))
    return pairs or None


def build_sft_config(SFTConfig, **wanted):
    """SFTConfig 가 실제로 가진 필드만 넘긴다. max_length ↔ max_seq_length · warmup 별칭 처리."""
    try:
        fields = {f.name: f for f in dataclasses.fields(SFTConfig)}
    except TypeError:
        import inspect
        fields = {k: None for k in inspect.signature(SFTConfig.__init__).parameters}
    names = set(fields)
    if "max_length" in wanted and "max_length" not in names and "max_seq_length" in names:
        wanted["max_seq_length"] = wanted.pop("max_length")
    if "warmup" in wanted:                       # transformers 5.x: warmup_steps(float<1=비율) · 4.x: warmup_ratio
        ratio = wanted.pop("warmup")
        ws = fields.get("warmup_steps")
        ws_float = ws is not None and "float" in str(getattr(ws, "type", ""))
        if "warmup_steps" in names and ws_float:
            wanted["warmup_steps"] = ratio
        elif "warmup_ratio" in names:
            wanted["warmup_ratio"] = ratio
    dropped = sorted(k for k in wanted if k not in names)
    if dropped:
        print(f"  WARNING: SFTConfig 에 없는 인자라 건너뜀 — {', '.join(dropped)}  (설정이 반영되지 않았다)")
    return SFTConfig(**{k: v for k, v in wanted.items() if k in names})


def main() -> None:
    args = parse_args()
    t0 = time.time()
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    cuda = torch.cuda.is_available()
    if not cuda:
        print("CUDA 없음 → fp32 · 4-bit 끔 · gradient checkpointing 끔 (스모크/CPU 경로)")
        args.load_4bit, args.compute_dtype = 0, "fp32"
    elif args.compute_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        print("이 GPU 는 bf16 을 지원하지 않음 → fp16 으로 전환 (캐글 T4·P100)")
        args.compute_dtype = "fp16"
    dtype = pick_dtype(torch, args.compute_dtype)
    device_name = torch.cuda.get_device_name(0) if cuda else "cpu"
    print(f"모델 {args.model_id} · 장치 {device_name} · 4bit={bool(args.load_4bit)} · dtype={args.compute_dtype}")

    # ── 데이터 → 텍스트 ───────────────────────────────────────
    tools, _ = load_tools(args.tools)
    rows = load_jsonl(Path(args.data_dir) / "train.jsonl", limit=args.max_train_rows or None)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    mode = detect_render_mode(tok, tools)
    texts = [render_chat(tok, r["messages"], r["tools"] or tools) for r in rows]
    lens = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
    srt = sorted(lens)
    p50, p95, mx = srt[len(srt) // 2], srt[int(len(srt) * 0.95)], srt[-1]
    keep = [i for i, n in enumerate(lens) if n <= args.max_seq]
    dropped = len(rows) - len(keep)
    print(f"학습 행 {len(rows)}개 · 템플릿 모드 {mode} · 토큰 길이 p50 {p50} / p95 {p95} / max {mx}")
    if dropped:
        print(f"  !!! --max_seq {args.max_seq} 를 넘는 행 {dropped}개는 버린다 (잘린 정답으로 학습하지 않는다). "
              f"전부 쓰려면 --max_seq {mx} 이상")
    if not keep:
        sys.exit(f"[중단] 모든 행이 --max_seq {args.max_seq} 를 넘는다 (최소 {srt[0]} 토큰). "
                 f"system+도구 정의만 약 1,950 토큰이라 2304 이상이 필요하다 — config.env 의 MAX_SEQ 를 올릴 것")
    rows = [rows[i] for i in keep]
    texts = [texts[i] for i in keep]
    print("── 샘플 ──\n" + texts[0][:800] + ("…" if len(texts[0]) > 800 else "") + "\n──────────")

    n_pairs = n_fallback = 0
    texts = [strip_turn_newline(t, tok.eos_token) for t in texts]      # 전체 텍스트 행도 eos 뒤 개행은 뗀다
    if args.loss == "completion":
        records = []
        for r, t in zip(rows, texts):
            pairs = build_pairs(tok, r["messages"], r["tools"] or tools, tok.eos_token)
            if pairs is None:                       # 템플릿이 prefix 성질을 안 지킴 → 이 행만 전체 텍스트
                records.append({"prompt": "", "completion": t}); n_fallback += 1
            else:
                records.extend({"prompt": p, "completion": c} for p, c in pairs); n_pairs += len(pairs)
        ds = Dataset.from_list(records)
        print(f"손실 completion-only · (prompt, completion) 쌍 {n_pairs}개 (행 {len(rows)}개) · 전체텍스트 폴백 {n_fallback}행")
        print("── completion 샘플 ──\n" + records[0]["completion"][:300] + "\n──────────")
    else:
        ds = Dataset.from_list([{"text": t} for t in texts])
        print(f"손실 full · 행 {len(rows)}개")

    # ── 모델 ───────────────────────────────────────────────────
    quant = None
    if args.load_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = from_pretrained_compat(AutoModelForCausalLM, args.model_id, dtype,
                                   quantization_config=quant, trust_remote_code=True,
                                   device_map="auto" if cuda else None)
    model.config.use_cache = False
    use_gc = bool(cuda)
    if args.load_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gc)
    elif use_gc:
        model.enable_input_require_grads()
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=LORA_TARGETS)
    try:
        model = get_peft_model(model, lora)
    except ValueError as e:  # 모듈 이름이 다른 아키텍처
        print(f"  LoRA 대상 모듈을 못 찾음({str(e)[:80]}) → all-linear 로 재시도")
        lora = dataclasses.replace(lora, target_modules="all-linear")
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ── 학습 설정 ──────────────────────────────────────────────
    sft_kwargs = dict(
        output_dir=args.output_dir, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup=0.03,
        logging_steps=args.logging_steps, save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        eval_strategy="no", bf16=(args.compute_dtype == "bf16"), fp16=(args.compute_dtype == "fp16"),
        gradient_checkpointing=use_gc, gradient_checkpointing_kwargs={"use_reentrant": False} if use_gc else None,
        optim="paged_adamw_8bit" if (args.load_4bit and cuda) else "adamw_torch",
        report_to="none", max_length=args.max_seq, packing=False,
        seed=args.seed, dataloader_num_workers=0, remove_unused_columns=True,
    )
    if args.loss == "completion":
        sft_kwargs["completion_only_loss"] = True        # prompt/completion 열 → trl 이 completion 토큰만 손실에 넣는다
    else:
        sft_kwargs["dataset_text_field"] = "text"
    if args.push_to_hub:
        # 학습 **중**에도 체크포인트가 Hub 에 올라가게 (캐글: 세션이 끊겨도 last-checkpoint 가 남는다). 마지막 어댑터는 아래서 따로 올린다
        sft_kwargs.update(push_to_hub=True, hub_model_id=args.push_to_hub, hub_strategy="checkpoint", hub_private_repo=True)
    cfg = build_sft_config(SFTConfig, **sft_kwargs)
    try:
        trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    except TypeError:
        trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, tokenizer=tok)

    resume = bool(args.resume and glob.glob(os.path.join(args.output_dir, "checkpoint-*")))
    print(f"학습 시작 {'(체크포인트에서 이어서)' if resume else ''}")
    result = trainer.train(resume_from_checkpoint=True if resume else None)

    # ── 저장 ───────────────────────────────────────────────────
    adapter_dir = ensure_dir(Path(args.output_dir) / "adapter")
    trainer.model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    run = {"model_id": args.model_id, "args": vars(args), "render_mode": mode, "device": device_name,
           "dtype": args.compute_dtype, "n_rows": len(rows), "n_rows_dropped_over_max_seq": dropped,
           "loss": args.loss, "n_pairs": n_pairs, "n_fulltext_fallback_rows": n_fallback,
           "token_len": {"p50": p50, "p95": p95, "max": mx},
           "train_loss": getattr(result, "training_loss", None), "global_step": getattr(result, "global_step", None),
           "elapsed_s": round(time.time() - t0, 1), "versions": lib_versions(),
           "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (Path(args.output_dir) / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.push_to_hub:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(args.push_to_hub, private=True, exist_ok=True, repo_type="model")
            api.upload_folder(folder_path=str(adapter_dir), repo_id=args.push_to_hub, repo_type="model")
            api.upload_file(path_or_fileobj=str(Path(args.output_dir) / "run.json"), path_in_repo="run.json",
                            repo_id=args.push_to_hub, repo_type="model")
            run["pushed_to"] = args.push_to_hub
            print(f"HF Hub 업로드 완료 → {args.push_to_hub}")
        except Exception as e:  # noqa: BLE001 — 업로드 실패가 학습 결과를 지우면 안 된다
            print(f"!!! HF Hub 업로드 실패: {e}  (어댑터는 {adapter_dir} 에 있다. 토큰·repo 이름 확인 후 다시 올릴 것)")

    print("\n══ 학습 완료 ══")
    print(f"  어댑터   {adapter_dir}")
    print(f"  run.json {Path(args.output_dir) / 'run.json'}")
    print(f"  step {run['global_step']} · loss {run['train_loss']} · {run['elapsed_s']}s · 템플릿 {mode}")
    print(f"  손실 {args.loss} · 행 {len(rows)}개(버림 {dropped}) · 쌍 {n_pairs}개 · 전체텍스트 폴백 {n_fallback}행")


if __name__ == "__main__":
    main()
