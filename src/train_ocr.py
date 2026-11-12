#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Donut 한국어 OCR 전체 미세조정 (SynthDoG-ko) — 모델 이름은 여기 없다. config.env 의 MODEL_ID → train_ocr.sh → env.

  bash train_ocr.sh --smoke      # 16행 · 4 step — 서버에서 가장 먼저: s/step · GPU 메모리 측정
  bash train_ocr.sh              # 본 학습 → $OCR_OUT_DIR/model (bf16 safetensors + processor + tokenizer) · run.json
  python src/train_ocr.py --max_steps 300 --batch 1 --grad_accum 16 ...   # 인자 기본값은 config.env

학습:  fp32 가중치(master) + bf16 autocast · gradient checkpointing · AdamW(wd 0.01) · linear warmup→decay · clip 1.0
labels: text_sequence + </s> (task 토큰 <s_synthdog> 은 넣지 않는다 — decoder_start_token_id 가 그 역할) · pad → -100
종료:  1 epoch 끝 · MAX_STEPS 도달 · TIME_BUDGET_MIN 경과 중 먼저 오는 것 → 마지막 step 뒤 저장
체크포인트: OCR_CKPT_DIR/step-<n>/ (최신 1개만 남긴다) · --resume <그 경로> 로 이어서
OOM 이면 자동 재시도 없이 멈춘다 → BATCH=1 GRAD_ACCUM=16 bash train_ocr.sh
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocr_common import (TASK_TOKEN, decode_image, gt_text, lib_versions, load_cfg, load_model,  # noqa: E402
                        load_split, pixel_values)

LOG_FILE: Path | None = None


def log(msg: str) -> None:
    """stdout + $OCR_OUT_DIR/train.log (셸의 tee 는 train_console.log 로 따로 받는다)."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_FILE is not None:
        with open(LOG_FILE, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")


def parse_args(argv=None) -> argparse.Namespace:
    c = load_cfg()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", default=c.get("MODEL_ID", ""), help="HF id 또는 경로 — config.env 의 MODEL_ID")
    ap.add_argument("--data_dir", default=c.get("OCR_DATA_DIR", ""), help="train-*.parquet 가 있는 곳 (OCR_DATA_DIR)")
    ap.add_argument("--out_dir", default=c.get("OCR_OUT_DIR", "outputs/donut-ocr"), help="model/ · run.json · train.log")
    ap.add_argument("--ckpt_dir", default=c.get("OCR_CKPT_DIR", "outputs/donut-ocr-ckpt"), help="step-<n>/ 체크포인트")
    ap.add_argument("--image_h", type=int, default=int(c.get("IMAGE_H", 1280)))
    ap.add_argument("--image_w", type=int, default=int(c.get("IMAGE_W", 960)))
    ap.add_argument("--max_len", type=int, default=int(c.get("MAX_LEN", 768)), help="labels 최대 토큰 (eos 포함)")
    ap.add_argument("--max_train_rows", type=int, default=int(c.get("MAX_TRAIN_ROWS", 0)), help="0=전부")
    ap.add_argument("--max_steps", type=int, default=int(c.get("MAX_STEPS", -1)), help="-1=1 epoch")
    ap.add_argument("--time_budget_min", type=float, default=float(c.get("TIME_BUDGET_MIN", 0)), help="0=제한 없음")
    ap.add_argument("--batch", type=int, default=int(c.get("BATCH", 2)))
    ap.add_argument("--grad_accum", type=int, default=int(c.get("GRAD_ACCUM", 8)))
    ap.add_argument("--lr", type=float, default=float(c.get("LR", 3e-5)))
    ap.add_argument("--warmup_steps", type=int, default=int(c.get("WARMUP_STEPS", 50)))
    ap.add_argument("--save_steps", type=int, default=int(c.get("SAVE_STEPS", 100)), help="0=중간 저장 안 함")
    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2, help="DataLoader num_workers")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default="", help="OCR_CKPT_DIR/step-<n> 경로 — 가중치·옵티마이저·step 을 이어서")
    ap.add_argument("--smoke", action="store_true",
                    help="16행 · max_steps 4 · grad_accum≤2 · 매 step 로그 · 2 step 마다 저장. 산출물은 OCR_CKPT_DIR/smoke-* (홈 아님)")
    a = ap.parse_args(argv)
    if a.smoke:
        a.max_train_rows, a.max_steps, a.grad_accum = 16, 4, min(a.grad_accum, 2)
        a.save_steps, a.logging_steps, a.time_budget_min = 2, 1, 0
        a.out_dir = str(Path(a.ckpt_dir) / "smoke-out")
        a.ckpt_dir = str(Path(a.ckpt_dir) / "smoke-ckpt")
    if not a.model_id:
        ap.error("MODEL_ID 가 비어 있습니다 (config.env)")
    return a


class Collator:
    """원시 행 묶음 → {pixel_values, labels}. 배치마다 그 자리에서 이미지 디코드·전처리 (DataLoader worker 에서 돈다).
    labels 는 배치 안 최장 길이로 pad(최대 max_len) → pad 자리 -100. 앞에 task 토큰 없음."""

    def __init__(self, processor, image_h: int, image_w: int, max_len: int):
        self.processor, self.h, self.w, self.max_len = processor, image_h, image_w, max_len

    def __call__(self, rows: list[dict]) -> dict:
        tok = self.processor.tokenizer
        images = [decode_image(r) for r in rows]
        texts = [gt_text(r) + tok.eos_token for r in rows]
        enc = tok(texts, add_special_tokens=False, max_length=self.max_len, truncation=True,
                  padding="longest", return_tensors="pt")
        labels = enc.input_ids.clone()
        labels[enc.attention_mask == 0] = -100
        return {"pixel_values": pixel_values(self.processor, images, self.h, self.w), "labels": labels}


def train_step(model, batch: dict, device, amp: bool, grad_accum: int) -> float:
    """forward(+bf16 autocast) → loss/grad_accum backward. 원래 loss 값을 돌려준다."""
    import torch
    pv = batch["pixel_values"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        out = model(pixel_values=pv, labels=labels, use_cache=False)
    loss = out.loss
    (loss / grad_accum).backward()
    return float(loss.detach())


def save_checkpoint(model, processor, optimizer, scheduler, ckpt_dir: Path, step: int, state: dict) -> Path:
    """ckpt_dir/step-<step>/ 에 저장한 뒤 다른 step-* 는 지운다 (최신 1개만)."""
    import torch
    d = ckpt_dir / f"step-{step}"
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(d))
    processor.save_pretrained(str(d))
    torch.save(optimizer.state_dict(), d / "optimizer.pt")
    torch.save(scheduler.state_dict(), d / "scheduler.pt")
    (d / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    for old in ckpt_dir.glob("step-*"):
        if old != d and old.is_dir() and re.fullmatch(r"step-\d+", old.name):
            shutil.rmtree(old, ignore_errors=True)
    return d


def save_final(model, processor, out: Path) -> Path:
    """bf16 safetensors + processor(image processor·tokenizer). 저장 뒤 모델은 bf16 이 된다(학습 끝난 뒤에만 부를 것)."""
    import torch
    out.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(str(out))
    processor.save_pretrained(str(out))
    processor.tokenizer.save_pretrained(str(out))
    return out


def main(argv=None) -> None:
    global LOG_FILE
    args = parse_args(argv)
    t0 = time.time()
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    out_dir, ckpt_dir = Path(args.out_dir), Path(args.ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    LOG_FILE = out_dir / "train.log"
    torch.manual_seed(args.seed)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    amp = cuda and torch.cuda.is_bf16_supported()
    dev_name = torch.cuda.get_device_name(0) if cuda else "cpu"
    if cuda:
        free, total = torch.cuda.mem_get_info()
        log(f"장치 {dev_name} · 여유 {free / 2**30:.1f} / {total / 2**30:.1f} GiB · bf16 autocast={amp}")
    else:
        log("CUDA 없음 → CPU fp32 (스모크 경로)")
    log(f"설정 {'[SMOKE] ' if args.smoke else ''}model={args.model_id} image={args.image_h}x{args.image_w} "
        f"max_len={args.max_len} batch={args.batch}x{args.grad_accum} lr={args.lr} rows={args.max_train_rows} "
        f"max_steps={args.max_steps} budget={args.time_budget_min}min out={out_dir} ckpt={ckpt_dir}")

    # ── 데이터 ─────────────────────────────────────────────────
    ds = load_split(args.data_dir, "train", args.max_train_rows)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(ds), generator=g).tolist()
    ds = ds.select(perm)                                    # 셔플 순서를 고정해 --resume 때 같은 순서로 건너뛴다
    log(f"학습 행 {len(ds)}개 · 열 {ds.column_names}")

    # ── 모델 ───────────────────────────────────────────────────
    src = args.resume or args.model_id
    model, processor, task_id = load_model(src, args.image_h, args.image_w, args.max_len, torch.float32)
    log(f"task 토큰 {TASK_TOKEN}={task_id} · pad={processor.tokenizer.pad_token_id} · eos={processor.tokenizer.eos_token_id} "
        f"· vocab {len(processor.tokenizer)} · 파라미터 {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.to(device)
    model.train()

    micro_per_epoch = math.ceil(len(ds) / args.batch)
    steps_per_epoch = math.ceil(micro_per_epoch / args.grad_accum)
    total = steps_per_epoch if args.max_steps <= 0 else min(args.max_steps, steps_per_epoch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, min(args.warmup_steps, max(0, total - 1)), total)

    step = 0
    if args.resume:
        st = json.loads((Path(args.resume) / "state.json").read_text(encoding="utf-8"))
        opt.load_state_dict(torch.load(Path(args.resume) / "optimizer.pt", map_location=device))
        sched.load_state_dict(torch.load(Path(args.resume) / "scheduler.pt", map_location="cpu"))
        step = int(st["step"])
        log(f"이어서: {args.resume} · step {step} 부터")
    skip_rows = step * args.grad_accum * args.batch
    loader = DataLoader(ds.select(range(min(skip_rows, len(ds)), len(ds))), batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=Collator(processor, args.image_h, args.image_w, args.max_len),
                        pin_memory=cuda, persistent_workers=False)
    log(f"step 계획 {total} (epoch 당 {steps_per_epoch} · micro-batch {micro_per_epoch}) · warmup {args.warmup_steps}")

    # ── 학습 루프 ──────────────────────────────────────────────
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    budget_s = args.time_budget_min * 60 if args.time_budget_min > 0 else float("inf")
    t_train = time.time()
    t_win, win_losses, last_losses = time.time(), [], []
    step_start = step
    stop_reason = "epoch"
    micro = 0
    opt.zero_grad(set_to_none=True)

    def optimizer_step() -> None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

    try:
        for batch in loader:
            win_losses.append(train_step(model, batch, device, amp, args.grad_accum))
            micro += 1
            if micro % args.grad_accum:
                continue
            optimizer_step()
            step += 1
            if step % args.logging_steps == 0 or step == total:
                s_step = (time.time() - t_win) / max(1, len(win_losses) / args.grad_accum)
                done = step - step_start
                avg = (time.time() - t_train) / max(1, done)
                eta = min((total - step) * avg, max(0.0, budget_s - (time.time() - t0)))
                gib = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
                loss = sum(win_losses) / len(win_losses)
                last_losses.append(loss)
                log(f"step {step}/{total} · loss {loss:.4f} · lr {sched.get_last_lr()[0]:.2e} · {s_step:.2f} s/step "
                    f"· ETA {eta / 60:.1f}min · GPU {gib:.2f} GiB")
                t_win, win_losses = time.time(), []
            if args.save_steps > 0 and step % args.save_steps == 0 and step < total:
                d = save_checkpoint(model, processor, opt, sched, ckpt_dir, step,
                                    {"step": step, "total": total, "seed": args.seed, "max_train_rows": args.max_train_rows,
                                     "batch": args.batch, "grad_accum": args.grad_accum})
                log(f"체크포인트 {d}")
            if step >= total:
                stop_reason = "max_steps" if args.max_steps > 0 and total == args.max_steps else "epoch"
                break
            if time.time() - t0 >= budget_s:
                stop_reason = "time_budget"
                log(f"TIME_BUDGET_MIN {args.time_budget_min} 경과 → step {step} 에서 멈추고 저장")
                break
        else:
            if micro % args.grad_accum:                     # epoch 끝의 남은 micro-batch 도 한 step 으로
                optimizer_step()
                step += 1
                if win_losses:
                    last_losses.append(sum(win_losses) / len(win_losses))
    except torch.cuda.OutOfMemoryError:
        gib = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
        msg = (f"\n!!! GPU 메모리 부족(OOM) — step {step} · batch {args.batch} · image {args.image_h}x{args.image_w} · "
               f"peak {gib:.2f} GiB\n    자동 재시도는 하지 않는다. 유효 배치를 유지하려면:\n"
               f"      BATCH=1 GRAD_ACCUM={args.batch * args.grad_accum} bash train_ocr.sh\n"
               f"    그래도 OOM 이면 IMAGE_H=960 IMAGE_W=720 (평가도 같은 값으로)")
        log(msg)
        sys.exit(3)

    train_s = time.time() - t_train
    done = step - step_start
    s_per_step = train_s / max(1, done)
    peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
    final_loss = last_losses[-1] if last_losses else (sum(win_losses) / len(win_losses) if win_losses else None)
    log(f"학습 끝 ({stop_reason}) · step {step}/{total} · {train_s / 60:.1f}min · {s_per_step:.2f} s/step · peak {peak:.2f} GiB")

    # ── 저장 ───────────────────────────────────────────────────
    model_dir = save_final(model, processor, out_dir / "model")
    run = {"model_id": args.model_id, "task_token": TASK_TOKEN, "task_token_id": task_id, "args": vars(args),
           "device": dev_name, "precision": "fp32 master + bf16 autocast" if amp else "fp32",
           "saved_dtype": "bf16", "n_train_rows": len(ds), "planned_steps": total, "steps": step,
           "stop_reason": stop_reason, "final_loss": final_loss, "s_per_step": round(s_per_step, 3),
           "train_s": round(train_s, 1), "elapsed_s": round(time.time() - t0, 1), "peak_gib": round(peak, 2),
           "versions": lib_versions(), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n══ 학습 완료 ══")
    print(f"  모델     {model_dir}")
    print(f"  run.json {out_dir / 'run.json'}")
    print(f"  step {step} · loss {final_loss} · {s_per_step:.2f} s/step · peak {peak:.2f} GiB · 종료 {stop_reason}")
    if args.smoke and done and int(load_cfg().get("MAX_TRAIN_ROWS", 0) or 0) > 0:
        full = math.ceil(int(load_cfg()["MAX_TRAIN_ROWS"]) / args.batch)
        per_micro = train_s / max(1, micro)
        print(f"  [SMOKE] micro-batch 당 {per_micro:.2f}s → 본 학습(행 {full * args.batch}) 1 epoch ≈ {full * per_micro / 60:.0f}min "
              f"(데이터로더 첫 배치 포함이라 약간 과대)")


if __name__ == "__main__":
    main()
