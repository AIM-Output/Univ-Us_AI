# -*- coding: utf-8 -*-
"""Donut OCR 코드 CPU 스모크 — 모델·데이터 다운로드 없이.

가짜 parquet 2행(서버와 같은 스키마: image={bytes,path} · ground_truth=JSON 문자열) +
랜덤 초기화한 아주 작은 VisionEncoderDecoderModel(DonutSwin depths [1,1] + mBART 1층) + 손으로 만든 XLM-R 토크나이저를
경로로 저장해 MODEL_ID 자리에 넣고, 함수 단위와 main() 끝까지(train → eval base/finetuned → compare) 돌린다.

  PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest -q tests/test_ocr_smoke.py
"""
from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import ocr_common as oc  # noqa: E402

H, W, MAX_LEN = 64, 48, 24
TEXTS = ["가나 다라 마바", "사 아 자 차카 타파하"]
ENV_KEYS = ("MODEL_ID", "OCR_DATA_DIR", "OCR_OUT_DIR", "OCR_CKPT_DIR", "IMAGE_H", "IMAGE_W", "MAX_LEN", "BATCH",
            "GRAD_ACCUM", "LR", "WARMUP_STEPS", "MAX_TRAIN_ROWS", "MAX_STEPS", "TIME_BUDGET_MIN", "SAVE_STEPS", "EVAL_ROWS")


def _png(color) -> bytes:
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (200, 100), color).save(b, format="PNG")
    return b.getvalue()


def _write_parquet(path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    img_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    images = pa.array([{"bytes": _png((255, 255, 255)), "path": "a.png"}, {"bytes": _png((200, 200, 200)), "path": "b.png"}],
                      type=img_type)
    gts = pa.array([json.dumps({"gt_parse": {"text_sequence": t}}, ensure_ascii=False) for t in TEXTS])
    pq.write_table(pa.table({"image": images, "ground_truth": gts}), path)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("synthdog")
    _write_parquet(d / "train-00000-of-00001-fake.parquet")
    _write_parquet(d / "validation-00000-of-00001-fake.parquet")
    return d


def _tiny_tokenizer():
    from transformers import XLMRobertaTokenizer
    chars = sorted(set("".join(TEXTS).replace(" ", "")))
    vocab = [("<s>", 0.0), ("<pad>", 0.0), ("</s>", 0.0), ("<unk>", 0.0), ("▁", -1.0)] + [(c, -2.0) for c in chars] \
        + [("▁" + c, -1.5) for c in chars]
    tok = XLMRobertaTokenizer(vocab=vocab)
    tok.add_tokens([oc.TASK_TOKEN], special_tokens=True)     # donut-base 처럼 task 토큰을 특수 토큰으로
    return tok


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> Path:
    """랜덤 초기화 donut 꼴 모델 + 프로세서를 디렉터리로 저장 (MODEL_ID 로 쓴다)."""
    from transformers import (DonutImageProcessor, DonutProcessor, DonutSwinConfig, MBartConfig,
                              VisionEncoderDecoderConfig, VisionEncoderDecoderModel)
    tok = _tiny_tokenizer()
    enc = DonutSwinConfig(image_size=[H, W], patch_size=4, num_channels=3, embed_dim=16, depths=[1, 1],
                          num_heads=[1, 2], window_size=2)
    dec = MBartConfig(vocab_size=len(tok), d_model=32, decoder_layers=1, encoder_layers=1, decoder_attention_heads=2,
                      encoder_attention_heads=2, decoder_ffn_dim=64, encoder_ffn_dim=64, max_position_embeddings=64,
                      is_decoder=True, add_cross_attention=True, pad_token_id=tok.pad_token_id,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, scale_embedding=True)
    cfg = VisionEncoderDecoderConfig.from_encoder_decoder_configs(enc, dec)
    model = VisionEncoderDecoderModel(config=cfg)
    ip = DonutImageProcessor(size={"height": 2 * H, "width": 2 * W}, do_align_long_axis=True,
                             do_thumbnail=True, do_pad=True, do_normalize=True)
    proc = DonutProcessor(image_processor=ip, tokenizer=tok)
    d = tmp_path_factory.mktemp("tiny-donut")
    model.save_pretrained(str(d))
    proc.save_pretrained(str(d))
    return d


@pytest.fixture()
def env(monkeypatch, tmp_path, data_dir, tiny_model_dir):
    """config.env 대신 셸 env 로 전부 덮어쓴다 (load_cfg 의 마지막 순위)."""
    vals = {"MODEL_ID": str(tiny_model_dir), "OCR_DATA_DIR": str(data_dir), "OCR_OUT_DIR": str(tmp_path / "out"),
            "OCR_CKPT_DIR": str(tmp_path / "ckpt"), "IMAGE_H": str(H), "IMAGE_W": str(W), "MAX_LEN": str(MAX_LEN),
            "BATCH": "1", "GRAD_ACCUM": "1", "LR": "1e-3", "WARMUP_STEPS": "0", "MAX_TRAIN_ROWS": "0",
            "MAX_STEPS": "-1", "TIME_BUDGET_MIN": "0", "SAVE_STEPS": "1", "EVAL_ROWS": "2"}
    for k in ENV_KEYS:
        monkeypatch.setenv(k, vals[k])
    return vals


# ── 데이터·채점 ──────────────────────────────────────────────

def test_load_split_and_row_helpers(data_dir):
    ds = oc.load_split(data_dir, "train", 0)
    assert len(ds) == 2 and set(ds.column_names) == {"image", "ground_truth"}
    assert len(oc.load_split(data_dir, "validation", 1)) == 1
    row = ds[0]
    assert isinstance(row["image"], dict) and "bytes" in row["image"]          # 서버와 같은 비디코드 스키마
    assert oc.gt_text(row) == TEXTS[0]
    assert oc.gt_text({"ground_truth": {"gt_parse": {"text_sequence": "x"}}}) == "x"
    assert oc.gt_text({"ground_truth": "그냥 문자열"}) == "그냥 문자열"
    img = oc.decode_image(row)
    assert img.mode == "RGB" and img.size == (200, 100)


def test_metrics():
    assert oc.normalize("  미 늄   매\n트를  ") == "미 늄 매 트를"
    assert oc.cer("가나다", "가나다") == 0.0
    assert oc.cer("가나다라", "가다라") == pytest.approx(0.25)
    assert oc.cer("", "abc") == 3.0                                           # max(1, len(ref))
    assert oc.ned_sim("가나다라", "가다라") == pytest.approx(0.75)
    assert oc.ned_sim("", "") == 1.0
    a = oc.score_pair(" 가나  다 ", "가나 다")
    b = oc.score_pair("가나다라마", "")
    assert a["exact"] and a["cer"] == 0.0 and not b["exact"] and b["cer"] == 1.0
    a["sec"], b["sec"] = 0.2, 0.4
    s = oc.summarize([a, b])
    assert s["n"] == 2 and s["exact_rate"] == 0.5 and s["cer_mean"] == pytest.approx(0.5)
    assert s["cer_weighted"] == pytest.approx(5 / (4 + 5)) and s["sec_mean"] == pytest.approx(0.3)
    assert oc.clean_text("<s_synthdog> 가나 </s><pad><pad>") == "가나"


def test_load_cfg_precedence(tmp_path, monkeypatch):
    (tmp_path / "config.env").write_text("BATCH=2\nLR=3e-5   # 주석\nOCR_OUT_DIR=/a\n", encoding="utf-8")
    (tmp_path / "config.local.env").write_text("BATCH=4\n", encoding="utf-8")
    monkeypatch.delenv("BATCH", raising=False)
    monkeypatch.delenv("LR", raising=False)
    monkeypatch.setenv("OCR_OUT_DIR", "/b")
    c = oc.load_cfg(tmp_path)
    assert c["BATCH"] == "4" and c["LR"] == "3e-5" and c["OCR_OUT_DIR"] == "/b"


# ── 학습: 변환·labels·loss 한 step ───────────────────────────

def test_collator_labels_and_one_step(data_dir, tiny_model_dir):
    import torch
    import train_ocr as T
    model, proc, task_id = oc.load_model(str(tiny_model_dir), H, W, MAX_LEN, torch.float32)
    tok = proc.tokenizer
    assert task_id == tok.convert_tokens_to_ids(oc.TASK_TOKEN) != tok.unk_token_id
    assert model.config.decoder_start_token_id == task_id and model.config.pad_token_id == tok.pad_token_id
    assert list(model.config.encoder.image_size) == [H, W]

    ds = oc.load_split(data_dir, "train", 0)
    batch = T.Collator(proc, H, W, MAX_LEN)([ds[0], ds[1]])
    pv, labels = batch["pixel_values"], batch["labels"]
    assert tuple(pv.shape) == (2, 3, H, W) and pv.dtype == torch.float32
    assert (labels == task_id).sum() == 0                                      # task 토큰은 labels 에 없다
    assert (labels == tok.pad_token_id).sum() == 0                             # pad 는 전부 -100
    for i, t in enumerate(TEXTS):
        valid = labels[i][labels[i] != -100].tolist()
        assert valid[-1] == tok.eos_token_id and tok.decode(valid, skip_special_tokens=True).strip() == t
    assert (labels[0] == -100).any() or (labels[1] == -100).any()             # 길이가 달라 짧은 쪽이 pad 됨

    # shift_tokens_right 가 decoder 입력 첫 자리에 task 토큰을 넣는지
    dec_in = model.prepare_decoder_input_ids_from_labels(labels)
    assert (dec_in[:, 0] == task_id).all() and (dec_in != -100).all()

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.decoder.lm_head.weight.detach().clone()
    loss = T.train_step(model, batch, torch.device("cpu"), amp=False, grad_accum=1)
    assert math.isfinite(loss) and loss > 0
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    assert not torch.equal(before, model.decoder.lm_head.weight)


def test_checkpoint_keeps_latest_only(tmp_path, tiny_model_dir):
    import torch
    import train_ocr as T
    from transformers import get_linear_schedule_with_warmup
    model, proc, _ = oc.load_model(str(tiny_model_dir), H, W, MAX_LEN, torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sch = get_linear_schedule_with_warmup(opt, 0, 10)
    for s in (1, 2):
        T.save_checkpoint(model, proc, opt, sch, tmp_path, s, {"step": s})
    assert sorted(p.name for p in tmp_path.glob("step-*")) == ["step-2"]
    m2, p2, tid = oc.load_model(str(tmp_path / "step-2"), H, W, MAX_LEN, torch.float32)
    assert tid == m2.config.decoder_start_token_id


# ── main() 끝까지: train → eval base → eval finetuned → compare ──

def test_train_main_end_to_end(env):
    import train_ocr as T
    T.main(["--workers", "0"])
    out = Path(env["OCR_OUT_DIR"])
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["steps"] == 2 and run["stop_reason"] == "epoch" and run["n_train_rows"] == 2
    assert run["final_loss"] is not None and math.isfinite(run["final_loss"])
    assert (out / "train.log").read_text(encoding="utf-8").count("step ") >= 2
    st = json.loads((out / "model" / "config.json").read_text(encoding="utf-8"))
    assert st["encoder"]["image_size"] == [H, W]
    assert list((out / "model").glob("*.safetensors"))
    assert (out / "model" / "tokenizer.json").exists() or (out / "model" / "sentencepiece.bpe.model").exists()
    assert (out / "model" / "preprocessor_config.json").exists() or (out / "model" / "processor_config.json").exists()
    import torch
    m, _, _ = oc.load_model(str(out / "model"), H, W, MAX_LEN, "auto")
    assert next(m.parameters()).dtype == torch.bfloat16                         # bf16 로 저장됨
    ck = list(Path(env["OCR_CKPT_DIR"]).glob("step-*"))
    assert len(ck) == 1                                                         # 최신 1개만 (마지막 step 은 저장 안 함)


def test_train_max_steps_and_resume(env):
    import train_ocr as T
    T.main(["--workers", "0", "--max_steps", "1", "--save_steps", "0"])
    run = json.loads((Path(env["OCR_OUT_DIR"]) / "run.json").read_text(encoding="utf-8"))
    assert run["steps"] == 1 and run["stop_reason"] == "max_steps"
    # 1 step 마다 저장하는 2 step 학습의 step-1 에서 이어서 → 남은 1 step
    T.main(["--workers", "0", "--max_steps", "-1", "--save_steps", "1", "--out_dir", env["OCR_OUT_DIR"] + "_r"])
    T.main(["--workers", "0", "--resume", str(Path(env["OCR_CKPT_DIR"]) / "step-1"), "--out_dir", env["OCR_OUT_DIR"] + "_r2"])
    r2 = json.loads((Path(env["OCR_OUT_DIR"] + "_r2") / "run.json").read_text(encoding="utf-8"))
    assert r2["steps"] == 2 and r2["planned_steps"] == 2


def test_smoke_flag_redirects_outputs(env):
    import train_ocr as T
    a = T.parse_args(["--smoke"])
    assert a.max_train_rows == 16 and a.max_steps == 4 and a.grad_accum <= 2
    assert Path(a.out_dir).parent == Path(env["OCR_CKPT_DIR"]) and a.out_dir != env["OCR_OUT_DIR"]


def test_eval_main_and_compare(env):
    import eval_ocr as E
    import train_ocr as T
    T.main(["--workers", "0"])
    out = Path(env["OCR_OUT_DIR"])
    E.main(["--tag", "base", "--batch", "4"])
    E.main(["--model", str(out / "model"), "--tag", "finetuned", "--batch", "1"])
    for tag in ("base", "finetuned"):
        res = json.loads((out / f"eval_{tag}.json").read_text(encoding="utf-8"))
        assert res["summary"]["n"] == 2 and len(res["rows"]) == 2
        for r in res["rows"]:
            assert {"ref", "hyp", "cer", "sec", "ned_sim", "exact"} <= set(r)
            assert oc.TASK_TOKEN not in r["hyp"] and "</s>" not in r["hyp"] and "<pad>" not in r["hyp"]
            assert r["ref"] in TEXTS
        md = (out / f"eval_{tag}.md").read_text(encoding="utf-8")
        assert "CER 평균" in md and "가장 나쁜 10건" in md and "무작위 5건" in md
    E.main(["--compare"])
    cmp_md = (out / "compare.md").read_text(encoding="utf-8")
    assert "학습 전" in cmp_md and "학습 후" in cmp_md and "%p" in cmp_md


def test_generate_batch_directly(data_dir, tiny_model_dir):
    import torch
    import eval_ocr as E
    model, proc, task_id = oc.load_model(str(tiny_model_dir), H, W, MAX_LEN, torch.float32)
    model.eval()
    ds = oc.load_split(data_dir, "validation", 0)
    hyps = E.generate_batch(model, proc, [oc.decode_image(ds[i]) for i in range(2)], task_id, MAX_LEN, H, W)
    assert len(hyps) == 2 and all(isinstance(h, str) for h in hyps)
    assert all("<" not in h for h in hyps)
