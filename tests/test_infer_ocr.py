# -*- coding: utf-8 -*-
"""src/infer_ocr.py — 다운로드 없이 test_ocr_smoke 의 작은 랜덤 Donut 모델로.

  PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest -q tests/test_infer_ocr.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent / "src"))
sys.path.insert(0, str(TESTS))

import infer_ocr as io_ocr  # noqa: E402
import test_ocr_smoke as smoke  # noqa: E402

tiny_model_dir = smoke.tiny_model_dir          # 같은 픽스처(랜덤 DonutSwin + mBART 1층 + 손으로 만든 토크나이저)


def _png(color=(255, 255, 255), size=(200, 100)) -> bytes:
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, format="PNG")
    return b.getvalue()


@pytest.fixture()
def clean_env(monkeypatch):
    monkeypatch.delenv("UNIVUS_OCR", raising=False)
    monkeypatch.delenv(io_ocr.ENV_DIR, raising=False)
    yield monkeypatch
    io_ocr._avail = None
    io_ocr._engine = None


def test_class_available_and_mixed_inputs(tiny_model_dir, tmp_path):
    from PIL import Image
    eng = io_ocr.DonutOCR(tiny_model_dir, device="cpu", batch=2)
    a = eng.available()
    assert a["ok"] is True and a["engine"] == "donut-ocr-ko" and a["lang"] == "ko" and a["korean"] is True
    assert a["device"] == "cpu" and Path(a["model_dir"]) == Path(tiny_model_dir).resolve() and a["reason"] == ""
    assert (eng.image_h, eng.image_w) == (smoke.H, smoke.W)                  # 저장된 encoder.image_size 를 쓴다
    assert eng.task_id == eng.processor.tokenizer.convert_tokens_to_ids(io_ocr.TASK_TOKEN)

    p = tmp_path / "page.png"
    p.write_bytes(_png((230, 230, 230)))
    out = eng.image_texts([_png(), b"not an image", str(p)])
    assert len(out) == 3 and out[1] is None and isinstance(out[0], str) and isinstance(out[2], str)
    assert all("<" not in t for t in (out[0], out[2]))                        # 특수 토큰 제거
    # PIL · Path · 없는 파일 · 잘린 PNG · 빈 bytes
    more = eng.image_texts([Image.new("RGB", (60, 80), "white"), p, tmp_path / "nope.png", _png()[:40], b""], batch=4)
    assert isinstance(more[0], str) and isinstance(more[1], str) and more[2:] == [None, None, None]
    assert eng.image_text(p) == out[2] and eng.image_text(b"x") is None
    assert eng.image_texts([]) == []


def test_matches_eval_generate_batch(tiny_model_dir):
    """eval_ocr.generate_batch(CER 5.19% 평가에 쓴 호출)와 같은 글자."""
    import torch
    import eval_ocr as E
    import ocr_common as oc
    from PIL import Image
    imgs = [Image.new("RGB", (200, 100), c) for c in ("white", (120, 120, 120), (10, 200, 30))]
    model, proc, task_id = oc.load_model(str(tiny_model_dir), smoke.H, smoke.W, smoke.MAX_LEN, torch.float32)
    model.eval()
    ref = E.generate_batch(model, proc, imgs, task_id, smoke.MAX_LEN, smoke.H, smoke.W)
    eng = io_ocr.DonutOCR(tiny_model_dir, device="cpu", max_len=smoke.MAX_LEN)
    assert eng.image_texts(imgs, batch=3) == ref
    assert io_ocr.normalize("  미 늄   매\n트를 ") == oc.normalize("  미 늄   매\n트를 ")
    assert io_ocr.clean_text("<s_synthdog> 가나 </s><pad>") == oc.clean_text("<s_synthdog> 가나 </s><pad>") == "가나"


def test_module_singleton(clean_env, tiny_model_dir):
    clean_env.setenv(io_ocr.ENV_DIR, str(tiny_model_dir))
    a = io_ocr.available(refresh=True)
    assert a["ok"] is True and a["korean"] is True and a["engine"] == "donut-ocr-ko"
    eng = io_ocr._engine
    assert io_ocr.available(refresh=True) is not None and io_ocr._engine is eng   # 같은 폴더면 다시 올리지 않는다
    out = io_ocr.image_texts([_png(), b"broken"])
    assert len(out) == 2 and isinstance(out[0], str) and out[1] is None
    assert isinstance(io_ocr.image_text(_png()), str)
    assert io_ocr.image_texts([]) == []


def test_module_off(clean_env, tiny_model_dir):
    clean_env.setenv(io_ocr.ENV_DIR, str(tiny_model_dir))
    clean_env.setenv("UNIVUS_OCR", "off")
    a = io_ocr.available(refresh=True)
    assert a["ok"] is False and a["korean"] is False and "UNIVUS_OCR" in a["reason"]
    assert io_ocr.image_texts([_png(), _png()]) == [None, None]
    assert io_ocr.image_text(_png()) is None and io_ocr.pdf_text(b"%PDF-1.4") is None


def test_module_missing_model_dir(clean_env, tmp_path):
    clean_env.setenv(io_ocr.ENV_DIR, str(tmp_path / "no-model-here"))
    a = io_ocr.available(refresh=True)
    assert a["ok"] is False and io_ocr.ENV_DIR in a["reason"]
    assert io_ocr.image_texts([_png()]) == [None]


def test_pdf_text(clean_env, tiny_model_dir):
    from PIL import Image
    clean_env.setenv(io_ocr.ENV_DIR, str(tiny_model_dir))
    io_ocr.available(refresh=True)
    b = io.BytesIO()
    Image.new("RGB", (300, 400), "white").save(b, format="PDF")
    ok, why = io_ocr.pdf_support()
    res = io_ocr.pdf_text(b.getvalue(), pages=3)
    if ok:
        assert res is None or isinstance(res, str)
    else:
        assert res is None and "pypdfium2" in why
    assert io_ocr.pdf_text(b"") is None


def test_cli_json(tiny_model_dir, tmp_path, capsys):
    p = tmp_path / "a.png"
    p.write_bytes(_png())
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")
    assert io_ocr.main(["--model", str(tiny_model_dir), "--device", "cpu", "--batch", "2", "--json", str(p), str(bad)]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["path"] for r in rows] == [str(p), str(bad)]
    assert isinstance(rows[0]["text"], str) and rows[1]["text"] is None and all(r["sec"] >= 0 for r in rows)
