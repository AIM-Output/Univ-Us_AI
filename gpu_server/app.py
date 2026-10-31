#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU 서버(학교 pod) Flask — 학습 지휘 + 병합 + OpenAI 호환 추론을 **한 프로세스**가 상태를 바꿔 가며 맡는다.

  python gpu_server/app.py --port 8600
  (운영은 gpu_server/run_server.sh start|stop|status)

왜 상태 기계인가: 9.5GiB 한 조각을 학습과 서빙이 나눠 쓸 수 없다. 그래서 상태는 항상 하나다.
    idle ─ /train/start ─▶ training ─▶ idle
    idle ─ /model/merge ─▶ merging  ─▶ idle          (어댑터 + 베이스 → nf4 병합본, 홈에 ≈2.6GB)
    idle ─ /model/load  ─▶ loading  ─▶ serving ─ /model/unload ─▶ idle
  serving 상태에서만 /v1/chat/completions 가 답한다(아니면 503). 학습 중에는 추론을 받지 않는다.

설정(환경변수 · 인자): UNIVUS_API_KEY · LOCAL_SERVER_URL(기본 http://127.0.0.1:8601 = ssh -R 로 들어온 내 PC)
  PERSIST_DIR(기본 ~/univus-train-out, 남음) · SCRATCH_DIR(기본 ~/.venv/univus-train, pod 재시작 시 소멸)
  HF_HUB_CACHE(기본 $SCRATCH_DIR/hf_cache) · CONFIG_ENV(기본 저장소의 config.env — MODEL_ID 등은 여기서만 읽는다)
모델 이름은 이 파일에 없다. 메시지 본문은 로그에 남기지 않는다(개인정보).
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
import common  # noqa: E402

from flask import Flask, jsonify, request  # noqa: E402

# ── 설정 ─────────────────────────────────────────────────────

def _parse_env_file(path: Path) -> dict[str, str]:
    """bash KEY=value 파일(config.env) → dict. 주석·빈 줄 제거, ${VAR} 는 파일 안 값 → 환경변수 순으로 치환."""
    cfg: dict[str, str] = {}
    if not path.is_file():
        return cfg
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"').strip("'")
    for k, v in cfg.items():
        cfg[k] = re.sub(r"\$\{(\w+)\}", lambda m: cfg.get(m.group(1), os.environ.get(m.group(1), "")), v)
    return cfg


HOME = Path(os.environ.get("HOME") or os.environ.get("USERPROFILE") or "~").expanduser()
CONFIG_ENV = Path(os.environ.get("CONFIG_ENV", str(REPO_ROOT / "config.env")))
CFG = _parse_env_file(CONFIG_ENV)
CFG.update({k: v for k, v in _parse_env_file(REPO_ROOT / "config.local.env").items()})   # 내 PC/pod 전용 덮어쓰기
PERSIST_DIR = Path(os.environ.get("PERSIST_DIR") or CFG.get("PERSIST_DIR") or (HOME / "univus-train-out")).expanduser()
SCRATCH_DIR = Path(os.environ.get("SCRATCH_DIR") or CFG.get("SCRATCH_DIR") or (HOME / ".venv" / "univus-train")).expanduser()
os.environ.setdefault("HF_HUB_CACHE", str(SCRATCH_DIR / "hf_cache"))      # 가중치 캐시만 overlay — 토큰(HF_HOME)은 건드리지 않는다
os.environ.setdefault("PYTHONUTF8", "1")
API_KEY = (os.environ.get("UNIVUS_API_KEY") or CFG.get("UNIVUS_API_KEY") or "").strip()
LOCAL_SERVER_URL = (os.environ.get("LOCAL_SERVER_URL") or CFG.get("LOCAL_SERVER_URL") or "http://127.0.0.1:8601").rstrip("/")
TRAIN_DATA_DIR = Path(os.environ.get("TRAIN_DATA_DIR") or (REPO_ROOT / CFG.get("DATA_DIR", "data/f9")))
LOG_DIR = PERSIST_DIR / "logs"
MODELS_DIR = PERSIST_DIR / "models"
TOOLS_PATH = REPO_ROOT / "tools" / "f9_tools.json"
IM_END = "<|im_end|>"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

# ── 상태 ─────────────────────────────────────────────────────
STATE = {"state": "idle", "since": time.time(), "started_at": time.time()}
LOCK = threading.Lock()          # 상태 전이
GEN_LOCK = threading.Lock()      # 생성은 한 번에 하나
MODEL: dict[str, Any] = {"model": None, "tok": None, "name": None, "source": None, "render_mode": None}
TRAIN: dict[str, Any] = {"run_name": None, "proc": None, "log": None, "started_at": None, "finished_at": None,
                         "ok": None, "error": None, "upload": None, "phase": None}
MERGE: dict[str, Any] = {"run_name": None, "out_name": None, "started_at": None, "finished_at": None, "ok": None,
                         "error": None, "result": None}


def _set_state(new: str) -> None:
    STATE["state"] = new
    STATE["since"] = time.time()


def _transition(expected: str, new: str) -> bool:
    with LOCK:
        if STATE["state"] != expected:
            return False
        _set_state(new)
        return True


def _authorized() -> bool:
    if not API_KEY:
        return True
    auth = request.headers.get("Authorization", "")
    return auth.startswith("Bearer ") and hmac.compare_digest(auth[7:].strip(), API_KEY)


@app.before_request
def _auth_gate():
    if not _authorized():
        return jsonify({"error": {"message": "unauthorized — Authorization: Bearer <UNIVUS_API_KEY>", "type": "auth"}}), 401
    return None


def _log(name: str, line: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / name, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {line}\n")


def _disk(path: Path) -> dict:
    try:
        path.mkdir(parents=True, exist_ok=True)
        u = shutil.disk_usage(path)
        return {"total_gb": round(u.total / 1e9, 2), "free_gb": round(u.free / 1e9, 2)}
    except OSError as e:
        return {"error": str(e)}


def _vram() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cuda": False}
        free, total = torch.cuda.mem_get_info()
        return {"cuda": True, "device": torch.cuda.get_device_name(0), "free_gib": round(free / 1024**3, 2),
                "total_gib": round(total / 1024**3, 2), "bf16": torch.cuda.is_bf16_supported()}
    except Exception as e:  # noqa: BLE001
        return {"cuda": False, "error": str(e)[:120]}


def ensure_persist_links() -> None:
    """train.sh 와 같은 규칙 — TARGET=gpu-server 면 저장소의 data·outputs 를 PERSIST_DIR 로 심볼릭 링크."""
    if CFG.get("TARGET", "local") != "gpu-server":
        return
    for d in ("data", "outputs"):
        (PERSIST_DIR / d).mkdir(parents=True, exist_ok=True)
        link = REPO_ROOT / d
        if link.is_symlink():
            continue
        if link.exists():
            link.rename(PERSIST_DIR / f"{d}_moved_{int(time.time())}")
        link.symlink_to(PERSIST_DIR / d, target_is_directory=True)


# ── 학습: 데이터 받기 → train.sh → eval.sh → 결과 올리기 ───────

def _http():
    import requests
    s = requests.Session()
    if API_KEY:
        s.headers["Authorization"] = f"Bearer {API_KEY}"
    return s


def fetch_training_data(local_url: str, dest: Path) -> list[dict]:
    """로컬 서버 /data/manifest → 파일 받기 → sha256 검증. 반환: 받은 파일 목록."""
    s = _http()
    r = s.get(f"{local_url}/data/manifest", timeout=30)
    r.raise_for_status()
    files = r.json().get("files", [])
    names = {f["name"] for f in files}
    if not {"train.jsonl", "eval.jsonl"} <= names:
        raise RuntimeError(f"로컬 서버에 train.jsonl/eval.jsonl 이 없습니다 (있는 것: {sorted(names)})")
    dest.mkdir(parents=True, exist_ok=True)
    got = []
    for f in files:
        name = f["name"]
        with s.get(f"{local_url}/data/{name}", timeout=300, stream=True) as resp:
            resp.raise_for_status()
            h = hashlib.sha256()
            tmp = dest / (name + ".part")
            with open(tmp, "wb") as out:
                for chunk in resp.iter_content(1 << 20):
                    out.write(chunk); h.update(chunk)
        if h.hexdigest() != f["sha256"]:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{name} sha256 불일치 — 전송 중 깨짐")
        tmp.replace(dest / name)
        got.append({"name": name, "size": f["size"], "sha256": f["sha256"]})
    return got


def upload_results(local_url: str, run_name: str, out_dir: Path) -> dict:
    """outputs/<run> 의 어댑터·run.json·평가 결과를 로컬 서버 /results/<run>/… 로 올린다."""
    s = _http()
    wanted: list[Path] = []
    adapter = out_dir / "adapter"
    if adapter.is_dir():
        wanted += [p for p in adapter.rglob("*") if p.is_file()]
    for pat in ("run.json", "eval_*.md", "eval_*.json", "compare.md", "summary_row_*.md", "train.log"):
        wanted += [p for p in out_dir.glob(pat) if p.is_file()]
    done, failed = [], []
    for p in wanted:
        rel = p.relative_to(out_dir).as_posix()
        try:
            with open(p, "rb") as f:
                r = s.post(f"{local_url}/results/{run_name}/{rel}", data=f, timeout=600,
                           headers={"Content-Type": "application/octet-stream"})
            r.raise_for_status()
            done.append(rel)
        except Exception as e:  # noqa: BLE001
            failed.append({"file": rel, "error": str(e)[:160]})
    return {"uploaded": done, "failed": failed, "n": len(done)}


def _run_logged(cmd: list[str], env: dict, log_path: Path) -> int:
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(cmd)}\n"); log.flush()
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
        TRAIN["proc"] = proc
        rc = proc.wait()
        TRAIN["proc"] = None
        return rc


def _train_worker(run_name: str, local_url: str, extra_args: list[str]) -> None:
    log_path = LOG_DIR / f"{run_name}.log"
    TRAIN.update({"run_name": run_name, "log": str(log_path), "started_at": time.time(), "finished_at": None,
                  "ok": None, "error": None, "upload": None, "phase": "download"})
    env = dict(os.environ)
    env["RUN_NAME"] = run_name
    try:
        ensure_persist_links()
        got = fetch_training_data(local_url, TRAIN_DATA_DIR)
        _log(f"{run_name}.log", f"데이터 받음: {json.dumps(got, ensure_ascii=False)}")
        TRAIN["phase"] = "train"
        rc = _run_logged(["bash", "train.sh", *extra_args], env, log_path)
        if rc != 0:
            raise RuntimeError(f"train.sh 종료 코드 {rc}")
        TRAIN["phase"] = "eval"
        rc = _run_logged(["bash", "eval.sh"], env, log_path)
        if rc != 0:
            _log(f"{run_name}.log", f"eval.sh 종료 코드 {rc} — 평가 실패, 어댑터 업로드는 진행")
        TRAIN["phase"] = "upload"
        out_dir = REPO_ROOT / "outputs" / run_name
        TRAIN["upload"] = upload_results(local_url, run_name, out_dir)
        TRAIN["ok"] = True
    except Exception as e:  # noqa: BLE001
        TRAIN["ok"] = False
        TRAIN["error"] = str(e)[:400]
        _log(f"{run_name}.log", f"실패: {e!r}")
    finally:
        TRAIN["finished_at"] = time.time()
        TRAIN["phase"] = None
        with LOCK:
            _set_state("idle")


def _tail(path: Optional[str], n: int = 40) -> list[str]:
    if not path or not Path(path).is_file():
        return []
    dq: deque[str] = deque(maxlen=n)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            dq.append(line.rstrip("\n"))
    return list(dq)


_LOSS_RE = re.compile(r"\{'loss': '?([\d.]+)'?.*?'epoch': '?([\d.]+)'?")


def _parse_progress(lines: list[str]) -> dict:
    prog: dict = {}
    for ln in reversed(lines):
        m = _LOSS_RE.search(ln)
        if m:
            prog = {"loss": float(m.group(1)), "epoch": float(m.group(2))}
            break
    return prog


# ── 모델 적재 · 병합 · 생성 ─────────────────────────────────────

def _bnb_config(compute_dtype: str):
    import torch
    from transformers import BitsAndBytesConfig
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(compute_dtype, torch.bfloat16)
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=dt)


def _from_pretrained(cls, path: str, **kw):
    import torch
    try:
        return cls.from_pretrained(path, dtype=torch.bfloat16, **kw)
    except TypeError:
        return cls.from_pretrained(path, torch_dtype=torch.bfloat16, **kw)


def _resolve_adapter(run_name: Optional[str], adapter_path: Optional[str]) -> Path:
    if adapter_path:
        return Path(adapter_path).expanduser().resolve()
    run = run_name or CFG.get("RUN_NAME", "run1")
    for cand in (PERSIST_DIR / "outputs" / run / "adapter", REPO_ROOT / "outputs" / run / "adapter"):
        if (cand / "adapter_config.json").is_file():
            return cand
    raise FileNotFoundError(f"어댑터를 찾지 못함: outputs/{run}/adapter (PERSIST_DIR 과 저장소 둘 다)")


def load_model(source: str, name: Optional[str], adapter_path: Optional[str]) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cuda = torch.cuda.is_available()
    device_map = "auto" if cuda else None
    if source == "merged":
        path = MODELS_DIR / (name or _default_merged_name(CFG.get("RUN_NAME", "run1")))
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"병합본 없음: {path} — 먼저 POST /model/merge")
        tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
        model = _from_pretrained(AutoModelForCausalLM, str(path), device_map=device_map, trust_remote_code=True)
        label = path.name
    elif source == "adapter":
        from peft import PeftModel
        apath = _resolve_adapter(name, adapter_path)
        base_id = CFG.get("MODEL_ID")
        if not base_id:
            raise RuntimeError("config.env 의 MODEL_ID 가 비어 있습니다")
        tok = AutoTokenizer.from_pretrained(str(apath), trust_remote_code=True)
        quant = _bnb_config(CFG.get("COMPUTE_DTYPE", "bf16")) if (cuda and CFG.get("LOAD_4BIT", "1") == "1") else None
        model = _from_pretrained(AutoModelForCausalLM, base_id, device_map=device_map, trust_remote_code=True,
                                 quantization_config=quant)
        model = PeftModel.from_pretrained(model, str(apath))
        label = f"{base_id}+{apath.name}"
    else:
        raise ValueError("source 는 merged 또는 adapter")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model.eval()
    tools, _ = common.load_tools(TOOLS_PATH)
    mode = common.detect_render_mode(tok, tools)
    MODEL.update({"model": model, "tok": tok, "name": label, "source": source, "render_mode": mode,
                  "loaded_at": time.time()})
    return {"name": label, "source": source, "render_mode": mode, "vram": _vram()}


def unload_model() -> None:
    MODEL.update({"model": None, "tok": None, "name": None, "source": None, "render_mode": None})
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _default_merged_name(run_name: str) -> str:
    return f"{run_name}-merged-nf4"


def _estimate_nf4_bytes(model) -> dict:
    """nf4 병합본 크기 추정 — 선형층은 nf4+이중양자화(≈0.56B/param), 임베딩·lm_head 는 bf16 그대로(2B/param).
    실측(Qwen2.5-0.5B): 494M params → 469MB (임베딩 비중이 커서 0.95B/param). 4B 급은 ≈2.7~2.9GB."""
    total = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight.numel() if model.get_input_embeddings() is not None else 0
    head = model.get_output_embeddings()
    tied = getattr(getattr(model, "config", None), "tie_word_embeddings", False)
    if head is not None and not tied:
        emb += head.weight.numel()
    linear = max(total - emb, 0)
    est = int(linear * 0.56 + emb * 2.0) + 16 * 1024 * 1024      # + 토크나이저·config 여유
    return {"params": total, "embedding_params": emb, "estimated_bytes": est}


def _merge_worker(run_name: str, out_name: str, adapter_path: Optional[str]) -> None:
    MERGE.update({"run_name": run_name, "out_name": out_name, "started_at": time.time(), "finished_at": None,
                  "ok": None, "error": None, "result": None})
    tmp = SCRATCH_DIR / "merge_tmp"
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        base_id = CFG.get("MODEL_ID")
        if not base_id:
            raise RuntimeError("config.env 의 MODEL_ID 가 비어 있습니다")
        apath = _resolve_adapter(run_name, adapter_path)
        out_dir = MODELS_DIR / out_name
        _log("merge.log", f"시작 run={run_name} adapter={apath} → {out_dir}")

        # 1) 베이스 bf16 을 CPU 에 올려 어댑터 병합 (GPU 9.5GiB 를 안 쓴다)
        base = _from_pretrained(AutoModelForCausalLM, base_id, low_cpu_mem_usage=True, trust_remote_code=True)
        est = _estimate_nf4_bytes(base)
        n_params, est_bytes = est["params"], est["estimated_bytes"]
        free = shutil.disk_usage(PERSIST_DIR).free if PERSIST_DIR.exists() else 0
        if est_bytes > free * 0.95:
            raise RuntimeError(f"홈 여유 {free/1e9:.2f}GB < 병합본 예상 {est_bytes/1e9:.2f}GB — 먼저 공간을 비우세요")
        merged = PeftModel.from_pretrained(base, str(apath)).merge_and_unload()
        tok = AutoTokenizer.from_pretrained(str(apath), trust_remote_code=True)
        if tmp.exists():
            shutil.rmtree(tmp)
        merged.save_pretrained(str(tmp), safe_serialization=True)
        tok.save_pretrained(str(tmp))
        del merged, base
        import gc; gc.collect()

        # 2) 병합본을 nf4 로 다시 읽어(GPU) 4-bit 그대로 저장 — 홈 3GB 예산 안에 들어가는 형태
        if not torch.cuda.is_available():
            raise RuntimeError("nf4 저장에는 CUDA 가 필요합니다")
        q = _from_pretrained(AutoModelForCausalLM, str(tmp), device_map="auto", trust_remote_code=True,
                             quantization_config=_bnb_config(CFG.get("COMPUTE_DTYPE", "bf16")))
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        q.save_pretrained(str(out_dir), safe_serialization=True)
        tok.save_pretrained(str(out_dir))
        del q; gc.collect(); torch.cuda.empty_cache()
        shutil.rmtree(tmp, ignore_errors=True)

        size = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
        info = {"run_name": run_name, "out_name": out_name, "base_model": base_id, "adapter": str(apath),
                "params": n_params, "embedding_params": est["embedding_params"], "estimated_bytes": est_bytes, "size_bytes": size,
                "home_free_gb_after": round(shutil.disk_usage(PERSIST_DIR).free / 1e9, 2),
                "quant": "bnb nf4 double-quant", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "versions": common.lib_versions()}
        (out_dir / "merge.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        MERGE.update({"ok": True, "result": info})
        _log("merge.log", f"완료 {out_dir} {size/1e9:.2f}GB")
    except Exception as e:  # noqa: BLE001
        MERGE.update({"ok": False, "error": str(e)[:400]})
        _log("merge.log", f"실패: {e!r}")
        shutil.rmtree(tmp, ignore_errors=True)
    finally:
        MERGE["finished_at"] = time.time()
        with LOCK:
            _set_state("idle")


def _render_prompt(tok, messages: list[dict], tools: Optional[list[dict]], json_mode: bool) -> str:
    """도구가 있으면 학습과 같은 render_chat, 없으면 템플릿만. (빈 tools 로 render_chat 을 부르면 모드가 잘못 캐시된다)"""
    msgs = [dict(m) for m in messages]
    if json_mode:
        note = "\n\n답은 JSON 만 출력하세요."
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = (msgs[0].get("content") or "") + note
        else:
            msgs.insert(0, {"role": "system", "content": note.strip()})
    if tools:
        return common.render_chat(tok, msgs, tools, add_generation_prompt=True)
    flat = common.flatten_tool_calls(msgs)
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(flat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(flat, tokenize=False, add_generation_prompt=True)
    return common.manual_render(flat, add_generation_prompt=True)


def _parse_all_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in common._TOOL_CALL.finditer(text):             # noqa: SLF001 — 같은 프로젝트의 파서를 그대로 쓴다
        obj = common._first_json_object(m.group(1))        # noqa: SLF001
        if obj and "name" in obj:
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            calls.append({"name": str(obj["name"]), "arguments": args if isinstance(args, dict) else {}})
    if not calls:
        one = common.parse_tool_call(text)
        if one:
            calls.append(one)
    return calls


def generate_completion(body: dict) -> dict:
    import torch
    model, tok = MODEL["model"], MODEL["tok"]
    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 가 비어 있습니다")
    tools = body.get("tools") or None
    if body.get("tool_choice") == "none":
        tools = None
    json_mode = (body.get("response_format") or {}).get("type") == "json_object"
    temperature = float(body.get("temperature") or 0.0)
    max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 512)
    prompt = _render_prompt(tok, messages, tools, json_mode)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    eos_ids = {tok.eos_token_id}
    im_end = tok.convert_tokens_to_ids(IM_END)
    if isinstance(im_end, int) and im_end >= 0 and im_end != tok.unk_token_id:
        eos_ids.add(im_end)
    gen_kw: dict[str, Any] = {"max_new_tokens": max_new, "pad_token_id": tok.pad_token_id,
                              "eos_token_id": sorted(i for i in eos_ids if i is not None)}
    if temperature > 0:
        gen_kw.update(do_sample=True, temperature=temperature, top_p=float(body.get("top_p") or 0.95))
    else:
        gen_kw.update(do_sample=False)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, **gen_kw)
    new_ids = out[0, enc.input_ids.shape[1]:]
    n_new = int(new_ids.shape[0])
    raw = tok.decode(new_ids, skip_special_tokens=True)
    text = common.strip_think(raw)
    calls = _parse_all_tool_calls(text) if tools else []
    content = common._TOOL_CALL.sub("", text).strip() if calls else text.strip()   # noqa: SLF001
    hit_eos = bool(len(new_ids) and int(new_ids[-1]) in eos_ids)
    finish = "tool_calls" if calls else ("stop" if (hit_eos or n_new < max_new) else "length")
    elapsed = time.time() - t0
    _log("serve.log", f"chat prompt_tokens={int(enc.input_ids.shape[1])} completion_tokens={n_new} "
                      f"tools={len(tools) if tools else 0} calls={len(calls)} finish={finish} {elapsed:.2f}s")
    message: dict[str, Any] = {"role": "assistant", "content": content or (None if calls else "")}
    if calls:
        message["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                                  "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                                 for c in calls]
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
            "model": MODEL["name"], "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": int(enc.input_ids.shape[1]), "completion_tokens": n_new,
                      "total_tokens": int(enc.input_ids.shape[1]) + n_new},
            "univus": {"render_mode": MODEL["render_mode"], "latency_s": round(elapsed, 3)}}


# ── 엔드포인트 ───────────────────────────────────────────────

@app.get("/health")
def health():
    return jsonify({"ok": True, "state": STATE["state"], "state_since": int(STATE["since"]),
                    "uptime_s": round(time.time() - STATE["started_at"]),
                    "model": {"loaded": MODEL["model"] is not None, "name": MODEL["name"], "source": MODEL["source"],
                              "render_mode": MODEL["render_mode"]},
                    "config": {"config_env": str(CONFIG_ENV), "model_id": CFG.get("MODEL_ID"), "run_name": CFG.get("RUN_NAME"),
                               "target": CFG.get("TARGET"), "local_server_url": LOCAL_SERVER_URL,
                               "persist_dir": str(PERSIST_DIR), "scratch_dir": str(SCRATCH_DIR),
                               "hf_hub_cache": os.environ.get("HF_HUB_CACHE"), "auth": bool(API_KEY)},
                    "vram": _vram(), "disk": {"persist": _disk(PERSIST_DIR), "scratch": _disk(SCRATCH_DIR)},
                    "train": {k: v for k, v in TRAIN.items() if k != "proc"}, "merge": MERGE,
                    "versions": common.lib_versions()})


@app.post("/train/start")
def train_start():
    body = request.get_json(silent=True) or {}
    run_name = str(body.get("run_name") or CFG.get("RUN_NAME") or "run1")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", run_name):
        return jsonify({"error": "run_name 은 영문·숫자·._- 만"}), 400
    local_url = str(body.get("local_url") or LOCAL_SERVER_URL).rstrip("/")
    extra = [str(a) for a in (body.get("extra_args") or [])]
    if MODEL["model"] is not None:
        return jsonify({"error": "모델이 적재돼 있습니다 — 먼저 POST /model/unload (학습과 서빙은 VRAM 을 나눌 수 없음)"}), 409
    if not _transition("idle", "training"):
        return jsonify({"error": f"지금 상태가 {STATE['state']} 입니다 — idle 일 때만 학습을 시작합니다"}), 409
    th = threading.Thread(target=_train_worker, args=(run_name, local_url, extra), daemon=True)
    th.start()
    return jsonify({"started": True, "run_name": run_name, "local_url": local_url,
                    "log": str(LOG_DIR / f"{run_name}.log")}), 202


@app.get("/train/status")
def train_status():
    lines = _tail(TRAIN.get("log"))
    run_json = None
    if TRAIN.get("run_name"):
        p = REPO_ROOT / "outputs" / TRAIN["run_name"] / "run.json"
        if p.is_file():
            try:
                run_json = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                run_json = {"error": "run.json 파싱 실패"}
    return jsonify({"state": STATE["state"], **{k: v for k, v in TRAIN.items() if k != "proc"},
                    "running": TRAIN.get("proc") is not None, "progress": _parse_progress(lines),
                    "log_tail": lines, "run_json": run_json})


@app.post("/train/stop")
def train_stop():
    proc = TRAIN.get("proc")
    if STATE["state"] != "training" or proc is None:
        return jsonify({"stopped": False, "detail": "실행 중인 학습이 없습니다"}), 409
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
    return jsonify({"stopped": True})


@app.post("/model/merge")
def model_merge():
    body = request.get_json(silent=True) or {}
    run_name = str(body.get("run_name") or CFG.get("RUN_NAME") or "run1")
    out_name = str(body.get("out_name") or _default_merged_name(run_name))
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", out_name):
        return jsonify({"error": "out_name 은 영문·숫자·._- 만"}), 400
    if MODEL["model"] is not None:
        return jsonify({"error": "모델이 적재돼 있습니다 — 먼저 POST /model/unload"}), 409
    if not _transition("idle", "merging"):
        return jsonify({"error": f"지금 상태가 {STATE['state']} 입니다"}), 409
    threading.Thread(target=_merge_worker, args=(run_name, out_name, body.get("adapter_path")), daemon=True).start()
    return jsonify({"started": True, "run_name": run_name, "out_name": out_name,
                    "out_dir": str(MODELS_DIR / out_name), "status": "GET /health 의 merge 로 확인"}), 202


@app.get("/model/merge/status")
def merge_status():
    return jsonify({"state": STATE["state"], **MERGE})


@app.post("/model/load")
def model_load():
    body = request.get_json(silent=True) or {}
    source = str(body.get("source") or "merged")
    if not _transition("idle", "loading"):
        return jsonify({"error": f"지금 상태가 {STATE['state']} 입니다 — idle 일 때만 적재합니다"}), 409
    try:
        info = load_model(source, body.get("name") or body.get("run_name"), body.get("adapter_path"))
    except Exception as e:  # noqa: BLE001
        unload_model()
        with LOCK:
            _set_state("idle")
        return jsonify({"error": str(e)[:400]}), 400
    with LOCK:
        _set_state("serving")
    return jsonify({"loaded": True, **info})


@app.post("/model/unload")
def model_unload():
    if STATE["state"] not in ("serving", "loading"):
        return jsonify({"unloaded": False, "detail": f"상태 {STATE['state']}"}), 409
    with GEN_LOCK:
        unload_model()
        with LOCK:
            _set_state("idle")
    return jsonify({"unloaded": True, "vram": _vram()})


@app.get("/models")
def models_list():
    return jsonify({"merged": sorted(p.name for p in MODELS_DIR.iterdir() if (p / "config.json").is_file())
                    if MODELS_DIR.exists() else [],
                    "adapters": sorted(p.parent.parent.name for p in (PERSIST_DIR / "outputs").glob("*/adapter/adapter_config.json"))
                    if (PERSIST_DIR / "outputs").exists() else []})


@app.get("/v1/models")
def v1_models():
    data = [{"id": MODEL["name"], "object": "model", "owned_by": "univus"}] if MODEL["name"] else []
    return jsonify({"object": "list", "data": data})


@app.post("/v1/chat/completions")
def v1_chat():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": {"message": "JSON 본문이 필요합니다", "type": "invalid_request_error"}}), 400
    if body.get("stream"):
        return jsonify({"error": {"message": "stream 은 지원하지 않습니다 — stream:false 로 보내세요",
                                  "type": "invalid_request_error"}}), 400
    if STATE["state"] != "serving" or MODEL["model"] is None:
        return jsonify({"error": {"message": f"모델이 서빙 상태가 아닙니다 (state={STATE['state']}) — POST /model/load",
                                  "type": "server_error"}}), 503
    try:
        with GEN_LOCK:
            if MODEL["model"] is None:
                return jsonify({"error": {"message": "모델이 내려갔습니다", "type": "server_error"}}), 503
            return jsonify(generate_completion(body))
    except ValueError as e:
        return jsonify({"error": {"message": str(e), "type": "invalid_request_error"}}), 400
    except Exception as e:  # noqa: BLE001
        _log("serve.log", f"ERROR {type(e).__name__}: {str(e)[:200]}")
        return jsonify({"error": {"message": f"생성 실패: {type(e).__name__}", "type": "server_error"}}), 500


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0", help="pod 안에서만 열리고 밖에는 ssh -L 로만 보인다")
    ap.add_argument("--port", type=int, default=int(os.environ.get("SERVER_PORT", "8600")))
    ap.add_argument("--pidfile", default="")
    args = ap.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if args.pidfile:
        Path(args.pidfile).write_text(str(os.getpid()), encoding="utf-8")
    print(f"gpu_server  http://{args.host}:{args.port}  model_id={CFG.get('MODEL_ID')}  run={CFG.get('RUN_NAME')}  "
          f"persist={PERSIST_DIR}  scratch={SCRATCH_DIR}  local={LOCAL_SERVER_URL}  "
          f"auth={'on' if API_KEY else 'OFF (UNIVUS_API_KEY 비어 있음)'}")
    print(f"  vram={_vram()}  disk(persist)={_disk(PERSIST_DIR)}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
