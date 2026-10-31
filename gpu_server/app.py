#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU 서버(학교 pod) Flask — 학습 지휘 + 병합 + OpenAI 호환 추론을 **한 프로세스**가 상태를 바꿔 가며 맡는다.

  python gpu_server/app.py --port 8600            # 127.0.0.1 에만 묶인다 (ssh -L 로만 보인다)
  (운영은 gpu_server/run_server.sh start|stop|status)

왜 상태 기계인가: 9.5GiB 한 조각을 학습과 서빙이 나눠 쓸 수 없다. 그래서 상태는 항상 하나다.
    idle ─ /train/start ─▶ training ─▶ idle
    idle ─ /model/merge ─▶ merging  ─▶ idle          (어댑터 + 베이스 → nf4 병합본, 홈에 ≈2.8GB)
    idle ─ /model/load  ─▶ loading  ─▶ serving ─ /model/unload ─▶ idle
  serving 상태에서만 /v1/chat/completions 가 답한다(아니면 503). 학습 중에는 추론을 받지 않는다.

설정(환경변수 · 인자): UNIVUS_API_KEY(루프백이 아닌 host 에 열 때 필수) · LOCAL_SERVER_URL(기본 http://127.0.0.1:8601 = ssh -R 로 들어온 내 PC)
  PERSIST_DIR(기본 ~/univus-train-out, 남음) · SCRATCH_DIR(기본 ~/.venv/univus-train, pod 재시작 시 소멸)
  HF_HUB_CACHE(기본 $SCRATCH_DIR/hf_cache) · CONFIG_ENV(기본 저장소의 config.env — MODEL_ID 등은 여기서만 읽는다)
  env 파일 키는 [A-Za-z0-9_], 값은 '#' 뒤가 잘린다.
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
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
import common  # noqa: E402

from flask import Flask, jsonify, request  # noqa: E402

# ── 설정 ─────────────────────────────────────────────────────

def _parse_env_file(path: Path) -> dict[str, str]:
    """bash KEY=value 파일(config.env) → dict. 키는 [A-Za-z0-9_], '#' 뒤는 주석, ${VAR} 는 파일 안 값 → 환경변수 순으로 치환."""
    cfg: dict[str, str] = {}
    if not path.is_file():
        return cfg
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if m:
            cfg[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    for k, v in cfg.items():
        cfg[k] = re.sub(r"\$\{(\w+)\}", lambda mm: cfg.get(mm.group(1), os.environ.get(mm.group(1), "")), v)
    return cfg


HOME = Path(os.environ.get("HOME") or os.environ.get("USERPROFILE") or "~").expanduser()
CONFIG_ENV = Path(os.environ.get("CONFIG_ENV", str(REPO_ROOT / "config.env")))
CFG = _parse_env_file(CONFIG_ENV)
CFG.update(_parse_env_file(REPO_ROOT / "config.local.env"))                 # 내 PC/pod 전용 덮어쓰기 (gitignore)
PERSIST_DIR = Path(os.environ.get("PERSIST_DIR") or CFG.get("PERSIST_DIR") or (HOME / "univus-train-out")).expanduser().resolve()
SCRATCH_DIR = Path(os.environ.get("SCRATCH_DIR") or CFG.get("SCRATCH_DIR") or (HOME / ".venv" / "univus-train")).expanduser().resolve()
os.environ.setdefault("HF_HUB_CACHE", str(SCRATCH_DIR / "hf_cache"))      # 가중치 캐시만 overlay — 토큰(HF_HOME)은 건드리지 않는다
os.environ.setdefault("PYTHONUTF8", "1")
API_KEY = (os.environ.get("UNIVUS_API_KEY") or CFG.get("UNIVUS_API_KEY") or "").strip()
API_KEY_B = API_KEY.encode("utf-8")
LOCAL_SERVER_URL = (os.environ.get("LOCAL_SERVER_URL") or CFG.get("LOCAL_SERVER_URL") or "http://127.0.0.1:8601").rstrip("/")
TRAIN_DATA_DIR = Path(os.environ.get("TRAIN_DATA_DIR") or (REPO_ROOT / CFG.get("DATA_DIR", "data/f9")))
LOG_DIR = PERSIST_DIR / "logs"
MODELS_DIR = PERSIST_DIR / "models"
TOOLS_PATH = REPO_ROOT / "tools" / "f9_tools.json"
IM_END = "<|im_end|>"
LOOPBACK = ("127.0.0.1", "localhost", "::1")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
DATA_WHITELIST = ("train.jsonl", "eval.jsonl", "README.md")
EXTRA_ARGS = {"--max_steps": True, "--max_train_rows": True, "--epochs": True, "--lr": True, "--resume": False}   # 값 필요 여부
MAX_NEW_TOKENS_CAP = 2048
MAX_PROMPT_TOKENS = 8192
MIN_CPU_MERGE_MEM = 10 * 1024**3

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

# ── 상태 ─────────────────────────────────────────────────────
STATE = {"state": "idle", "since": time.time(), "started_at": time.time()}
LOCK = threading.Lock()          # 상태 전이
GEN_LOCK = threading.Lock()      # 생성·적재 해제는 한 번에 하나
MODEL: dict[str, Any] = {"model": None, "tok": None, "name": None, "source": None, "render_mode": None}
TRAIN: dict[str, Any] = {"run_name": None, "proc": None, "log": None, "started_at": None, "finished_at": None,
                         "ok": None, "error": None, "upload": None, "phase": None, "stopped": False, "cleaned": None}
MERGE: dict[str, Any] = {"run_name": None, "out_name": None, "mode": None, "started_at": None, "finished_at": None,
                         "ok": None, "error": None, "result": None}


def _set_state(new: str) -> None:
    STATE["state"] = new
    STATE["since"] = time.time()


def _transition(expected: str, new: str) -> bool:
    """expected 상태일 때만 new 로 바꾼다 (원자적)."""
    with LOCK:
        if STATE["state"] != expected:
            return False
        _set_state(new)
        return True


def _authorized() -> bool:
    if not API_KEY:
        return True
    auth = request.headers.get("Authorization", "")
    return auth.startswith("Bearer ") and hmac.compare_digest(auth[7:].strip().encode("utf-8"), API_KEY_B)


@app.before_request
def _auth_gate():
    if not _authorized():
        return _err(401, "unauthorized — Authorization: Bearer <UNIVUS_API_KEY>", "auth")
    return None


def _err(code: int, message: str, etype: str = "invalid_request_error"):
    return jsonify({"error": {"message": message, "type": etype, "code": code}}), code


@app.errorhandler(404)
@app.errorhandler(405)
@app.errorhandler(413)
@app.errorhandler(500)
def _http_err(e):
    code = getattr(e, "code", 500)
    return _err(code, getattr(e, "description", str(e)), "server_error" if code >= 500 else "invalid_request_error")


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


def _valid_name(s: Any) -> Optional[str]:
    s = str(s or "")
    return s if NAME_RE.fullmatch(s) else None


def _under(path: Path, root: Path) -> bool:
    try:
        p = path.resolve()
        r = root.resolve()
        return p == r or r in p.parents
    except OSError:
        return False


def _is_loopback(url: str) -> bool:
    try:
        return (urlparse(url).hostname or "") in LOOPBACK
    except ValueError:
        return False


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


# ── 프로세스 그룹 ────────────────────────────────────────────

def popen_group(cmd: list[str], env: dict, log_fh) -> subprocess.Popen:
    """자식(bash → python | tee)을 한 프로세스 그룹으로 띄운다 — 중지 때 통째로 죽이기 위해."""
    kw: dict[str, Any] = {}
    if os.name == "posix":
        kw["start_new_session"] = True
    else:
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log_fh, stderr=subprocess.STDOUT, **kw)


def kill_process_group(proc: subprocess.Popen, timeout: float = 20.0) -> bool:
    """SIGTERM → 기다림 → SIGKILL 을 **그룹 전체**에. 반환: 그룹이 끝났는가."""
    if proc.poll() is not None:
        return True
    if os.name == "posix":
        for sig, wait in ((signal.SIGTERM, timeout), (signal.SIGKILL, 10.0)):
            try:
                os.killpg(os.getpgid(proc.pid), sig)
            except ProcessLookupError:
                return True
            try:
                proc.wait(timeout=wait)
                return True
            except subprocess.TimeoutExpired:
                continue
        return proc.poll() is not None
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)   # Windows(개발 PC): 트리째
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False


# ── 학습: 데이터 받기 → train.sh → eval.sh → 결과 올리기 ───────

def _http():
    import requests
    s = requests.Session()
    if API_KEY:
        s.headers["Authorization"] = f"Bearer {API_KEY}"
    return s


def fetch_training_data(local_url: str, dest: Path) -> list[dict]:
    """로컬 서버 /data/manifest → 화이트리스트 파일만 받기 → sha256 검증."""
    s = _http()
    r = s.get(f"{local_url}/data/manifest", timeout=30)
    r.raise_for_status()
    files = [f for f in r.json().get("files", []) if f.get("name") in DATA_WHITELIST]
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
        if h.hexdigest() != f.get("sha256"):
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{name} sha256 불일치 — 전송 중 깨짐")
        tmp.replace(dest / name)
        got.append({"name": name, "size": f.get("size"), "sha256": f.get("sha256")})
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
    return {"uploaded": done, "failed": failed, "n": len(done), "n_failed": len(failed)}


def cleanup_checkpoints(out_dir: Path) -> list[str]:
    """홈 10GB 예산 — 업로드가 끝난 run 의 checkpoint-* 를 지운다 (adapter/ · run.json · 평가 파일은 남긴다)."""
    removed = []
    for d in sorted(out_dir.glob("checkpoint-*")):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d.name)
    return removed


def validate_extra_args(extra: Any) -> list[str]:
    if extra is None:
        return []
    if not isinstance(extra, list) or not all(isinstance(a, (str, int, float)) for a in extra):
        raise ValueError("extra_args 는 문자열 목록이어야 합니다")
    out, i = [], 0
    while i < len(extra):
        flag = str(extra[i])
        if flag not in EXTRA_ARGS:
            raise ValueError(f"허용되지 않는 인자: {flag} (가능: {', '.join(EXTRA_ARGS)})")
        out.append(flag)
        if EXTRA_ARGS[flag]:
            if i + 1 >= len(extra) or not re.fullmatch(r"-?\d+(\.\d+)?([eE]-?\d+)?", str(extra[i + 1])):
                raise ValueError(f"{flag} 뒤에는 숫자 값이 와야 합니다")
            out.append(str(extra[i + 1])); i += 1
        i += 1
    return out


def _run_logged(cmd: list[str], env: dict, log_path: Path) -> int:
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(cmd)}\n"); log.flush()
        proc = popen_group(cmd, env, log)
        TRAIN["proc"] = proc
        rc = proc.wait()
        TRAIN["proc"] = None
        return rc


def _train_worker(run_name: str, local_url: str, extra_args: list[str]) -> None:
    log_path = LOG_DIR / f"{run_name}.log"
    TRAIN.update({"run_name": run_name, "log": str(log_path), "started_at": time.time(), "finished_at": None,
                  "ok": None, "error": None, "upload": None, "phase": "download", "stopped": False, "cleaned": None})
    env = dict(os.environ)
    env["RUN_NAME"] = run_name
    env["PYTHONUNBUFFERED"] = "1"          # loss 줄이 바로 로그에 보이게
    try:
        ensure_persist_links()
        got = fetch_training_data(local_url, TRAIN_DATA_DIR)
        _log(f"{run_name}.log", f"데이터 받음: {json.dumps(got, ensure_ascii=False)}")
        TRAIN["phase"] = "train"
        rc = _run_logged(["bash", "train.sh", *extra_args], env, log_path)
        if TRAIN["stopped"]:
            raise RuntimeError("사용자가 중지함")
        if rc != 0:
            raise RuntimeError(f"train.sh 종료 코드 {rc}")
        TRAIN["phase"] = "eval"
        rc = _run_logged(["bash", "eval.sh"], env, log_path)
        if TRAIN["stopped"]:
            raise RuntimeError("사용자가 중지함")
        if rc != 0:
            _log(f"{run_name}.log", f"eval.sh 종료 코드 {rc} — 평가 실패, 어댑터 업로드는 진행")
        TRAIN["phase"] = "upload"
        out_dir = REPO_ROOT / "outputs" / run_name
        up = upload_results(local_url, run_name, out_dir)
        TRAIN["upload"] = up
        if up["n_failed"]:
            TRAIN["ok"] = "partial"
            TRAIN["error"] = f"업로드 실패 {up['n_failed']}건 — 어댑터는 pod 의 {out_dir} 에 남아 있다 (checkpoint 도 보존)"
        else:
            TRAIN["cleaned"] = cleanup_checkpoints(out_dir)
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
    for ln in reversed(lines):
        m = _LOSS_RE.search(ln)
        if m:
            return {"loss": float(m.group(1)), "epoch": float(m.group(2))}
    return {}


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
        p = Path(str(adapter_path)).expanduser()
        if not _under(p, PERSIST_DIR):
            raise ValueError(f"adapter_path 는 {PERSIST_DIR} 아래만 허용")
        if not (p / "adapter_config.json").is_file():
            raise FileNotFoundError(f"어댑터 없음: {p}")
        return p.resolve()
    run = _valid_name(run_name or CFG.get("RUN_NAME", "run1"))
    if not run:
        raise ValueError("run_name 형식 오류")
    for cand in (PERSIST_DIR / "outputs" / run / "adapter", REPO_ROOT / "outputs" / run / "adapter"):
        if (cand / "adapter_config.json").is_file():
            return cand.resolve()
    raise FileNotFoundError(f"어댑터를 찾지 못함: outputs/{run}/adapter (PERSIST_DIR 과 저장소 둘 다)")


def _default_merged_name(run_name: str) -> str:
    return f"{run_name}-merged-nf4"


def _estimate_nf4_bytes(model) -> dict:
    """nf4 병합본 크기 추정 — 선형층 nf4+이중양자화 ≈0.56B/param, 임베딩·lm_head 는 bf16 그대로 2B/param.
    실측(Qwen2.5-0.5B): 494M params → 469MB (추정 489MB, +4%). Qwen3-4B ≈ 2.83GB."""
    total = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight.numel() if model.get_input_embeddings() is not None else 0
    head = model.get_output_embeddings()
    tied = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))
    if head is not None and not tied:
        emb += head.weight.numel()
    linear = max(total - emb, 0)
    est = int(linear * 0.56 + emb * 2.0) + 16 * 1024 * 1024
    return {"params": total, "embedding_params": emb, "estimated_bytes": est}


def _estimate_from_id(model_id: str) -> dict:
    """가중치를 읽지 않고(meta device) 구조만으로 파라미터 수를 센다 — 4-bit 로 적재하면 numel 이 절반으로 보여서."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    return _estimate_nf4_bytes(m)


def _cgroup_mem_limit() -> Optional[int]:
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(p).read_text().strip()
        except OSError:
            continue
        if v and v != "max":
            try:
                return int(v)
            except ValueError:
                pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def load_model(source: str, name: Optional[str], adapter_path: Optional[str]) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cuda = torch.cuda.is_available()
    device_map = "auto" if cuda else None
    if source == "merged":
        nm = _valid_name(name or _default_merged_name(CFG.get("RUN_NAME", "run1")))
        if not nm:
            raise ValueError("name 형식 오류 (영문·숫자로 시작, ._- 만)")
        path = MODELS_DIR / nm
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"병합본 없음: {path} — 먼저 POST /model/merge")
        tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
        model = _from_pretrained(AutoModelForCausalLM, str(path), device_map=device_map, trust_remote_code=True)
        label = nm
    elif source == "adapter":
        from peft import PeftModel
        if name is not None and not _valid_name(name):
            raise ValueError("run_name 형식 오류")
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
    MODEL.update({"model": model, "tok": tok, "name": label, "source": source, "render_mode": mode, "loaded_at": time.time()})
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


def _merge_worker(run_name: str, out_name: str, adapter_path: Optional[str], mode: str) -> None:
    MERGE.update({"run_name": run_name, "out_name": out_name, "mode": mode, "started_at": time.time(),
                  "finished_at": None, "ok": None, "error": None, "result": None})
    tmp = SCRATCH_DIR / "merge_tmp"
    out_dir = MODELS_DIR / out_name
    partial = MODELS_DIR / f"{out_name}.partial"
    try:
        import gc
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        base_id = CFG.get("MODEL_ID")
        if not base_id:
            raise RuntimeError("config.env 의 MODEL_ID 가 비어 있습니다")
        apath = _resolve_adapter(run_name, adapter_path)
        _log("merge.log", f"시작 mode={mode} run={run_name} adapter={apath} → {out_dir}")

        # 0) 공간 — 가중치를 읽지 않고 구조만으로 추정. 덮어쓸 기존 병합본 크기는 여유로 쳐 준다
        est = _estimate_from_id(base_id)
        n_params, est_bytes = est["params"], est["estimated_bytes"]
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(PERSIST_DIR).free + _dir_size(out_dir)
        if est_bytes > free * 0.95:
            raise RuntimeError(f"홈 여유 {free/1e9:.2f}GB < 병합본 예상 {est_bytes/1e9:.2f}GB — 먼저 공간을 비우세요")
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True, exist_ok=True)
        tok = AutoTokenizer.from_pretrained(str(apath), trust_remote_code=True)

        if mode == "gpu-nf4":
            # 1) 베이스를 nf4 로 GPU 에 올리고 어댑터를 그 위에 병합 (peft 가 4-bit Linear 로 병합을 지원) → 그대로 저장
            if not torch.cuda.is_available():
                raise RuntimeError("gpu-nf4 병합에는 CUDA 가 필요합니다 (CPU 만이면 mode=cpu-bf16)")
            base = _from_pretrained(AutoModelForCausalLM, base_id, device_map="auto", trust_remote_code=True,
                                    quantization_config=_bnb_config(CFG.get("COMPUTE_DTYPE", "bf16")))
            merged = PeftModel.from_pretrained(base, str(apath)).merge_and_unload()
            merged.save_pretrained(str(partial), safe_serialization=True)
            tok.save_pretrained(str(partial))
            del merged, base
        elif mode == "cpu-bf16":
            # 1') 베이스 bf16 을 CPU RAM 에 통째로 (4B ≈ 8GB) — cgroup 메모리 한도가 10GB 미만이면 거절
            lim = _cgroup_mem_limit()
            if lim is not None and lim < MIN_CPU_MERGE_MEM:
                raise RuntimeError(f"컨테이너 메모리 한도 {lim/1024**3:.1f}GiB < 10GiB — mode=gpu-nf4 로 병합하세요")
            if not torch.cuda.is_available():
                raise RuntimeError("nf4 저장에는 CUDA 가 필요합니다")
            base = _from_pretrained(AutoModelForCausalLM, base_id, low_cpu_mem_usage=True, trust_remote_code=True)
            merged = PeftModel.from_pretrained(base, str(apath)).merge_and_unload()
            shutil.rmtree(tmp, ignore_errors=True)
            merged.save_pretrained(str(tmp), safe_serialization=True)
            tok.save_pretrained(str(tmp))
            del merged, base
            gc.collect()
            q = _from_pretrained(AutoModelForCausalLM, str(tmp), device_map="auto", trust_remote_code=True,
                                 quantization_config=_bnb_config(CFG.get("COMPUTE_DTYPE", "bf16")))
            q.save_pretrained(str(partial), safe_serialization=True)
            tok.save_pretrained(str(partial))
            del q
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            raise ValueError("mode 는 gpu-nf4 또는 cpu-bf16")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2) 완성본만 제자리로 (기존 것은 그때 교체)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        partial.rename(out_dir)
        size = _dir_size(out_dir)
        info = {"run_name": run_name, "out_name": out_name, "mode": mode, "base_model": base_id, "adapter": str(apath),
                "params": n_params, "embedding_params": est["embedding_params"], "estimated_bytes": est_bytes,
                "size_bytes": size, "home_free_gb_after": round(shutil.disk_usage(PERSIST_DIR).free / 1e9, 2),
                "quant": "bnb nf4 double-quant", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "versions": common.lib_versions()}
        (out_dir / "merge.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        MERGE.update({"ok": True, "result": info})
        _log("merge.log", f"완료 {out_dir} {size/1e9:.2f}GB")
    except Exception as e:  # noqa: BLE001
        MERGE.update({"ok": False, "error": str(e)[:400]})
        _log("merge.log", f"실패: {e!r}")
        shutil.rmtree(partial, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)
    finally:
        MERGE["finished_at"] = time.time()
        try:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        with LOCK:
            _set_state("idle")


def _validate_messages(messages: Any) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 는 비어 있지 않은 목록이어야 합니다")
    out = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"messages[{i}] 는 role(system|user|assistant|tool) 이 있는 객체여야 합니다")
        content = m.get("content")
        if content is not None and not isinstance(content, str):
            if isinstance(content, list):   # OpenAI 멀티파트 — 텍스트 조각만 합친다
                content = "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
            else:
                raise ValueError(f"messages[{i}].content 는 문자열이어야 합니다")
        if m.get("tool_calls") is not None and not isinstance(m["tool_calls"], list):
            raise ValueError(f"messages[{i}].tool_calls 는 목록이어야 합니다")
        mm = {"role": m["role"], "content": content}
        if m.get("tool_calls"):
            mm["tool_calls"] = m["tool_calls"]
        out.append(mm)
    return out


def _json_note(response_format: Any) -> Optional[str]:
    """response_format: json_object | json_schema → system 에 덧붙일 안내. 아니면 None."""
    if not isinstance(response_format, dict):
        return None
    t = response_format.get("type")
    if t == "json_object":
        return "답은 JSON 만 출력하세요."
    if t == "json_schema":
        js = response_format.get("json_schema") or {}
        schema = js.get("schema", js) if isinstance(js, dict) else js
        return "답은 아래 JSON 스키마를 따르는 JSON 만 출력하세요:\n" + json.dumps(schema, ensure_ascii=False)[:4000]
    return None


def _render_prompt(tok, messages: list[dict], tools: Optional[list[dict]], note: Optional[str]) -> str:
    """도구가 있으면 학습과 같은 render_chat, 없으면 템플릿만. (빈 tools 로 render_chat 을 부르면 모드가 잘못 캐시된다)"""
    msgs = [dict(m) for m in messages]
    if note:
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = (msgs[0].get("content") or "") + "\n\n" + note
        else:
            msgs.insert(0, {"role": "system", "content": note})
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
    for m in common._TOOL_CALL.finditer(text):             # noqa: SLF001 — 같은 프로젝트의 파서
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
    messages = _validate_messages(body.get("messages"))
    tools = body.get("tools") or None
    if tools is not None and (not isinstance(tools, list) or not all(isinstance(t, dict) and isinstance(t.get("function"), dict)
                                                                       and t["function"].get("name") for t in tools)):
        raise ValueError("tools 는 {type:function, function:{name,…}} 목록이어야 합니다")
    if body.get("tool_choice") == "none":
        tools = None
    allowed = {t["function"]["name"] for t in tools} if tools else set()
    note = _json_note(body.get("response_format"))
    temperature = float(body.get("temperature") or 0.0)
    mt = body.get("max_tokens", body.get("max_completion_tokens"))      # 0 과 None 을 구분한다 (0 은 400)
    try:
        max_new = 512 if mt is None else int(mt)
    except (TypeError, ValueError):
        raise ValueError("max_tokens 는 정수")
    if max_new < 1:
        raise ValueError("max_tokens 는 1 이상")
    max_new = min(max_new, MAX_NEW_TOKENS_CAP)
    prompt = _render_prompt(tok, messages, tools, note)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    n_prompt = int(enc.input_ids.shape[1])
    if n_prompt > MAX_PROMPT_TOKENS:
        raise ValueError(f"프롬프트가 {n_prompt} 토큰 — 상한 {MAX_PROMPT_TOKENS}")
    enc = enc.to(model.device)
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
    new_ids = out[0, n_prompt:]
    n_new = int(new_ids.shape[0])
    text = common.strip_think(tok.decode(new_ids, skip_special_tokens=True))
    calls = [c for c in _parse_all_tool_calls(text) if c["name"] in allowed] if tools else []
    content = common._TOOL_CALL.sub("", text).strip() if calls else text.strip()   # noqa: SLF001
    hit_eos = bool(n_new and int(new_ids[-1]) in eos_ids)
    finish = "tool_calls" if calls else ("stop" if (hit_eos or n_new < max_new) else "length")
    elapsed = time.time() - t0
    _log("serve.log", f"chat prompt_tokens={n_prompt} completion_tokens={n_new} tools={len(allowed)} calls={len(calls)} "
                      f"json={'y' if note else 'n'} finish={finish} {elapsed:.2f}s")
    message: dict[str, Any] = {"role": "assistant", "content": content or (None if calls else "")}
    if calls:
        message["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                                  "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                                 for c in calls]
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
            "model": MODEL["name"], "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_new, "total_tokens": n_prompt + n_new},
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
    run_name = _valid_name(body.get("run_name") or CFG.get("RUN_NAME") or "run1")
    if not run_name:
        return _err(400, "run_name 은 영문·숫자로 시작, 영문·숫자·._- 80자 이내")
    local_url = str(body.get("local_url") or "").rstrip("/")
    if not local_url or not _is_loopback(local_url):
        local_url = LOCAL_SERVER_URL            # 루프백(ssh -R)이 아닌 주소는 무시한다
    try:
        extra = validate_extra_args(body.get("extra_args"))
    except ValueError as e:
        return _err(400, str(e))
    if MODEL["model"] is not None:
        return _err(409, "모델이 적재돼 있습니다 — 먼저 POST /model/unload (학습과 서빙은 VRAM 을 나눌 수 없음)", "state")
    if not _transition("idle", "training"):
        return _err(409, f"지금 상태가 {STATE['state']} 입니다 — idle 일 때만 학습을 시작합니다", "state")
    threading.Thread(target=_train_worker, args=(run_name, local_url, extra), daemon=True).start()
    return jsonify({"started": True, "run_name": run_name, "local_url": local_url, "extra_args": extra,
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
    up = TRAIN.get("upload") or {}
    return jsonify({"state": STATE["state"], **{k: v for k, v in TRAIN.items() if k != "proc"},
                    "running": TRAIN.get("proc") is not None, "progress": _parse_progress(lines),
                    "upload_failed": up.get("failed", []), "log_tail": lines, "run_json": run_json})


@app.post("/train/stop")
def train_stop():
    proc = TRAIN.get("proc")
    if STATE["state"] != "training" or proc is None:
        return _err(409, "실행 중인 학습이 없습니다", "state")
    TRAIN["stopped"] = True
    gone = kill_process_group(proc)
    # 상태는 워커 스레드가 프로세스 그룹 종료를 확인한 뒤 idle 로 돌린다 — 여기서 잠깐 기다려 준다
    for _ in range(50):
        if STATE["state"] == "idle":
            break
        time.sleep(0.2)
    return jsonify({"stopped": gone, "state": STATE["state"]}), (200 if gone else 500)


@app.post("/model/merge")
def model_merge():
    body = request.get_json(silent=True) or {}
    run_name = _valid_name(body.get("run_name") or CFG.get("RUN_NAME") or "run1")
    if not run_name:
        return _err(400, "run_name 형식 오류 (영문·숫자로 시작, ._- 만)")
    out_name = _valid_name(body.get("out_name") or _default_merged_name(run_name))
    if not out_name:
        return _err(400, "out_name 형식 오류 (영문·숫자로 시작, ._- 만)")
    mode = str(body.get("mode") or "gpu-nf4")
    if mode not in ("gpu-nf4", "cpu-bf16"):
        return _err(400, "mode 는 gpu-nf4 또는 cpu-bf16")
    ap = body.get("adapter_path")
    if ap is not None and not _under(Path(str(ap)).expanduser(), PERSIST_DIR):
        return _err(400, f"adapter_path 는 {PERSIST_DIR} 아래만 허용")
    if MODEL["model"] is not None:
        return _err(409, "모델이 적재돼 있습니다 — 먼저 POST /model/unload", "state")
    if not _transition("idle", "merging"):
        return _err(409, f"지금 상태가 {STATE['state']} 입니다", "state")
    threading.Thread(target=_merge_worker, args=(run_name, out_name, ap, mode), daemon=True).start()
    return jsonify({"started": True, "run_name": run_name, "out_name": out_name, "mode": mode,
                    "out_dir": str(MODELS_DIR / out_name), "status": "GET /model/merge/status"}), 202


@app.get("/model/merge/status")
def merge_status():
    return jsonify({"state": STATE["state"], **MERGE})


@app.post("/model/load")
def model_load():
    body = request.get_json(silent=True) or {}
    source = str(body.get("source") or "merged")
    name = body.get("name") or body.get("run_name")
    if name is not None and not _valid_name(name):
        return _err(400, "name/run_name 형식 오류 (영문·숫자로 시작, ._- 만)")
    ap = body.get("adapter_path")
    if ap is not None and not _under(Path(str(ap)).expanduser(), PERSIST_DIR):
        return _err(400, f"adapter_path 는 {PERSIST_DIR} 아래만 허용")
    if not _transition("idle", "loading"):
        return _err(409, f"지금 상태가 {STATE['state']} 입니다 — idle 일 때만 적재합니다", "state")
    try:
        info = load_model(source, name, ap)
    except Exception as e:  # noqa: BLE001
        unload_model()
        with LOCK:
            _set_state("idle")
        return _err(400, str(e)[:400])
    if not _transition("loading", "serving"):     # 원자적 — 그 사이 다른 전이가 끼어들 수 없다
        unload_model()
        return _err(500, "상태 전이 실패", "server_error")
    return jsonify({"loaded": True, **info})


@app.post("/model/unload")
def model_unload():
    with GEN_LOCK:
        if STATE["state"] != "serving":
            return _err(409, f"serving 상태에서만 내릴 수 있습니다 (지금 {STATE['state']})", "state")
        unload_model()
        _transition("serving", "idle")
    return jsonify({"unloaded": True, "vram": _vram()})


@app.get("/models")
def models_list():
    merged = sorted(p.name for p in MODELS_DIR.iterdir() if (p / "config.json").is_file()) if MODELS_DIR.exists() else []
    outs = PERSIST_DIR / "outputs"
    adapters = sorted(p.parent.parent.name for p in outs.glob("*/adapter/adapter_config.json")) if outs.exists() else []
    return jsonify({"merged": merged, "adapters": adapters})


@app.get("/v1/models")
def v1_models():
    data = [{"id": MODEL["name"], "object": "model", "owned_by": "univus"}] if MODEL["name"] else []
    return jsonify({"object": "list", "data": data})


@app.post("/v1/chat/completions")
def v1_chat():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _err(400, "JSON 본문이 필요합니다")
    if body.get("stream"):
        return _err(400, "stream 은 지원하지 않습니다 — stream:false 로 보내세요")
    if STATE["state"] != "serving" or MODEL["model"] is None:
        return _err(503, f"모델이 서빙 상태가 아닙니다 (state={STATE['state']}) — POST /model/load", "server_error")
    try:
        with GEN_LOCK:
            if MODEL["model"] is None:
                return _err(503, "모델이 내려갔습니다", "server_error")
            return jsonify(generate_completion(body))
    except ValueError as e:
        return _err(400, str(e))
    except Exception as e:  # noqa: BLE001
        _log("serve.log", f"ERROR {type(e).__name__}: {str(e)[:200]}")
        return _err(500, f"생성 실패: {type(e).__name__}", "server_error")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("SERVER_HOST", "127.0.0.1"),
                    help="기본 127.0.0.1 — ssh -L 터널은 루프백이면 충분하다. 다른 주소로 열려면 UNIVUS_API_KEY 필수")
    ap.add_argument("--port", type=int, default=int(os.environ.get("SERVER_PORT", "8600")))
    ap.add_argument("--pidfile", default="")
    args = ap.parse_args()
    if not API_KEY and args.host not in LOOPBACK:
        sys.exit(f"UNIVUS_API_KEY 가 비어 있는데 host={args.host} 로 열 수 없습니다 — 키를 넣거나 --host 127.0.0.1")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if args.pidfile:
        Path(args.pidfile).write_text(str(os.getpid()), encoding="utf-8")
    print(f"gpu_server  http://{args.host}:{args.port}  model_id={CFG.get('MODEL_ID')}  run={CFG.get('RUN_NAME')}  "
          f"persist={PERSIST_DIR}  scratch={SCRATCH_DIR}  local={LOCAL_SERVER_URL}  "
          f"auth={'on' if API_KEY else 'OFF (루프백 전용)'}", flush=True)
    print(f"  vram={_vram()}  disk(persist)={_disk(PERSIST_DIR)}", flush=True)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
