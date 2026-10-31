# gpu_server — pod 에서 도는 Flask 하나로 학습 · 병합 · 추론

학교 GPU 서버(pod)는 **홈 10GB 중 3.1GB 만 남아 있고**(2026-10-30 실측), overlay 는 넓지만 pod 재시작 때 지워집니다.
그래서 pod 에는 **코드와 병합한 4-bit 가중치만** 두고, 학습 데이터와 결과는 내 PC 의 `local_server` 가 들고 있습니다.
pod 는 바깥으로 포트를 열 수 없다고 보고(열린 건 SSH 30562 뿐) **모든 통신을 SSH 터널 하나로** 묶습니다.

```
[내 PC]                                                  [pod]
WE-Meet 앱 ─ http://127.0.0.1:8600/v1 ─┐
local_server (Flask :8601)              │  ssh -L 8600  →  gpu_server (Flask :8600)
  data/f9/*.jsonl  ←──── GET /data/… ───┼─ ssh -R 8601 ←   ├ train.sh / eval.sh   (state=training)
  results/<run>/   ←── POST /results/… ─┘                  ├ 어댑터 병합 → nf4     (state=merging)
                                                            └ /v1/chat/completions (state=serving)
```

> **LoRA 어댑터만으로는 추론이 안 됩니다.** 어댑터(수십 MB)는 베이스 가중치 위에 얹는 차분이라 베이스(또는 병합본)가 반드시 같이 있어야 합니다.
> 홈 3GB 예산에 맞추는 방법이 `POST /model/merge` 입니다 — 베이스 + 어댑터를 병합한 뒤 **nf4 4-bit 로 저장**하면 Qwen3-4B 가 약 2.7~2.9GB 가 되어 홈에 영구 보관됩니다(임베딩은 bf16 그대로라 0.65B/param 보다 큽니다).

## 상태 기계 — 한 번에 한 가지만

9.5GiB 한 조각을 학습과 서빙이 나눌 수 없으므로 서버는 항상 상태 하나를 가집니다.

| 상태 | 들어가는 길 | 나오는 길 | 이때 되는 것 |
| --- | --- | --- | --- |
| `idle` | 시작 · 작업 끝 | | 학습·병합·적재 시작 |
| `training` | `POST /train/start` | 끝나면 자동 `idle` | `/train/status` `/train/stop` |
| `merging` | `POST /model/merge` | 끝나면 자동 `idle` | `/model/merge/status` |
| `loading` → `serving` | `POST /model/load` | `POST /model/unload` | **`/v1/chat/completions`** |

`serving` 이 아닐 때 `/v1/chat/completions` 는 **503** 을 돌려줍니다. 학습하려면 먼저 `unload`.

## 운영 절차 (처음부터 끝까지)

| # | 어디서 | 명령 | 설명 |
| :---: | --- | --- | --- |
| 0 | 내 PC | `copy scripts\server.env.example scripts\server.env` 후 값 채움 · `config.local.env` 에 `UNIVUS_API_KEY=<같은 값>` | 두 서버가 같은 Bearer 키를 쓴다 |
| 1 | 내 PC | `git switch gpu/qwen3-4b` → `.\scripts\deploy_to_server.ps1 -IncludeLocalEnv` | 브랜치의 추적 파일을 `~/univus-train` 로 (git archive → scp → tar) |
| 2 | pod | `cd ~/univus-train; bash gpu_server/setup_server.sh` | venv(overlay) + torch + requirements + 가중치 prefetch. **pod 재시작 후마다** |
| 3 | pod | `bash gpu_server/run_server.sh start` | `:8600` 에 Flask. `status` `stop` `log` |
| 4 | 내 PC | `.\scripts\tunnel.ps1` (창 하나 계속 켜 둠) | `-L 8600` + `-R 8601`, 끊기면 재연결 |
| 5 | 내 PC | `.\.venv\Scripts\python.exe src\prepare_f9.py --n 1500 --seed 42` → `.\local_server\run_local.ps1` | 데이터 서버 `:8601` |
| 6 | 내 PC | 아래 curl 로 `/train/start` → `/train/status` 로 지켜봄 | 끝나면 어댑터·평가가 `results/<run>/` 에 내려와 있다 |
| 7 | 내 PC | `/model/merge` → `/model/load {"source":"merged"}` | 홈에 nf4 병합본 생성 후 서빙 |
| 8 | 내 PC | WE-Meet `.env` 설정 (아래) | 앱이 pod 모델로 답한다 |

### curl 예시 (내 PC, 터널이 떠 있는 상태 · `KEY` 는 UNIVUS_API_KEY)

```bash
H='-H "Authorization: Bearer $KEY" -H "Content-Type: application/json"'
curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:8600/health | python -m json.tool

# 학습 (데이터는 pod 가 ssh -R 로 내 PC 에서 받아 간다)
curl -s -X POST -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
     -d '{"run_name":"qwen3-4b-f9"}' http://127.0.0.1:8600/train/start
curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:8600/train/status | python -c "import json,sys;d=json.load(sys.stdin);print(d['state'],d['phase'],d['progress']);print('\n'.join(d['log_tail'][-8:]))"
curl -s -X POST -H "Authorization: Bearer $KEY" http://127.0.0.1:8600/train/stop

# 병합(nf4, 홈에 ≈2.6GB) → 적재 → 추론
curl -s -X POST -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"run_name":"qwen3-4b-f9"}' http://127.0.0.1:8600/model/merge
curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:8600/model/merge/status
curl -s -X POST -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"source":"merged"}' http://127.0.0.1:8600/model/load
#   병합 전에 어댑터로 바로 써 보려면:  -d '{"source":"adapter","run_name":"qwen3-4b-f9"}'  (베이스를 4-bit 로 받아 얹는다 — 가중치는 overlay 캐시)

# OpenAI 호환 — WE-Meet 가 보내는 모양 그대로 (tools 는 tools/f9_tools.json 의 "tools" 배열)
curl -s -X POST -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" http://127.0.0.1:8600/v1/chat/completions -d '{
  "model": "univus", "temperature": 0,
  "messages": [{"role":"system","content":"오늘은 2026-10-30(금) 14:00 입니다. 주 시작은 월요일입니다."},
               {"role":"user","content":"이번 주 마감 뭐 남았어?"}],
  "tools": [{"type":"function","function":{"name":"list_events","description":"기간 안의 일정·마감 조회",
             "parameters":{"type":"object","properties":{"from":{"type":"string"},"to":{"type":"string"},
             "kinds":{"type":"array","items":{"type":"string"}}},"required":["from","to"]}}}]
}'
# → choices[0].message.tool_calls[0].function = {"name":"list_events","arguments":"{\"from\": \"2026-10-26\", ...}"}, finish_reason "tool_calls"
```

`stream: true` 는 400(지원 안 함), `response_format: {"type":"json_object"}` 는 받아서 system 에 "JSON 만 출력" 을 덧붙입니다.
`tool_choice: "none"` 이면 도구 없이 렌더합니다. 요청 본문은 로그에 남기지 않고 토큰 수·지연만 `logs/serve.log` 에 적습니다.

### WE-Meet 쪽 `.env`

```
LLM_MAIN_BASE_URL=http://127.0.0.1:8600/v1
LLM_MAIN_API_KEY=<UNIVUS_API_KEY 와 같은 값>
LLM_MAIN_MODEL=univus-f9            # 서버는 이 값을 무시한다 (적재된 모델 하나뿐)
```

터널(`scripts\tunnel.ps1`)이 떠 있어야 하고, pod 쪽 상태가 `serving` 이어야 합니다. 학습 중에는 F1·F11·F12·C4 의 LLM 보조 기능이 "LLM 미설정" 과 같은 경로(규칙만)로 동작합니다.

## 저장 공간 — 무엇이 어디에 남나

| 위치 | 남나 | 담는 것 | 크기 |
| --- | :---: | --- | --- |
| `~/univus-train-out/models/<run>-merged-nf4/` | ✅ 홈 | **병합 nf4 가중치 + 토크나이저** — 서빙용 | Qwen3-4B ≈ **2.7~2.9GB** (선형층 0.56B/param + 임베딩 2B/param · 0.5B 실측 469MB) |
| `~/univus-train-out/outputs/<run>/adapter/` | ✅ 홈 | LoRA 어댑터 | 30~60MB |
| `~/univus-train-out/outputs/<run>/` | ✅ 홈 | run.json · eval_*.md/json · compare.md · checkpoint-*(최근 2개) | 체크포인트 포함 수백 MB — 끝난 run 의 `checkpoint-*` 는 지워도 됨 |
| `~/univus-train-out/data/f9/` · `logs/` | ✅ 홈 | 받은 JSONL · 서버/학습 로그 | 수 MB |
| `~/.venv/univus-train/.venv` | ❌ overlay | venv (torch 포함) | ~8GB |
| `~/.venv/univus-train/hf_cache` | ❌ overlay | 베이스 가중치(학습·`adapter` 적재·병합 때 필요) | Qwen3-4B bf16 ≈ 8GB |
| `~/.venv/univus-train/merge_tmp` | ❌ overlay | 병합 중간 bf16 (끝나면 삭제) | ≈ 8GB |

**홈 예산 3.1GB** 안에 병합본(≈2.8GB) 하나 + 어댑터 + 결과가 들어갑니다. 병합본을 두 개 두면 넘칩니다 — `/model/merge` 는 저장 전에 여유 공간을 재고 부족하면 거절합니다.
`/health` 의 `disk.persist.free_gb` 로 수시로 확인하세요.

## pod 재시작 후 복구 순서

1. `bash gpu_server/setup_server.sh` — venv 와 HF 캐시를 다시 만든다(정상). 홈의 병합본·어댑터·데이터는 그대로 있다.
2. `bash gpu_server/run_server.sh start`
3. 서빙만 할 거면 `POST /model/load {"source":"merged"}` — 병합본은 홈에 있으므로 **가중치 재다운로드 없이** 바로 뜬다.
   다시 학습·병합하려면 베이스 가중치가 필요하므로 setup 의 prefetch 가 끝나 있어야 한다.

## 파일

| 파일 | 역할 |
| --- | --- |
| `app.py` | Flask — 상태 기계 · `/train/*` · `/model/*` · `/v1/chat/completions` (`src/common.py` 의 렌더·파서를 그대로 씀) |
| `setup_server.sh` | pod 환경 (처음 1회 + 재시작 후). venv 는 `train.sh` 와 공유 |
| `run_server.sh` | `start` `stop` `status` `restart` `log` — pidfile `~/univus-train-out/gpu_server.pid` |
| `../scripts/deploy_to_server.ps1` | 내 PC → pod 코드 배포 |
| `../scripts/tunnel.ps1` | SSH 터널 유지 |
| `../local_server/` | 내 PC 데이터·결과 서버 |

설정은 `config.env`(브랜치) → `config.local.env`(비밀·pod 전용) → 환경변수 순으로 읽습니다. `app.py` 안에 모델 이름은 없습니다.
