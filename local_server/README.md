# local_server — 내 PC 의 데이터·결과 서버

pod 는 저장 공간이 작고 밖으로 나갈 수 없으므로, **학습 데이터는 내 PC 가 들고 있고 pod 가 가져가며**, 학습 결과(어댑터·평가)는 **pod 가 내 PC 로 올려 둡니다**.
pod 에서 이 서버는 `ssh -R` 역터널을 타고 `http://127.0.0.1:8601` 로 보입니다(`scripts\tunnel.ps1`). 그래서 `127.0.0.1` 에만 묶습니다.

```powershell
.\.venv\Scripts\python.exe src\prepare_f9.py --n 1500 --seed 42   # data/f9/train.jsonl · eval.jsonl
.\local_server\run_local.ps1                                        # scripts\server.env 의 UNIVUS_API_KEY · LOCAL_PORT 사용
```

| 메서드 · 경로 | 설명 |
| --- | --- |
| `GET /health` | 데이터 폴더·파일 목록·인증 여부 |
| `GET /data/manifest` | `train.jsonl` `eval.jsonl` `README.md` 의 크기·sha256 — pod 가 받은 뒤 검증한다 |
| `GET /data/<name>` | 위 세 파일만. 다른 이름·경로는 404 |
| `POST /results/<run>/<path>` | pod 가 올리는 파일 → `results/<run>/<path>` (파일당 100MB, `.part` 로 받고 완료 시 교체) |
| `GET /results` | `results/` 트리 |

- `UNIVUS_API_KEY` 가 있으면 모든 요청에 `Authorization: Bearer <키>` 가 필요합니다(gpu_server 와 같은 키). 비워 두면 인증 없이 뜹니다 — 역터널 전용이지만 켜 두는 것을 권장합니다.
- `results/` 는 gitignore 입니다. 끝난 run 의 `results/<run>/adapter/` 가 **어댑터의 사본**이므로 pod 홈이 지워져도 복구할 수 있습니다.
- 평가셋은 팀이 같은 파일을 써야 비교가 되므로 `data/f9/` 를 함부로 다시 생성하지 않습니다(같은 `--seed` 면 같은 파일).
