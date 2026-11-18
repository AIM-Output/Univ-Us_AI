# gpu/qwen3-4b-v2 — AI 팀 F9 모델(Qwen3-4B QLoRA) 이어서 학습

**처음이면 [docs/F9_학습_가이드.md](docs/F9_학습_가이드.md) 부터 읽는다.** 사람별 할 일, 데이터 쓰는 법, 학습 방법이 거기에 있다.

| 항목 | 값 |
| --- | --- |
| 출발점 | `gpu/qwen3-4b` (`f69b052`) — 코드와 학습 설정은 그대로 두고 **데이터를 늘려** 다시 학습한다 |
| 모델 | `Qwen/Qwen3-4B` · 4-bit(nf4) QLoRA · LoRA r=16 · MAX_SEQ 2560 · batch 1 × accum 8 · completion-only 손실 |
| 학습 데이터 | v1(AI 팀, seed 42) + v2(AI 팀, seed 7, 1만 행 — 이번에 처음 학습) + v3(우리 말투 틀, seed 11) |
| 시험 | `eval.jsonl`(AI 팀 358턴, 기준선 77.7%) + `real_dev` · `real_test`(우리가 쓴 실제 문장) |
| 학습 장소 | 하람 데스크톱(RTX 5060 8GB). 8GB 에 안 들어가면 RunPod (가이드 8장 · 부록 C) |
| 기록 | [docs/f9/RESULTS.md](docs/f9/RESULTS.md) · `docs/eval/<run>/` · Release `<run>` |

## 이 브랜치에서 더한 것

| 파일 | 무엇 |
| --- | --- |
| `data_src/real/<이름>.tsv` | 사람이 쓴 실제 문장 — 시험 문제(학습에 넣지 않는다) |
| `data_src/frames/<이름>.tsv` | 사람이 쓴 말투 틀 — v3 학습 데이터의 재료 |
| `src/make_real.py` | 실제 문장 검사(`--check`) · 검수용 표(`--show`) · `real_dev.jsonl` / `real_test.jsonl` 만들기 |
| `src/prepare_f9_v3.py` | 말투 틀 검사(`--check`) · 표본 보기(`--show`) · `extra_v3.jsonl` 만들기 — 정답 계산은 v2 의 함수를 쓴다 |
| `src/datasrc.py` | 위 두 스크립트가 같이 쓰는 TSV 읽기 · 표기 도우미 |
| `scripts/peek_rows.py` | 데이터 파일에서 문장을 뽑아 표로 보기 (생성 문장 검수) |
| `scripts/peek_fail.py` | 평가 결과에서 틀린 턴만 뽑아 보기 (실패 분석) |
| `tests/test_f9_data_tools.py` | 위 도구들의 자가 점검 — `python -m unittest discover -s tests` |
| `.github/workflows/f9-data-check.yml` | `data_src/` 를 고친 PR 을 자동 검사 |
| `requirements.lock.txt` | AI 팀 1차 학습 때의 라이브러리 버전 |
| `docs/` | 가이드 · 성적표 |

## 자주 쓰는 명령

```bash
# 데이터 (순서 그대로 — v2 는 eval.jsonl 이 먼저 있어야 한다)
python src/prepare_f9.py --n 1500 --seed 42
python src/prepare_f9_v2.py --n 10000 --seed 7
python src/prepare_f9_v3.py --n 3000 --seed 11
python src/prepare_f9.py --n 1500 --seed 42 --extra data/f9/extra_v2.jsonl data/f9/extra_v3.jsonl
python src/make_real.py

# 검사
python src/make_real.py --check
python src/prepare_f9_v3.py --check
python -m unittest discover -s tests

# 학습 · 평가 (Git Bash)
RUN_NAME=f9-r1 bash train.sh
RUN_NAME=f9-r1 bash eval.sh
```

## AI 팀 브랜치(`gpu/qwen3-4b`)와 다른 점

- `config.env`: `TARGET=local` · `EPOCHS=1` · `SAVE_STEPS=100` · `RUN_NAME=f9-dev`. 나머지 학습 설정은 같다.
- 기존 공통 파일(`src/common.py` · `prepare_f9.py` · `prepare_f9_v2.py` · `train.py` · `eval.py` · `train.sh` · `eval.sh` · setup 스크립트 · `tools/`)은 **고치지 않았다.** 그래서 `git merge common` 이 그대로 된다.
- 새 도구는 이 브랜치에만 있다. 다른 모델 브랜치에서도 쓰게 되면 `common` 으로 옮긴다.
- AI 팀의 이 브랜치 설명(학교 서버 메모리 예산 등)은 `git show gpu/qwen3-4b:README_BRANCH.md` 로 볼 수 있다.
