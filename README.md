# archive — 더 이상 쓰지 않는 과거 브랜치 모음

2026-11-17 에 아래 브랜치 4개를 이 브랜치 하나로 묶고 원래 브랜치는 지웠다. 각 브랜치의 **마지막 트리가 폴더로** 들어 있고,
**커밋 이력도 그대로 남아 있다**(이 브랜치의 merge 부모) — `git log --first-parent` 로 폴더별 머지 커밋을, `git log <sha>` 로 원래 이력을 볼 수 있다.

| 폴더 | 원래 브랜치 | 마지막 커밋 | 무엇이었나 | 왜 접었나 |
| --- | --- | --- | --- | --- |
| `collect_ai-agent_logan0741/` | `collect_ai-agent_logan0741` | `fcfb783` 2026-09-01 (logan0741) | 2026-09 1차 개발 — 공지 수집 크롤러, 학습 폴더 초안, 문서(`AI_에이전트_개발계획서.md`, `docs/GPU-환경과-모델-선정.md`, `docs/서버-작업환경-설정.md`) | 학습 코드는 `common`/`gpu/*` 로 새로 만들었고 크롤러는 제품(WE-Meet)이 규칙 기반으로 직접 구현 |
| `feature-pic_ai-agent/` | `feature/pic_ai-agent` | `0337c78` 2026-09-14 (thisisuuk) | 강의계획서 OCR 골격 — Donut 기반 pdf→image, Pydantic 스키마, dataset/train/inference 스크립트 | 강의계획서 기능이 2026-09-13 제품에서 빠짐. 문서 OCR 은 `gpu/donut-ocr`(SynthDoG-ko 미세조정, CER 5.2%) 로 대체 |
| `gpu-hyperclovax-1.5b/` | `gpu/hyperclovax-1.5b` | `571b392` 2026-11-16 (logan0741) | F9 비교 후보 — HyperCLOVAX-SEED-Text-Instruct-1.5B, bf16 LoRA. `common` 과 다른 파일은 `config.env` `setup.sh` `README_BRANCH.md` 뿐 | 2026-10-30 Qwen3-4B 로 결정(9.5GiB 에서 학습·서빙 둘 다 가능, tool calling 템플릿 내장). 학습 결과 없음 |
| `kaggle-qwen3-8b/` | `kaggle/qwen3-8b` | `dc3217f` 2026-11-16 (logan0741) | F9 비교 후보 — Qwen3-8B 4-bit fp16, 캐글 T4×2 용 노트북(`kaggle/`)·`setup_kaggle.md` | 같은 이유. 학습 결과 없음 |

## 되살리기

이력이 남아 있으므로 브랜치만 다시 만들면 된다.

```bash
git branch gpu/hyperclovax-1.5b 571b392     # 또는 kaggle/qwen3-8b dc3217f · feature/pic_ai-agent 0337c78 · collect_ai-agent_logan0741 fcfb783
```

지금 쓰는 브랜치와 가중치는 [`main` README](../../blob/main/README.md) 의 "저장소 구조" 표를 본다.
